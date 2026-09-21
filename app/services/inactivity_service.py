"""
Servicio de expiración de contexto conversacional por inactividad.

Cada dispositivo de WhatsApp (numero_paciente) recibe su propio temporizador
asyncio completamente independiente. Cuando el usuario no escribe durante el
tiempo configurado (SESSION_TTL_SECONDS, por defecto 900 s = 15 min), el
servicio:
  1. Detecta si la conversación quedó a mitad de un agendamiento de cita.
  2. Envía el mensaje contextual correspondiente.
  3. Purga el hilo en PostgreSQL y limpia cachés en memoria.
  4. Registra el cierre en la base de datos de auditoría.

Cada número funciona de forma completamente aislada: los temporizadores,
cachés y estados de LangGraph son indexados estrictamente por numero_paciente
y jamás se comparten entre usuarios distintos.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Mensajes de cierre
# ─────────────────────────────────────────────────────────────────────────────

MENSAJE_INACTIVIDAD_CITA_INCOMPLETA = (
    "Por no haber completado la información requerida, no pudimos continuar "
    "con el agendamiento de tu cita. 😔🦷\n\n"
    "Si deseas agendar una nueva cita o consultar algún servicio, solo escríbeme "
    "y con mucho gusto te ayudaré. ¡Aquí estaré para cuando lo necesites! 😊"
)

MENSAJE_INACTIVIDAD_GENERAL = (
    "He cerrado nuestra conversación por inactividad. 🦷\n\n"
    "Recuerda que aquí estaré cuando desees agendar una cita, consultar nuestros "
    "servicios o resolver cualquier duda. ¡Solo escríbeme cuando lo necesites! 😊👋"
)

# ─────────────────────────────────────────────────────────────────────────────
# Indicadores que revelan que el bot estaba en medio de un agendamiento
# ─────────────────────────────────────────────────────────────────────────────
_MARCADORES_CITA_PENDIENTE = [
    "📋 *propuesta de cita:",
    "📋 *propuesta de cambio de cita:",
    "¿confirmas estos datos para agendar",
    "¿confirmas estos datos para reprogramar",
    "para gestionar tu cita, por favor indícame tu",
    "¡con el mayor gusto! para registrar tu cita",
    "indica la fecha y hora de tu preferencia",
    "¿para qué fecha te gustaría la cita",
    "¿qué día y horario te queda mejor",
    "¿con cuál de nuestros especialistas",
    "indícame qué servicio o tratamiento necesitas",
    "para agendar tu cita, por favor indícame tu",
    "¿cuándo te gustaría la cita",
    "¿qué horario prefieres",
]

_MARCADORES_CITA_EXITOSA = [
    "cita agendada exitosamente",
    "cita agendada con éxito",
    "ha sido agendada con éxito",
    "tu cita ha sido registrada",
    "cita confirmada",
    "tu cita fue cancelada",
    "tu cita ha sido reprogramada",
    "cita reprogramada exitosamente",
    "tu cita fue agendada",
    "quedó agendada para",
    "cita cancelada exitosamente",
    "agendada exitosamente",
    "agendada con éxito",
]


class InactivityService:
    """
    Gestor de temporizadores de inactividad por usuario de WhatsApp.

    Cada numero_paciente mantiene su propia tarea asyncio independiente.
    Los temporizadores se crean, renuevan y cancelan estrictamente por
    numero_paciente sin ningún tipo de estado compartido entre usuarios.
    """

    def __init__(self) -> None:
        # {numero_paciente: asyncio.Task}
        self._tasks: dict[str, asyncio.Task] = {}

    # ──────────────────────────────────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────────────────────────────────

    def touch(self, numero_paciente: str, ttl_seconds: int = 900) -> None:
        """
        Reinicia (o crea) el temporizador de inactividad para numero_paciente.

        Debe invocarse cada vez que el bot termina de responder al usuario,
        para reiniciar la cuenta regresiva de 15 minutos.
        """
        self._cancel(numero_paciente)
        self._tasks[numero_paciente] = asyncio.create_task(
            self._expiry_worker(numero_paciente, ttl_seconds),
            name=f"inactivity-{numero_paciente}",
        )

    def cancel(self, numero_paciente: str) -> None:
        """Cancela el temporizador de inactividad (p. ej. conversación escalada)."""
        self._cancel(numero_paciente)

    async def stop_all(self) -> None:
        """Cancela y espera a que terminen todos los temporizadores activos.
        Se invoca en el shutdown de FastAPI para evitar tareas huérfanas.
        """
        tasks = list(self._tasks.items())
        for _num, task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*[t for _, t in tasks], return_exceptions=True)
        self._tasks.clear()

    async def procesar_expiraciones_pendientes(self, ttl_seconds: int = 900) -> None:
        """
        Job periódico del Sweeper: consulta en PostgreSQL todas las sesiones
        cuya inactividad supere ttl_seconds (900s = 15 min), y ejecuta
        el cierre ordenado (notificación a WhatsApp, registro en auditoría y purga).
        Garantiza que incluso tras un reinicio del contenedor o fallo de red,
        no queden sesiones huérfanas en la base de datos.
        """
        from app.session.postgres_checkpointer import get_checkpointer_instance
        checkpointer = get_checkpointer_instance()
        if not checkpointer:
            return

        try:
            expiradas = await checkpointer.obtener_sesiones_expiradas(ttl_seconds)
            if not expiradas:
                return

            logger.info(
                "[Inactivity Sweeper] Detectadas %d sesiones expiradas en PostgreSQL: %s",
                len(expiradas),
                expiradas,
            )
            for thread_id in expiradas:
                self._cancel(thread_id)
                try:
                    await self._handle_expiry(thread_id)
                except Exception as exc:
                    logger.error(
                        "[Inactivity Sweeper] Error procesando cierre para %s: %s",
                        thread_id,
                        exc,
                        exc_info=True,
                    )
        except Exception as e:
            logger.error(
                "[Inactivity Sweeper] Error al consultar sesiones expiradas: %s",
                e,
                exc_info=True,
            )

    # ──────────────────────────────────────────────────────────────────────
    # Lógica interna
    # ──────────────────────────────────────────────────────────────────────

    def _cancel(self, numero_paciente: str) -> None:
        existing = self._tasks.pop(numero_paciente, None)
        if existing and not existing.done():
            existing.cancel()

    async def _expiry_worker(self, numero_paciente: str, ttl_seconds: int) -> None:
        """
        Espera ttl_seconds y luego ejecuta el cierre de sesión para este usuario.
        Si se cancela antes (porque el usuario volvió a escribir), simplemente termina.
        """
        try:
            await asyncio.sleep(ttl_seconds)
        except asyncio.CancelledError:
            return  # El usuario escribió antes de que expirara el tiempo

        # Remover la tarea del registro (ya expiró)
        self._tasks.pop(numero_paciente, None)

        try:
            await self._handle_expiry(numero_paciente)
        except Exception as exc:
            logger.error(
                "[Inactivity] Error manejando expiración de sesión para %s: %s",
                numero_paciente,
                exc,
                exc_info=True,
            )

    async def _handle_expiry(self, numero_paciente: str) -> None:
        """Cierra la sesión del usuario tras la expiración del temporizador."""
        # Importaciones lazy para evitar ciclos de importación
        from app.clients.evolution_client import evolution_client
        from app.clients.dotnet_client import dotnet_client
        from app.session.postgres_checkpointer import get_checkpointer_instance
        from app.services.chat.message_processor import (
            is_escalated as _is_escalated,
            _USER_LAST_ACTIVE,
        )
        from app.services.chat.chat_orchestrator import _USER_PUSH_NAMES
        from app.api.routes.webhook import _PATIENT_CONTEXT_CACHE
        from app.services.whatsapp_identity import (
            obtener_telefono_canonico,
            obtener_destino_envio,
        )

        logger.info(
            "[Inactivity] Sesión expirada por inactividad para %s. Evaluando estado...",
            numero_paciente,
        )

        # No interrumpir conversaciones en atención humana
        try:
            if await _is_escalated(numero_paciente):
                logger.info(
                    "[Inactivity] Conversación de %s está escalada/en atención humana. "
                    "Se omite el cierre automático por inactividad.",
                    numero_paciente,
                )
                return
        except Exception:
            pass

        # Determinar el mensaje de cierre adecuado
        cita_en_curso = await self._detect_incomplete_appointment(numero_paciente)
        mensaje_cierre = (
            MENSAJE_INACTIVIDAD_CITA_INCOMPLETA
            if cita_en_curso
            else MENSAJE_INACTIVIDAD_GENERAL
        )

        logger.info(
            "[Inactivity] %s → %s",
            numero_paciente,
            "CITA INCOMPLETA" if cita_en_curso else "CIERRE GENERAL",
        )

        # 1. Enviar mensaje de cierre por WhatsApp
        try:
            await evolution_client.enviar_mensaje(numero_paciente, mensaje_cierre)
        except Exception as e:
            logger.warning(
                "[Inactivity] No se pudo enviar mensaje de cierre a %s: %s",
                numero_paciente,
                e,
            )

        # 2. Registrar el mensaje de cierre en la base de datos de auditoría
        try:
            asyncio.create_task(
                dotnet_client.registrar_mensaje(
                    chat_identifier=numero_paciente,
                    rol="CHATBOT",
                    contenido=mensaje_cierre,
                )
            )
        except Exception as e:
            logger.warning(
                "[Inactivity] No se pudo registrar mensaje de cierre en DB para %s: %s",
                numero_paciente,
                e,
            )

        # 3. Purgar el hilo de LangGraph en PostgreSQL
        canon = obtener_telefono_canonico(numero_paciente)
        targets_clear = {numero_paciente, canon}
        dest_envio = obtener_destino_envio(numero_paciente)
        if dest_envio:
            targets_clear.add(dest_envio)

        checkpointer = get_checkpointer_instance()
        for t in targets_clear:
            try:
                if checkpointer:
                    await checkpointer.clear_thread(t)
                dotnet_client.limpiar_cache_conversacion(t)
            except Exception as e:
                logger.debug(
                    "[Inactivity] Error limpiando hilo %s: %s", t, e
                )

        # 4. Limpiar cachés en memoria — exclusivamente para este usuario
        _PATIENT_CONTEXT_CACHE.pop(numero_paciente, None)
        _PATIENT_CONTEXT_CACHE.pop(canon, None)
        _USER_PUSH_NAMES.pop(numero_paciente, None)
        _USER_LAST_ACTIVE.pop(numero_paciente, None)

        logger.info(
            "[Inactivity] Contexto de %s limpiado completamente. "
            "El próximo mensaje del usuario iniciará una sesión nueva.",
            numero_paciente,
        )

    async def _detect_incomplete_appointment(self, numero_paciente: str) -> bool:
        """
        Analiza los mensajes recientes del hilo de LangGraph para determinar
        si el bot había presentado una propuesta de cita o estaba recopilando
        datos para agendar, y la cita NO fue completada.

        Retorna True si hay una cita incompleta, False en caso contrario.
        """
        try:
            from app.session.memory_store import get_thread_config
            from app.graph.builder import get_graph
            from langchain_core.messages import AIMessage

            config = get_thread_config(numero_paciente)
            snapshot = await get_graph().aget_state(config)
            if not snapshot or not snapshot.values:
                return False

            messages = snapshot.values.get("messages", [])
            if not messages:
                return False

            # Verificar los últimos 12 mensajes del bot para detectar el estado
            ai_messages_recientes = [
                m for m in messages[-12:]
                if isinstance(m, AIMessage) and m.content and isinstance(m.content, str)
            ]

            if not ai_messages_recientes:
                return False

            # 1. Inspección estructurada de ToolMessages de agendamiento
            from langchain_core.messages import ToolMessage
            for m in reversed(messages[-12:]):
                if isinstance(m, ToolMessage):
                    tool_name = getattr(m, "name", "")
                    content_str = str(getattr(m, "content", "")).lower()
                    if tool_name == "agendar_cita_tool" and ("¡cita confirmada" in content_str or "éxito" in content_str):
                        return False

            # 2. Si el último mensaje del bot indica éxito en la cita, NO hay cita incompleta
            ultimo_ai = ai_messages_recientes[-1]
            contenido_ultimo = (ultimo_ai.content or "").lower()
            for marcador in _MARCADORES_CITA_EXITOSA:
                if marcador in contenido_ultimo:
                    return False

            # 3. Verificar si el bot presentó una propuesta formal o solicitó datos para agendar
            for m in reversed(ai_messages_recientes):
                c = (m.content or "").lower()
                # Tarjeta de propuesta visual o solicitud explícita de confirmación
                if "propuesta de cita" in c or "propuesta de cambio" in c or "¿confirmas estos datos" in c:
                    return True
                # Búsqueda complementaria de marcadores de agendamiento en curso
                if any(marcador in c for marcador in _MARCADORES_CITA_PENDIENTE):
                    return True

            return False

        except Exception as exc:
            logger.debug(
                "[Inactivity] No se pudo determinar estado de cita para %s: %s",
                numero_paciente,
                exc,
            )
            return False


# Singleton del servicio — se reutiliza en todo el proceso
inactivity_service = InactivityService()
