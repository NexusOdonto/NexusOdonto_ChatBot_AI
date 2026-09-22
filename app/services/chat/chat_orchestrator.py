"""Orquestador de mensajes entrantes de chat: Anti-Spam, Debouncing (2.0s) y Filtro de Galimatías."""

import time
import re
import asyncio
import logging
from typing import Dict, List, Optional, Callable, Awaitable, Any

from app.clients.evolution_client import evolution_client
from app.clients.dotnet_client import dotnet_client
from app.services.inactivity_service import inactivity_service

logger = logging.getLogger(__name__)

# Parámetros de Anti-Spam y Debounce
SPAM_WINDOW_SECONDS = 5.0
SPAM_MAX_BURST = 3
SPAM_PENALTY_SECONDS = 15.0
SPAM_WARN_COOLDOWN = 30.0
DEBOUNCE_WAIT_SECONDS = 2.0

_SPAM_TIMESTAMPS: Dict[str, List[float]] = {}
_SPAM_BLOCKED_UNTIL: Dict[str, float] = {}
_SPAM_WARNED_AT: Dict[str, float] = {}

_USER_MESSAGE_BUFFERS: Dict[str, List[str]] = {}
_USER_DEBOUNCE_TASKS: Dict[str, asyncio.Task] = {}
_USER_PUSH_NAMES: Dict[str, str] = {}
_USER_PROCESSING: Dict[str, bool] = {}
_USER_CALLBACKS: Dict[str, Any] = {}
_USER_LAST_PROCESSED: Dict[str, tuple[str, float]] = {}

KEYWORDS_CLINICA = {
    "cita", "citas", "agendar", "agenda", "apartar", "programar", "horario", "horarios",
    "doctor", "doctora", "odontologo", "odontologa", "precio", "precios", "costo",
    "cuanto", "cuánto", "vale", "servicio", "servicios", "limpieza", "diseño",
    "ortodoncia", "bracket", "brackets", "calza", "resina", "extraccion", "extracción",
    "cordal", "cordales", "corona", "implante", "dolor", "urgencia", "emergencia",
    "cedula", "cédula", "documento", "nombre", "hola", "buenos", "buenas", "tardes",
    "dias", "días", "noches", "gracias", "cancelar", "modificar", "reprogramar",
    "confirmar", "asistir", "consulta", "telefono", "teléfono", "direccion", "dirección",
    "si", "sí", "no", "ok", "vale", "listo", "dale", "bien", "perfecto"
}


class ChatOrchestrator:
    """Gestiona la recepción de mensajes, filtrando spam y agrupando ráfagas continuas de WhatsApp."""

    @staticmethod
    def is_user_spam_blocked(phone: str) -> bool:
        """Determina si un usuario está penalizado temporalmente por ráfagas de spam."""
        return time.monotonic() < _SPAM_BLOCKED_UNTIL.get(phone, 0.0)

    @classmethod
    def check_and_trigger_spam(cls, phone: str) -> bool:
        """Verifica si los mensajes recientes del usuario constituyen spam en tiempo real."""
        now = time.monotonic()
        timestamps = _SPAM_TIMESTAMPS.get(phone, [])
        timestamps = [t for t in timestamps if now - t < SPAM_WINDOW_SECONDS]
        timestamps.append(now)
        _SPAM_TIMESTAMPS[phone] = timestamps

        if len(timestamps) > SPAM_MAX_BURST:
            _SPAM_BLOCKED_UNTIL[phone] = now + SPAM_PENALTY_SECONDS
            _USER_MESSAGE_BUFFERS.pop(phone, None)
            existing = _USER_DEBOUNCE_TASKS.pop(phone, None)
            if existing and not existing.done():
                existing.cancel()

            last_warn = _SPAM_WARNED_AT.get(phone, 0.0)
            if now - last_warn > SPAM_WARN_COOLDOWN:
                _SPAM_WARNED_AT[phone] = now
                logger.warning(f"[Anti-Spam] Ráfaga detectada para {phone}. Enviando advertencia.")
                spam_msg = (
                    "⚠️ Estás enviando muchos mensajes seguidos.\n"
                    "Por favor espera un momento y escribe tu consulta en un solo mensaje para poder atenderte bien. 😊"
                )
                asyncio.create_task(evolution_client.enviar_mensaje(phone, spam_msg))
            return True
        return False

    @staticmethod
    def is_nonsense_or_gibberish(texto: str) -> bool:
        """Detecta aporreo de teclado, caracteres repetidos o galimatías sin significado clínico."""
        raw = texto.strip()
        if not raw:
            return False

        palabras_lower = [p.strip(".,;:!?()[]\"'").lower() for p in raw.split()]
        if any(p in KEYWORDS_CLINICA for p in palabras_lower):
            return False

        solo_digitos = "".join(c for c in raw if c.isdigit())
        chars_sin_separadores = raw.replace(" ", "").replace(".", "").replace("-", "")
        if solo_digitos and len(solo_digitos) == len(chars_sin_separadores):
            if 7 <= len(solo_digitos) <= 12 or solo_digitos in ("1", "2", "3", "4", "5"):
                return False
            return True

        if len(palabras_lower) <= 3:
            if re.search(r"(.)\1{3,}", raw, re.IGNORECASE):
                return True
            if re.search(r"^(.{2,4})\1{2,}$", raw.replace(" ", ""), re.IGNORECASE):
                return True
            for p in palabras_lower:
                solo_letras = re.sub(r"[^a-záéíóúñ]", "", p)
                if len(solo_letras) >= 6:
                    vocales = len(re.findall(r"[aeiouáéíóú]", solo_letras))
                    if vocales == 0 or (len(solo_letras) >= 8 and (vocales / len(solo_letras)) < 0.15):
                        return True

        return False

    @classmethod
    def enqueue_message(
        cls,
        phone: str,
        message: str,
        push_name: str,
        process_callback: Callable[[str, str, str], Awaitable[None]],
    ) -> bool:
        """Encola el mensaje aplicando un worker serializado por usuario con debounce.
        Evita tareas en paralelo compitiendo en el lock y consolida todas las ráfagas
        en una sola respuesta coherente sin duplicados.
        """
        if cls.is_user_spam_blocked(phone) or cls.check_and_trigger_spam(phone):
            return False

        if push_name:
            _USER_PUSH_NAMES[phone] = push_name.strip()

        _USER_CALLBACKS[phone] = process_callback

        # Cancelar temporizador de inactividad mientras el usuario escribe
        inactivity_service.cancel(phone)

        # Enviar feedback de presencia inmediato en WhatsApp ("Escribiendo...")
        from app.clients.evolution_client import evolution_client
        asyncio.create_task(evolution_client.enviar_presencia(phone, "composing"))

        # Registrar de inmediato en la base de datos de auditoría
        asyncio.create_task(
            dotnet_client.registrar_mensaje(chat_identifier=phone, rol="USUARIO", contenido=message)
        )

        # Acumular en el buffer del usuario sin duplicar cadenas idénticas consecutivas
        clean_msg = message.strip()
        if phone not in _USER_MESSAGE_BUFFERS:
            _USER_MESSAGE_BUFFERS[phone] = []
        if not _USER_MESSAGE_BUFFERS[phone] or _USER_MESSAGE_BUFFERS[phone][-1] != clean_msg:
            _USER_MESSAGE_BUFFERS[phone].append(clean_msg)

        # Si ya hay un worker (debounce o procesando), solo acumular en buffer.
        # Evita dos workers en paralelo que producen respuestas duplicadas (flood).
        existing_task = _USER_DEBOUNCE_TASKS.get(phone)
        if _USER_PROCESSING.get(phone, False) or (existing_task and not existing_task.done()):
            if existing_task and not existing_task.done() and not _USER_PROCESSING.get(phone, False):
                # Reiniciar solo la ventana de debounce cancelando el sleep pendiente
                existing_task.cancel()
            else:
                logger.debug(
                    f"[Orchestrator] Usuario {phone} ya tiene worker activo. Mensaje acumulado en buffer."
                )
                return True

        async def _run_user_worker():
            try:
                await asyncio.sleep(DEBOUNCE_WAIT_SECONDS)
            except asyncio.CancelledError:
                # Si se canceló para reiniciar debounce, otro worker será creado por enqueue.
                return

            _USER_DEBOUNCE_TASKS.pop(phone, None)
            if _USER_PROCESSING.get(phone, False):
                return
            _USER_PROCESSING[phone] = True

            try:
                while True:
                    mensajes = _USER_MESSAGE_BUFFERS.pop(phone, [])
                    if not mensajes:
                        break

                    name = _USER_PUSH_NAMES.get(phone, "")
                    cb = _USER_CALLBACKS.get(phone)
                    texto_consolidado = " ".join(mensajes).strip()

                    now_mono = time.monotonic()
                    last_processed = _USER_LAST_PROCESSED.get(phone)
                    if last_processed:
                        prev_text, prev_time = last_processed
                        if prev_text == texto_consolidado and (now_mono - prev_time) < 4.0:
                            logger.info(
                                f"[Orchestrator] Texto idéntico ignorado por repetición rápida para {phone}: '{texto_consolidado[:40]}'"
                            )
                            continue

                    if texto_consolidado and cb:
                        _USER_LAST_PROCESSED[phone] = (texto_consolidado, now_mono)
                        logger.info(
                            f"[Debounce Flush] Mensajes agrupados ({len(mensajes)}) para {phone}: '{texto_consolidado}'"
                        )
                        await cb(phone, texto_consolidado, name)

                    if _USER_MESSAGE_BUFFERS.get(phone):
                        await asyncio.sleep(1.0)
            except Exception as e:
                logger.error(f"[Orchestrator] Error en loop de procesamiento de {phone}: {e}", exc_info=True)
            finally:
                _USER_PROCESSING[phone] = False
                # Relanzar solo si hay buffer con contenido nuevo
                if _USER_MESSAGE_BUFFERS.get(phone):
                    pending = _USER_DEBOUNCE_TASKS.get(phone)
                    if not pending or pending.done():
                        _USER_DEBOUNCE_TASKS[phone] = asyncio.create_task(_run_user_worker())

        _USER_DEBOUNCE_TASKS[phone] = asyncio.create_task(_run_user_worker())
        return True


chat_orchestrator = ChatOrchestrator()
