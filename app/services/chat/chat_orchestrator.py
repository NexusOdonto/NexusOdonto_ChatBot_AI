"""Orquestador de mensajes entrantes de chat: Anti-Spam y Debouncing (0.5s)."""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Callable, Awaitable, Any

from app.clients.evolution_client import evolution_client
from app.clients.dotnet_client import dotnet_client
from app.services.inactivity_service import inactivity_service

logger = logging.getLogger(__name__)

# Parámetros de Anti-Spam y Debounce
SPAM_WINDOW_SECONDS = 5.0
SPAM_MAX_BURST = 6
SPAM_PENALTY_SECONDS = 15.0
SPAM_WARN_COOLDOWN = 30.0
DEBOUNCE_WAIT_SECONDS = 0.5

_SPAM_TIMESTAMPS: Dict[str, List[float]] = {}
_SPAM_BLOCKED_UNTIL: Dict[str, float] = {}
_SPAM_WARNED_AT: Dict[str, float] = {}

_USER_MESSAGE_BUFFERS: Dict[str, List[str]] = {}
_USER_DEBOUNCE_TASKS: Dict[str, asyncio.Task] = {}
_USER_PUSH_NAMES: Dict[str, str] = {}
_USER_PROCESSING: Dict[str, bool] = {}
_USER_CALLBACKS: Dict[str, Any] = {}
_USER_LAST_PROCESSED: Dict[str, tuple[str, float]] = {}

class ChatOrchestrator:
    """Gestiona la recepción de mensajes, filtrando spam y agrupando ráfagas continuas de WhatsApp."""

    @staticmethod
    def is_user_spam_blocked(phone: str) -> bool:
        """Determina si un usuario está penalizado temporalmente por ráfagas de spam."""
        return time.monotonic() < _SPAM_BLOCKED_UNTIL.get(phone, 0.0)

    @classmethod
    def check_and_trigger_spam(cls, phone: str) -> bool:
        """Verifica si los mensajes recientes del usuario constituyen spam en tiempo real.
        Si se detecta flood, se penaliza silenciosamente sin enviar mensajes salientes
        para proteger la línea de WhatsApp contra baneos automáticos de Meta.
        """
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

            logger.warning(
                f"[Anti-Spam] Ráfaga excesiva detectada para {phone} ({len(timestamps)} msgs en {SPAM_WINDOW_SECONDS}s). "
                f"Mensajes descartados silenciosamente sin emisión saliente (prevención de baneos Meta/Evolution)."
            )
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
            t_debounce = time.perf_counter()
            try:
                await asyncio.sleep(DEBOUNCE_WAIT_SECONDS)
            except asyncio.CancelledError:
                # Si se canceló para reiniciar debounce, otro worker será creado por enqueue.
                return
            debounce_elapsed = time.perf_counter() - t_debounce

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
                        logger.info(
                            f"[Latency] phone={phone} debounce_wait={debounce_elapsed:.3f}s "
                            f"(configured={DEBOUNCE_WAIT_SECONDS})"
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
