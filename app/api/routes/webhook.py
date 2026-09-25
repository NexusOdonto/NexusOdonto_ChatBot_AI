"""Controlador HTTP de Webhook para WhatsApp (Evolution API).
Procesa eventos de mensajería entrantes, deduplicación de entregas y delegación
asíncrona limpia al orquestador de chat y procesador de mensajes.
"""

import time
import asyncio
import logging
from collections import OrderedDict
from typing import Any
from fastapi import APIRouter, Request

from app.schemas.chat import EvolutionWebhookPayload, unwrap_message_dict, extract_interactive_selection
from app.clients.evolution_client import (
    evolution_client,
    is_bot_message_id,
    is_recent_bot_text,
    extract_evolution_message_id,
)
from app.services.semantic_cache import purgar_cache_semantico
from app.services.chat.chat_orchestrator import (
    chat_orchestrator,
    _USER_PUSH_NAMES,
)
from app.services.chat.message_processor import (
    process_whatsapp_message,
    process_whatsapp_audio,
    process_whatsapp_unsupported_media,
    registrar_mensaje_asesor,
    is_escalated as _is_escalated,
    with_chat_lock,
    _USER_LAST_ACTIVE,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["WhatsApp Webhook"])

# Deduplicación de webhooks entrantes (TTL 90 segundos)
_MESSAGE_DEDUPE_TTL_SECONDS = 90
_SEEN_MESSAGE_IDS: OrderedDict[str, float] = OrderedDict()
_PATIENT_CONTEXT_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}

# Deduplicación por contenido reciente por usuario (TTL 4 segundos)
_RECENT_USER_CONTENT: OrderedDict[str, float] = OrderedDict()
_RECENT_USER_CONTENT_TTL = 4.0


def _prune_seen_message_ids(now: float) -> None:
    while _SEEN_MESSAGE_IDS:
        oldest_id, seen_at = next(iter(_SEEN_MESSAGE_IDS.items()))
        if now - seen_at <= _MESSAGE_DEDUPE_TTL_SECONDS:
            break
        _SEEN_MESSAGE_IDS.popitem(last=False)


def _is_duplicate_message_id(message_id: str | None) -> bool:
    """Retorna True si este key.id ya fue procesado dentro de la ventana de tiempo."""
    if not message_id:
        return False
    now = time.monotonic()
    _prune_seen_message_ids(now)
    if message_id in _SEEN_MESSAGE_IDS:
        return True
    _SEEN_MESSAGE_IDS[message_id] = now
    _SEEN_MESSAGE_IDS.move_to_end(message_id)
    return False


def _is_duplicate_user_content(phone: str, text: str) -> bool:
    """Retorna True si un usuario envió exactamente el mismo texto hace menos de 4 segundos."""
    if not phone or not text:
        return False
    norm_text = " ".join(text.strip().lower().split())
    if not norm_text:
        return False
    key = f"{phone}:{norm_text}"
    now = time.monotonic()
    while _RECENT_USER_CONTENT:
        oldest_k, seen_at = next(iter(_RECENT_USER_CONTENT.items()))
        if now - seen_at <= _RECENT_USER_CONTENT_TTL:
            break
        _RECENT_USER_CONTENT.popitem(last=False)
    if key in _RECENT_USER_CONTENT:
        return True
    _RECENT_USER_CONTENT[key] = now
    _RECENT_USER_CONTENT.move_to_end(key)
    return False


@router.post("/cache/purge")
async def purge_cache_endpoint():
    """Purga el caché semántico en memoria y en Qdrant."""
    exito = purgar_cache_semantico()
    if exito:
        return {
            "status": "ok",
            "message": "Caché semántico purgado y colección recreada exitosamente."
        }
    return {
        "status": "error",
        "message": "No se pudo purgar la colección en Qdrant (verificar conectividad)."
    }


@router.post("/whatsapp")
async def receive_whatsapp_message(request: Request):
    """Endpoint principal de webhook para Evolution API.
    Retorna HTTP 200 de inmediato y desacopla el procesamiento mediante tareas asíncronas.
    """
    try:
        raw_json = await request.json()
        payload = EvolutionWebhookPayload(**raw_json)

        # Solo mensajes nuevos. MESSAGES_UPDATE genera ecos/status que re-disparan fromMe
        # y pueden clasificar respuestas del bot como "asesor humano".
        event_name = (payload.event or raw_json.get("event") or "").strip().lower()
        if event_name and event_name not in (
            "messages.upsert",
            "messages_upsert",
            "message.upsert",
        ):
            return {"status": "ignored", "reason": f"event_{event_name}"}

        if not payload.data:
            return {"status": "ignored", "reason": "empty_data"}

        data = payload.data

        # 1. Mensajes enviados desde el propio dispositivo vinculado (fromMe)
        if data.key and data.key.fromMe:
            msg_id_from_me = getattr(data.key, "id", None) or ""
            if is_bot_message_id(msg_id_from_me):
                return {"status": "ignored", "reason": "self_message_bot"}

            remote_jid_me = data.key.remoteJid if (data.key and data.key.remoteJid) else ""
            from app.services.whatsapp_identity import extraer_identidad_webhook, obtener_telefono_canonico
            destinatario, _ = extraer_identidad_webhook(raw_json, data)
            if not destinatario:
                participant_me = getattr(data.key, "participant", None) or getattr(data.key, "remoteJidAlt", None)
                if "@lid" in str(remote_jid_me) and participant_me and "@s.whatsapp.net" in str(participant_me):
                    destinatario = str(participant_me)
                else:
                    destinatario = remote_jid_me
                destinatario = obtener_telefono_canonico(destinatario)

            raw_msg_me = data.message or {}
            unwrapped_me = unwrap_message_dict(raw_msg_me)
            texto_asesor = (
                unwrapped_me.get("conversation")
                or (unwrapped_me.get("extendedTextMessage") or {}).get("text", "")
                or ""
            )
            if not texto_asesor:
                for k in ["imageMessage", "videoMessage", "documentMessage"]:
                    if k in unwrapped_me and isinstance(unwrapped_me[k], dict):
                        cap = unwrapped_me[k].get("caption", "")
                        if cap:
                            texto_asesor = cap
                            break

            # Eco del bot: match por texto (± destino LID/teléfono). fromMe ya prueba origen local.
            if texto_asesor and (
                is_recent_bot_text(destinatario, texto_asesor)
                or is_recent_bot_text(str(remote_jid_me or ""), texto_asesor)
            ):
                if msg_id_from_me:
                    from app.clients.evolution_client import register_bot_message_id
                    register_bot_message_id(str(msg_id_from_me))
                logger.info(f"[fromMe-BotEcho] Eco de mensaje enviado por el bot hacia {destinatario} descartado.")
                return {"status": "ignored", "reason": "self_message_bot_echo"}

            if texto_asesor and destinatario:
                asyncio.create_task(registrar_mensaje_asesor(destinatario, texto_asesor))
                logger.info(f"[fromMe-Humano] Mensaje manual del asesor hacia {destinatario}: '{texto_asesor}'")
            return {"status": "ignored", "reason": "self_message_human_registered"}

        # 2. Descartar entregas duplicadas
        message_id = (
            getattr(data.key, "id", None) if (data.key and getattr(data.key, "id", None))
            else extract_evolution_message_id(raw_json)
        )
        if _is_duplicate_message_id(message_id):
            logger.info(f"[Webhook] Mensaje duplicado ignorado (key.id={message_id})")
            return {"status": "ignored", "reason": "duplicate_message_id"}

        remote_jid = data.key.remoteJid if (data.key and data.key.remoteJid) else ""
        if not remote_jid:
            return {"status": "ignored", "reason": "no_remote_jid"}

        from app.services.whatsapp_identity import (
            es_identificador_lid,
            extraer_identidad_webhook,
            telefono_para_almacenar,
        )
        numero_paciente, destino_envio = extraer_identidad_webhook(raw_json, data)
        # Si Evolution mandó solo @lid, intentar recuperar el teléfono real (remoteJidAlt histórico)
        if es_identificador_lid(numero_paciente):
            try:
                phone_e164 = await evolution_client.resolver_telefono_desde_lid(numero_paciente)
                if phone_e164:
                    from app.services.whatsapp_identity import registrar_asociacion_lid, jid_desde_telefono
                    registrar_asociacion_lid(phone_e164, numero_paciente)
                    numero_paciente = jid_desde_telefono(phone_e164)
                    logger.info(
                        "[Webhook] Identidad LID reparada → teléfono %s (dest=%s)",
                        telefono_para_almacenar(numero_paciente),
                        destino_envio,
                    )
            except Exception as resolve_err:
                logger.debug("[Webhook] No se pudo resolver LID a teléfono: %s", resolve_err)

        push_name = str(getattr(data, "pushName", None) or "").strip()

        raw_message = data.message or {}
        unwrapped_message = unwrap_message_dict(raw_message)
        message_type = data.messageType or ""

        # 3. Audio / Notas de voz
        is_audio = (message_type == "audioMessage" or "audioMessage" in unwrapped_message)
        if is_audio:
            if chat_orchestrator.is_user_spam_blocked(numero_paciente) or chat_orchestrator.check_and_trigger_spam(numero_paciente):
                return {"status": "ignored", "reason": "spam_blocked"}
            logger.info(f"[Webhook] Audio recibido de {numero_paciente}. Encolando transcripción...")
            raw_payload_data = raw_json.get("data", {})
            asyncio.create_task(
                with_chat_lock(
                    numero_paciente,
                    process_whatsapp_audio(numero_paciente, raw_payload_data, unwrapped_message, push_name=push_name),
                )
            )
            return {"status": "queued"}

        # 4. Medios no soportados (imágenes, stickers, documentos, videos, ubicación)
        unsupported_keys = {
            "imageMessage", "videoMessage", "ptvMessage", "documentMessage",
            "documentWithCaptionMessage", "stickerMessage", "contactMessage",
            "contactsArrayMessage", "locationMessage", "liveLocationMessage",
        }
        is_unsupported = (message_type in unsupported_keys or any(k in unwrapped_message for k in unsupported_keys))
        if is_unsupported:
            if chat_orchestrator.is_user_spam_blocked(numero_paciente) or chat_orchestrator.check_and_trigger_spam(numero_paciente):
                return {"status": "ignored", "reason": "spam_blocked"}
            caption = ""
            for k in ["imageMessage", "videoMessage", "documentMessage"]:
                if k in unwrapped_message and isinstance(unwrapped_message[k], dict):
                    caption = unwrapped_message[k].get("caption", "") or caption

            logger.info(f"[Webhook] Medio no soportado recibido de {numero_paciente}. Encolando aviso...")
            asyncio.create_task(
                with_chat_lock(
                    numero_paciente,
                    process_whatsapp_unsupported_media(numero_paciente, caption),
                )
            )
            return {"status": "queued"}

        # 5. Mensaje de Texto (interactivo o regular)
        mensaje_texto = ""
        interactive_selection = extract_interactive_selection(unwrapped_message)
        if interactive_selection:
            mensaje_texto = interactive_selection
        elif "conversation" in unwrapped_message and unwrapped_message["conversation"]:
            mensaje_texto = unwrapped_message["conversation"]
        elif "extendedTextMessage" in unwrapped_message:
            ext = unwrapped_message["extendedTextMessage"]
            if isinstance(ext, dict):
                mensaje_texto = ext.get("text", "")
            elif hasattr(ext, "text"):
                mensaje_texto = ext.text or ""

        if mensaje_texto:
            if _is_duplicate_user_content(numero_paciente, mensaje_texto):
                logger.info(f"[Webhook] Entrega duplicada de contenido en <4s ignorada para {numero_paciente}: '{mensaje_texto[:40]}'")
                return {"status": "ignored", "reason": "duplicate_content_rapid_repeat"}

            logger.info(f"[Webhook] Mensaje de texto de {numero_paciente}: '{mensaje_texto}'")

            async def _on_debounce_flush(phone: str, text: str, name: str):
                await with_chat_lock(
                    phone,
                    process_whatsapp_message(phone, text, push_name=name, registrar_usuario_db=False),
                )

            enqueued = chat_orchestrator.enqueue_message(
                phone=numero_paciente,
                message=mensaje_texto,
                push_name=push_name,
                process_callback=_on_debounce_flush,
            )
            if not enqueued:
                return {"status": "ignored", "reason": "spam_blocked"}
        else:
            logger.warning(f"[Webhook] Mensaje no reconocido o vacío de {numero_paciente}: {unwrapped_message}")

        return {"status": "queued"}

    except Exception as e:
        logger.error(f"[Webhook Error] Fallo al procesar el webhook: {e}", exc_info=True)
        return {"status": "error", "message": "Error procesando el payload"}