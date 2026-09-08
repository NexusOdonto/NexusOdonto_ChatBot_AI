import logging
import asyncio
from fastapi import APIRouter, HTTPException, Security, Depends, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Optional

from app.clients.evolution_client import evolution_client
from app.clients.dotnet_client import dotnet_client
from app.core.config import settings
from app.graph.builder import get_graph
from app.session.memory_store import get_thread_config
from app.session.postgres_checkpointer import get_checkpointer_instance

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Agent Handoff & Messaging"])

# Seguridad básica mediante API Key o Secret Key interno
api_key_header = APIKeyHeader(name="X-Internal-Secret", auto_error=False)


def verify_internal_secret(api_key: Optional[str] = Depends(api_key_header)):
    expected_secret = settings.agent_internal_secret or settings.webhook_secret
    if expected_secret and api_key != expected_secret:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Acceso denegado: Token interno inválido o ausente."
        )
    return True


class SendMessageRequest(BaseModel):
    phone_number: str = Field(..., description="Número de teléfono o remoteJid del paciente en WhatsApp")
    message: str = Field(..., description="Contenido del mensaje a enviar al paciente")
    sender_role: str = Field(default="AGENTE_HUMANO", description="Rol del remitente: AGENTE_HUMANO, SISTEMA, CHATBOT")
    skip_db_register: bool = Field(default=False, description="Si es True, omite registrar el mensaje en .NET DB porque ya fue grabado")


class ResumeConversationRequest(BaseModel):
    phone_number: str = Field(..., description="Número de teléfono del paciente para reactivar el bot")
    clear_history: bool = Field(default=False, description="Si es True, limpia la memoria previa del bot")


@router.post("/conversations/resume", dependencies=[Depends(verify_internal_secret)])
async def resume_conversation(request: ResumeConversationRequest):
    """Reactivador de conversación.

    Llamado por el Backend .NET o el Frontend cuando un asesor humano finaliza
    la atención o devuelve la conversación al Chatbot AI.
    """
    phone_number = request.phone_number.strip()
    if not phone_number:
        raise HTTPException(status_code=400, detail="El número de teléfono es requerido.")

    try:
        targets = [phone_number]
        ALIAS_MAP = {
            "233783743803574@lid": "573001112233@s.whatsapp.net",
            "573001112233": "233783743803574@lid",
            "573001112233@s.whatsapp.net": "233783743803574@lid",
            "189515549421795@lid": "573226688304@s.whatsapp.net",
            "573226688304": "189515549421795@lid",
            "573226688304@s.whatsapp.net": "189515549421795@lid",
            "57213628510462@lid": "573238891073@s.whatsapp.net",
            "573238891073": "57213628510462@lid",
            "573238891073@s.whatsapp.net": "57213628510462@lid",
        }
        if phone_number in ALIAS_MAP and ALIAS_MAP[phone_number] not in targets:
            targets.append(ALIAS_MAP[phone_number])

        for t in targets:
            try:
                config = get_thread_config(t)
                checkpointer = get_checkpointer_instance()
                if checkpointer:
                    await checkpointer.clear_thread(t)
                dotnet_client.limpiar_cache_conversacion(t)
                logger.info(f"[Handoff] Estado reactivado y limpiado para {t}")
            except Exception as t_err:
                logger.warning(f"[Handoff] Error reactivando hilo {t}: {t_err}")

        # Mensaje de notificación amigable al paciente
        mensaje_retorno = (
            "🤖 *Nexus Odonto Asistente Virtual*\n\n"
            "La atención con nuestro asesor ha finalizado. Mi sistema ha sido reactivado. "
            "¿Hay algo más en lo que pueda colaborarte hoy? 🦷✨"
        )
        await evolution_client.enviar_mensaje(phone_number, mensaje_retorno)
        
        # Persistir mensaje de notificación en Oracle DB
        asyncio.create_task(
            dotnet_client.registrar_mensaje(phone_number, "CHATBOT", mensaje_retorno)
        )

        return {
            "status": "success",
            "message": f"Conversación reactivada con éxito para {phone_number}",
            "conversation_status": "ACTIVA"
        }

    except Exception as e:
        logger.error(f"[Handoff Error] Error al reanudar conversación para {phone_number}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno al reanudar conversación: {str(e)}")


@router.post("/agent/send-message", dependencies=[Depends(verify_internal_secret)])
async def send_agent_message(request: SendMessageRequest):
    """Envía un mensaje redactado por un asesor humano hacia el WhatsApp del paciente.

    Invocado por el Backend .NET o Frontend Web cuando el asesor escribe en el panel.
    Envía el mensaje por Evolution API y lo registra en Oracle DB.
    """
    phone_number = request.phone_number.strip()
    message_text = request.message.strip()

    if not phone_number or not message_text:
        raise HTTPException(status_code=400, detail="phone_number y message son requeridos.")

    try:
        # 1. Enviar mensaje al teléfono del paciente por Evolution API
        sent_ok = await evolution_client.enviar_mensaje(phone_number, message_text)
        if not sent_ok:
            raise HTTPException(status_code=502, detail="Error enviando el mensaje a través de Evolution API.")

        # 2. Registrar el mensaje en la base de datos Oracle vía .NET client (solo si no viene registrado por .NET)
        if not request.skip_db_register:
            asyncio.create_task(
                dotnet_client.registrar_mensaje(
                    chat_identifier=phone_number,
                    rol=request.sender_role,
                    contenido=message_text,
                )
            )

        return {
            "status": "success",
            "message": "Mensaje enviado y registrado exitosamente.",
            "phone_number": phone_number
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Handoff Error] Error enviando mensaje de agente a {phone_number}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error al procesar envío de mensaje: {str(e)}")
