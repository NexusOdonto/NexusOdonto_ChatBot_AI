"""Nodo de detección de emergencias médicas / odontológicas severas.
Escala inmediatamente a nivel CRÍTICO si se detectan hemorragias, dolor intolerable,
flemones con fiebre o traumatismos maxilofaciales graves.
"""

import asyncio
import logging
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from app.clients.dotnet_client import dotnet_client
from app.clients.evolution_client import evolution_client
from app.security.emergency_detector import detect_severe_emergency, MENSAJE_EMERGENCIA_URGENCIAS
from app.graph.state import AgentState

logger = logging.getLogger(__name__)


async def emergency_check_node(state: AgentState, config: RunnableConfig) -> dict:
    """Evalúa si el mensaje del usuario describe una emergencia severa que requiera atención médica inmediata."""
    messages = state.get("messages", [])
    if not messages:
        return {"conversation_status": state.get("conversation_status", "ACTIVA"), "emergency_detected": False}

    last_message = messages[-1]
    if not isinstance(last_message, HumanMessage):
        return {"conversation_status": state.get("conversation_status", "ACTIVA"), "emergency_detected": False}

    user_text = last_message.content
    if not isinstance(user_text, str) or not user_text.strip():
        return {"conversation_status": state.get("conversation_status", "ACTIVA"), "emergency_detected": False}

    is_emergency, reason = await detect_severe_emergency(user_text)
    if is_emergency:
        logger.warning(f"[Emergency Detector] Emergencia detectada ({reason}) en mensaje: '{user_text}'")
        thread_id = config.get("configurable", {}).get("thread_id")

        # 1. Enviar alerta por WhatsApp inmediatamente
        if thread_id:
            try:
                await evolution_client.enviar_mensaje(numero=thread_id, texto=MENSAJE_EMERGENCIA_URGENCIAS)
                asyncio.create_task(
                    dotnet_client.registrar_mensaje(thread_id, "CHATBOT", MENSAJE_EMERGENCIA_URGENCIAS)
                )
            except Exception as e:
                logger.error(f"[Emergency Detector] Error enviando alerta por WhatsApp a {thread_id}: {e}")

        # 2. Escalar ticket a nivel CRÍTICO, crear notificación y actualizar estado en .NET
        if thread_id:
            try:
                conv_id = await dotnet_client.obtener_o_crear_conversacion(thread_id)
                if conv_id:
                    await dotnet_client.actualizar_estado_conversacion(conv_id, dotnet_client.STATUS_ESCALADA)
                await dotnet_client.crear_ticket_soporte(
                    telefono=thread_id,
                    motivo=f"EMERGENCIA_MEDICA: {reason}",
                    prioridad="CRITICO",
                )
                await dotnet_client.crear_notificacion(
                    titulo=f"Emergencia odontológica: {reason}",
                    mensaje=f"Atención urgente solicitada por {thread_id}: {user_text}",
                    prioridad="CRITICO",
                    conversation_id=conv_id,
                    telefono=thread_id,
                )
            except Exception as e:
                logger.warning(f"[Emergency Detector] Error registrando ticket/notificación CRÍTICA en .NET para {thread_id}: {e}")

        emergency_message = AIMessage(content=MENSAJE_EMERGENCIA_URGENCIAS)
        return {
            "messages": [emergency_message],
            "conversation_status": "ESCALADA",
            "emergency_detected": True,
            "emergency_reason": reason,
        }

    return {
        "conversation_status": state.get("conversation_status", "ACTIVA"),
        "emergency_detected": False,
    }
