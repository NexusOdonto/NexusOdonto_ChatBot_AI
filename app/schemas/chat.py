from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Dict, Any

class BaseChatModel(BaseModel):
    model_config = ConfigDict(extra="allow")

class ExtendedTextMessage(BaseChatModel):
    text: Optional[str] = None

class MessageKey(BaseChatModel):
    remoteJid: Optional[str] = None
    remoteJidAlt: Optional[str] = None
    participant: Optional[str] = None
    participantAlt: Optional[str] = None
    senderPn: Optional[str] = None
    addressingMode: Optional[str] = None
    fromMe: Optional[bool] = False
    id: Optional[str] = None


class MessageData(BaseChatModel):
    key: Optional[MessageKey] = None
    pushName: Optional[str] = None
    participant: Optional[str] = None
    senderPn: Optional[str] = None
    messageType: Optional[str] = None
    message: Optional[Dict[str, Any]] = None
    base64: Optional[str] = None

class EvolutionWebhookPayload(BaseChatModel):
    event: Optional[str] = None
    instance: Optional[str] = None
    data: Optional[MessageData] = None


def unwrap_message_dict(raw_message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Desempaqueta mensajes anidados de WhatsApp (ephemeralMessage, viewOnceMessage,
    documentWithCaptionMessage, etc.) para llegar al mensaje interno real.
    """
    if not isinstance(raw_message, dict):
        return {}

    curr = raw_message
    for _ in range(5):  # Máximo 5 niveles de anidamiento
        if "ephemeralMessage" in curr and isinstance(curr["ephemeralMessage"], dict):
            curr = curr["ephemeralMessage"].get("message", curr["ephemeralMessage"])
        elif "viewOnceMessage" in curr and isinstance(curr["viewOnceMessage"], dict):
            curr = curr["viewOnceMessage"].get("message", curr["viewOnceMessage"])
        elif "viewOnceMessageV2" in curr and isinstance(curr["viewOnceMessageV2"], dict):
            curr = curr["viewOnceMessageV2"].get("message", curr["viewOnceMessageV2"])
        elif "documentWithCaptionMessage" in curr and isinstance(curr["documentWithCaptionMessage"], dict):
            curr = curr["documentWithCaptionMessage"].get("message", curr["documentWithCaptionMessage"])
        else:
            break
    return curr


def extract_interactive_selection(unwrapped_message: Dict[str, Any]) -> Optional[str]:
    """Extrae el texto/ID seleccionado de un mensaje interactivo (botón o lista).

    Evolution API envía las respuestas de botones como buttonsResponseMessage y
    las respuestas de listas como listResponseMessage. Esta función extrae
    el identificador seleccionado para que el flujo de registro lo procese.

    Returns:
        El texto/ID de la opción seleccionada, o None si no es un mensaje interactivo.
    """
    # Respuesta a botones de WhatsApp
    btn_response = unwrapped_message.get("buttonsResponseMessage") or unwrapped_message.get("buttonResponseMessage")
    if isinstance(btn_response, dict):
        # Puede venir como selectedButtonId o selectedDisplayText
        return (
            btn_response.get("selectedButtonId")
            or btn_response.get("selectedDisplayText")
            or btn_response.get("selectedId")
            or ""
        )

    # Respuesta a listas de WhatsApp
    list_response = unwrapped_message.get("listResponseMessage")
    if isinstance(list_response, dict):
        # El rowId es el identificador técnico de la opción seleccionada
        single = list_response.get("singleSelectReply") or {}
        return (
            single.get("selectedRowId")
            or list_response.get("title")
            or list_response.get("selectedRowId")
            or ""
        )

    # Respuesta a mensajes interactivos nativos (nativeFlowResponseMessage)
    interactive = unwrapped_message.get("interactiveResponseMessage")
    if isinstance(interactive, dict):
        body = interactive.get("nativeFlowResponseMessage") or {}
        return body.get("selectedId") or body.get("body") or ""

    return None

