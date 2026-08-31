from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Dict, Any

class BaseChatModel(BaseModel):
    model_config = ConfigDict(extra="allow")

class ExtendedTextMessage(BaseChatModel):
    text: Optional[str] = None

class MessageKey(BaseChatModel):
    remoteJid: Optional[str] = None
    fromMe: Optional[bool] = False
    id: Optional[str] = None

class MessageDetail(BaseChatModel):
    conversation: Optional[str] = None
    extendedTextMessage: Optional[ExtendedTextMessage] = None
    imageMessage: Optional[Dict[str, Any]] = None
    audioMessage: Optional[Dict[str, Any]] = None
    videoMessage: Optional[Dict[str, Any]] = None
    ptvMessage: Optional[Dict[str, Any]] = None
    documentMessage: Optional[Dict[str, Any]] = None
    documentWithCaptionMessage: Optional[Dict[str, Any]] = None
    stickerMessage: Optional[Dict[str, Any]] = None
    contactMessage: Optional[Dict[str, Any]] = None
    contactsArrayMessage: Optional[Dict[str, Any]] = None
    locationMessage: Optional[Dict[str, Any]] = None
    liveLocationMessage: Optional[Dict[str, Any]] = None
    viewOnceMessage: Optional[Dict[str, Any]] = None
    viewOnceMessageV2: Optional[Dict[str, Any]] = None
    ephemeralMessage: Optional[Dict[str, Any]] = None

class MessageData(BaseChatModel):
    key: Optional[MessageKey] = None
    pushName: Optional[str] = None
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
