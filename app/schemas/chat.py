from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ExtendedTextMessage(BaseModel):
    text: Optional[str] = None

class MessageDetail(BaseModel):
    conversation: Optional[str] = None
    extendedTextMessage: Optional[ExtendedTextMessage] = None

class MessageKey(BaseModel):
    remoteJid: Optional[str] = None
    fromMe: Optional[bool] = False

class MessageData(BaseModel):
    key: Optional[MessageKey] = None
    message: Optional[MessageDetail] = None

class EvolutionWebhookPayload(BaseModel):
    data: Optional[MessageData] = None
