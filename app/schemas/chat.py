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
    messageTimestamp: Optional[Any] = None
    date_time: Optional[Any] = None

class EvolutionWebhookPayload(BaseChatModel):
    event: Optional[str] = None
    instance: Optional[str] = None
    data: Optional[MessageData] = None
    date_time: Optional[Any] = None


def extract_message_timestamp(payload: Optional[EvolutionWebhookPayload] = None, raw_json: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """Extrae el timestamp en segundos (UNIX epoch) de un mensaje entrante de Evolution API.
    
    Soporta enteros/flotantes en segundos o milisegundos, objetos Baileys Long ({"low": ...}),
    y cadenas ISO (date_time / dateTime). Retorna None si no se puede determinar.
    """
    raw_data = (raw_json or {}).get("data") if isinstance(raw_json, dict) else {}
    if not isinstance(raw_data, dict):
        raw_data = {}

    data_obj = getattr(payload, "data", None) if payload else None

    # 1. Candidatos directos de messageTimestamp
    ts_val = None
    if data_obj is not None:
        ts_val = getattr(data_obj, "messageTimestamp", None)
    if ts_val is None:
        ts_val = raw_data.get("messageTimestamp")
    if ts_val is None and isinstance(raw_json, dict):
        ts_val = raw_json.get("messageTimestamp")

    # Si es dict tipo Long de Baileys / protobuf: {"low": 1711929600, ...}
    if isinstance(ts_val, dict):
        ts_val = ts_val.get("low")

    if ts_val is not None:
        try:
            val_f = float(ts_val)
            # En milisegundos (>1e11)
            if val_f > 1e11:
                return val_f / 1000.0
            # En segundos (>1e8)
            if val_f > 1e8:
                return val_f
        except (ValueError, TypeError):
            pass

    # 2. Candidatos de fecha en string ISO (date_time / dateTime)
    dt_str = None
    if data_obj is not None:
        dt_str = getattr(data_obj, "date_time", None)
    if not dt_str:
        dt_str = raw_data.get("date_time") or raw_data.get("dateTime")
    if not dt_str and isinstance(raw_json, dict):
        dt_str = raw_json.get("date_time") or raw_json.get("dateTime")

    if dt_str and isinstance(dt_str, str):
        try:
            from datetime import datetime
            clean_dt = dt_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_dt)
            return dt.timestamp()
        except Exception:
            pass

    return None


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

