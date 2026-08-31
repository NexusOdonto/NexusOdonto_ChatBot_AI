import base64
import io
import logging
from typing import Optional, Dict, Any
import httpx
from openai import AsyncOpenAI

from app.clients.evolution_client import evolution_client
from app.core.config import settings

logger = logging.getLogger(__name__)

# Mensaje amigable cuando el audio está vacío o inaudible
MENSAJE_AUDIO_NO_ENTENDIDO = (
    "No logré escuchar con claridad tu nota de voz 🎧.\n\n"
    "Por favor, intenta grabarla nuevamente en un lugar con menos ruido o escríbenos tu consulta por texto para poder atenderte."
)

MENSAJE_ERROR_PROCESANDO_AUDIO = (
    "Tuvimos un inconveniente técnico al procesar tu nota de voz 🎧.\n\n"
    "Por favor, intenta enviarla nuevamente o escríbenos tu consulta por texto."
)


def _clean_base64_string(b64_str: str) -> str:
    """Elimina el prefijo data URI (ej. 'data:audio/ogg;base64,') si está presente."""
    if "," in b64_str and b64_str.startswith("data:"):
        return b64_str.split(",", 1)[1]
    return b64_str


async def extraer_bytes_audio(raw_payload_data: Dict[str, Any], raw_message: Dict[str, Any]) -> Optional[tuple[bytes, str]]:
    """
    Extrae los bytes del audio y el mimetype correspondiente.
    Intenta:
    1. Base64 embebido en el mensaje o payload.
    2. Consulta a Evolution API mediante /chat/getBase64FromMediaMessage.
    3. Descarga directa desde URL si está disponible.
    
    Retorna (audio_bytes, mimetype) o None si no fue posible obtener el audio.
    """
    audio_msg = raw_message.get("audioMessage", {})
    mimetype = audio_msg.get("mimetype", "audio/ogg; codecs=opus")

    # 1. Base64 directo en audioMessage o en data
    b64_str = audio_msg.get("base64") or raw_payload_data.get("base64")
    if b64_str and isinstance(b64_str, str):
        try:
            cleaned = _clean_base64_string(b64_str)
            audio_bytes = base64.b64decode(cleaned)
            if audio_bytes:
                logger.info(f"[Audio Service] Audio obtenido desde base64 directo ({len(audio_bytes)} bytes)")
                return audio_bytes, mimetype
        except Exception as e:
            logger.warning(f"[Audio Service] Error decodificando base64 directo: {e}")

    # 2. Solicitar base64 a Evolution API
    try:
        logger.info("[Audio Service] Solicitando base64 a Evolution API...")
        media_resp = await evolution_client.obtener_base64_media(raw_message)
        if media_resp:
            # Puede venir en base64 o data.base64
            b64_val = media_resp.get("base64")
            if not b64_val and isinstance(media_resp.get("data"), dict):
                b64_val = media_resp["data"].get("base64")
            
            resp_mimetype = media_resp.get("mimetype") or mimetype

            if b64_val and isinstance(b64_val, str):
                cleaned = _clean_base64_string(b64_val)
                audio_bytes = base64.b64decode(cleaned)
                if audio_bytes:
                    logger.info(f"[Audio Service] Audio obtenido desde Evolution API ({len(audio_bytes)} bytes)")
                    return audio_bytes, resp_mimetype
    except Exception as e:
        logger.warning(f"[Audio Service] Error al obtener base64 desde Evolution API: {e}")

    # 3. Descarga desde URL si existe
    url = audio_msg.get("url")
    if url and isinstance(url, str) and url.startswith("http"):
        try:
            logger.info(f"[Audio Service] Intentando descarga directa desde URL: {url}")
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200 and resp.content:
                    logger.info(f"[Audio Service] Audio descargado desde URL ({len(resp.content)} bytes)")
                    return resp.content, mimetype
        except Exception as e:
            logger.warning(f"[Audio Service] Error descargando audio desde URL: {e}")

    logger.error("[Audio Service] No se pudo obtener el contenido del audio por ningún medio.")
    return None


async def transcribir_audio(audio_bytes: bytes, mimetype: str = "audio/ogg") -> Optional[str]:
    """
    Transcribe los bytes de audio a texto usando OpenAI Whisper (whisper-1).
    Configurado en español y con contexto léxico de la clínica Nexus Odonto.
    """
    if not settings.openai_api_key:
        logger.error("[Audio Service] No hay OPENAI_API_KEY configurada para transcribir audios.")
        return None

    # Determinar extensión apropiada
    ext = "ogg"
    if "mp4" in mimetype:
        ext = "m4a"
    elif "mpeg" in mimetype or "mp3" in mimetype:
        ext = "mp3"
    elif "wav" in mimetype:
        ext = "wav"

    filename = f"audio.{ext}"

    try:
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename

        logger.info(f"[Audio Service] Enviando audio ({len(audio_bytes)} bytes, {filename}) a Whisper...")
        transcription = await client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="es",
            prompt="Nexus Odonto, consultorio odontológico, citas, doctores, tratamientos, limpieza, ortodoncia, endodoncia, diseño de sonrisa, implantes."
        )

        texto = transcription.text.strip()
        logger.info(f"[Audio Service] Transcripción exitosa: '{texto}'")
        return texto
    except Exception as e:
        logger.error(f"[Audio Service] Error al transcribir audio con Whisper: {e}", exc_info=True)
        return None
