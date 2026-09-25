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
    "No alcancé a escuchar bien tu nota de voz.\n\n"
    "¿La grabas otra vez en un lugar más silencioso, o me lo escribes por texto?"
)

MENSAJE_ERROR_PROCESANDO_AUDIO = (
    "Se nos complicó procesar tu nota de voz.\n\n"
    "¿La intentas de nuevo o me cuentas por texto lo que necesitas?"
)


def _clean_base64_string(b64_str: str) -> str:
    """Elimina el prefijo data URI (ej. 'data:audio/ogg;base64,') si está presente."""
    if "," in b64_str and b64_str.startswith("data:"):
        return b64_str.split(",", 1)[1]
    return b64_str


def _decrypt_whatsapp_enc_media(enc_bytes: bytes, media_key_b64: str, app_info: bytes = b"WhatsApp Audio Keys") -> Optional[bytes]:
    """Descifra el binario .enc descargado de los servidores CDN de WhatsApp utilizando HKDF y AES-256-CBC."""
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        from cryptography.hazmat.primitives import hashes

        media_key = base64.b64decode(media_key_b64)
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=112,
            salt=None,
            info=app_info,
        )
        key_stream = hkdf.derive(media_key)
        iv = key_stream[:16]
        cipher_key = key_stream[16:48]

        # WhatsApp incluye 10 bytes de firma MAC al final del archivo .enc
        ciphertext = enc_bytes[:-10] if len(enc_bytes) > 10 else enc_bytes
        cipher = Cipher(algorithms.AES(cipher_key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        decrypted = decryptor.update(ciphertext) + decryptor.finalize()

        # Quitar el padding PKCS7 si es válido
        if decrypted:
            pad_len = decrypted[-1]
            if 1 <= pad_len <= 16:
                decrypted = decrypted[:-pad_len]

        logger.info(f"[Audio Service] Audio .enc descifrado exitosamente ({len(decrypted)} bytes)")
        return decrypted
    except Exception as e:
        logger.warning(f"[Audio Service] Error al descifrar binario .enc de WhatsApp: {e}")
        return None


async def extraer_bytes_audio(raw_payload_data: Dict[str, Any], raw_message: Dict[str, Any]) -> Optional[tuple[bytes, str]]:
    """
    Extrae los bytes del audio y el mimetype correspondiente.
    Intenta:
    1. Base64 embebido en el mensaje o payload.
    2. Consulta a Evolution API mediante /chat/getBase64FromMediaMessage con la estructura completa del mensaje.
    3. Descarga directa desde URL de WhatsApp con descifrado .enc automático.
    
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

    # 2. Solicitar base64 a Evolution API enviando el contenedor completo del mensaje
    message_payloads = [
        raw_payload_data,
        {"key": raw_payload_data.get("key"), "message": raw_payload_data.get("message") or raw_message},
        {"message": raw_message, "key": raw_payload_data.get("key")},
    ]

    for p in message_payloads:
        try:
            logger.info("[Audio Service] Solicitando base64 a Evolution API...")
            media_resp = await evolution_client.obtener_base64_media(p)
            if media_resp:
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
                break
        except Exception as e:
            logger.warning(f"[Audio Service] Error al obtener base64 desde Evolution API: {e}")

    # 3. Descarga desde URL de WhatsApp con descifrado .enc automático si aplica
    url = audio_msg.get("url")
    media_key = audio_msg.get("mediaKey")
    if url and isinstance(url, str) and url.startswith("http"):
        try:
            logger.info(f"[Audio Service] Intentando descarga directa desde URL: {url}")
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200 and resp.content:
                    raw_downloaded = resp.content
                    logger.info(f"[Audio Service] Audio descargado desde URL ({len(raw_downloaded)} bytes)")

                    # Si el archivo está cifrado (.enc) o viene mediaKey, descifrarlo
                    if (".enc" in url or media_key) and media_key:
                        decrypted = _decrypt_whatsapp_enc_media(raw_downloaded, media_key)
                        if decrypted:
                            return decrypted, mimetype
                    return raw_downloaded, mimetype
        except Exception as e:
            logger.warning(f"[Audio Service] Error descargando audio desde URL: {e}")

    logger.error("[Audio Service] No se pudo obtener el contenido del audio por ningún medio.")
    return None


async def transcribir_audio(audio_bytes: bytes, mimetype: str = "audio/ogg") -> Optional[str]:
    """
    Transcribe los bytes de audio a texto usando Gemini Multimodal o OpenAI Whisper
    según el proveedor configurado.
    """
    prov = (settings.llm_provider or "openai").lower().strip()

    # 1. Si el proveedor activo es Gemini o si hay clave de Gemini y no de OpenAI
    if (prov == "gemini" and settings.gemini_api_key) or (not settings.openai_api_key.startswith("sk-") and settings.gemini_api_key):
        try:
            import asyncio
            import google.generativeai as genai

            genai.configure(api_key=settings.gemini_api_key)
            model_name = "gemini-1.5-flash" if "flash" in settings.gemini_model else settings.gemini_model
            model = genai.GenerativeModel(model_name)
            clean_mime = mimetype.split(";")[0].strip() if mimetype else "audio/ogg"
            prompt = (
                "Transcribe el siguiente audio exactamente como se escucha en español. "
                "Contexto: consultorio odontológico Nexus Odonto (citas, doctores, tratamientos, ortodoncia, etc.). "
                "No agregues comentarios ni formato adicional, únicamente devuelve el texto transcrito."
            )
            logger.info(f"[Audio Service] Transcribiendo audio ({len(audio_bytes)} bytes) con Gemini ({model_name})...")

            response = await asyncio.to_thread(
                model.generate_content,
                [
                    {"mime_type": clean_mime, "data": audio_bytes},
                    prompt,
                ],
            )
            texto = (response.text or "").strip()
            logger.info(f"[Audio Service] Transcripción exitosa con Gemini: '{texto}'")
            return texto
        except Exception as exc:
            logger.error(f"[Audio Service] Error al transcribir con Gemini: {exc}", exc_info=True)
            if not settings.openai_api_key:
                return None

    # 2. Si hay clave válida de OpenAI
    if settings.openai_api_key and settings.openai_api_key.startswith("sk-"):
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
            logger.info(f"[Audio Service] Transcripción exitosa con Whisper: '{texto}'")
            return texto
        except Exception as e:
            logger.error(f"[Audio Service] Error al transcribir audio con Whisper: {e}", exc_info=True)
            return None

    logger.error("[Audio Service] No hay API Key válida para transcribir notas de voz (OpenAI o Gemini).")
    return None
