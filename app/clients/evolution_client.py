import os
import json
import logging
import time
from collections import OrderedDict
from typing import Optional, Dict, Any
import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─── IDs de mensajes enviados por el bot ──────────────────────────────────────────────
# Guardamos el key.id de cada mensaje que el bot envía a través de Evolution.
# Cuando el webhook recibe el eco fromMe con ese mismo ID, lo reconoce como bot
# y no lo registra como mensaje de asesor humano.
_BOT_SENT_IDS: OrderedDict[str, float] = OrderedDict()
_BOT_SENT_TTL = 120  # segundos hasta descartar el ID


def register_bot_message_id(msg_id: str) -> None:
    """Registra el ID de un mensaje enviado por el bot para ignorar su eco fromMe."""
    if not msg_id:
        return
    now = time.monotonic()
    # Limpiar entradas viejas
    to_remove = [k for k, ts in _BOT_SENT_IDS.items() if now - ts > _BOT_SENT_TTL]
    for k in to_remove:
        del _BOT_SENT_IDS[k]
    _BOT_SENT_IDS[msg_id] = now


def is_bot_message_id(msg_id: str) -> bool:
    """Retorna True si el ID corresponde a un mensaje enviado por el bot."""
    return bool(msg_id) and msg_id in _BOT_SENT_IDS


class EvolutionClient:
    def __init__(self):
        # Lectura de variables de entorno 
        self.base_url: str = os.getenv("EVOLUTION_API_URL", "http://localhost:8080").rstrip("/")
        self.instance_name: str = os.getenv("EVOLUTION_INSTANCE_NAME", os.getenv("INSTANCE_NAME", "clinica_odonto"))
        self.api_key: str = os.getenv("EVOLUTION_API_KEY", "")
        self.timeout: float = float(os.getenv("EVOLUTION_API_TIMEOUT", "15.0"))
        self._default_delay_ms: int = int(os.getenv("EVOLUTION_SEND_DELAY_MS", "400"))

    def _get_headers(self) -> Dict[str, str]:
        """Encabezados requeridos por Evolution API."""
        return {
            "Content-Type": "application/json; charset=utf-8",
            "apikey": self.api_key
        }

    def _normalize_destination(self, numero: str) -> str:
        from app.services.whatsapp_identity import obtener_destino_envio
        return obtener_destino_envio(numero)

    async def enviar_mensaje(self, numero: str, texto: str, delay: int | None = None) -> Optional[Dict[str, Any]]:
        """
        Envía un mensaje de texto a través de Evolution API v2.
        Endpoint: POST /message/sendText/{instance_name}
        Payload: {"number": numero, "text": texto, "delay": delay}
        """
        if delay is None:
            delay = self._default_delay_ms
        url = f"{self.base_url}/message/sendText/{self.instance_name}"
        target_number = self._normalize_destination(numero)
        
        payload = {
            "number": target_number,
            "options": {
                "delay": delay,
                "presence": "composing"
            },
            "text": texto
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                response = await client.post(url, content=body_bytes, headers=self._get_headers())
                response.raise_for_status()
                logger.info(f"[Evolution API] Mensaje enviado exitosamente a {numero}")
                resp_data = response.json()
                # Registrar el ID del mensaje para ignorar el eco fromMe del webhook
                sent_id = None
                if isinstance(resp_data, dict):
                    sent_id = (
                        resp_data.get("key", {}).get("id")
                        or resp_data.get("id")
                    )
                if sent_id:
                    register_bot_message_id(str(sent_id))
                return resp_data
            except httpx.HTTPStatusError as e:
                logger.error(f"[Evolution API] Error HTTP {e.response.status_code} al enviar mensaje: {e.response.text}")
                return None
            except httpx.RequestError as e:
                logger.error(f"[Evolution API] Error de conexión/timeout con Evolution API (¿contenedor apagado?): {str(e)}")
                return None

    async def enviar_botones(
        self,
        numero: str,
        titulo: str,
        descripcion: str,
        botones: list[Dict[str, str]],
        pie: str = "",
        delay: int | None = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Envía un mensaje interactivo con botones de respuesta rápida vía Evolution API v2.
        Endpoint: POST /message/sendButtons/{instance_name}

        Cada botón es un dict con:
          - "buttonId": identificador técnico (ej. "CC")
          - "buttonText": {"displayText": "Cédula de Ciudadanía"}

        WhatsApp permite máximo 3 botones por mensaje.
        """
        if delay is None:
            delay = self._default_delay_ms
        url = f"{self.base_url}/message/sendButtons/{self.instance_name}"
        payload = {
            "number": numero,
            "options": {"delay": delay, "presence": "composing"},
            "buttonMessage": {
                "title": titulo,
                "description": descripcion,
                "footerText": pie,
                "buttons": botones,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                response = await client.post(url, content=body_bytes, headers=self._get_headers())
                response.raise_for_status()
                logger.info(f"[Evolution API] Botones enviados exitosamente a {numero}")
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"[Evolution API] Error HTTP {e.response.status_code} al enviar botones: {e.response.text}"
                )
                # Fallback: enviar como texto plano con opciones numeradas
                texto_fallback = f"*{titulo}*\n\n{descripcion}\n\n"
                for i, b in enumerate(botones, 1):
                    display = b.get("buttonText", {}).get("displayText", b.get("buttonId", ""))
                    texto_fallback += f"{i}. {display}\n"
                if pie:
                    texto_fallback += f"\n_{pie}_"
                return await self.enviar_mensaje(numero, texto_fallback, delay)
            except httpx.RequestError as e:
                logger.error(f"[Evolution API] Error de conexión al enviar botones: {str(e)}")
                return None

    async def enviar_lista(
        self,
        numero: str,
        titulo: str,
        descripcion: str,
        texto_boton: str,
        secciones: list[Dict[str, Any]],
        pie: str = "",
        delay: int | None = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Envía un mensaje interactivo tipo lista de selección vía Evolution API v2.
        Endpoint: POST /message/sendList/{instance_name}

        Cada sección tiene:
          - "title": nombre de la sección
          - "rows": [{"title": "Opción", "description": "Detalle", "rowId": "ID_TECNICO"}]
        """
        if delay is None:
            delay = self._default_delay_ms
        url = f"{self.base_url}/message/sendList/{self.instance_name}"
        payload = {
            "number": numero,
            "options": {"delay": delay, "presence": "composing"},
            "listMessage": {
                "title": titulo,
                "description": descripcion,
                "footerText": pie,
                "buttonText": texto_boton,
                "sections": secciones,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                response = await client.post(url, content=body_bytes, headers=self._get_headers())
                response.raise_for_status()
                logger.info(f"[Evolution API] Lista enviada exitosamente a {numero}")
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"[Evolution API] Error HTTP {e.response.status_code} al enviar lista: {e.response.text}"
                )
                # Fallback: enviar como texto plano con opciones numeradas
                texto_fallback = f"*{titulo}*\n\n{descripcion}\n\n"
                for sec in secciones:
                    for row in sec.get("rows", []):
                        texto_fallback += f"• {row.get('title', '')} — {row.get('description', '')}\n"
                if pie:
                    texto_fallback += f"\n_{pie}_"
                texto_fallback += f"\n\nResponde con el nombre de la opción que deseas."
                return await self.enviar_mensaje(numero, texto_fallback, delay)
            except httpx.RequestError as e:
                logger.error(f"[Evolution API] Error de conexión al enviar lista: {str(e)}")
                return None

    async def obtener_base64_media(self, message: Dict[str, Any], convert_to_mp4: bool = False) -> Optional[Dict[str, Any]]:
        """
        Obtiene el contenido binario/base64 de un mensaje multimedia desde Evolution API v2.
        Endpoint: POST /chat/getBase64FromMediaMessage/{instance_name}
        Payload: {"message": message, "convertToMp4": convert_to_mp4}
        """
        url = f"{self.base_url}/chat/getBase64FromMediaMessage/{self.instance_name}"
        payload = {
            "message": message,
            "convertToMp4": convert_to_mp4
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                response = await client.post(url, content=body_bytes, headers=self._get_headers())
                response.raise_for_status()
                data = response.json()
                logger.info("[Evolution API] Base64 multimedia obtenido exitosamente")
                return data
            except httpx.HTTPStatusError as e:
                logger.error(f"[Evolution API] Error HTTP {e.response.status_code} al obtener base64: {e.response.text}")
                return None
            except httpx.RequestError as e:
                logger.error(f"[Evolution API] Error de conexión al solicitar base64 multimedia: {str(e)}")
                return None

# Instancia singleton para reutilizar en el agente
evolution_client = EvolutionClient()