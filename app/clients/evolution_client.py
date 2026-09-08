import os
import logging
from typing import Optional, Dict, Any
import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class EvolutionClient:
    def __init__(self):
        # Lectura de variables de entorno 
        self.base_url: str = os.getenv("EVOLUTION_API_URL", "http://localhost:8080").rstrip("/")
        self.instance_name: str = os.getenv("EVOLUTION_INSTANCE_NAME", os.getenv("INSTANCE_NAME", "clinica_odonto"))
        self.api_key: str = os.getenv("EVOLUTION_API_KEY", "")
        self.timeout: float = float(os.getenv("EVOLUTION_API_TIMEOUT", "15.0"))

    def _get_headers(self) -> Dict[str, str]:
        """Encabezados requeridos por Evolution API."""
        return {
            "Content-Type": "application/json",
            "apikey": self.api_key
        }

    def _normalize_destination(self, numero: str) -> str:
        dest = str(numero).strip()
        # Aliases de LIDs que fueron mapeados o provienen de chats con LIDs de WhatsApp
        LID_ALIASES = {
            "573001112233": "233783743803574@lid",
            "573001112233@s.whatsapp.net": "233783743803574@lid",
            "573226688304": "189515549421795@lid",
            "573226688304@s.whatsapp.net": "189515549421795@lid",
            "57213628510462": "57213628510462@lid",
            "57213628510462@s.whatsapp.net": "57213628510462@lid",
            "573238891073": "57213628510462@lid",
            "573238891073@s.whatsapp.net": "57213628510462@lid",
        }
        if dest in LID_ALIASES:
            return LID_ALIASES[dest]

        # Si son más de 13 dígitos numéricos puros (característica de LID de WhatsApp)
        digits = "".join(ch for ch in dest if ch.isdigit())
        if len(digits) >= 14 and not dest.endswith("@lid"):
            return f"{digits}@lid"

        return dest

    async def enviar_mensaje(self, numero: str, texto: str, delay: int = 1200) -> Optional[Dict[str, Any]]:
        """
        Envía un mensaje de texto a través de Evolution API v2.
        Endpoint: POST /message/sendText/{instance_name}
        Payload: {"number": numero, "text": texto, "delay": delay}
        """
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
                response = await client.post(url, json=payload, headers=self._get_headers())
                response.raise_for_status()
                logger.info(f"[Evolution API] Mensaje enviado exitosamente a {numero}")
                return response.json()
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
        delay: int = 1200,
    ) -> Optional[Dict[str, Any]]:
        """
        Envía un mensaje interactivo con botones de respuesta rápida vía Evolution API v2.
        Endpoint: POST /message/sendButtons/{instance_name}

        Cada botón es un dict con:
          - "buttonId": identificador técnico (ej. "CC")
          - "buttonText": {"displayText": "Cédula de Ciudadanía"}

        WhatsApp permite máximo 3 botones por mensaje.
        """
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
                response = await client.post(url, json=payload, headers=self._get_headers())
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
        delay: int = 1200,
    ) -> Optional[Dict[str, Any]]:
        """
        Envía un mensaje interactivo tipo lista de selección vía Evolution API v2.
        Endpoint: POST /message/sendList/{instance_name}

        Cada sección tiene:
          - "title": nombre de la sección
          - "rows": [{"title": "Opción", "description": "Detalle", "rowId": "ID_TECNICO"}]
        """
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
                response = await client.post(url, json=payload, headers=self._get_headers())
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
                response = await client.post(url, json=payload, headers=self._get_headers())
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