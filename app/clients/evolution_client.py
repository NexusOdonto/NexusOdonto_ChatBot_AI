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
        self.instance_name: str = os.getenv("EVOLUTION_INSTANCE_NAME", "clinica_odonto")
        self.api_key: str = os.getenv("EVOLUTION_API_KEY", "")
        self.timeout: float = float(os.getenv("EVOLUTION_API_TIMEOUT", "10.0"))

    def _get_headers(self) -> Dict[str, str]:
        """Encabezados requeridos por Evolution API."""
        return {
            "Content-Type": "application/json",
            "apikey": self.api_key
        }

    async def enviar_mensaje(self, numero: str, texto: str, delay: int = 1200) -> Optional[Dict[str, Any]]:
        """
        Envía un mensaje de texto a través de Evolution API v2.
        Endpoint: POST /message/sendText/{instance_name}
        Payload: {"number": numero, "text": texto, "delay": delay}
        """
        url = f"{self.base_url}/message/sendText/{self.instance_name}"
        
        # Limpieza básica del número para dejar solo dígitos
        numero_limpio = "".join(filter(str.isdigit, numero))

        payload = {
            "number": numero_limpio,
            "text": texto,
            "delay": delay
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, json=payload, headers=self._get_headers())
                response.raise_for_status()
                logger.info(f"[Evolution API] Mensaje enviado exitosamente a {numero_limpio}")
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"[Evolution API] Error HTTP {e.response.status_code} al enviar mensaje: {e.response.text}")
                return None
            except httpx.RequestError as e:
                logger.error(f"[Evolution API] Error de conexión/timeout con Evolution API (¿contenedor apagado?): {str(e)}")
                return None

# Instancia singleton para reutilizar en el agente
evolution_client = EvolutionClient()