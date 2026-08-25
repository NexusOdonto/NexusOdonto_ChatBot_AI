import os
import logging
from typing import Optional, Dict, Any, List
import httpx
from dotenv import load_dotenv

# Cargar variables del entorno
load_dotenv()

logger = logging.getLogger(__name__)

class DotNetClient:
    def __init__(self):
        self.base_url: str = os.getenv("DOTNET_API_URL", "http://localhost:5000/api").rstrip("/")
        self.secret_token: str = os.getenv("AGENT_INTERNAL_SECRET", "")
        self.timeout: float = float(os.getenv("DOTNET_API_TIMEOUT", "10.0"))

    def _get_headers(self) -> Dict[str, str]:
        """Encabezados con token de seguridad para que C# permita el acceso."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if self.secret_token:
            headers["Authorization"] = f"Bearer {self.secret_token}"
            headers["x-api-key"] = self.secret_token
        return headers

    async def consultar_disponibilidad(
        self, profesional_id: Optional[int] = None, fecha: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Consulta horarios disponibles en el backend .NET.
        Maneja timeouts y errores de conexión.
        """
        url = f"{self.base_url}/citas/disponibilidad"
        params: Dict[str, Any] = {}
        if profesional_id:
            params["profesionalId"] = profesional_id
        if fecha:
            params["fecha"] = fecha

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(url, params=params, headers=self._get_headers())
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"[.NET Client] Error HTTP {e.response.status_code}: {e.response.text}")
                return None
            except httpx.RequestError as e:
                logger.error(f"[.NET Client] Error de conexión/timeout con backend .NET: {str(e)}")
                return None

    async def agendar_cita(self, datos_cita: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Registra una cita en el backend .NET.
        Maneja timeouts y errores de conexión.
        """
        url = f"{self.base_url}/citas"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, json=datos_cita, headers=self._get_headers())
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"[.NET Client] Error HTTP {e.response.status_code}: {e.response.text}")
                return None
            except httpx.RequestError as e:
                logger.error(f"[.NET Client] Error de conexión/timeout al agendar cita: {str(e)}")
                return None

# Instancia reutilizable para el bot y las tools
dotnet_client = DotNetClient()