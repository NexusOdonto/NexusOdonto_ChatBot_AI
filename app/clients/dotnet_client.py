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

        # El backend definirá si valida JWT, API key o ambos encabezados.
        """Encabezados con token de seguridad para que C# permita el acceso."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if self.secret_token:
            headers["Authorization"] = f"Bearer {self.secret_token}"
            headers["X-Api-Key"] = self.secret_token
            headers["x-api-key"] = self.secret_token
        return headers

    async def consultar_disponibilidad(
        self,
        profesional_id: Optional[int] = None,
        fecha: Optional[str] = None,
        servicio_id: Optional[int] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Consulta horarios disponibles en el backend .NET.
        Maneja timeouts y errores de conexión.
        """
        if profesional_id is None:
			# Sin profesional no es posible calcular sus espacios disponibles.
            logger.error("[.NET Client] profesional_id es obligatorio para consultar horarios")
            return None

        url = f"{self.base_url}/profesionales/{profesional_id}/horarios-disponibles"
        params: Dict[str, Any] = {}
        if fecha:
			# La fecha limita la búsqueda a un día específico del calendario.
            params["fecha"] = fecha
        if servicio_id:
			# El servicio determina la duración necesaria del espacio.
            params["servicioId"] = servicio_id

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

    async def consultar_citas(self, fecha: str) -> Optional[Any]:
        """Consulta las citas de una fecha en el backend .NET."""
        url = f"{self.base_url}/citas"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                page = 1
                all_items: List[Dict[str, Any]] = []
                while True:
					# Se recorren todas las páginas para no dejar citas sin recordar.
                    response = await client.get(
                        url,
                        params={"fecha": fecha, "page": page, "pageSize": 100},
                        headers=self._get_headers(),
                    )
                    response.raise_for_status()
                    payload = response.json()

                    if isinstance(payload, list):
                        return payload
                    if not isinstance(payload, dict):
                        return payload

                    items = payload.get("items", [])
                    if isinstance(items, list):
						# Se normaliza la respuesta paginada a una única colección local.
                        all_items.extend(item for item in items if isinstance(item, dict))

                    total_pages = payload.get("totalPages", page)
                    if page >= total_pages or not items:
                        return {**payload, "items": all_items, "totalItems": len(all_items)}
                    page += 1
        except httpx.HTTPStatusError as e:
            logger.error(f"[.NET Client] Error HTTP {e.response.status_code}: {e.response.text}")
            return None
        except httpx.RequestError as e:
            logger.error(f"[.NET Client] Error de conexión/timeout al consultar citas: {str(e)}")
            return None

    async def crear_ticket_soporte(self, telefono: str, motivo: str, prioridad: str = "MEDIA") -> Optional[Dict[str, Any]]:
        """Crea un ticket en la API de .NET. Si falla, no interrumpe el chatbot."""
        url = f"{self.base_url}/tickets"
        payload = {
            "telefono": telefono,
            "motivo": motivo,
            "prioridad": prioridad
        }
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.post(url, json=payload, headers=self._get_headers())
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.warning(f"[.NET Client] No se pudo registrar el ticket (servicio no disponible): {str(e)}")
            return None

    async def obtener_servicios(self) -> Optional[List[Dict[str, Any]]]:
        """Obtiene la lista de servicios activos de la clínica."""
        url = f"{self.base_url}/servicios"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(url, headers=self._get_headers())
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, list):
                    return payload
                if isinstance(payload, dict):
                    return payload.get("items", [])
                return None
            except Exception as e:
                logger.error(f"[.NET Client] Error al obtener servicios: {str(e)}")
                return None

    async def obtener_profesionales(self, especialidad_id: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
        """Obtiene la lista de profesionales, opcionalmente filtrados por especialidad."""
        url = f"{self.base_url}/profesionales"
        params = {}
        if especialidad_id is not None:
            params["especialidadId"] = especialidad_id
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(url, params=params, headers=self._get_headers())
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, list):
                    return payload
                if isinstance(payload, dict):
                    return payload.get("items", [])
                return None
            except Exception as e:
                logger.error(f"[.NET Client] Error al obtener profesionales: {str(e)}")
                return None

    async def obtener_especialidades(self) -> Optional[List[Dict[str, Any]]]:
        """Obtiene la lista de especialidades de la clínica."""
        url = f"{self.base_url}/especialidades"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(url, headers=self._get_headers())
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, list):
                    return payload
                if isinstance(payload, dict):
                    return payload.get("items", [])
                return None
            except Exception as e:
                logger.error(f"[.NET Client] Error al obtener especialidades: {str(e)}")
                return None

    async def obtener_contexto_conversacion(self, conversacion_chatbot_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene el contexto de una conversación de chatbot, incluyendo el paciente vinculado si existe."""
        url = f"{self.base_url}/conversaciones-chatbot/{conversacion_chatbot_id}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(url, headers=self._get_headers())
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.error(f"[.NET Client] Error al obtener contexto de conversación: {str(e)}")
                return None

    async def buscar_pacientes(self, search: str) -> Optional[List[Dict[str, Any]]]:
        """Busca pacientes por teléfono, nombre o documento."""
        url = f"{self.base_url}/pacientes"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(url, params={"search": search}, headers=self._get_headers())
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, list):
                    return payload
                if isinstance(payload, dict):
                    return payload.get("items", [])
                return None
            except Exception as e:
                logger.error(f"[.NET Client] Error al buscar pacientes: {str(e)}")
                return None

    async def vincular_paciente(self, conversacion_chatbot_id: str, paciente_id: int) -> bool:
        """Vincula un paciente a una conversación de chatbot."""
        url = f"{self.base_url}/conversaciones-chatbot/{conversacion_chatbot_id}/paciente"
        payload = {"pacienteId": paciente_id}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.patch(url, json=payload, headers=self._get_headers())
                response.raise_for_status()
                return True
            except Exception as e:
                logger.error(f"[.NET Client] Error al vincular paciente: {str(e)}")
                return False

# Instancia reutilizable para el bot y las tools
dotnet_client = DotNetClient()