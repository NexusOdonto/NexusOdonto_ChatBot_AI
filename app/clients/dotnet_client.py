import os
import logging
import asyncio
from typing import Optional, Dict, Any, List
import httpx
from dotenv import load_dotenv
from app.core.config import settings

# Cargar variables del entorno
load_dotenv()

logger = logging.getLogger(__name__)


class DotNetClient:
    """Cliente HTTP asíncrono para consumir la API de backend .NET de Nexus Odonto."""

    def __init__(self):
        self.base_url: str = os.getenv("DOTNET_API_URL", settings.dotnet_api_url).rstrip("/")
        self.auth_login: str = os.getenv("DOTNET_AUTH_LOGIN", settings.dotnet_auth_login)
        self.auth_password: str = os.getenv("DOTNET_AUTH_PASSWORD", settings.dotnet_auth_password)
        self.secret_token: str = os.getenv("AGENT_INTERNAL_SECRET", settings.agent_internal_secret)
        self.timeout: float = float(os.getenv("DOTNET_API_TIMEOUT", settings.dotnet_api_timeout))

        self._jwt_token: Optional[str] = None
        self._auth_lock = asyncio.Lock()
        self._reasons_cache: Optional[List[Dict[str, Any]]] = None
        self._priorities_cache: Optional[List[Dict[str, Any]]] = None

    @property
    def _auth_url(self) -> str:
        """Calcula la URL base del servidor (quitando /api/v1 o /api) para las rutas de autenticación."""
        url = self.base_url
        if url.endswith("/api/v1"):
            url = url[:-7]
        elif url.endswith("/api"):
            url = url[:-4]
        return f"{url}/api/auth"

    async def _get_valid_token(self, force_refresh: bool = False) -> str:
        """Obtiene un token JWT válido iniciando sesión automáticamente en .NET si es necesario."""
        if not force_refresh and self._jwt_token:
            return self._jwt_token

        async with self._auth_lock:
            # Doble verificación tras adquirir el lock
            if not force_refresh and self._jwt_token:
                return self._jwt_token

            # Si hay credenciales de usuario configuradas, iniciar sesión contra /api/auth/login
            if self.auth_login and self.auth_password:
                login_endpoint = f"{self._auth_url}/login"
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        response = await client.post(
                            login_endpoint,
                            json={"login": self.auth_login, "password": self.auth_password},
                            headers={"Content-Type": "application/json", "Accept": "application/json"},
                        )
                        if response.status_code == 200:
                            data = response.json()
                            token = data.get("token")
                            if token:
                                self._jwt_token = token
                                logger.info("[.NET Client] Sesión autenticada exitosamente con backend .NET (JWT)")
                                return self._jwt_token
                        else:
                            logger.warning(
                                f"[.NET Client] No se pudo autenticar en {login_endpoint}: {response.status_code} - {response.text}"
                            )
                except Exception as e:
                    logger.warning(f"[.NET Client] Error al conectar con endpoint de login ({login_endpoint}): {e}")

            # Fallback a token secreto estático
            return self.secret_token or ""

    async def _get_headers(self, force_refresh: bool = False) -> Dict[str, str]:
        """Encabezados con token JWT de autorización para que C# permita el acceso."""
        token = await self._get_valid_token(force_refresh=force_refresh)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
            headers["X-Api-Key"] = token
            headers["x-api-key"] = token
        return headers

    async def _request_with_retry(
        self, method: str, url: str, params: Optional[Dict[str, Any]] = None, json: Optional[Any] = None
    ) -> Optional[httpx.Response]:
        """Ejecuta una petición HTTP con manejo automático de reintentos en 401 (token expirado)."""
        headers = await self._get_headers()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.request(method, url, params=params, json=json, headers=headers)
                if response.status_code == 401:
                    logger.info("[.NET Client] Token JWT expirado o no válido (401). Renovando sesión...")
                    self._jwt_token = None
                    headers = await self._get_headers(force_refresh=True)
                    response = await client.request(method, url, params=params, json=json, headers=headers)
                return response
            except Exception as e:
                logger.error(f"[.NET Client] Error en {method} {url}: {e}")
                return None

    async def obtener_servicios(self) -> Optional[List[Dict[str, Any]]]:
        """Obtiene la lista de servicios activos de la clínica."""
        url = f"{self.base_url}/Services"
        response = await self._request_with_retry("GET", url)
        if response and response.status_code == 200:
            payload = response.json()
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict):
                return payload.get("items", [])
        return None

    async def obtener_especialidades(self) -> Optional[List[Dict[str, Any]]]:
        """Obtiene la lista de especialidades de la clínica."""
        url = f"{self.base_url}/Specialties"
        response = await self._request_with_retry("GET", url)
        if response and response.status_code == 200:
            payload = response.json()
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict):
                return payload.get("items", [])
        return None

    async def obtener_profesionales(
        self, especialidad_id: Optional[Any] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Obtiene la lista de profesionales, opcionalmente filtrados por especialidad."""
        url = f"{self.base_url}/Professionals"
        params = {}
        if especialidad_id is not None:
            params["especialidadId"] = str(especialidad_id)
            params["specialtyId"] = str(especialidad_id)

        response = await self._request_with_retry("GET", url, params=params if params else None)
        if response and response.status_code == 200:
            payload = response.json()
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict):
                return payload.get("items", [])
        return None

    async def consultar_disponibilidad(
        self,
        profesional_id: Optional[Any] = None,
        fecha: Optional[str] = None,
        servicio_id: Optional[Any] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Consulta horarios disponibles en el backend .NET."""
        url = f"{self.base_url}/Availabilities"
        params: Dict[str, Any] = {}
        if profesional_id:
            params["profesionalId"] = str(profesional_id)
            params["professionalId"] = str(profesional_id)
        if fecha:
            params["fecha"] = str(fecha)
            params["date"] = str(fecha)
        if servicio_id:
            params["servicioId"] = str(servicio_id)
            params["serviceId"] = str(servicio_id)

        response = await self._request_with_retry("GET", url, params=params)
        if response and response.status_code == 200:
            payload = response.json()
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict):
                return payload.get("items", [])
        return None

    async def agendar_cita(self, datos_cita: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Registra una cita en el backend .NET."""
        url = f"{self.base_url}/Appointments"
        response = await self._request_with_retry("POST", url, json=datos_cita)
        if response and response.status_code in (200, 201):
            return response.json()
        return None

    async def consultar_citas(self, fecha: str) -> Optional[Any]:
        """Consulta las citas de una fecha en el backend .NET."""
        url = f"{self.base_url}/Appointments"
        params = {"fecha": fecha, "date": fecha}
        response = await self._request_with_retry("GET", url, params=params)
        if response and response.status_code == 200:
            return response.json()
        return None

    async def _obtener_ticket_reasons(self) -> List[Dict[str, Any]]:
        """Obtiene y cachea los motivos de ticket de soporte disponibles en .NET."""
        if self._reasons_cache:
            return self._reasons_cache
        url = f"{self.base_url}/SupportTicketReasons"
        response = await self._request_with_retry("GET", url)
        if response and response.status_code == 200:
            self._reasons_cache = response.json() if isinstance(response.json(), list) else []
            return self._reasons_cache
        return []

    async def _obtener_notification_priorities(self) -> List[Dict[str, Any]]:
        """Obtiene y cachea las prioridades de notificación disponibles en .NET."""
        if self._priorities_cache:
            return self._priorities_cache
        url = f"{self.base_url}/NotificationPriorities"
        response = await self._request_with_retry("GET", url)
        if response and response.status_code == 200:
            self._priorities_cache = response.json() if isinstance(response.json(), list) else []
            return self._priorities_cache
        return []

    async def crear_ticket_soporte(
        self, telefono: str, motivo: str, prioridad: str = "MEDIA"
    ) -> Optional[Dict[str, Any]]:
        """Crea un ticket en la API de .NET. Si falla, no interrumpe el chatbot."""
        url = f"{self.base_url}/SupportTickets"

        # 1. Resolver UUID de motivo y prioridad si existen en el catálogo
        reasons = await self._obtener_ticket_reasons()
        priorities = await self._obtener_notification_priorities()

        reason_id = None
        motivo_upper = motivo.upper()
        for r in reasons:
            code = (r.get("code") or "").upper()
            if "EMERGENCIA" in motivo_upper and code in ("CONSULTA_COMPLEJA", "SOLICITUD_USUARIO"):
                reason_id = r.get("id")
                break
            if "SOLICITUD" in motivo_upper and "SOLICITUD" in code:
                reason_id = r.get("id")
                break
            if "RAG" in motivo_upper and "RAG" in code:
                reason_id = r.get("id")
                break
        if not reason_id and reasons:
            reason_id = reasons[0].get("id")

        priority_id = None
        prio_upper = prioridad.upper()
        for p in priorities:
            code = (p.get("code") or "").upper()
            if prio_upper in ("CRITICO", "CRÍTICO", "URGENTE") and code == "URGENTE":
                priority_id = p.get("id")
                break
            if prio_upper in ("ALTA", "HIGH") and code == "ALTA":
                priority_id = p.get("id")
                break
            if prio_upper in ("MEDIA", "NORMAL") and code == "NORMAL":
                priority_id = p.get("id")
                break
            if prio_upper in ("BAJA", "LOW") and code == "BAJA":
                priority_id = p.get("id")
                break
        if not priority_id and priorities:
            priority_id = priorities[0].get("id")

        payload = {
            "title": f"Soporte Chatbot: {motivo}",
            "description": f"Solicitud desde WhatsApp ({telefono}). Motivo: {motivo}",
            "chatbotSummary": f"Paciente {telefono} reportó: {motivo}",
            "telefono": telefono,
            "motivo": motivo,
            "prioridad": prioridad,
        }
        if reason_id:
            payload["ticketReasonId"] = reason_id
        if priority_id:
            payload["notificationPriorityId"] = priority_id

        response = await self._request_with_retry("POST", url, json=payload)
        if response and response.status_code in (200, 201):
            logger.info(f"[.NET Client] Ticket de soporte registrado exitosamente para {telefono}")
            return response.json()

        logger.warning(
            f"[.NET Client] No se pudo registrar el ticket (HTTP {response.status_code if response else 'None'})"
        )
        return None

    async def obtener_contexto_conversacion(
        self, conversacion_chatbot_id: str
    ) -> Optional[Dict[str, Any]]:
        """Obtiene el contexto de una conversación de chatbot, incluyendo el paciente vinculado si existe."""
        url = f"{self.base_url}/ChatbotConversations/{conversacion_chatbot_id}"
        response = await self._request_with_retry("GET", url)
        if response and response.status_code == 200:
            return response.json()
        return None

    async def buscar_pacientes(self, search: str) -> Optional[List[Dict[str, Any]]]:
        """Busca pacientes por teléfono, nombre o documento."""
        url = f"{self.base_url}/Patients"
        response = await self._request_with_retry("GET", url, params={"search": search})
        if response and response.status_code == 200:
            payload = response.json()
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict):
                return payload.get("items", [])
        return None

    async def vincular_paciente(self, conversacion_chatbot_id: str, paciente_id: Any) -> bool:
        """Vincula un paciente a una conversación de chatbot."""
        url = f"{self.base_url}/ChatbotConversations/{conversacion_chatbot_id}"
        payload = {"patientId": str(paciente_id), "pacienteId": str(paciente_id)}
        response = await self._request_with_retry("PATCH", url, json=payload)
        if response and response.status_code in (200, 204):
            return True
        return False


# Instancia reutilizable para el bot y las tools
dotnet_client = DotNetClient()