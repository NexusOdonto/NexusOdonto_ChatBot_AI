import os
import json as json_lib
import logging
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta, timezone
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
        self._conversations_lock = asyncio.Lock()
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
                login_candidates = [
                    {"loginId": self.auth_login, "password": self.auth_password},
                ]
                for payload in login_candidates:
                    try:
                        async with httpx.AsyncClient(timeout=self.timeout) as client:
                            headers = {
                                "Content-Type": "application/json; charset=utf-8",
                                "Accept": "application/json",
                            }
                            # BOT_SERVICE login is rejected without the shared API↔bot secret.
                            if self.secret_token:
                                headers["X-Internal-Secret"] = self.secret_token
                            response = await client.post(
                                login_endpoint,
                                json=payload,
                                headers=headers,
                            )
                            if response.status_code == 200:
                                data = response.json()
                                token = data.get("token")
                                if token:
                                    self._jwt_token = token
                                    logger.info("[.NET Client] Sesión autenticada exitosamente con backend .NET (JWT)")
                                    return self._jwt_token
                    except Exception as e:
                        logger.debug(f"[.NET Client] Intento fallido con payload {list(payload.keys())}: {e}")

                logger.warning(f"[.NET Client] No se pudo autenticar con ningún candidato en {login_endpoint}")

            # Fallback a token secreto estático
            return self.secret_token or ""

    async def _get_headers(self, force_refresh: bool = False) -> Dict[str, str]:
        """Encabezados con token JWT de autorización para que C# permita el acceso."""
        token = await self._get_valid_token(force_refresh=force_refresh)
        headers = {
            "Content-Type": "application/json; charset=utf-8",
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

    async def obtener_empleados(self) -> List[Dict[str, Any]]:
        """Obtiene la lista de empleados activos en el backend .NET."""
        url = f"{self.base_url}/Employees"
        response = await self._request_with_retry("GET", url)
        if response and response.status_code == 200:
            payload = response.json()
            return payload if isinstance(payload, list) else payload.get("items", [])
        return []

    async def obtener_personas(self) -> List[Dict[str, Any]]:
        """Obtiene la lista de personas registradas en el backend .NET."""
        url = f"{self.base_url}/Persons"
        response = await self._request_with_retry("GET", url)
        if response and response.status_code == 200:
            payload = response.json()
            return payload if isinstance(payload, list) else payload.get("items", [])
        return []

    async def obtener_profesionales(
        self, especialidad_id: Optional[Any] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Obtiene la lista de profesionales enriquecidos con su nombre completo."""
        url = f"{self.base_url}/Professionals"
        params = {}
        if especialidad_id is not None:
            params["especialidadId"] = str(especialidad_id)
            params["specialtyId"] = str(especialidad_id)

        response = await self._request_with_retry("GET", url, params=params if params else None)
        if response and response.status_code == 200:
            payload = response.json()
            profs = payload if isinstance(payload, list) else payload.get("items", [])
            if not profs:
                return []

            # Enriquecer con nombres reales consultando empleados y personas
            try:
                empleados = await self.obtener_empleados()
                personas = await self.obtener_personas()
                emp_to_person = {e.get("id"): e.get("personId") for e in empleados if isinstance(e, dict)}
                person_names = {
                    p.get("id"): f"{p.get('firstName', '')} {p.get('lastName', '')}".strip()
                    for p in personas if isinstance(p, dict)
                }

                for p in profs:
                    if isinstance(p, dict):
                        emp_id = p.get("employeeId")
                        per_id = emp_to_person.get(emp_id)
                        nombre_raw = person_names.get(per_id)
                        if nombre_raw:
                            prefijo = "Dra." if any(n in nombre_raw.lower() for n in ["laura", "maria", "ana", "camila", "valentina", "sofia"]) else "Dr."
                            p["name"] = f"{prefijo} {nombre_raw}"
                            p["nombre"] = f"{prefijo} {nombre_raw}"
                            p["nombreCompleto"] = f"{prefijo} {nombre_raw}"
            except Exception as enh_err:
                logger.debug(f"[.NET Client] No se pudieron enriquecer nombres de profesionales: {enh_err}")

            return profs
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

    async def agendar_cita(self, datos_cita: Dict[str, Any]) -> Dict[str, Any]:
        """Registra una cita en el backend .NET."""
        url = f"{self.base_url}/Appointments"
        response = await self._request_with_retry("POST", url, json=datos_cita)
        if response and response.status_code in (200, 201):
            res_json = response.json() if isinstance(response.json(), dict) else {"id": str(response.json())}
            res_json["success"] = True
            return res_json
        
        error_msg = ""
        status_code = response.status_code if response else 500
        if response:
            try:
                res_err = response.json()
                if isinstance(res_err, dict):
                    error_msg = (
                        res_err.get("message")
                        or res_err.get("detail")
                        or res_err.get("title")
                        or str(res_err.get("errors") or "")
                    )
            except Exception:
                error_msg = response.text[:300]
            logger.warning(
                f"[.NET Client] Error creando cita (HTTP {status_code}): {error_msg or response.text[:300]}"
            )
        return {
            "success": False,
            "error": error_msg or "No fue posible registrar la cita en el sistema.",
            "status_code": status_code,
        }

    async def consultar_citas(self, fecha: str = "") -> Optional[Any]:
        """Consulta las citas de una fecha en el backend .NET. Si fecha está vacía o el filtro falla, consulta todas sin query params."""
        url = f"{self.base_url}/Appointments"
        clean_f = str(fecha).strip() if fecha else ""
        if clean_f:
            response = await self._request_with_retry("GET", url, params={"date": clean_f})
            if response and response.status_code == 200:
                payload = response.json()
                items = payload if isinstance(payload, list) else payload.get("items", [])
                if items:
                    return payload
            response = await self._request_with_retry("GET", url, params={"fecha": clean_f})
            if response and response.status_code == 200:
                payload = response.json()
                items = payload if isinstance(payload, list) else payload.get("items", [])
                if items:
                    return payload

        # Fallback seguro: consultar sin parámetros para que el llamador filtre por fecha en memoria
        response = await self._request_with_retry("GET", url)
        if response and response.status_code == 200:
            return response.json()
        return None

    async def obtener_catalogo(self, nombre_catalogo: str) -> List[Dict[str, Any]]:
        """Obtiene un catálogo genérico de la API (ej: AppointmentStatuses, DocumentTypes, Sexes, etc.)."""
        url = f"{self.base_url}/{nombre_catalogo}"
        response = await self._request_with_retry("GET", url)
        if response and response.status_code == 200:
            payload = response.json()
            return payload if isinstance(payload, list) else payload.get("items", [])
        return []

    async def obtener_citas_paciente(self, patient_id: str) -> List[Dict[str, Any]]:
        """Obtiene las citas registradas exclusivamente para un paciente específico, enriquecidas con profesional y servicio."""
        if not patient_id:
            return []
        url = f"{self.base_url}/Appointments"
        response = await self._request_with_retry("GET", url)
        if not response or response.status_code != 200:
            return []

        payload = response.json()
        all_citas = payload if isinstance(payload, list) else payload.get("items", [])
        
        # Filtrar estrictamente solo las citas del paciente dado
        citas_paciente = [
            c for c in all_citas
            if str(c.get("patientId") or c.get("pacienteId", "")).lower() == str(patient_id).lower()
        ]

        if not citas_paciente:
            return []

        # Enriquecer con nombres de profesionales, servicios y estados
        try:
            profs = await self.obtener_profesionales() or []
            servs = await self.obtener_servicios() or []
            statuses = await self.obtener_catalogo("AppointmentStatuses") or []

            prof_dict = {str(p.get("id")).lower(): p.get("name") or p.get("nombre") for p in profs if isinstance(p, dict)}
            serv_dict = {str(s.get("id")).lower(): s.get("name") or s.get("nombre") for s in servs if isinstance(s, dict)}
            status_dict = {str(st.get("id")).lower(): st.get("name") or st.get("nombre") or st.get("code") for st in statuses if isinstance(st, dict)}

            for c in citas_paciente:
                p_id = str(c.get("professionalId") or "").lower()
                s_id = str(c.get("serviceId") or "").lower()
                st_id = str(c.get("appointmentStatusId") or "").lower()

                c["professionalName"] = prof_dict.get(p_id, "Especialista Odontológico")
                c["serviceName"] = serv_dict.get(s_id, "Consulta Odontológica")
                
                known_names = {
                    "10000000-0000-0000-0000-000000000001": "Programada",
                    "10000000-0000-0000-0000-000000000002": "Confirmada",
                    "10000000-0000-0000-0000-000000000003": "En Atención",
                    "10000000-0000-0000-0000-000000000004": "Completada",
                    "10000000-0000-0000-0000-000000000005": "Cancelada",
                    "10000000-0000-0000-0000-000000000006": "No Asistió",
                }
                if c.get("cancelledAt"):
                    c["statusName"] = "Cancelada"
                elif st_id in known_names:
                    c["statusName"] = known_names[st_id]
                else:
                    raw_st = status_dict.get(st_id, "Programada")
                    raw_upper = str(raw_st).upper()
                    if "NO_ASIST" in raw_upper or "NO ASIST" in raw_upper:
                        c["statusName"] = "No Asistió"
                    elif "CONFIRM" in raw_upper:
                        c["statusName"] = "Confirmada"
                    elif "ATENC" in raw_upper:
                        c["statusName"] = "En Atención"
                    elif "COMPLET" in raw_upper:
                        c["statusName"] = "Completada"
                    elif "CANCEL" in raw_upper:
                        c["statusName"] = "Cancelada"
                    else:
                        c["statusName"] = "Programada"
        except Exception as enrich_err:
            logger.warning(f"[.NET Client] Error enriqueciendo citas de paciente: {enrich_err}")

        # Ordenar por fecha startsAt
        try:
            citas_paciente.sort(key=lambda x: str(x.get("startsAt") or x.get("fechaHoraInicio") or ""))
        except Exception:
            pass

        return citas_paciente


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

    async def actualizar_estado_conversacion(
        self, conversacion_id: str, status_id: str, patient_id: Optional[str] = None
    ) -> bool:
        """Actualiza el estado de una conversación en el backend .NET / Oracle.
        
        Si el estado es STATUS_ACTIVA, desasigna cualquier asesor humano y reabre la conversación.
        """
        try:
            url = f"{self.base_url}/ChatbotConversations/{conversacion_id}"
            payload: Dict[str, Any] = {
                "conversationStatusId": status_id,
                "patientId": patient_id,
            }
            if status_id.lower() == self.STATUS_ACTIVA.lower():
                payload["assignedEmployeeId"] = None
                payload["employeeId"] = None
                payload["assignedUserId"] = None
                payload["closedAt"] = None

            resp = await self._request_with_retry("PUT", url, json=payload)
            if resp and resp.status_code in (200, 204):
                logger.info(f"[.NET Client] Estado de conversación {conversacion_id} actualizado a {status_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"[.NET Client] Error actualizando estado de conversación {conversacion_id}: {e}")
            return False

    async def resolver_tickets_conversacion(self, conversacion_id: str) -> int:
        """Cierra o resuelve cualquier ticket de soporte abierto para una conversación específica en .NET."""
        if not conversacion_id:
            return 0
        try:
            tickets = await self.obtener_catalogo("SupportTickets") or []
            STATUS_RESUELTO = "c0000000-0000-0000-0000-000000000004"
            STATUS_CERRADO = "c0000000-0000-0000-0000-000000000005"
            conv_id_clean = str(conversacion_id).strip().lower()

            cerrados = 0
            for t in tickets:
                if not isinstance(t, dict):
                    continue
                t_conv = str(t.get("chatbotConversationId", "")).strip().lower()
                t_status = str(t.get("ticketStatusId", "")).strip().lower()
                if t_conv == conv_id_clean and t_status not in (STATUS_RESUELTO, STATUS_CERRADO):
                    ticket_id = t.get("id")
                    if ticket_id:
                        url = f"{self.base_url}/SupportTickets/{ticket_id}"
                        payload = dict(t)
                        payload["ticketStatusId"] = STATUS_RESUELTO
                        payload["resolvedAt"] = datetime.now(timezone.utc).isoformat()
                        payload["resolutionNotes"] = "Atención finalizada por asesor. Conversación reactivada con NexusBot."
                        resp = await self._request_with_retry("PUT", url, json=payload)
                        if resp and resp.status_code in (200, 204):
                            cerrados += 1
                            logger.info(f"[.NET Client] Ticket {ticket_id} resuelto para conversación {conversacion_id}")
            return cerrados
        except Exception as e:
            logger.warning(f"[.NET Client] Error resolviendo tickets para conversación {conversacion_id}: {e}")
            return 0


    async def crear_ticket_soporte(
        self, telefono: str, motivo: str, prioridad: str = "MEDIA"
    ) -> Optional[Dict[str, Any]]:
        """Crea un ticket en la API de .NET vinculado a la conversación. Si falla, no interrumpe el chatbot."""
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
        if not reason_id:
            if "RAG" in motivo_upper:
                reason_id = "d0000000-0000-0000-0000-000000000002"
            else:
                reason_id = "d0000000-0000-0000-0000-000000000001"

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
        if not priority_id:
            if prio_upper in ("CRITICO", "CRÃTICO", "URGENTE"):
                priority_id = "50000000-0000-0000-0000-000000000004"
            elif prio_upper in ("ALTA", "HIGH"):
                priority_id = "50000000-0000-0000-0000-000000000003"
            else:
                priority_id = "50000000-0000-0000-0000-000000000002"

        # 2. Obtener o crear la conversación en .NET para vincularla al ticket
        conv_id = await self.obtener_o_crear_conversacion(telefono)
        patient_id = None
        try:
            persona = await self.buscar_persona_por_telefono(telefono)
            if persona and persona.get("id"):
                paciente = await self.buscar_paciente_por_person_id(str(persona["id"]))
                if paciente and paciente.get("id"):
                    patient_id = str(paciente["id"])
        except Exception as e:
            logger.debug(f"[.NET Client] Error buscando paciente para ticket: {e}")

        import re
        clean_tel = re.sub(r"@.*$", "", telefono).strip()
        formatted_phone = clean_tel if clean_tel.startswith("+") else f"+{clean_tel}"

        human_reason = "El paciente solicitó atención personalizada con un asesor humano" if "SOLICITUD" in motivo.upper() \
                  else ("Baja confianza en la respuesta automática de la IA" if "RAG" in motivo.upper() \
                  else f"Atención requerida: {motivo}")

        payload = {
            "chatbotConversationId": conv_id,
            "patientId": patient_id,
            "title": f"Escalamiento WhatsApp: {human_reason}",
            "description": f"Solicitud desde WhatsApp ({formatted_phone}). Motivo: {human_reason}",
            "chatbotSummary": human_reason,
            "telefono": formatted_phone,
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
            # Actualizar estado de la conversación a ESCALADA en Oracle DB
            if conv_id:
                asyncio.create_task(
                    self.actualizar_estado_conversacion(conv_id, self.STATUS_ESCALADA, patient_id)
                )
            # Emitir también notificación en /Notifications para que el panel web avise inmediatamente
            asyncio.create_task(
                self.crear_notificacion(
                    titulo=f"Alerta: {human_reason}",
                    mensaje=f"Paciente {formatted_phone}: {human_reason}",
                    prioridad=prioridad,
                    patient_id=patient_id,
                    conversation_id=conv_id,
                    telefono=formatted_phone,
                )
            )
            return response.json()

        logger.warning(
            f"[.NET Client] No se pudo registrar el ticket (HTTP {response.status_code if response else 'None'})"
        )
        return None

    async def crear_notificacion(
        self,
        titulo: str,
        mensaje: str,
        prioridad: str = "URGENTE",
        patient_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        telefono: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Crea una notificación en el backend .NET (/Notifications) para alertar al personal en el panel de recepción."""
        try:
            url = f"{self.base_url}/Notifications"
            priorities = await self._obtener_notification_priorities()
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
            if not priority_id and priorities:
                priority_id = priorities[0].get("id")
            if not priority_id:
                if prio_upper in ("CRITICO", "CRÍTICO", "URGENTE"):
                    priority_id = "50000000-0000-0000-0000-000000000004"
                elif prio_upper in ("ALTA", "HIGH"):
                    priority_id = "50000000-0000-0000-0000-000000000003"
                else:
                    priority_id = "50000000-0000-0000-0000-000000000002"

            payload = {
                "title": titulo,
                "message": mensaje,
                "description": mensaje,
                "content": mensaje,
                "isRead": False,
            }
            if priority_id:
                payload["notificationPriorityId"] = priority_id
            if patient_id:
                payload["patientId"] = patient_id
            if conversation_id:
                payload["chatbotConversationId"] = conversation_id
            if telefono:
                payload["phone"] = telefono

            resp = await self._request_with_retry("POST", url, json=payload)
            if resp and resp.status_code in (200, 201):
                logger.info(f"[.NET Client] Notificación creada en /Notifications para {telefono or 'recepción'}")
                return resp.json()
            elif resp:
                logger.debug(f"[.NET Client] /Notifications retornó HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as notif_err:
            logger.debug(f"[.NET Client] Error intentando crear notificación en /Notifications: {notif_err}")
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

    async def obtener_appointment_status_id(self, code: str = "AGENDADA") -> str:
        """Obtiene el ID del estado de cita por código o nombre."""
        c_upper = str(code).strip().upper().replace(" ", "_")
        fallback_map = {
            "PENDIENTE": "10000000-0000-0000-0000-000000000001",
            "AGENDADA": "10000000-0000-0000-0000-000000000001",
            "PROGRAMADA": "10000000-0000-0000-0000-000000000001",
            "SCHEDULED": "10000000-0000-0000-0000-000000000001",
            "CONFIRMADA": "10000000-0000-0000-0000-000000000002",
            "CONFIRMED": "10000000-0000-0000-0000-000000000002",
            "EN_ATENCION": "10000000-0000-0000-0000-000000000003",
            "ATTENDING": "10000000-0000-0000-0000-000000000003",
            "COMPLETADA": "10000000-0000-0000-0000-000000000004",
            "COMPLETED": "10000000-0000-0000-0000-000000000004",
            "CANCELADA": "10000000-0000-0000-0000-000000000005",
            "CANCELLED": "10000000-0000-0000-0000-000000000005",
            "NO_ASISTIO": "10000000-0000-0000-0000-000000000006",
            "NO_SHOW": "10000000-0000-0000-0000-000000000006",
            "NOSHOW": "10000000-0000-0000-0000-000000000006",
        }
        url = f"{self.base_url}/AppointmentStatuses"
        response = await self._request_with_retry("GET", url)
        if response and response.status_code == 200:
            statuses = response.json() if isinstance(response.json(), list) else []
            for s in statuses:
                s_code = str(s.get("code") or "").upper()
                s_name = str(s.get("name") or "").upper().replace(" ", "_")
                if s_code == c_upper or s_name == c_upper:
                    return str(s.get("id"))
        return fallback_map.get(c_upper, "10000000-0000-0000-0000-000000000001")

    async def obtener_appointment_origin_id(self, code: str = "AGENTE_BOT") -> str:
        """Obtiene el ID del origen de cita por código."""
        url = f"{self.base_url}/AppointmentOrigins"
        response = await self._request_with_retry("GET", url)
        if response and response.status_code == 200:
            origins = response.json() if isinstance(response.json(), list) else []
            for o in origins:
                if str(o.get("code")).upper() == code.upper():
                    return str(o.get("id"))
        return "20000000-0000-0000-0000-000000000002"

    async def buscar_pacientes(self, search: str) -> Optional[List[Dict[str, Any]]]:
        """Busca pacientes por teléfono, nombre o documento cruzando con Personas."""
        try:
            url = f"{self.base_url}/Patients"
            response = await self._request_with_retry("GET", url)
            if not response or response.status_code != 200:
                return None

            patients = response.json() if isinstance(response.json(), list) else response.json().get("items", [])
            if not patients:
                return []

            personas = await self.obtener_personas()
            person_map = {p.get("id"): p for p in personas if isinstance(p, dict)}

            clean_search = "".join(c for c in str(search) if c.isdigit())
            text_search = str(search).lower().strip()

            matching_patients = []
            for pt in patients:
                if not isinstance(pt, dict):
                    continue
                per_id = pt.get("personId")
                per = person_map.get(per_id, {})

                phone_digits = "".join(c for c in str(per.get("phone") or "") if c.isdigit())
                doc_digits = "".join(c for c in str(per.get("documentNumber") or "") if c.isdigit())
                doc_raw = str(per.get("documentNumber") or "").lower()
                first_name = str(per.get("firstName") or "").lower()
                last_name = str(per.get("lastName") or "").lower()
                full_name = f"{first_name} {last_name}".strip()

                # Comparación flexible por teléfono
                matched = False
                if clean_search and phone_digits:
                    if clean_search.endswith(phone_digits) or phone_digits.endswith(clean_search) or clean_search in phone_digits or phone_digits in clean_search:
                        matched = True

                # Comparación flexible por documento
                if not matched and clean_search and (clean_search == doc_digits or clean_search in doc_raw):
                    matched = True

                # Comparación por texto/nombre
                if not matched and text_search and (text_search in full_name or full_name in text_search or text_search in doc_raw):
                    matched = True

                if matched:
                    pt["persona"] = per
                    pt["nombre"] = full_name
                    pt["telefono"] = per.get("phone")
                    pt["documento"] = per.get("documentNumber")
                    matching_patients.append(pt)

            if matching_patients:
                return matching_patients

            # Si no hubo búsqueda específica, devolver todos enriquecidos
            if not search or not search.strip():
                for pt in patients:
                    per_id = pt.get("personId")
                    per = person_map.get(per_id, {})
                    pt["persona"] = per
                    pt["nombre"] = f"{per.get('firstName', '')} {per.get('lastName', '')}".strip()
                return patients

            return []
        except Exception as e:
            logger.error(f"[.NET Client] Error buscando pacientes: {e}")
            return None

    async def vincular_paciente(self, conversacion_chatbot_id: str, paciente_id: Any) -> bool:
        """Vincula un paciente a una conversación de chatbot."""
        url = f"{self.base_url}/ChatbotConversations/{conversacion_chatbot_id}"
        payload = {"patientId": str(paciente_id), "pacienteId": str(paciente_id)}
        response = await self._request_with_retry("PATCH", url, json=payload)
        if response and response.status_code in (200, 204):
            return True
        return False

    # ─────────────────────────────────────────────────────────────
    # Métodos para el flujo de registro / login de pacientes
    # ─────────────────────────────────────────────────────────────

    async def obtener_tipos_documento(self) -> List[Dict[str, Any]]:
        """Obtiene el catálogo de tipos de documento (CC, TI, CE, PP, etc.) desde .NET."""
        url = f"{self.base_url}/DocumentTypes"
        response = await self._request_with_retry("GET", url)
        if response and response.status_code == 200:
            payload = response.json()
            return payload if isinstance(payload, list) else payload.get("items", [])
        return []

    async def obtener_sexos(self) -> List[Dict[str, Any]]:
        """Obtiene el catálogo de sexos (Masculino, Femenino, Otro) desde .NET."""
        url = f"{self.base_url}/Sexes"
        response = await self._request_with_retry("GET", url)
        if response and response.status_code == 200:
            payload = response.json()
            return payload if isinstance(payload, list) else payload.get("items", [])
        return []

    async def registrar_paciente(self, datos_onboard: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Registra un paciente completo mediante el endpoint de onboarding.

        El endpoint POST /api/v1/Patients/onboard crea en una sola transacción:
        - PersonEntity (datos personales)
        - UserEntity (credenciales con hash BCrypt)
        - PatientEntity (contacto de emergencia)
        - ClinicalHistoryEntity (historial clínico vacío)
        - UserRoleEntity (rol PACIENTE)

        Args:
            datos_onboard: dict con los campos de OnboardPatientDto:
                documentTypeId, documentNumber, firstName, lastName,
                dateOfBirth, sexId, phone, email, address,
                emergencyContact, emergencyPhone, password
        """
        url = f"{self.base_url}/Patients/onboard"
        response = await self._request_with_retry("POST", url, json=datos_onboard)
        if response and response.status_code in (200, 201):
            logger.info(f"[.NET Client] Paciente registrado exitosamente: {datos_onboard.get('documentNumber')}")
            return response.json()
        if response:
            logger.warning(
                f"[.NET Client] Error en onboarding (HTTP {response.status_code}): {response.text}"
            )
        return None

    async def crear_paciente_basico(
        self,
        cedula: str,
        nombre: str,
        telefono_whatsapp: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Crea un perfil de paciente básico en el sistema usando datos mínimos.
        Utilizado por el agente de WhatsApp cuando un paciente nuevo confirma su cita.
        """
        cedula_clean = str(cedula).strip()
        parts = str(nombre).strip().split()
        first_name = parts[0] if parts else "Paciente"
        last_name = " ".join(parts[1:]) if len(parts) > 1 else "Nexus"

        # Tipo de documento CC por defecto (Citizenship Card)
        doc_type_id = "e0000000-0000-0000-0000-000000000001"
        try:
            doc_types = await self.obtener_tipos_documento()
            if doc_types:
                for dt in doc_types:
                    c = (dt.get("code") or dt.get("name") or "").upper()
                    if "CC" in c or "CEDULA" in c or "CITIZEN" in c:
                        doc_type_id = str(dt.get("id"))
                        break
                if not doc_type_id and doc_types:
                    doc_type_id = str(doc_types[0].get("id"))
        except Exception:
            pass

        # Sexo por defecto
        sex_id = "f0000000-0000-0000-0000-000000000002"
        try:
            sexes = await self.obtener_sexos()
            if sexes:
                sex_id = str(sexes[0].get("id"))
        except Exception:
            pass

        # Si el teléfono de whatsapp es un LID, intentar resolver su teléfono canónico
        from app.services.whatsapp_identity import obtener_telefono_canonico
        resolved_phone = obtener_telefono_canonico(str(telefono_whatsapp or ""))
        clean_phone = "".join(ch for ch in resolved_phone if ch.isdigit())
        if len(clean_phone) > 12:
            clean_phone = ""
        if not clean_phone:
            clean_phone = "+573000000000"
        elif not clean_phone.startswith("+"):
            clean_phone = f"+{clean_phone}"

        onboard_payload = {
            "documentTypeId": doc_type_id,
            "documentNumber": cedula_clean,
            "firstName": first_name,
            "lastName": last_name,
            "dateOfBirth": "2000-01-01",
            "sexId": sex_id,
            "phone": clean_phone,
            "email": f"paciente_{cedula_clean}@nexusodonto.com",
            "address": "Consultorio Nexus Odonto",
            "emergencyContact": "Recepción Nexus",
            "emergencyPhone": "+573246030217",
            "password": cedula_clean,
            "mustChangePassword": True,
        }
        res = await self.registrar_paciente(onboard_payload)
        if res:
            logger.info(f"[.NET Client] Paciente básico creado para {cedula_clean} con ID: {res.get('patientId')}")
        return res

    async def login_paciente(self, document_number: str, password: str) -> Optional[Dict[str, Any]]:
        """Autentica un paciente mediante documento y contraseña.

        Endpoint: POST /api/auth/login
        Payload: {loginId, password}  (loginId = cédula / documentNumber)
        Retorna el JWT y datos de sesión si es exitoso.
        """
        login_endpoint = f"{self._auth_url}/login"
        payload = {"loginId": document_number, "password": password}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    login_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json; charset=utf-8", "Accept": "application/json"},
                )
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"[.NET Client] Login exitoso para documento: {document_number}")
                    return data
                logger.warning(
                    f"[.NET Client] Login fallido para {document_number} (HTTP {response.status_code})"
                )
                return None
        except Exception as e:
            logger.error(f"[.NET Client] Error en login: {e}")
            return None

    async def buscar_persona_por_telefono(self, telefono: str) -> Optional[Dict[str, Any]]:
        """Busca si existe una persona registrada con el teléfono dado.

        Reglas estrictas de privacidad y validación:
        - Descarta identificadores de WhatsApp tipo LID (@lid o > 11 dígitos no estándar).
        - Descarta teléfonos dummy/placeholder (+573000000000, 3000000000, 0000000000).
        - Requiere coincidencia exacta de los 10 dígitos móviles colombianos (3XXXXXXXXX).
        """
        from app.services.whatsapp_identity import obtener_telefono_canonico
        raw_str = obtener_telefono_canonico(str(telefono or "")).strip()
        if not raw_str or "@lid" in raw_str.lower():
            return None

        # Normalizar: quedarse solo con dígitos
        tel_digits = "".join(c for c in raw_str if c.isdigit())
        if not tel_digits or len(tel_digits) > 12:
            return None

        # Quitar prefijo de país colombiano 57 si aplica
        if tel_digits.startswith("57") and len(tel_digits) == 12:
            tel_digits = tel_digits[2:]

        # Solo números móviles colombianos válidos (10 dígitos que empiezan por 3)
        if len(tel_digits) != 10 or not tel_digits.startswith("3"):
            return None

        DUMMY_PHONES = {"3000000000", "0000000000", "1111111111", "1234567890"}
        if tel_digits in DUMMY_PHONES:
            return None

        personas = await self.obtener_personas()
        if not personas:
            return None

        for persona in personas:
            if not isinstance(persona, dict):
                continue
            phone = str(persona.get("phone") or "")
            phone_digits = "".join(c for c in phone if c.isdigit())
            if phone_digits.startswith("57") and len(phone_digits) == 12:
                phone_digits = phone_digits[2:]

            if phone_digits in DUMMY_PHONES or len(phone_digits) != 10 or not phone_digits.startswith("3"):
                continue

            if tel_digits == phone_digits:
                return persona

        return None

    async def buscar_persona_por_documento(self, document_number: str) -> Optional[Dict[str, Any]]:
        """Busca si existe una persona registrada con el número de documento dado, comparando dígitos."""
        if not document_number:
            return None
        personas = await self.obtener_personas()
        if not personas:
            return None

        clean_doc = str(document_number).strip().lower()
        doc_digits = "".join(c for c in clean_doc if c.isdigit())

        for persona in personas:
            if not isinstance(persona, dict):
                continue
            doc = str(persona.get("documentNumber") or "").strip().lower()
            p_digits = "".join(c for c in doc if c.isdigit())
            if doc == clean_doc or (doc_digits and doc_digits == p_digits):
                return persona
        return None

    async def crear_paciente_para_persona(
        self,
        person_id: str,
        contacto_emergencia: str = "Recepción Nexus",
        telefono_emergencia: str = "+573246030217",
    ) -> Optional[Dict[str, Any]]:
        """Crea el registro en la tabla Patients para una persona que ya existe en Persons."""
        url = f"{self.base_url}/Patients"
        payload = {
            "personId": str(person_id),
            "emergencyContact": contacto_emergencia,
            "emergencyPhone": telefono_emergencia,
        }
        res = await self._request_with_retry("POST", url, json=payload)
        if res and res.status_code in (200, 201):
            logger.info(f"[.NET Client] Registro de paciente creado exitosamente para personId: {person_id}")
            return res.json()
        if res:
            logger.warning(f"[.NET Client] Error creando paciente para personId {person_id} (HTTP {res.status_code}): {res.text}")
        return None

    async def buscar_paciente_por_person_id(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Busca un paciente por su personId para obtener el patientId."""
        url = f"{self.base_url}/Patients"
        response = await self._request_with_retry("GET", url)
        if response and response.status_code == 200:
            patients = response.json() if isinstance(response.json(), list) else response.json().get("items", [])
            for pt in patients:
                if isinstance(pt, dict) and str(pt.get("personId")).lower() == str(person_id).lower():
                    return pt
        return None

    async def buscar_citas_por_cedula(self, cedula: str) -> List[Dict[str, Any]]:
        """Busca todas las citas de un paciente identificado por su número de cédula.
        
        Flujo:
        1. Buscar la persona por documento.
        2. Encontrar el paciente asociado a esa persona.
        3. Retornar sus citas enriquecidas.
        """
        persona = await self.buscar_persona_por_documento(cedula)
        if not persona:
            return []
        person_id = persona.get("id")
        if not person_id:
            return []
        paciente = await self.buscar_paciente_por_person_id(str(person_id))
        if not paciente:
            return []
        patient_id = paciente.get("id")
        if not patient_id:
            return []
        return await self.obtener_citas_paciente(str(patient_id))

    async def cancelar_cita(self, cita_id: str, datos_cita: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Cancela una cita por su ID usando PUT con el estado CANCELADA.
        
        El backend usa CrudControllerBase que expone PUT (no PATCH).
        Retorna un dict con success y mensaje de error si aplica.
        """
        status_id = await self.obtener_appointment_status_id("CANCELADA")
        
        # Si no se pasó datos_cita o faltan fechas, intentar obtener la cita del backend
        if not datos_cita or not datos_cita.get("startsAt"):
            try:
                resp = await self._request_with_retry("GET", f"{self.base_url}/Appointments/{cita_id}")
                if resp and resp.status_code == 200:
                    datos_cita = resp.json()
            except Exception as e:
                logger.debug(f"[.NET Client] No se pudo obtener la cita {cita_id}: {e}")

        datos_cita = datos_cita or {}
        starts_at = datos_cita.get("startsAt") or datos_cita.get("fechaHoraInicio") or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        ends_at = datos_cita.get("endsAt") or datos_cita.get("fechaHoraFin") or (datetime.utcnow() + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")

        payload: Dict[str, Any] = {
            "patientId": str(datos_cita.get("patientId") or datos_cita.get("pacienteId") or "00000000-0000-0000-0000-000000000000"),
            "professionalId": str(datos_cita.get("professionalId") or "00000000-0000-0000-0000-000000000000"),
            "serviceId": str(datos_cita.get("serviceId") or "00000000-0000-0000-0000-000000000000"),
            "startsAt": starts_at,
            "endsAt": ends_at,
            "appointmentStatusId": str(status_id),
            "appointmentOriginId": str(datos_cita.get("appointmentOriginId") or "20000000-0000-0000-0000-000000000002"),
            "reasonForVisit": datos_cita.get("reasonForVisit") or "Cancelada vía Chatbot",
            "notes": datos_cita.get("notes"),
            "cancellationReason": "Cancelada por el paciente vía chatbot",
        }

        url = f"{self.base_url}/Appointments/{cita_id}"
        response = await self._request_with_retry("PUT", url, json=payload)
        if response and response.status_code in (200, 204):
            logger.info(f"[.NET Client] Cita {cita_id} cancelada exitosamente")
            return {"success": True}
        
        error_msg = "Error desconocido al cancelar la cita"
        if response is not None:
            try:
                err_json = response.json()
                error_msg = err_json.get("message") or err_json.get("title") or response.text[:300]
            except Exception:
                error_msg = response.text[:300]
            logger.warning(
                f"[.NET Client] No se pudo cancelar la cita {cita_id} "
                f"(HTTP {response.status_code}): {error_msg}"
            )
            return {"success": False, "status_code": response.status_code, "error": error_msg}
        else:
            logger.warning(f"[.NET Client] No se pudo cancelar la cita {cita_id} (sin respuesta)")
            return {"success": False, "status_code": None, "error": "No hubo respuesta del servidor"}

    async def modificar_cita(self, cita_id: str, datos_actualizacion: Dict[str, Any]) -> Dict[str, Any]:
        """Modifica los campos de una cita existente (fecha, horario, profesional, etc.).
        
        El backend usa CrudControllerBase que expone PUT, por lo que se usa PUT directamente.
        Retorna un dict con success, data y error si aplica.
        """
        url = f"{self.base_url}/Appointments/{cita_id}"
        response = await self._request_with_retry("PUT", url, json=datos_actualizacion)
        if response and response.status_code in (200, 204):
            logger.info(f"[.NET Client] Cita {cita_id} modificada exitosamente")
            try:
                data = response.json() if response.content else datos_actualizacion
            except Exception:
                data = datos_actualizacion
            return {"success": True, "data": data}
        
        error_msg = "Error desconocido al modificar la cita"
        if response is not None:
            try:
                err_json = response.json()
                error_msg = err_json.get("message") or err_json.get("title") or response.text[:300]
            except Exception:
                error_msg = response.text[:300]
            logger.warning(
                f"[.NET Client] No se pudo modificar la cita {cita_id} "
                f"(HTTP {response.status_code}): {error_msg}"
            )
            return {"success": False, "status_code": response.status_code, "error": error_msg}
        else:
            logger.warning(f"[.NET Client] No se pudo modificar la cita {cita_id} (sin respuesta)")
            return {"success": False, "status_code": None, "error": "No hubo respuesta del servidor"}

    # -------------------------------------------------------------
    # GESTIÓN DE CONVERSACIONES Y MENSAJES (ORACLE DB VIA .NET API)
    # -------------------------------------------------------------
    CHANNEL_WHATSAPP: str = "90000000-0000-0000-0000-000000000001"
    CHANNEL_TELEGRAM: str = "90000000-0000-0000-0000-000000000002"
    CHANNEL_WEBCHAT: str = "90000000-0000-0000-0000-000000000003"

    ROLE_USUARIO: str = "a0000000-0000-0000-0000-000000000001"
    ROLE_CHATBOT: str = "a0000000-0000-0000-0000-000000000002"
    ROLE_AGENTE_HUMANO: str = "a0000000-0000-0000-0000-000000000003"
    ROLE_SISTEMA: str = "a0000000-0000-0000-0000-000000000004"

    STATUS_ACTIVA: str = "b0000000-0000-0000-0000-000000000001"
    STATUS_ESCALADA: str = "b0000000-0000-0000-0000-000000000002"
    STATUS_ATENDIDA_HUMANO: str = "b0000000-0000-0000-0000-000000000003"
    STATUS_CERRADA: str = "b0000000-0000-0000-0000-000000000004"

    _conversations_cache: Dict[str, str] = {}

    async def obtener_o_crear_conversacion(
        self,
        chat_identifier: str,
        channel_id: Optional[str] = None,
        patient_id: Optional[str] = None,
    ) -> Optional[str]:
        """Obtiene el ID de una conversación activa o crea una nueva en el backend .NET / Oracle.
        
        Mantiene una caché en memoria para evitar llamadas redundantes de búsqueda.
        """
        from app.services.whatsapp_identity import obtener_telefono_canonico
        ident = obtener_telefono_canonico(str(chat_identifier or "")).strip()
        if not ident:
            return None

        # 1. Verificar caché en memoria rápido
        if ident in self._conversations_cache:
            return self._conversations_cache[ident]

        async with self._conversations_lock:
            # Re-verificar tras adquirir el lock por si otra corrutina concurrente ya la creó
            if ident in self._conversations_cache:
                return self._conversations_cache[ident]

            # 2. Consultar conversaciones existentes en el backend .NET
            url_get = f"{self.base_url}/ChatbotConversations"
            resp = await self._request_with_retry("GET", url_get)
            if resp and resp.status_code == 200:
                try:
                    items = resp.json()
                    if isinstance(items, dict):
                        items = items.get("items", [])
                    for conv in items:
                        if str(conv.get("chatIdentifier", "")).strip() == ident:
                            conv_id = str(conv.get("id"))
                            # Si la conversación ya existe (incluso cerrada o en otro estado), se reutiliza.
                            # Si estaba cerrada o en estado distinto de ACTIVA, se reabre en .NET.
                            # IMPORTANTE: Si estÃ¡ en ESCALADA o ATENDIDA_HUMANO, NUNCA sobreescribir a ACTIVA.
                            conv_status = str(conv.get("conversationStatusId", "")).lower()
                            is_human_or_escalated = conv_status in (
                                self.STATUS_ESCALADA.lower(),
                                self.STATUS_ATENDIDA_HUMANO.lower(),
                            )
                            if (conv.get("closedAt") or conv_status == self.STATUS_CERRADA.lower()) and not is_human_or_escalated:
                                try:
                                    url_put = f"{self.base_url}/ChatbotConversations/{conv_id}"
                                    payload_put = {
                                        "conversationStatusId": self.STATUS_ACTIVA,
                                        "patientId": conv.get("patientId") or patient_id,
                                        "closedAt": None,
                                    }
                                    await self._request_with_retry("PUT", url_put, json=payload_put)
                                    logger.info(f"[.NET Client] Conversación {conv_id} reabierta como ACTIVA para {ident}")
                                except Exception as e_put:
                                    logger.warning(f"[.NET Client] Error al reabrir conversación {conv_id}: {e_put}")

                            self._conversations_cache[ident] = conv_id
                            logger.info(f"[.NET Client] Conversación recuperada para {ident}: {conv_id}")
                            return conv_id
                except Exception as e:
                    logger.debug(f"[.NET Client] Error parseando conversaciones existentes: {e}")

            # 3. Si no se especificó patient_id, intentar buscar si el teléfono pertenece a un paciente registrado
            if not patient_id:
                try:
                    persona = await self.buscar_persona_por_telefono(ident)
                    if persona and persona.get("id"):
                        paciente = await self.buscar_paciente_por_person_id(str(persona["id"]))
                        if paciente and paciente.get("id"):
                            patient_id = str(paciente["id"])
                except Exception as pat_err:
                    logger.debug(f"[.NET Client] No se pudo autovincular paciente por teléfono: {pat_err}")

            # 4. Si no existe conversación activa, crear una nueva
            payload = {
                "chatIdentifier": ident,
                "chatChannelId": channel_id or self.CHANNEL_WHATSAPP,
                "patientId": patient_id,
            }
            resp_post = await self._request_with_retry("POST", url_get, json=payload)
            if resp_post and resp_post.status_code in (200, 201):
                try:
                    data = resp_post.json()
                    conv_id = str(data.get("id"))
                    if conv_id:
                        self._conversations_cache[ident] = conv_id
                        logger.info(f"[.NET Client] Nueva conversación creada en DB para {ident}: {conv_id}")
                        return conv_id
                except Exception as e:
                    logger.error(f"[.NET Client] Error parseando respuesta de creación de conversación: {e}")

            logger.warning(f"[.NET Client] No se pudo crear/obtener conversación en DB para {ident}")
            return None

    async def guardar_mensaje_conversacion(
        self,
        conversation_id: str,
        rol: str,
        contenido: str,
        rag_confidence: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Guarda un mensaje en la tabla chatbot_messages de la base de datos vía el endpoint de .NET."""
        if not conversation_id or not contenido:
            return None

        # Resolver el MessageRoleId según el rol
        rol_upper = str(rol).upper().strip()
        if rol_upper in ("USUARIO", "USER", "HUMAN"):
            role_id = self.ROLE_USUARIO
        elif rol_upper in ("CHATBOT", "ASSISTANT", "BOT"):
            role_id = self.ROLE_CHATBOT
        elif rol_upper in ("AGENTE_HUMANO", "ASESOR", "HUMAN_AGENT"):
            role_id = self.ROLE_AGENTE_HUMANO
        else:
            role_id = self.ROLE_SISTEMA

        url = f"{self.base_url}/ChatbotMessages"
        payload = {
            "chatbotConversationId": conversation_id,
            "messageRoleId": role_id,
            "content": str(contenido).strip(),
            "ragConfidence": float(rag_confidence) if rag_confidence is not None else None,
        }

        resp = await self._request_with_retry("POST", url, json=payload)
        if resp and resp.status_code in (200, 201):
            try:
                data = resp.json()
                logger.info(f"[.NET Client] Mensaje ({rol_upper}) guardado en DB para conv {conversation_id}")
                return data
            except Exception:
                return payload
        else:
            status = resp.status_code if resp else "Sin respuesta"
            logger.warning(f"[.NET Client] No se pudo guardar mensaje ({rol_upper}) en DB (HTTP {status})")
            return None

    async def registrar_mensaje(
        self,
        chat_identifier: str,
        rol: str,
        contenido: str,
        rag_confidence: Optional[float] = None,
        patient_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Registra un mensaje asegurando que la conversación exista en base de datos.
        
        Obtiene o crea la conversación en Oracle y luego inserta el mensaje en chatbot_messages.
        """
        try:
            if not contenido or not str(contenido).strip():
                return None
            conv_id = await self.obtener_o_crear_conversacion(
                chat_identifier=chat_identifier,
                patient_id=patient_id,
            )
            if not conv_id:
                logger.warning(f"[.NET Client] No se pudo obtener/crear conversación para {chat_identifier}")
                return None
            return await self.guardar_mensaje_conversacion(
                conversation_id=conv_id,
                rol=rol,
                contenido=contenido,
                rag_confidence=rag_confidence,
            )
        except Exception as e:
            logger.error(f"[.NET Client] Error registrando mensaje para {chat_identifier}: {e}")
            return None

    def limpiar_cache_conversacion(self, chat_identifier: str) -> None:
        """Limpia el ID en caché cuando la conversación se reinicia."""
        self._conversations_cache.pop(str(chat_identifier).strip(), None)

    async def obtener_citas_agendadas_para_recordatorio(self, fecha: str) -> List[Dict[str, Any]]:
        """Obtiene las citas programadas para una fecha específica, enriquecidas con paciente, teléfono, servicio y doctor.
        
        Solo incluye citas activas no canceladas para emitir los recordatorios por WhatsApp.
        """
        clean_fecha = str(fecha).strip()
        # 1. Intentar consultar citas filtradas por fecha o todas sin filtro
        citas_raw = await self.consultar_citas(clean_fecha)
        if not citas_raw:
            citas_raw = await self.consultar_citas("")
        if not citas_raw:
            return []
        
        citas_list = citas_raw if isinstance(citas_raw, list) else citas_raw.get("items", [])
        
        # 2. Resolver dinámicamente el ID del estado AGENDADA
        status_agendada_id = await self.obtener_appointment_status_id("AGENDADA")
        valid_status_ids = {status_agendada_id.lower(), "10000000-0000-0000-0000-000000000001"}
        
        # Filtrar por fecha de inicio y descartar canceladas
        citas_filtradas = []
        for c in citas_list:
            if not isinstance(c, dict):
                continue
            starts = str(c.get("startsAt") or c.get("fechaHoraInicio") or "")
            status_id = str(c.get("appointmentStatusId") or "").lower()
            is_cancelled = bool(c.get("cancelledAt")) or "cancel" in str(c.get("statusName", "")).lower()
            
            date_matches = clean_fecha in starts or starts.startswith(clean_fecha)
            status_matches = (status_id in valid_status_ids) or not status_id
            
            if date_matches and not is_cancelled and status_matches:
                citas_filtradas.append(c)
                
        if not citas_filtradas:
            return []

        # Cargar catálogos y entidades para enriquecer
        try:
            from datetime import datetime
            patients = await self.obtener_catalogo("Patients") or []
            persons = await self.obtener_personas() or []
            profs = await self.obtener_profesionales() or []
            servs = await self.obtener_servicios() or []
            convs = await self.obtener_catalogo("ChatbotConversations") or []

            patient_map = {str(p.get("id")).lower(): p for p in patients if isinstance(p, dict)}
            person_map = {str(p.get("id")).lower(): p for p in persons if isinstance(p, dict)}
            prof_map = {str(p.get("id")).lower(): p.get("name") or p.get("nombre") for p in profs if isinstance(p, dict)}
            serv_map = {str(s.get("id")).lower(): s.get("name") or s.get("nombre") for s in servs if isinstance(s, dict)}
            
            # Mapeo de paciente a chatIdentifier de WhatsApp
            patient_conv_map = {}
            for cv in convs:
                if isinstance(cv, dict):
                    pid = str(cv.get("patientId") or "").lower()
                    chat_id = cv.get("chatIdentifier") or ""
                    if pid and chat_id and "@" in chat_id:
                        patient_conv_map[pid] = chat_id

            citas_enriquecidas = []
            for c in citas_filtradas:
                cita_id = str(c.get("id"))
                patient_id = str(c.get("patientId") or "").lower()
                prof_id = str(c.get("professionalId") or "").lower()
                serv_id = str(c.get("serviceId") or "").lower()

                patient_obj = patient_map.get(patient_id, {})
                person_id = str(patient_obj.get("personId") or "").lower()
                person_obj = person_map.get(person_id, {})

                first_name = (person_obj.get("firstName") or "").strip()
                last_name = (person_obj.get("lastName") or "").strip()
                full_name = f"{first_name} {last_name}".strip() or "Paciente"
                doc_number = (person_obj.get("documentNumber") or "").strip()

                # Resolver teléfono: Person.phone -> ChatbotConversation -> Patient.emergencyPhone
                phone_raw = (person_obj.get("phone") or "").strip()
                if not phone_raw and patient_id in patient_conv_map:
                    phone_raw = patient_conv_map[patient_id]
                if not phone_raw:
                    phone_raw = (patient_obj.get("emergencyPhone") or "").strip()

                # Normalizar teléfono para WhatsApp (Evolution API)
                if "@lid" in phone_raw:
                    phone_clean = phone_raw
                else:
                    digits = "".join(ch for ch in phone_raw if ch.isdigit())
                    if len(digits) == 10:
                        phone_clean = f"57{digits}"
                    elif len(digits) > 10:
                        phone_clean = digits
                    else:
                        phone_clean = digits

                # Formatear hora (ej. 11:30 AM)
                starts_at_str = str(c.get("startsAt") or c.get("fechaHoraInicio") or "")
                time_formatted = starts_at_str
                try:
                    dt = datetime.fromisoformat(starts_at_str)
                    time_formatted = dt.strftime("%I:%M %p").lstrip("0")
                except Exception:
                    pass

                citas_enriquecidas.append({
                    "id": cita_id,
                    "startsAt": starts_at_str,
                    "timeFormatted": time_formatted,
                    "date": clean_fecha,
                    "patientId": patient_id,
                    "patientName": full_name,
                    "documentNumber": doc_number,
                    "phone": phone_clean,
                    "professionalId": prof_id,
                    "professionalName": prof_map.get(prof_id, "Especialista Odontológico"),
                    "serviceId": serv_id,
                    "serviceName": serv_map.get(serv_id, "Consulta Odontológica"),
                    "reasonForVisit": c.get("reasonForVisit") or "Consulta Odontológica",
                })

            return citas_enriquecidas
        except Exception as e:
            logger.error(f"[.NET Client] Error enriqueciendo citas para recordatorio: {e}", exc_info=True)
            return []

    async def confirmar_estado_cita(self, cita_id: str) -> Dict[str, Any]:
        """Actualiza el estado de una cita a CONFIRMADA (10000000-0000-0000-0000-000000000002)."""
        STATUS_CONFIRMADA = "10000000-0000-0000-0000-000000000002"
        # Obtener datos de la cita actual para mantener los demás campos
        url_get = f"{self.base_url}/Appointments/{cita_id}"
        resp_get = await self._request_with_retry("GET", url_get)
        if not resp_get or resp_get.status_code != 200:
            return {"success": False, "error": f"No se pudo consultar la cita {cita_id}"}
        
        cita = resp_get.json()
        payload = {
            "patientId": cita.get("patientId"),
            "professionalId": cita.get("professionalId"),
            "serviceId": cita.get("serviceId"),
            "startsAt": cita.get("startsAt"),
            "endsAt": cita.get("endsAt"),
            "appointmentStatusId": STATUS_CONFIRMADA,
            "appointmentOriginId": cita.get("appointmentOriginId"),
            "reasonForVisit": cita.get("reasonForVisit"),
            "notes": ((cita.get("notes") or "") + " | Confirmada vía Chatbot WhatsApp").strip(),
        }
        url_put = f"{self.base_url}/Appointments/{cita_id}"
        resp_put = await self._request_with_retry("PUT", url_put, json=payload)
        if resp_put and resp_put.status_code in (200, 204):
            logger.info(f"[.NET Client] Cita {cita_id} confirmada exitosamente en el sistema.")
            return {"success": True, "citaId": cita_id, "status": "CONFIRMADA"}
        
        err = resp_put.text[:300] if resp_put else "Sin respuesta"
        logger.warning(f"[.NET Client] Error confirmando cita {cita_id}: {err}")
        return {"success": False, "error": err}


# Instancia reutilizable para el bot y las tools
dotnet_client = DotNetClient()
