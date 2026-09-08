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
                            response = await client.post(
                                login_endpoint,
                                json=payload,
                                headers={"Content-Type": "application/json", "Accept": "application/json"},
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

    async def agendar_cita(self, datos_cita: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Registra una cita en el backend .NET."""
        url = f"{self.base_url}/Appointments"
        response = await self._request_with_retry("POST", url, json=datos_cita)
        if response and response.status_code in (200, 201):
            return response.json()
        if response:
            logger.warning(
                f"[.NET Client] Error creando cita (HTTP {response.status_code}): {response.text[:500]}"
            )
        return None

    async def consultar_citas(self, fecha: str) -> Optional[Any]:
        """Consulta las citas de una fecha en el backend .NET."""
        url = f"{self.base_url}/Appointments"
        params = {"fecha": fecha, "date": fecha}
        response = await self._request_with_retry("GET", url, params=params)
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
                c["statusName"] = status_dict.get(st_id, "Agendada")
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
        """Actualiza el estado de una conversación en el backend .NET / Oracle."""
        try:
            url = f"{self.base_url}/ChatbotConversations/{conversacion_id}"
            payload = {
                "conversationStatusId": status_id,
                "patientId": patient_id,
            }
            resp = await self._request_with_retry("PUT", url, json=payload)
            if resp and resp.status_code in (200, 204):
                logger.info(f"[.NET Client] Estado de conversación {conversacion_id} actualizado a {status_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"[.NET Client] Error actualizando estado de conversación {conversacion_id}: {e}")
            return False

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

    async def obtener_appointment_status_id(self, code: str = "AGENDADA") -> str:
        """Obtiene el ID del estado de cita por código."""
        url = f"{self.base_url}/AppointmentStatuses"
        response = await self._request_with_retry("GET", url)
        if response and response.status_code == 200:
            statuses = response.json() if isinstance(response.json(), list) else []
            for s in statuses:
                if str(s.get("code")).upper() == code.upper():
                    return str(s.get("id"))
        return "10000000-0000-0000-0000-000000000001"

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
        # Sexo Masculino por defecto
        sex_id = "f0000000-0000-0000-0000-000000000002"

        # Si el teléfono de whatsapp es un LID (>13 dígitos), no usarlo como número telefónico móvil
        clean_phone = "".join(ch for ch in str(telefono_whatsapp or "") if ch.isdigit())
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
            "password": f"Nexus{cedula_clean}!"
        }
        res = await self.registrar_paciente(onboard_payload)
        if res:
            logger.info(f"[.NET Client] Paciente básico creado para {cedula_clean} con ID: {res.get('patientId')}")
        return res

    async def login_paciente(self, document_number: str, password: str) -> Optional[Dict[str, Any]]:
        """Autentica un paciente mediante documento y contraseña.

        Endpoint: POST /api/auth/login
        Payload: {documentNumber, password}
        Retorna el JWT y datos de sesión si es exitoso.
        """
        login_endpoint = f"{self._auth_url}/login"
        payload = {"documentNumber": document_number, "password": password}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    login_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
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

        Compara los últimos 10 dígitos del teléfono para manejar variaciones de formato
        (con/sin código de país, con/sin sufijo de WhatsApp).
        """
        personas = await self.obtener_personas()
        if not personas:
            return None

        # Normalizar: quedarse solo con dígitos
        tel_digits = "".join(c for c in str(telefono) if c.isdigit())
        # Usar últimos 10 dígitos para comparación (sin código de país)
        tel_suffix = tel_digits[-10:] if len(tel_digits) >= 10 else tel_digits

        for persona in personas:
            if not isinstance(persona, dict):
                continue
            phone = persona.get("phone") or ""
            phone_digits = "".join(c for c in str(phone) if c.isdigit())
            phone_suffix = phone_digits[-10:] if len(phone_digits) >= 10 else phone_digits

            if tel_suffix and phone_suffix and tel_suffix == phone_suffix:
                return persona

        return None

    async def buscar_persona_por_documento(self, document_number: str) -> Optional[Dict[str, Any]]:
        """Busca si existe una persona registrada con el número de documento dado."""
        if not document_number:
            return None
        personas = await self.obtener_personas()
        if not personas:
            return None

        clean_doc = str(document_number).strip().lower()
        for persona in personas:
            if not isinstance(persona, dict):
                continue
            doc = str(persona.get("documentNumber") or "").strip().lower()
            if doc == clean_doc:
                return persona
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
        ident = str(chat_identifier).strip()
        if not ident:
            return None

        LID_MAPPING = {
            "233783743803574@lid": "573001112233@s.whatsapp.net",
            "189515549421795@lid": "573226688304@s.whatsapp.net",
            "57213628510462@lid": "573238891073@s.whatsapp.net",
        }
        if ident in LID_MAPPING:
            ident = LID_MAPPING[ident]

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
                        if (
                            str(conv.get("chatIdentifier", "")).strip() == ident
                            and not conv.get("closedAt")
                        ):
                            conv_id = str(conv.get("id"))
                            self._conversations_cache[ident] = conv_id
                            logger.info(f"[.NET Client] Conversación activa recuperada para {ident}: {conv_id}")
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


# Instancia reutilizable para el bot y las tools
dotnet_client = DotNetClient()