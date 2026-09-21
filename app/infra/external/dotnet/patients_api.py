"""Endpoints de Pacientes y Personas en la API .NET."""

import logging
from typing import Optional, List, Dict, Any
import httpx
from app.infra.external.dotnet.http_transport import dotnet_transport, DotNetHttpTransport
from app.infra.external.dotnet.catalog_api import catalog_api

logger = logging.getLogger(__name__)


class DotNetPatientsApi:
    def __init__(self, transport: Optional[DotNetHttpTransport] = None):
        self.transport = transport or dotnet_transport

    async def buscar_pacientes(self, search: str = "") -> Optional[List[Dict[str, Any]]]:
        """Obtiene la lista de pacientes desde el backend .NET."""
        params = {"pageSize": 100}
        if search:
            params["search"] = search
        response = await self.transport.request("GET", "Patients", params=params)
        if response and response.status_code == 200:
            payload = response.json()
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict):
                return payload.get("items", [])
        return None

    async def vincular_paciente(self, conversacion_chatbot_id: str, paciente_id: Any) -> bool:
        """Asocia un paciente a una conversación de chatbot existente."""
        payload = {"patientId": str(paciente_id)}
        response = await self.transport.request(
            "PATCH", f"ChatbotConversations/{conversacion_chatbot_id}", json=payload
        )
        if response and response.status_code in (200, 204):
            logger.info(f"[PatientsApi] Paciente {paciente_id} vinculado a conversación {conversacion_chatbot_id}")
            return True
        return False

    async def registrar_paciente(self, datos_onboard: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Registra un paciente completo usando el endpoint de onboard de .NET."""
        response = await self.transport.request("POST", "Patients/onboard", json=datos_onboard)
        if response and response.status_code in (200, 201):
            return response.json()
        status = response.status_code if response else "Sin respuesta"
        logger.warning(f"[PatientsApi] Error en onboarding de paciente (HTTP {status})")
        return None

    async def crear_paciente_basico(
        self,
        cedula: str = "",
        nombre: str = "",
        telefono_whatsapp: Optional[str] = None,
        document_number: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        phone: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Crea un perfil de paciente básico en el sistema usando datos mínimos al agendar."""
        doc_num = (cedula or document_number or "").strip()
        full_name = (nombre or f"{first_name or ''} {last_name or ''}").strip()
        tel = telefono_whatsapp or phone
        cedula_clean = str(doc_num).strip()
        parts = full_name.split()
        first_name_val = parts[0] if parts else "Paciente"
        last_name_val = " ".join(parts[1:]) if len(parts) > 1 else "Nexus"

        doc_type_id = "e0000000-0000-0000-0000-000000000001"
        try:
            doc_types = await catalog_api.obtener_tipos_documento()
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

        sex_id = "f0000000-0000-0000-0000-000000000002"
        try:
            sexes = await catalog_api.obtener_sexos()
            if sexes:
                sex_id = str(sexes[0].get("id"))
        except Exception:
            pass

        from app.services.whatsapp_identity import obtener_telefono_canonico
        resolved_phone = obtener_telefono_canonico(str(tel or ""))
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
            "firstName": first_name_val,
            "lastName": last_name_val,
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
            logger.info(f"[PatientsApi] Paciente básico creado para {cedula_clean} con ID: {res.get('patientId')}")
        return res

    async def login_paciente(self, document_number: str, password: str) -> Optional[Dict[str, Any]]:
        """Autentica un paciente mediante documento y contraseña."""
        login_endpoint = f"{self.transport.auth_url}/login"
        payload = {"loginId": document_number, "password": password}
        try:
            async with httpx.AsyncClient(timeout=self.transport.timeout) as client:
                response = await client.post(
                    login_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json; charset=utf-8", "Accept": "application/json"},
                )
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"[PatientsApi] Login exitoso para documento: {document_number}")
                    return data
                return None
        except Exception as e:
            logger.error(f"[PatientsApi] Error en login: {e}")
            return None

    async def buscar_persona_por_telefono(self, telefono: str) -> Optional[Dict[str, Any]]:
        """Busca si existe una persona registrada con el teléfono dado con reglas de privacidad."""
        from app.services.whatsapp_identity import obtener_telefono_canonico
        raw_str = obtener_telefono_canonico(str(telefono or "")).strip()
        if not raw_str or "@lid" in raw_str.lower():
            return None

        tel_digits = "".join(c for c in raw_str if c.isdigit())
        if not tel_digits or len(tel_digits) > 12:
            return None

        if tel_digits.startswith("57") and len(tel_digits) == 12:
            tel_digits = tel_digits[2:]

        if len(tel_digits) != 10 or not tel_digits.startswith("3"):
            return None

        DUMMY_PHONES = {"3000000000", "0000000000", "1111111111", "1234567890"}
        if tel_digits in DUMMY_PHONES:
            return None

        personas = await catalog_api.obtener_personas()
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
        """Busca si existe una persona registrada con el número de documento dado."""
        if not document_number:
            return None
        personas = await catalog_api.obtener_personas()
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
        """Crea el registro en la tabla Patients para una persona existente."""
        payload = {
            "personId": str(person_id),
            "emergencyContact": contacto_emergencia,
            "emergencyPhone": telefono_emergencia,
        }
        res = await self.transport.request("POST", "Patients", json=payload)
        if res and res.status_code in (200, 201):
            logger.info(f"[PatientsApi] Registro de paciente creado para personId: {person_id}")
            return res.json()
        return None

    async def buscar_paciente_por_person_id(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Busca un paciente por su personId para obtener el patientId."""
        response = await self.transport.request("GET", "Patients")
        if response and response.status_code == 200:
            payload = response.json()
            patients = payload if isinstance(payload, list) else payload.get("items", [])
            for pt in patients:
                if isinstance(pt, dict) and str(pt.get("personId")).lower() == str(person_id).lower():
                    return pt
        return None


patients_api = DotNetPatientsApi()
