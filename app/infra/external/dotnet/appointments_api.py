"""Endpoints de Citas y Agendamiento en la API .NET."""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from app.infra.external.dotnet.http_transport import dotnet_transport, DotNetHttpTransport
from app.infra.external.dotnet.catalog_api import catalog_api
from app.infra.external.dotnet.patients_api import patients_api

logger = logging.getLogger(__name__)


class DotNetAppointmentsApi:
    def __init__(self, transport: Optional[DotNetHttpTransport] = None):
        self.transport = transport or dotnet_transport

    async def agendar_cita(self, datos_cita: Dict[str, Any]) -> Dict[str, Any]:
        """Registra una cita en el backend .NET."""
        response = await self.transport.request("POST", "Appointments", json=datos_cita)
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
                f"[AppointmentsApi] Error creando cita (HTTP {status_code}): {error_msg or response.text[:300]}"
            )
        return {
            "success": False,
            "error": error_msg or "No fue posible registrar la cita en el sistema.",
            "status_code": status_code,
        }

    async def consultar_citas(self, fecha: str = "") -> Optional[Any]:
        """Consulta las citas de una fecha en el backend .NET con fallback seguro."""
        clean_f = str(fecha).strip() if fecha else ""
        if clean_f:
            response = await self.transport.request("GET", "Appointments", params={"date": clean_f})
            if response and response.status_code == 200:
                payload = response.json()
                items = payload if isinstance(payload, list) else payload.get("items", [])
                if items:
                    return payload
            response = await self.transport.request("GET", "Appointments", params={"fecha": clean_f})
            if response and response.status_code == 200:
                payload = response.json()
                items = payload if isinstance(payload, list) else payload.get("items", [])
                if items:
                    return payload

        # Fallback: consultar sin parámetros
        response = await self.transport.request("GET", "Appointments")
        if response and response.status_code == 200:
            return response.json()
        return None

    async def obtener_citas_paciente(self, patient_id: str) -> List[Dict[str, Any]]:
        """Obtiene todas las citas de un paciente dado su patientId."""
        citas_raw = await self.consultar_citas()
        if not citas_raw:
            return []
        items = citas_raw.get("items", citas_raw) if isinstance(citas_raw, dict) else citas_raw
        if not isinstance(items, list):
            return []
        return [
            c for c in items
            if str(c.get("patientId") or c.get("pacienteId") or "").lower() == str(patient_id).lower()
        ]

    async def buscar_citas_por_cedula(self, cedula: str) -> List[Dict[str, Any]]:
        """Busca todas las citas de un paciente identificado por su cédula."""
        persona = await patients_api.buscar_persona_por_documento(cedula)
        if not persona:
            return []
        person_id = persona.get("id")
        if not person_id:
            return []
        paciente = await patients_api.buscar_paciente_por_person_id(str(person_id))
        if not paciente:
            return []
        patient_id = paciente.get("id")
        if not patient_id:
            return []
        return await self.obtener_citas_paciente(str(patient_id))

    async def cancelar_cita(self, cita_id: str, datos_cita: Optional[Any] = None) -> Dict[str, Any]:
        """Cancela una cita por su ID usando PUT con el estado CANCELADA."""
        status_id = await catalog_api.obtener_appointment_status_id("CANCELADA")

        motivo_extra = datos_cita if isinstance(datos_cita, str) else ""
        if not isinstance(datos_cita, dict) or not datos_cita.get("startsAt"):
            try:
                resp = await self.transport.request("GET", f"Appointments/{cita_id}")
                if resp and resp.status_code == 200:
                    datos_cita = resp.json()
            except Exception as e:
                logger.debug(f"[AppointmentsApi] No se pudo obtener la cita {cita_id}: {e}")

        datos_cita = datos_cita if isinstance(datos_cita, dict) else {}
        starts_at = (
            datos_cita.get("startsAt")
            or datos_cita.get("fechaHoraInicio")
            or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        ends_at = (
            datos_cita.get("endsAt")
            or datos_cita.get("fechaHoraFin")
            or (datetime.utcnow() + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        )

        cancellation_reason = (
            motivo_extra
            or datos_cita.get("cancellationReason")
            or "Cancelada por el paciente vía chatbot"
        )

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
            "cancellationReason": cancellation_reason,
        }

        response = await self.transport.request("PUT", f"Appointments/{cita_id}", json=payload)
        if response and response.status_code in (200, 204):
            logger.info(f"[AppointmentsApi] Cita {cita_id} cancelada exitosamente")
            return {"success": True}

        error_msg = "Error desconocido al cancelar la cita"
        if response is not None:
            try:
                err_json = response.json()
                error_msg = err_json.get("message") or err_json.get("title") or response.text[:300]
            except Exception:
                error_msg = response.text[:300]
            return {"success": False, "status_code": response.status_code, "error": error_msg}
        return {"success": False, "status_code": None, "error": "No hubo respuesta del servidor"}

    async def modificar_cita(self, cita_id: str, datos_actualizacion: Dict[str, Any]) -> Dict[str, Any]:
        """Modifica una cita existente (fecha, horario, profesional) usando PUT."""
        response = await self.transport.request("PUT", f"Appointments/{cita_id}", json=datos_actualizacion)
        if response and response.status_code in (200, 204):
            logger.info(f"[AppointmentsApi] Cita {cita_id} modificada exitosamente")
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
            return {"success": False, "status_code": response.status_code, "error": error_msg}
        return {"success": False, "status_code": None, "error": "No hubo respuesta del servidor"}

    async def confirmar_estado_cita(self, cita_id: str) -> Dict[str, Any]:
        """Actualiza el estado de una cita a CONFIRMADA (10000000-0000-0000-0000-000000000002)."""
        STATUS_CONFIRMADA = "10000000-0000-0000-0000-000000000002"
        resp_get = await self.transport.request("GET", f"Appointments/{cita_id}")
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
        resp_put = await self.transport.request("PUT", f"Appointments/{cita_id}", json=payload)
        if resp_put and resp_put.status_code in (200, 204):
            logger.info(f"[AppointmentsApi] Cita {cita_id} confirmada exitosamente en el sistema.")
            return {"success": True, "citaId": cita_id, "status": "CONFIRMADA"}

        err = resp_put.text[:300] if resp_put else "Sin respuesta"
        logger.warning(f"[AppointmentsApi] Error confirmando cita {cita_id}: {err}")
        return {"success": False, "error": err}


appointments_api = DotNetAppointmentsApi()
