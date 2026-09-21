"""Servicio de dominio y aplicación para Citas Médicas en Nexus Odonto.
Encapsula la lógica de negocio para disponibilidad, validación de franjas de almuerzo,
creación automática de pacientes y enriquecimiento de citas para recordatorios.
"""

import logging
from datetime import datetime, date, time
from typing import Optional, List, Dict, Any, Tuple

from app.domain.clinical.schedule_rules import (
    es_horario_almuerzo,
    es_hora_laboral_valida,
    es_dia_laboral,
    redondear_a_bloque_30_minutos,
    obtener_primer_turno_tarde,
)
from app.domain.security.habeas_data import validar_cedula
from app.infra.external.dotnet.appointments_api import appointments_api, DotNetAppointmentsApi
from app.infra.external.dotnet.patients_api import patients_api, DotNetPatientsApi
from app.infra.external.dotnet.catalog_api import catalog_api, DotNetCatalogApi

logger = logging.getLogger(__name__)


class AppointmentService:
    """Orquesta las operaciones de citas aplicando las reglas clínicas de Nexus Odonto."""

    def __init__(
        self,
        appt_api: Optional[DotNetAppointmentsApi] = None,
        pat_api: Optional[DotNetPatientsApi] = None,
        cat_api: Optional[DotNetCatalogApi] = None,
    ):
        self.appt_api = appt_api or appointments_api
        self.pat_api = pat_api or patients_api
        self.cat_api = cat_api or catalog_api

    def validar_horario_cita(self, dt: datetime) -> Tuple[bool, str]:
        """Valida que una fecha y hora cumplan con las reglas clínicas de Nexus Odonto."""
        fecha = dt.date()
        hora = dt.time()

        # 1. Validar día laboral
        es_laboral, motivo_dia = es_dia_laboral(fecha)
        if not es_laboral:
            return False, motivo_dia

        # 2. Validar almuerzo y jornada
        es_sabado = fecha.weekday() == 5
        es_valido, motivo_hora = es_hora_laboral_valida(hora, es_sabado=es_sabado)
        if not es_valido:
            return False, motivo_hora

        return True, "Horario clínico válido"

    async def asegurar_paciente(
        self,
        cedula: str,
        nombre_completo: str,
        telefono: Optional[str] = None,
        conversacion_id: Optional[str] = None,
    ) -> Optional[str]:
        """Busca o crea al paciente en el sistema .NET por su cédula y devuelve su patientId."""
        is_valid, clean_doc, err_msg = validar_cedula(cedula)
        if not is_valid or not clean_doc:
            logger.warning(f"[AppointmentService] Cédula inválida al asegurar paciente: {cedula} ({err_msg})")
            return None

        # 1. Buscar si el paciente ya existe en .NET
        pacientes = await self.pat_api.buscar_pacientes(clean_doc)
        if pacientes:
            for p in pacientes:
                doc = str(p.get("documentNumber") or p.get("numeroDocumento") or "")
                if "".join(c for c in doc if c.isdigit()) == clean_doc:
                    pid = str(p.get("id") or "")
                    if conversacion_id and pid:
                        await self.pat_api.vincular_paciente(conversacion_id, pid)
                    return pid

        # 2. Buscar si la persona existe pero no como paciente
        persona = await self.pat_api.buscar_persona_por_documento(clean_doc)
        if persona:
            person_id = str(persona.get("id") or "")
            if person_id:
                nuevo_p = await self.pat_api.crear_paciente_para_persona(person_id)
                if nuevo_p:
                    pid = str(nuevo_p.get("id") or "")
                    if conversacion_id and pid:
                        await self.pat_api.vincular_paciente(conversacion_id, pid)
                    return pid

        # 3. Crear paciente desde cero con datos básicos
        partes = nombre_completo.strip().split()
        first_name = partes[0] if partes else "Paciente"
        last_name = " ".join(partes[1:]) if len(partes) > 1 else "Nexus"

        nuevo_paciente = await self.pat_api.crear_paciente_basico(
            cedula=clean_doc,
            nombre=nombre_completo,
            telefono_whatsapp=telefono,
        )
        if nuevo_paciente:
            pid = str(nuevo_paciente.get("id") or "")
            if conversacion_id and pid:
                await self.pat_api.vincular_paciente(conversacion_id, pid)
            return pid

        return None

    async def enriquecer_citas_para_recordatorio(self, fecha: str) -> List[Dict[str, Any]]:
        """Cruza y enriquece las citas programadas de una fecha con datos de pacientes, doctores y servicios.
        Sustituye la lógica acoplada que antes vivía dentro del cliente HTTP.
        """
        citas_raw = await self.appt_api.consultar_citas(fecha=fecha)
        if not citas_raw:
            return []

        citas_list = citas_raw.get("items", citas_raw) if isinstance(citas_raw, dict) else citas_raw
        if not isinstance(citas_list, list):
            return []

        # Filtrar solo citas agendadas o confirmadas activas para esa fecha
        citas_activas = [
            c for c in citas_list
            if str(c.get("status") or c.get("estado") or "").upper() in ("AGENDADA", "CONFIRMADA", "PROGRAMADA")
        ]

        if not citas_activas:
            return []

        # Cargar catálogos auxiliares para el cruce de datos
        patients = await self.pat_api.buscar_pacientes("") or []
        persons = await self.cat_api.obtener_personas() or []
        profs = await self.cat_api.obtener_profesionales() or []
        servs = await self.cat_api.obtener_servicios() or []
        convs = await self.cat_api.obtener_catalogo("ChatbotConversations") or []

        patient_map = {str(p.get("id")).lower(): p for p in patients if isinstance(p, dict)}
        person_map = {str(p.get("id")).lower(): p for p in persons if isinstance(p, dict)}
        prof_map = {str(p.get("id")).lower(): (p.get("name") or p.get("nombre")) for p in profs if isinstance(p, dict)}
        serv_map = {str(s.get("id")).lower(): (s.get("name") or s.get("nombre")) for s in servs if isinstance(s, dict)}

        # Mapeo de paciente a teléfono / chatIdentifier
        patient_phone_map = {}
        for cv in convs:
            if isinstance(cv, dict):
                pid = str(cv.get("patientId") or "").lower()
                chat_id = cv.get("chatIdentifier") or ""
                if pid and chat_id:
                    patient_phone_map[pid] = chat_id

        citas_enriquecidas = []
        for c in citas_activas:
            cita_id = str(c.get("id"))
            patient_id = str(c.get("patientId") or "").lower()
            prof_id = str(c.get("professionalId") or "").lower()
            serv_id = str(c.get("serviceId") or "").lower()

            patient_obj = patient_map.get(patient_id, {})
            person_id = str(patient_obj.get("personId") or "").lower()
            person_obj = person_map.get(person_id, {})

            # Extraer nombres
            p_name = (
                person_obj.get("fullName")
                or person_obj.get("nombreCompleto")
                or f"{person_obj.get('firstName', '')} {person_obj.get('lastName', '')}".strip()
                or patient_obj.get("fullName")
                or "Paciente"
            )
            doc_number = str(person_obj.get("documentNumber") or patient_obj.get("documentNumber") or "")
            phone_raw = (
                patient_phone_map.get(patient_id)
                or str(person_obj.get("phone") or patient_obj.get("phone") or "")
            ).strip()

            if "@lid" in phone_raw:
                phone_clean = phone_raw
            else:
                digits = "".join(ch for ch in phone_raw if ch.isdigit())
                if len(digits) == 10:
                    phone_clean = f"57{digits}"
                else:
                    phone_clean = digits

            prof_name = prof_map.get(prof_id) or "Especialista Nexus Odonto"
            serv_name = serv_map.get(serv_id) or "Consulta Odontológica"

            # Formatear horario
            start_iso = str(c.get("startsAt") or c.get("fechaHoraInicio") or c.get("startTime") or "")
            time_formatted = ""
            date_formatted = fecha
            if start_iso:
                try:
                    dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
                    time_formatted = dt.strftime("%I:%M %p").lstrip("0")
                    date_formatted = dt.strftime("%Y-%m-%d")
                except Exception:
                    time_formatted = start_iso

            citas_enriquecidas.append({
                "id": cita_id,
                "startsAt": start_iso,
                "patientId": patient_id,
                "patientName": p_name,
                "documentNumber": doc_number,
                "phone": phone_clean,
                "professionalId": prof_id,
                "professionalName": prof_name,
                "serviceId": serv_id,
                "serviceName": serv_name,
                "date": date_formatted,
                "timeFormatted": time_formatted,
                "status": c.get("status") or c.get("estado"),
            })

        return citas_enriquecidas


appointment_service = AppointmentService()
