from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


class AppointmentStatus(str, Enum):
    AGENDADA = "AGENDADA"
    CONFIRMADA = "CONFIRMADA"
    CANCELADA = "CANCELADA"
    REPROGRAMADA = "REPROGRAMADA"
    COMPLETADA = "COMPLETADA"
    NO_ASISTIO = "NO_ASISTIO"


class AppointmentOrigin(str, Enum):
    AGENTE_BOT = "AGENTE_BOT"
    MANUAL = "MANUAL"
    RECEPCION = "RECEPCION"
    PACIENTE_WEB = "PACIENTE_WEB"


class Appointment(BaseModel):
    id: str
    patient_id: str
    professional_id: str
    service_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    reason: Optional[str] = None
    status: AppointmentStatus = AppointmentStatus.AGENDADA
    origin: AppointmentOrigin = AppointmentOrigin.AGENTE_BOT

    # Campos enriquecidos para comunicación amigable con el paciente
    patient_name: Optional[str] = None
    patient_document: Optional[str] = None
    patient_phone: Optional[str] = None
    professional_name: Optional[str] = None
    service_name: Optional[str] = None
    date_formatted: Optional[str] = None
    time_formatted: Optional[str] = None
