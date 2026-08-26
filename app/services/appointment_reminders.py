import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable
from zoneinfo import ZoneInfo

from app.clients.dotnet_client import dotnet_client
from app.clients.evolution_client import evolution_client
from app.core.config import settings

logger = logging.getLogger(__name__)


def _first_value(appointment: Dict[str, Any], *names: str) -> str:
    for name in names:
        value = appointment.get(name)
        if value is None:
            value = next(
                (candidate for key, candidate in appointment.items() if key.lower() == name.lower()),
                None,
            )
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _nested_value(appointment: Dict[str, Any], parent_names: tuple[str, ...], *names: str) -> str:
    for parent_name in parent_names:
        parent = next(
            (value for key, value in appointment.items() if key.lower() == parent_name.lower()),
            None,
        )
        if isinstance(parent, dict):
            value = _first_value(parent, *names)
            if value:
                return value
    return ""


def _appointment_items(response: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(response, list):
        return (item for item in response if isinstance(item, dict))
    if isinstance(response, dict):
        for key in ("data", "items", "citas", "results"):
            value = response.get(key)
            if isinstance(value, list):
                return (item for item in value if isinstance(item, dict))
    return ()


async def enviar_recordatorios_citas() -> None:
    """Envía por WhatsApp los recordatorios de las citas de mañana."""
    timezone = ZoneInfo(settings.reminder_timezone)
    tomorrow = (datetime.now(timezone) + timedelta(days=1)).date().isoformat()
    response = await dotnet_client.consultar_citas(tomorrow)

    if response is None:
        logger.warning("No se pudieron consultar las citas para %s", tomorrow)
        return

    for appointment in _appointment_items(response):
        patient = _first_value(
            appointment, "nombrePaciente", "pacienteNombre", "patientName", "nombre"
        ) or _nested_value(appointment, ("paciente", "patient"), "nombre", "nombreCompleto", "name")
        service = _first_value(appointment, "serviceName", "nombreServicio", "service") or _nested_value(
            appointment, ("servicio", "service"), "nombre", "name"
        )
        phone = _first_value(
            appointment, "telefono", "telefonoPaciente", "phone", "phoneNumber", "celular", "whatsapp"
        ) or _nested_value(appointment, ("paciente", "patient"), "telefono", "phone", "celular", "whatsapp")

        if not patient or not service or not phone:
            logger.warning("Se omite cita sin paciente, servicio o teléfono: %s", appointment)
            continue

        message = (
            f"Hola {patient}, te recordamos tu cita de {service} para mañana. "
            "Si necesitas reprogramarla, responde a este mensaje."
        )
        await evolution_client.enviar_mensaje(phone, message)
