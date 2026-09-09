import logging
from typing import Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.core.config import settings
from app.services.appointment_reminders import (
    enviar_recordatorios_citas,
    enviar_recordatorios_30_minutos,
    obtener_preview_recordatorios,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reminders", tags=["Recordatorios de Citas"])


class SendRemindersRequest(BaseModel):
    fecha: Optional[str] = None
    dry_run: bool = False


@router.get("/status")
async def get_reminders_status():
    """Retorna el estado de configuración de los recordatorios automáticos de citas."""
    return {
        "status": "active",
        "schedule": {
            "daily_next_day": {
                "hour": settings.reminder_schedule_hour,
                "minute": settings.reminder_schedule_minute,
                "timezone": settings.reminder_timezone,
            },
            "same_day_advance_30m": {
                "interval": "cada 2 minutos",
                "window": "citas que inician en 10 a 45 minutos",
                "timezone": settings.reminder_timezone,
            },
        },
        "description": "Se ejecutan recordatorios diarios para citas de mañana a las 8:00 AM y recordatorios 30 minutos antes en tiempo real.",
    }


@router.get("/preview")
async def preview_reminders(
    fecha: Optional[str] = Query(
        None,
        description="Fecha objetivo en formato YYYY-MM-DD. Por defecto evalúa el día de mañana.",
    )
):
    """Permite previsualizar los recordatorios que se enviarían para una fecha específica sin emitirlos."""
    items = await obtener_preview_recordatorios(fecha)
    return {
        "total": len(items),
        "fecha": fecha or "mañana (calculada)",
        "recordatorios": items,
    }


@router.post("/send")
async def trigger_reminders(req: Optional[SendRemindersRequest] = None):
    """Ejecuta el envío de recordatorios de citas de forma manual o simulada (dry_run)."""
    fecha = req.fecha if req else None
    dry_run = req.dry_run if req else False

    resultado = await enviar_recordatorios_citas(fecha=fecha, dry_run=dry_run)
    return {
        "status": "success",
        "resultado": resultado,
    }


@router.get("/preview-30m")
async def preview_reminders_30m():
    """Previsualiza las citas de hoy que calificarían para recordatorio de 30 minutos de anticipación."""
    resultado = await enviar_recordatorios_30_minutos(dry_run=True)
    return {
        "status": "success",
        "resultado": resultado,
    }


@router.post("/send-30m")
async def trigger_reminders_30m(dry_run: bool = False):
    """Dispara de forma manual el escaneo y envío de recordatorios de 30 minutos antes para citas de hoy."""
    resultado = await enviar_recordatorios_30_minutos(dry_run=dry_run)
    return {
        "status": "success",
        "resultado": resultado,
    }
