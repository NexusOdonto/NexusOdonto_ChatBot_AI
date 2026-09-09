import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from zoneinfo import ZoneInfo

from app.clients.dotnet_client import dotnet_client
from app.clients.evolution_client import evolution_client
from app.core.config import settings

logger = logging.getLogger(__name__)

# Registro en memoria de recordatorios enviados por fecha e ID de cita para evitar duplicados (Anti-Spam)
_RECORDATORIOS_ENVIADOS: Set[str] = set()


def _generar_clave_idempotencia(cita_id: str, fecha: str) -> str:
    """Genera una clave única para identificar el recordatorio enviado a una cita en una fecha dada."""
    return f"{fecha}:{cita_id}"


def construir_mensaje_recordatorio(cita: Dict[str, Any]) -> str:
    """Construye un mensaje de WhatsApp enriquecido, profesional y con opciones interactivas."""
    paciente = cita.get("patientName") or "Estimado/a paciente"
    servicio = cita.get("serviceName") or "Consulta Odontológica"
    doctor = cita.get("professionalName") or "nuestro especialista"
    hora = cita.get("timeFormatted") or "su hora programada"
    fecha = cita.get("date") or "mañana"

    # Formatear fecha a formato legible (ej. Jueves, 10 de Septiembre de 2026)
    fecha_legible = fecha
    try:
        dt = datetime.strptime(fecha, "%Y-%m-%d")
        dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        meses = [
            "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
            "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
        ]
        dia_nombre = dias[dt.weekday()]
        mes_nombre = meses[dt.month - 1]
        fecha_legible = f"{dia_nombre}, {dt.day} de {mes_nombre} de {dt.year}"
    except Exception:
        pass

    return (
        f"¡Hola, {paciente}! 👋✨\n\n"
        f"Te saludamos de *Nexus Odonto* 🦷 para recordarte tu cita programada:\n\n"
        f"📋 *Detalles de tu cita:*\n"
        f"• 🦷 *Tratamiento:* {servicio}\n"
        f"• 👨‍⚕️ *Especialista:* {doctor}\n"
        f"• 📅 *Fecha:* {fecha_legible}\n"
        f"• ⏰ *Horario:* {hora}\n"
        f"• 📍 *Sede:* Cr 24 #35-12, Santander\n"
        f"• 📞 *Línea de atención:* +57 324 6030217\n\n"
        f"💡 *Recomendación:* Por favor llega 10 a 15 minutos antes de tu hora para tu comodidad.\n\n"
        f"👉 *Por favor responde a este mensaje para gestionar tu turno:*\n"
        f"• Escribe *CONFIRMAR* para asegurar tu asistencia ✅\n"
        f"• Escribe *REPROGRAMAR* si necesitas cambiar de horario 🔄\n"
        f"• Escribe *CANCELAR* si ya no podrás asistir ❌\n\n"
        f"¡Estamos listos para cuidar de tu sonrisa! 😊🦷"
    )


async def obtener_preview_recordatorios(fecha: Optional[str] = None) -> List[Dict[str, Any]]:
    """Obtiene la lista de citas que calificarían para recordatorio con su mensaje formateado sin enviar nada."""
    try:
        tz = ZoneInfo(settings.reminder_timezone)
    except Exception:
        tz = ZoneInfo("America/Bogota")

    fecha_objetivo = fecha or (datetime.now(tz) + timedelta(days=1)).date().isoformat()
    citas = await dotnet_client.obtener_citas_agendadas_para_recordatorio(fecha_objetivo)

    preview_list = []
    for c in citas:
        clave = _generar_clave_idempotencia(c["id"], fecha_objetivo)
        preview_list.append({
            "citaId": c["id"],
            "fecha": fecha_objetivo,
            "paciente": c.get("patientName"),
            "cedula": c.get("documentNumber"),
            "telefono": c.get("phone"),
            "doctor": c.get("professionalName"),
            "servicio": c.get("serviceName"),
            "hora": c.get("timeFormatted"),
            "yaEnviado": clave in _RECORDATORIOS_ENVIADOS,
            "mensaje": construir_mensaje_recordatorio(c),
        })

    return preview_list


async def enviar_recordatorios_citas(
    fecha: Optional[str] = None, dry_run: bool = False
) -> Dict[str, Any]:
    """Envía por WhatsApp los recordatorios de las citas programadas para una fecha dada (por defecto mañana).
    
    Args:
        fecha: Fecha en formato YYYY-MM-DD. Si no se pasa, toma la fecha de mañana en la zona horaria configurada.
        dry_run: Si es True, no envía los mensajes reales ni los persiste en BD, solo simula la ejecución.

    Returns:
        Diccionario con métricas y resultados de los recordatorios procesados.
    """
    try:
        tz = ZoneInfo(settings.reminder_timezone)
    except Exception:
        tz = ZoneInfo("America/Bogota")

    fecha_objetivo = fecha or (datetime.now(tz) + timedelta(days=1)).date().isoformat()
    logger.info(f"[Recordatorios] Iniciando proceso para fecha {fecha_objetivo} (dry_run={dry_run})")

    citas = await dotnet_client.obtener_citas_agendadas_para_recordatorio(fecha_objetivo)
    if not citas:
        logger.info(f"[Recordatorios] No se encontraron citas agendadas pendientes para {fecha_objetivo}.")
        return {
            "fecha": fecha_objetivo,
            "total": 0,
            "enviados": 0,
            "omitidos": 0,
            "citas": [],
            "mensaje": f"No hay citas agendadas para {fecha_objetivo}",
        }

    enviados = 0
    omitidos = 0
    detalles = []

    for c in citas:
        cita_id = c["id"]
        telefono = c.get("phone")
        clave = _generar_clave_idempotencia(cita_id, fecha_objetivo)

        if not telefono:
            logger.warning(f"[Recordatorios] Se omite cita {cita_id} ({c.get('patientName')}) por no tener teléfono registrado.")
            omitidos += 1
            detalles.append({"citaId": cita_id, "estado": "omitida_sin_telefono", "paciente": c.get("patientName")})
            continue

        if clave in _RECORDATORIOS_ENVIADOS and not dry_run:
            logger.info(f"[Recordatorios] Se omite cita {cita_id} porque ya se envió recordatorio hoy ({clave}).")
            omitidos += 1
            detalles.append({"citaId": cita_id, "estado": "omitida_ya_enviada", "paciente": c.get("patientName")})
            continue

        mensaje = construir_mensaje_recordatorio(c)

        if dry_run:
            enviados += 1
            detalles.append({
                "citaId": cita_id,
                "estado": "simulada",
                "telefono": telefono,
                "paciente": c.get("patientName"),
            })
            continue

        # Envío real a través de Evolution API
        try:
            exito = await evolution_client.enviar_mensaje(numero=telefono, texto=mensaje)
            if exito:
                enviados += 1
                _RECORDATORIOS_ENVIADOS.add(clave)
                detalles.append({
                    "citaId": cita_id,
                    "estado": "enviada",
                    "telefono": telefono,
                    "paciente": c.get("patientName"),
                })
                logger.info(f"[Recordatorios] Recordatorio enviado a {c.get('patientName')} ({telefono}) para cita {cita_id}")

                # Persistir en el historial de mensajes de la conversación
                try:
                    await dotnet_client.registrar_mensaje(
                        chat_identifier=telefono,
                        rol="CHATBOT",
                        contenido=mensaje,
                        patient_id=c.get("patientId"),
                    )
                except Exception as db_err:
                    logger.debug(f"[Recordatorios] No se pudo persistir mensaje de recordatorio en DB: {db_err}")
            else:
                logger.warning(f"[Recordatorios] Evolution API no confirmó el envío para cita {cita_id} ({telefono})")
                detalles.append({
                    "citaId": cita_id,
                    "estado": "fallo_envio_evolution",
                    "telefono": telefono,
                    "paciente": c.get("patientName"),
                })
        except Exception as env_err:
            logger.error(f"[Recordatorios] Error al enviar mensaje para cita {cita_id}: {env_err}")
            detalles.append({
                "citaId": cita_id,
                "estado": "error",
                "error": str(env_err),
                "paciente": c.get("patientName"),
            })

    return {
        "fecha": fecha_objetivo,
        "total": len(citas),
        "enviados": enviados,
        "omitidos": omitidos,
        "dryRun": dry_run,
        "detalles": detalles,
    }


def construir_mensaje_recordatorio_30m(cita: Dict[str, Any]) -> str:
    """Construye un recordatorio amigable para enviar ~30 minutos antes del inicio de la cita."""
    paciente = cita.get("patientName") or "Estimado/a paciente"
    servicio = cita.get("serviceName") or "Consulta Odontológica"
    doctor = cita.get("professionalName") or "nuestro especialista"
    hora = cita.get("timeFormatted") or "su hora programada"

    return (
        f"⏰ *¡Recordatorio de tu cita en 30 minutos!* 🦷✨\n\n"
        f"¡Hola, {paciente}! 👋 Te saludamos de *Nexus Odonto* para recordarte que tu cita es hoy a las *{hora}* (faltan aproximadamente 30 minutos).\n\n"
        f"📋 *Detalles de tu turno:*\n"
        f"• 🦷 *Tratamiento:* {servicio}\n"
        f"• 👨‍⚕️ *Especialista:* {doctor}\n"
        f"• ⏰ *Horario:* {hora}\n"
        f"• 📍 *Sede:* Cr 24 #35-12, Santander\n"
        f"• 📞 *Línea de atención:* +57 324 6030217\n\n"
        f"💡 *Recomendación:* Por favor sal con tiempo hacia la clínica para evitar demoras. ¡Estamos listos para atenderte! 😊👍"
    )


async def enviar_recordatorios_30_minutos(dry_run: bool = False) -> Dict[str, Any]:
    """Escanea las citas agendadas para hoy y envía un recordatorio a las que inicien en ~30 minutos (ventana de 10 a 45 min)."""
    try:
        tz = ZoneInfo(settings.reminder_timezone)
    except Exception:
        tz = ZoneInfo("America/Bogota")

    now = datetime.now(tz)
    today_str = now.date().isoformat()

    citas = await dotnet_client.obtener_citas_agendadas_para_recordatorio(today_str)
    if not citas:
        return {"horaActual": now.strftime("%I:%M %p"), "totalCitasHoy": 0, "enviados": 0, "detalles": []}

    enviados = 0
    detalles = []

    for c in citas:
        cita_id = c["id"]
        telefono = c.get("phone")
        starts_at_str = c.get("startsAt")
        if not telefono or not starts_at_str:
            continue

        clave = f"reminder_30m:{cita_id}"
        if clave in _RECORDATORIOS_ENVIADOS and not dry_run:
            continue

        try:
            clean_starts = str(starts_at_str).replace("Z", "")
            dt_cita = datetime.fromisoformat(clean_starts)
            if dt_cita.tzinfo is None:
                dt_cita = dt_cita.replace(tzinfo=tz)

            minutos_restantes = (dt_cita - now).total_seconds() / 60
        except Exception as dt_err:
            logger.debug(f"[Recordatorios 30m] Error calculando tiempo para cita {cita_id}: {dt_err}")
            continue

        # Ventana de disparo: entre 10 y 45 minutos antes de la cita
        if 10 <= minutos_restantes <= 45:
            mensaje = construir_mensaje_recordatorio_30m(c)

            if dry_run:
                enviados += 1
                detalles.append({
                    "citaId": cita_id,
                    "paciente": c.get("patientName"),
                    "telefono": telefono,
                    "minutosRestantes": round(minutos_restantes, 1),
                    "estado": "simulado",
                })
                continue

            try:
                exito = await evolution_client.enviar_mensaje(numero=telefono, texto=mensaje)
                if exito:
                    enviados += 1
                    _RECORDATORIOS_ENVIADOS.add(clave)
                    detalles.append({
                        "citaId": cita_id,
                        "paciente": c.get("patientName"),
                        "telefono": telefono,
                        "minutosRestantes": round(minutos_restantes, 1),
                        "estado": "enviado",
                    })
                    logger.info(
                        f"[Recordatorios 30m] Recordatorio de 30 minutos enviado a {c.get('patientName')} "
                        f"({telefono}) para cita a las {c.get('timeFormatted')} (en {round(minutos_restantes, 1)} min)"
                    )

                    try:
                        await dotnet_client.registrar_mensaje(
                            chat_identifier=telefono,
                            rol="CHATBOT",
                            contenido=mensaje,
                            patient_id=c.get("patientId"),
                        )
                    except Exception as db_err:
                        logger.debug(f"[Recordatorios 30m] No se pudo persistir mensaje en DB: {db_err}")
            except Exception as env_err:
                logger.error(f"[Recordatorios 30m] Error enviando mensaje a {telefono}: {env_err}")

    return {
        "horaActual": now.strftime("%I:%M %p"),
        "totalCitasHoy": len(citas),
        "enviados": enviados,
        "detalles": detalles,
    }

