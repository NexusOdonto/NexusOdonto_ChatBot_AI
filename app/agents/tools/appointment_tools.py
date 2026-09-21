"""Herramientas de gestión de citas odontológicas para Nexus Odonto:
Agendamiento (con creación automática de paciente), consulta por cédula,
reprogramación, cancelación y confirmación formal de asistencia.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.runnables import RunnableConfig

from app.clients.dotnet_client import dotnet_client
from app.agents.tools.agenda_helpers import (
    _normalizar_texto,
    _obtener_valor,
    _validar_cedula,
    _run_sync,
    _formatear_hora_ampm,
    _parsear_fecha_hora_flexible,
    _validar_horario_cita,
    _resolver_cita_por_selector,
    _buscar_servicio_por_texto,
    _buscar_especialidad_por_texto,
    _servicios_activos,
    _servicios_relacionados_a_especialidad,
    _etiqueta_servicio,
    _mensaje_catalogo_no_encontrado,
    _mensaje_especialidad_sin_servicio_unico,
    _es_servicio_activo,
)

logger = logging.getLogger(__name__)


async def _agendar_cita_impl(
    cedula: str,
    nombre_paciente: str,
    profesional_id: Any,
    servicio_id: Any,
    fecha_hora_inicio: str,
    motivo_consulta: str,
    config: Optional[RunnableConfig] = None,
) -> str:
    """Agenda una cita buscando o creando el paciente por su cédula en el backend .NET."""
    try:
        thread_id = ""
        if config and isinstance(config, dict):
            thread_id = config.get("configurable", {}).get("thread_id", "")

        cedula = (cedula or "").strip()
        if not cedula:
            return (
                "Para poder registrar tu cita en el sistema necesito tu *número de cédula* 🆔 para vincular tu historial. "
                "¿Me la podrías indicar por favor? 😊"
            )

        error_cedula = _validar_cedula(cedula)
        if error_cedula:
            return error_cedula

        # 1. Resolver paciente por cédula
        persona = await dotnet_client.buscar_persona_por_documento(cedula)
        paciente_id = None
        nombre_clean = (nombre_paciente or "").strip()
        nombre_display = nombre_clean

        n_lower = nombre_clean.lower()
        es_placeholder = (
            not nombre_clean
            or len(nombre_clean) < 3
            or any(p in n_lower for p in ["paciente", "nexus", "nexusodonto", "desconocido", "anonimo", "anónimo", "n/a", "none", "usuario", "cliente"])
        )

        if persona:
            person_id = persona.get("id")
            nombre_bd = f"{persona.get('firstName', '')} {persona.get('lastName', '')}".strip()
            # Si en BD hay un nombre real previo (no placeholder), lo usamos; de lo contrario preferimos el nuevo provisto
            bd_lower = nombre_bd.lower()
            if nombre_bd and not any(p in bd_lower for p in ["paciente", "nexus"]):
                nombre_display = nombre_bd
            elif not es_placeholder:
                nombre_display = nombre_clean

            if person_id:
                paciente = await dotnet_client.buscar_paciente_por_person_id(str(person_id))
                if paciente:
                    paciente_id = paciente.get("id")
                else:
                    logger.info(f"[Agenda] Persona {person_id} existe pero no tiene registro en Patients. Creando paciente...")
                    nuevo_pac = await dotnet_client.crear_paciente_para_persona(str(person_id))
                    if nuevo_pac:
                        paciente_id = nuevo_pac.get("id")

        if not paciente_id:
            # Si el paciente no existe previamente en la base de datos, el nombre real es ESTRICTAMENTE OBLIGATORIO
            if es_placeholder:
                return (
                    "Para poder registrar tu cita en el sistema y crear tu ficha clínica, necesito obligatoriamente tu *nombre completo* (nombre y apellido) 👤.\n\n"
                    "¿Me podrías indicar cómo te llamas por favor? 😊"
                )

            logger.info(f"[Agenda] Paciente con cédula {cedula} no encontrado. Creando perfil básico con nombre: '{nombre_display}'...")
            resultado_registro = await dotnet_client.crear_paciente_basico(
                cedula=cedula,
                nombre=nombre_display,
                telefono_whatsapp=thread_id,
            )
            if resultado_registro:
                paciente_id = resultado_registro.get("patientId") or resultado_registro.get("id")
                logger.info(f"[Agenda] Paciente creado exitosamente con ID: {paciente_id}")
            else:
                logger.info(f"[Agenda] Onboarding básico no retornó ID. Buscando persona para vincular paciente...")
                persona_reintento = await dotnet_client.buscar_persona_por_documento(cedula.strip())
                if persona_reintento and persona_reintento.get("id"):
                    nuevo_pac = await dotnet_client.crear_paciente_para_persona(str(persona_reintento["id"]))
                    if nuevo_pac:
                        paciente_id = nuevo_pac.get("id")
                        logger.info(f"[Agenda] Paciente vinculado tras resolución de conflicto con ID: {paciente_id}")

            if not paciente_id:
                return (
                    "⚠️ No pude registrar tus datos en el sistema. "
                    "Por favor comunícate con recepción al *+57 324 6030217* para que te atiendan. 😊"
                )

        # 2. Resolver profesional
        profs = await dotnet_client.obtener_profesionales() or []
        resolved_prof = None
        for p in profs:
            if str(p.get("id")).lower() == str(profesional_id).lower():
                resolved_prof = p
                break
            p_name = _normalizar_texto(p.get("name") or "")
            norm_target = _normalizar_texto(str(profesional_id))
            if norm_target and (norm_target in p_name or p_name in norm_target):
                resolved_prof = p
                break
        if not resolved_prof and profs:
            resolved_prof = profs[0]

        resolved_prof_id = resolved_prof.get("id") if resolved_prof else profesional_id
        prof_nombre_display = resolved_prof.get("name") if resolved_prof else "Especialista Odontológico"

        # 3. Resolver servicio y duración
        servs = await dotnet_client.obtener_servicios() or []
        resolved_serv = None
        norm_target = _normalizar_texto(str(servicio_id or ""))
        for s in servs:
            if not _es_servicio_activo(s):
                continue
            if str(_obtener_valor(s, "id", "servicioId", "serviceId") or "").lower() == str(servicio_id).lower():
                resolved_serv = s
                break

        if not resolved_serv and norm_target and norm_target not in ("none", "null", "n/a", ""):
            resolved_serv = _buscar_servicio_por_texto(norm_target, servs)

        if not resolved_serv and norm_target and norm_target not in ("none", "null", "n/a", ""):
            especialidades = await dotnet_client.obtener_especialidades() or []
            esp_match = _buscar_especialidad_por_texto(norm_target, especialidades)
            if esp_match:
                relacionados = _servicios_relacionados_a_especialidad(esp_match, servs, norm_target)
                if len(relacionados) == 1:
                    resolved_serv = relacionados[0]
                else:
                    esp_nombre = _obtener_valor(esp_match, "name", "nombre") or str(servicio_id)
                    return _mensaje_especialidad_sin_servicio_unico(
                        str(servicio_id), str(esp_nombre), relacionados, servs
                    )

        if not resolved_serv:
            if norm_target and norm_target not in ("none", "null", "n/a", ""):
                return _mensaje_catalogo_no_encontrado(str(servicio_id), servs)
            return (
                "No pude identificar el servicio a agendar. "
                + _mensaje_catalogo_no_encontrado(str(servicio_id or "desconocido"), servs)
            )

        resolved_serv_id = _obtener_valor(resolved_serv, "id", "servicioId", "serviceId") or servicio_id
        serv_nombre_display = _etiqueta_servicio(resolved_serv) or "Consulta odontológica"
        duracion_min = int(
            _obtener_valor(resolved_serv, "durationMinutes", "duracionMinutos") or 45
        )

        # 4. Calcular y validar horario de atención y almuerzo
        raw = str(fecha_hora_inicio or "").strip()
        if not raw or raw.lower() in ("none", "null", "n/a", ""):
            return (
                f"Con gusto te ayudo a agendar tu cita de *{serv_nombre_display}* con *{prof_nombre_display}* 😊.\n\n"
                "¿Qué día y horario te quedaría mejor? O si prefieres, dime la fecha y con gusto te muestro los turnos disponibles."
            )

        starts_dt = _parsear_fecha_hora_flexible(raw)
        if not starts_dt:
            return (
                f"No pude interpretar la fecha y hora *'{raw}'* para agendar tu cita.\n\n"
                "Por favor indícame la fecha y hora deseada (ej: *2026-09-20 10:00 AM* o *mañana a las 2:00 PM*). 😊"
            )

        horario_valido, msg_horario = _validar_horario_cita(starts_dt, duracion_min, prof_nombre_display)
        if not horario_valido and msg_horario:
            return msg_horario

        ends_dt = starts_dt + timedelta(minutes=duracion_min)
        starts_at_iso = starts_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        ends_at_iso = ends_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # 5. Obtener IDs y registrar cita en .NET
        status_id = await dotnet_client.obtener_appointment_status_id("AGENDADA")
        origin_id = await dotnet_client.obtener_appointment_origin_id("AGENTE_BOT")

        payload = {
            "patientId": str(paciente_id),
            "professionalId": str(resolved_prof_id),
            "serviceId": str(resolved_serv_id),
            "startsAt": starts_at_iso,
            "endsAt": ends_at_iso,
            "appointmentStatusId": str(status_id),
            "appointmentOriginId": str(origin_id),
            "reasonForVisit": motivo_consulta or f"Cita de {serv_nombre_display}",
            "notes": "Agendado por Nexus Odonto Chatbot",
        }

        respuesta = await dotnet_client.agendar_cita(payload)
        if respuesta and respuesta.get("success"):
            cita_id = _obtener_valor(respuesta, "id", "citaId", "appointmentId")
            hora_inicio_str = _formatear_hora_ampm(starts_dt.strftime("%H:%M"))
            hora_fin_str = _formatear_hora_ampm(ends_dt.strftime("%H:%M"))
            fecha_str = starts_dt.strftime("%d/%m/%Y")

            web_url = os.getenv("WEB_PORTAL_URL", "https://nexusodonto.chatcampuslands.com/login")
            return (
                f"¡Cita Confirmada con Éxito! 🎉🦷✨\n\n"
                f"📋 *Resumen de tu Cita:*\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"• 🆔 *Cédula:* {cedula}\n"
                f"• 👤 *Paciente:* {nombre_display}\n"
                f"• 👨‍⚕️ *Especialista:* {prof_nombre_display}\n"
                f"• 🦷 *Tratamiento:* {serv_nombre_display}\n"
                f"• 📅 *Fecha:* {fecha_str}\n"
                f"• ⏰ *Horario:* {hora_inicio_str} a {hora_fin_str}\n"
                f"• 🆔 *Código de Cita:* `{cita_id}`\n"
                f"━━━━━━━━━━━━━━━━━━━━\n\n"
                f"🌐 *Consulta y gestiona tus citas en nuestra plataforma web:*\n"
                f"🔗 {web_url}\n"
                f"• 👤 *Usuario:* `{cedula}`\n"
                f"• 🔑 *Contraseña temporal:* `{cedula}`\n\n"
                f"💡 *Nota de Seguridad:* Por tu seguridad, una vez ingreses a la plataforma web deberás cambiar esta contraseña inicial. Ten en cuenta que por políticas de seguridad el bot no puede modificar o cambiar contraseñas.\n\n"
                f"📍 *Sede:* Nexus Odonto — Calle 100 # 15-20, Centro Médico Odontológico\n"
                f"📞 *Atención:* +57 324 6030217\n\n"
                f"¡Será un placer cuidar de tu sonrisa! 😊✨"
            )
        else:
            err_msg = str((respuesta.get("error") if isinstance(respuesta, dict) else "") or "").lower()
            hora_sol = _formatear_hora_ampm(starts_dt.strftime("%H:%M"))

            if any(w in err_msg for w in ["almuerzo", "lunch", "receso", "descanso"]):
                return (
                    f"⚠️ El turno de las *{hora_sol}* coincide con la franja de almuerzo del especialista (12:00 PM a 2:00 PM) 🍽️.\n\n"
                    f"Con gusto podemos agendarte en la jornada de la mañana (8:00 AM a 12:00 PM) o en la tarde a partir de las *2:00 PM* con {prof_nombre_display}. ¿Cuál te queda mejor? 😊"
                )
            elif any(w in err_msg for w in ["horario", "schedule", "disponib", "fuera", "outside"]):
                return (
                    f"⚠️ {prof_nombre_display} no tiene disponibilidad registrada a las *{hora_sol}* para esa fecha.\n\n"
                    f"¿Te gustaría que te muestre los horarios disponibles para que elijas otro turno cómodo? 😊"
                )
            elif any(w in err_msg for w in ["overlap", "ocupad", "conflict", "traslap", "already has"]):
                return (
                    f"⚠️ El turno de las *{hora_sol}* ya se encuentra reservado.\n\n"
                    f"¿Deseas consultar los horarios libres más cercanos para hoy o para otra fecha? 😊"
                )
            else:
                return (
                    f"⚠️ En este momento no fue posible confirmar la cita a las *{hora_sol}* en el sistema de agenda.\n\n"
                    f"Por favor consulta los horarios disponibles con {prof_nombre_display} o comunícate con recepción al *+57 324 6030217*. 😊"
                )
    except Exception as exc:
        logger.error(f"Error al agendar cita: {exc}", exc_info=True)
        return "Lo siento, ocurrió un problema de conexión al registrar la cita. Por favor intenta de nuevo en unos minutos."


async def _consultar_cita_por_cedula_impl(cedula: str) -> str:
    """Busca y formatea las citas de un paciente discriminando próximas vs historial."""
    try:
        cedula = (cedula or "").strip()
        if not cedula:
            return "Por favor, indícame tu número de cédula para poder consultar tus citas. 🆔"
        error_cedula = _validar_cedula(cedula)
        if error_cedula:
            return error_cedula

        citas = await dotnet_client.buscar_citas_por_cedula(cedula)

        if citas is None:
            return (
                f"❌ No encontré ningún paciente registrado con la cédula *{cedula}*.\n\n"
                "Si acabas de agendar una cita, asegúrate de usar la misma cédula con la que te registraste.\n"
                "¿Deseas verificar con otra cédula o necesitas ayuda? 😊"
            )

        if not citas:
            return (
                f"📋 *Consulta de Citas* 🦷✨\n\n"
                f"🆔 *Cédula:* {cedula}\n\n"
                "Actualmente no tienes citas registradas en nuestro sistema.\n\n"
                "💡 ¿Te gustaría agendar una nueva cita? Con gusto te ayudo. 😊"
            )

        try:
            from zoneinfo import ZoneInfo
            now_colombia = datetime.now(ZoneInfo("America/Bogota"))
        except Exception:
            now_colombia = datetime.now()

        proximas = []
        historial = []

        for c in citas:
            prof_nom = c.get("professionalName", "Especialista Odontológico")
            serv_nom = c.get("serviceName", "Consulta Odontológica")
            estado = c.get("statusName", "Programada")
            starts_at_raw = c.get("startsAt") or c.get("fechaHoraInicio") or ""
            ends_at_raw = c.get("endsAt") or c.get("fechaHoraFin") or ""
            cita_id = c.get("id") or c.get("citaId") or c.get("appointmentId") or "N/A"
            status_id = str(c.get("appointmentStatusId") or "").lower()

            dt_start = None
            fecha_display = ""
            hora_display = ""

            try:
                if starts_at_raw:
                    clean_start = str(starts_at_raw).replace("Z", "").split(".")[0].replace(" ", "T")
                    dt_start = datetime.fromisoformat(clean_start)
                    if dt_start.tzinfo is None and now_colombia.tzinfo:
                        dt_start = dt_start.replace(tzinfo=now_colombia.tzinfo)
                    fecha_display = dt_start.strftime("%d/%m/%Y")
                    hora_start_str = dt_start.strftime("%I:%M %p")

                    if ends_at_raw:
                        clean_end = str(ends_at_raw).replace("Z", "").split(".")[0].replace(" ", "T")
                        dt_end = datetime.fromisoformat(clean_end)
                        if dt_end.tzinfo is None and now_colombia.tzinfo:
                            dt_end = dt_end.replace(tzinfo=now_colombia.tzinfo)
                        hora_end_str = dt_end.strftime("%I:%M %p")
                        hora_display = f"{hora_start_str} - {hora_end_str}"
                    else:
                        hora_display = hora_start_str
            except Exception:
                fecha_display = str(starts_at_raw)[:10]
                hora_display = str(starts_at_raw)[11:16]

            is_cancelled = bool(c.get("cancelledAt")) or status_id == "10000000-0000-0000-0000-000000000005" or "cancel" in estado.lower()
            is_completed = status_id == "10000000-0000-0000-0000-000000000004" or "complet" in estado.lower()
            is_noshow = status_id == "10000000-0000-0000-0000-000000000006" or "no_asist" in estado.lower() or "no asist" in estado.lower()

            if is_noshow:
                estado = "No Asistió"

            if dt_start and not is_cancelled and not is_completed and not is_noshow:
                if dt_start < (now_colombia - timedelta(minutes=45)):
                    estado = "No Asistió (Vencida)"
                    is_noshow = True

            c_info = {
                "id": cita_id,
                "profesional": prof_nom,
                "servicio": serv_nom,
                "estado": estado,
                "fecha": fecha_display,
                "hora": hora_display,
                "dt": dt_start,
                "raw": c,
            }

            if is_cancelled or is_completed or is_noshow or (dt_start and dt_start < (now_colombia - timedelta(minutes=45))):
                historial.append(c_info)
            else:
                proximas.append(c_info)

        max_aware = datetime.max.replace(tzinfo=now_colombia.tzinfo) if now_colombia.tzinfo else datetime.max
        min_aware = datetime.min.replace(tzinfo=now_colombia.tzinfo) if now_colombia.tzinfo else datetime.min
        proximas.sort(key=lambda x: x["dt"] or max_aware)
        historial.sort(key=lambda x: x["dt"] or min_aware, reverse=True)

        resumen = [f"📋 *Citas Registradas para la Cédula:* `{cedula}` 🦷✨\n"]

        if proximas:
            resumen.append("📅 *PRÓXIMAS CITAS PROGRAMADAS:*")
            for i, c in enumerate(proximas, 1):
                icon_est = "🟢" if "confirm" in c["estado"].lower() else "🟡"
                resumen.append(
                    f"{i}️⃣ *Cita #{i}* {icon_est}\n"
                    f"   • 🦷 *Tratamiento:* {c['servicio']}\n"
                    f"   • 👨‍⚕️ *Especialista:* {c['profesional']}\n"
                    f"   • 📅 *Fecha:* {c['fecha']}\n"
                    f"   • ⏰ *Horario:* {c['hora']}\n"
                    f"   • 📌 *Estado:* {c['estado']}"
                )
            resumen.append("")

        if historial:
            resumen.append("📜 *HISTORIAL RECIENTE:*")
            for c in historial[:3]:
                icon_est = "🔴" if "cancel" in c["estado"].lower() else ("⚪" if "no asist" in c["estado"].lower() else "✅")
                resumen.append(
                    f"• {icon_est} *{c['fecha']}* — {c['servicio']} con {c['profesional']} ({c['estado']})"
                )
            resumen.append("")

        if proximas:
            resumen.append(
                "💡 *¿Necesitas gestionar alguna de tus citas activas?*\n"
                "Dime si deseas *reprogramarla*, *cancelarla* o *confirmar tu asistencia*. 😊"
            )
        else:
            resumen.append(
                "💡 No tienes citas programadas pendientes. ¿Te gustaría agendar una nueva cita? Con gusto te colaboro. 😊"
            )

        return "\n".join(resumen)
    except Exception as exc:
        logger.error(f"Error al consultar citas para cédula {cedula}: {exc}", exc_info=True)
        return "Lo siento, ocurrió un error al consultar tus citas en el sistema. Por favor intenta de nuevo en unos momentos."


async def _cancelar_cita_impl(cedula: str, cita_id: Optional[str] = None) -> str:
    """Cancela una cita verificando primero que la cédula corresponda al paciente."""
    try:
        cedula = (cedula or "").strip()
        cita_id = (cita_id or "").strip()

        if not cedula:
            return "Para cancelar tu cita, necesito tu *número de cédula* 🆔. ¿Me la puedes indicar? 😊"
        error_cedula = _validar_cedula(cedula)
        if error_cedula:
            return error_cedula

        citas = await dotnet_client.buscar_citas_por_cedula(cedula)
        if not citas:
            return (
                f"📋 *Consulta de Citas* 🦷✨\n\n"
                f"No encontré citas activas registradas para la cédula *{cedula}*.\n\n"
                "¿Deseas agendar una nueva cita? Con gusto te ayudo. 😊"
            )

        citas_activas = [c for c in citas if str(c.get("statusName", "")).lower() != "cancelada"]
        if not citas_activas:
            return f"Todas las citas registradas para la cédula *{cedula}* ya se encuentran canceladas o atendidas. 😊"

        cita_encontrada, msg_opciones = _resolver_cita_por_selector(citas_activas, cita_id)
        if not cita_encontrada:
            return msg_opciones or "No se pudo identificar la cita a cancelar. Por favor indícame el número de la cita (ej: Cita 1). 😊"

        target_id = str(cita_encontrada.get("id") or cita_encontrada.get("citaId") or cita_id)
        prof_nom = cita_encontrada.get("professionalName", "Especialista Odontológico")
        serv_nom = cita_encontrada.get("serviceName", "Consulta Odontológica")
        starts_at_raw = cita_encontrada.get("startsAt") or ""
        fecha_display = str(starts_at_raw)[:10]
        try:
            if starts_at_raw:
                clean = str(starts_at_raw).replace("Z", "").split(".")[0]
                dt = datetime.fromisoformat(clean)
                fecha_display = dt.strftime("%d/%m/%Y a las %I:%M %p")
        except Exception:
            pass

        resultado = await dotnet_client.cancelar_cita(target_id, cita_encontrada)

        if resultado.get("success"):
            return (
                f"✅ *Cita Cancelada Exitosamente* 🦷\n\n"
                f"📋 *Resumen de la Cita Cancelada:*\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"• 🆔 *Cédula:* {cedula}\n"
                f"• 👨‍⚕️ *Especialista:* {prof_nom}\n"
                f"• 🦷 *Tratamiento:* {serv_nom}\n"
                f"• 📅 *Fecha y Hora:* {fecha_display}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n\n"
                f"Tu cita ha sido cancelada. Si deseas reagendar en otro horario, con gusto te ayudo. 😊\n"
                f"📞 *Atención:* +57 324 6030217"
            )
        else:
            err = resultado.get("error", "")
            return (
                f"⚠️ No fue posible cancelar la cita en este momento.\n\n"
                f"Detalle: {err}\n"
                "Por favor intenta de nuevo o comunícate con recepción al *+57 324 6030217*. 😊"
            )
    except Exception as exc:
        logger.error(f"Error cancelando cita para cédula {cedula}: {exc}", exc_info=True)
        return "Lo siento, ocurrió un problema al cancelar la cita. Por favor intenta de nuevo o llama a recepción."


async def _modificar_cita_impl(
    cedula: str,
    nueva_fecha_hora: str,
    cita_id: Optional[str] = None,
    nuevo_profesional_id: Optional[str] = None,
) -> str:
    """Modifica la fecha/horario de una cita existente con validación de almuerzo y jornada."""
    try:
        cedula = (cedula or "").strip()
        cita_id = (cita_id or "").strip()
        nueva_fecha_hora = (nueva_fecha_hora or "").strip()

        if not cedula or not nueva_fecha_hora:
            return (
                "Para reprogramar tu cita necesito:\n"
                "• Tu *número de cédula* 🆔\n"
                "• La *nueva fecha y horario* deseado (ej: 2026-09-04 14:00)\n\n"
                "¿Me puedes proporcionar esos datos? 😊"
            )
        error_cedula = _validar_cedula(cedula)
        if error_cedula:
            return error_cedula

        citas = await dotnet_client.buscar_citas_por_cedula(cedula)
        if not citas:
            return (
                f"❌ No encontré ninguna cita activa registrada para la cédula *{cedula}*.\n\n"
                "¿Deseas agendar una nueva cita? Con gusto te ayudo. 😊"
            )

        citas_activas = [c for c in citas if str(c.get("statusName", "")).lower() != "cancelada"]
        if not citas_activas:
            return f"No tienes citas activas para reprogramar con la cédula *{cedula}*. ¿Deseas agendar una nueva cita? 😊"

        cita_encontrada, msg_opciones = _resolver_cita_por_selector(citas_activas, cita_id)
        if not cita_encontrada:
            return msg_opciones or "No se pudo identificar la cita a reprogramar. Por favor indícame el número de la cita (ej: Cita 1). 😊"

        target_id = str(cita_encontrada.get("id") or cita_encontrada.get("citaId") or cita_id)

        try:
            raw = str(nueva_fecha_hora).strip()
            starts_dt = _parsear_fecha_hora_flexible(raw)
            if not starts_dt:
                raise ValueError("Formato no reconocido")
        except Exception:
            return (
                f"❌ No pude interpretar la fecha '{nueva_fecha_hora}'.\n"
                "Por favor usa el formato: *YYYY-MM-DD HH:MM AM/PM* (ej: 2026-09-04 02:00 PM o 14:00)"
            )

        serv_id = cita_encontrada.get("serviceId") or cita_encontrada.get("servicioId")
        duracion_min = 45
        if serv_id:
            servs = await dotnet_client.obtener_servicios() or []
            for s in servs:
                if str(s.get("id")).lower() == str(serv_id).lower():
                    duracion_min = int(s.get("durationMinutes") or 45)
                    break

        prof_nombre_display = cita_encontrada.get("professionalName", "Especialista")
        resolved_prof_id = cita_encontrada.get("professionalId")
        if nuevo_profesional_id and str(nuevo_profesional_id).strip():
            profs = await dotnet_client.obtener_profesionales() or []
            for p in profs:
                p_name = _normalizar_texto(p.get("name") or "")
                norm_target = _normalizar_texto(str(nuevo_profesional_id))
                if str(p.get("id")).lower() == str(nuevo_profesional_id).lower() or (norm_target and norm_target in p_name):
                    resolved_prof_id = str(p.get("id"))
                    prof_nombre_display = p.get("name", prof_nombre_display)
                    break

        horario_valido, msg_horario = _validar_horario_cita(starts_dt, duracion_min, prof_nombre_display)
        if not horario_valido and msg_horario:
            return msg_horario

        ends_dt = starts_dt + timedelta(minutes=duracion_min)
        starts_at_iso = starts_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        ends_at_iso = ends_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        patient_id = cita_encontrada.get("patientId") or cita_encontrada.get("pacienteId")
        if not patient_id:
            try:
                per = await dotnet_client.buscar_persona_por_documento(cedula)
                if per and per.get("id"):
                    pac = await dotnet_client.buscar_paciente_por_person_id(str(per["id"]))
                    if pac and pac.get("id"):
                        patient_id = pac.get("id")
            except Exception as e_pac:
                logger.warning(f"[Modificar Cita] No se pudo resolver patientId para {cedula}: {e_pac}")

        datos_actualizacion: Dict[str, Any] = {
            "patientId": str(patient_id or "00000000-0000-0000-0000-000000000000"),
            "professionalId": str(resolved_prof_id or "00000000-0000-0000-0000-000000000000"),
            "serviceId": str(serv_id or "00000000-0000-0000-0000-000000000000"),
            "startsAt": starts_at_iso,
            "endsAt": ends_at_iso,
            "appointmentStatusId": str(cita_encontrada.get("appointmentStatusId") or "10000000-0000-0000-0000-000000000001"),
            "appointmentOriginId": str(cita_encontrada.get("appointmentOriginId") or "20000000-0000-0000-0000-000000000002"),
            "reasonForVisit": cita_encontrada.get("reasonForVisit") or "Reprogramada por el paciente",
            "notes": cita_encontrada.get("notes") or "Reprogramado por Nexus Odonto Chatbot",
        }

        resultado = await dotnet_client.modificar_cita(target_id, datos_actualizacion)

        if resultado.get("success"):
            fecha_display = starts_dt.strftime("%d/%m/%Y")
            hora_display = f"{starts_dt.strftime('%I:%M %p')} - {ends_dt.strftime('%I:%M %p')}"

            return (
                f"✅ *Cita Reprogramada Exitosamente* 🦷✨\n\n"
                f"📋 *Nuevos Datos de tu Cita:*\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"• 🆔 *Cédula:* {cedula}\n"
                f"• 👨‍⚕️ *Especialista:* {prof_nombre_display}\n"
                f"• 📅 *Nueva Fecha:* {fecha_display}\n"
                f"• ⏰ *Nuevo Horario:* {hora_display}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n\n"
                f"Por favor llega 10 minutos antes de tu hora programada. ¡Hasta pronto! 😊\n"
                f"📞 *Atención:* +57 324 6030217"
            )
        else:
            err_msg = str(resultado.get("error") or "").lower()
            hora_req = starts_dt.strftime('%I:%M %p')
            fecha_req = starts_dt.strftime('%Y-%m-%d')

            if "availability" in err_msg or "outside" in err_msg or resultado.get("status_code") == 409:
                return (
                    f"⚠️ El horario solicitado (*{hora_req}* del `{fecha_req}`) no está disponible para {prof_nombre_display}.\n\n"
                    "💡 Puede deberse al receso de almuerzo (12:00 PM a 2:00 PM) o a que está fuera de su jornada de atención.\n"
                    "Por favor consulta los horarios disponibles (mañana de 8:00 AM a 12:00 PM o tarde a partir de las 2:00 PM). 😊"
                )
            elif "overlap" in err_msg or "already has" in err_msg:
                return (
                    f"⚠️ {prof_nombre_display} ya tiene otra cita programada a las *{hora_req}*.\n\n"
                    "Por favor elige otro horario disponible para apartar tu turno. 😊"
                )
            else:
                return (
                    f"⚠️ No fue posible reprogramar tu cita en este momento.\n\n"
                    f"Detalle: {resultado.get('error')}\n"
                    "Por favor intenta de nuevo o comunícate con recepción: *+57 324 6030217* 😊"
                )
    except Exception as exc:
        logger.error(f"Error modificando cita para cédula {cedula}: {exc}", exc_info=True)
        return "Lo siento, ocurrió un problema al modificar la cita. Por favor intenta de nuevo o llama a recepción."


async def _confirmar_cita_impl(cedula: str, cita_id: Optional[str] = None) -> str:
    """Confirma formalmente la asistencia del paciente a una cita activa."""
    try:
        cedula = (cedula or "").strip()
        cita_id = (cita_id or "").strip()

        if not cedula:
            return "Para confirmar tu cita, por favor indícame tu *número de cédula* 🆔. 😊"
        error_cedula = _validar_cedula(cedula)
        if error_cedula:
            return error_cedula

        citas = await dotnet_client.buscar_citas_por_cedula(cedula)
        if not citas:
            return (
                f"📋 *Consulta de Citas* 🦷✨\n\n"
                f"No encontré citas registradas para la cédula *{cedula}*.\n\n"
                "Si deseas agendar una nueva cita, ¡con gusto te ayudo! 😊"
            )

        citas_activas = [
            c for c in citas
            if str(c.get("statusName", "")).lower() not in ("cancelada", "completed", "atendida")
        ]
        if not citas_activas:
            return f"No tienes citas pendientes por confirmar para la cédula *{cedula}*. Todas se encuentran completadas o canceladas. 😊"

        cita_a_confirmar, msg_opciones = _resolver_cita_por_selector(citas_activas, cita_id)
        if not cita_a_confirmar:
            if msg_opciones:
                return msg_opciones
            cita_a_confirmar = citas_activas[0]

        target_id = str(cita_a_confirmar.get("id") or cita_a_confirmar.get("citaId"))
        res = await dotnet_client.confirmar_estado_cita(target_id)
        if not res.get("success"):
            return (
                "Hubo un pequeño problema al confirmar tu cita en el sistema. "
                "Por favor comunícate directamente con recepción al +57 324 6030217 para asegurar tu asistencia. 😊"
            )

        doctor = cita_a_confirmar.get("professionalName", "Especialista Odontológico")
        servicio = cita_a_confirmar.get("serviceName", "Consulta Odontológica")
        starts_at_raw = cita_a_confirmar.get("startsAt") or ""

        fecha_display = starts_at_raw[:10]
        hora_display = starts_at_raw[11:16]
        try:
            clean_start = str(starts_at_raw).replace("Z", "").split(".")[0]
            dt = datetime.fromisoformat(clean_start)
            dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
            meses = [
                "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
            ]
            fecha_display = f"{dias[dt.weekday()]}, {dt.day} de {meses[dt.month - 1]} de {dt.year}"
            hora_display = dt.strftime("%I:%M %p").lstrip("0")
        except Exception:
            pass

        return (
            f"¡Excelente! Tu cita ha sido *confirmada exitosamente* en nuestro sistema 🎉✅\n\n"
            f"📋 *Resumen de tu Cita Confirmada:*\n"
            f"• 👤 *Cédula:* {cedula}\n"
            f"• 👨‍⚕️ *Especialista:* {doctor}\n"
            f"• 🦷 *Tratamiento:* {servicio}\n"
            f"• 📅 *Fecha:* {fecha_display}\n"
            f"• ⏰ *Horario:* {hora_display}\n"
            f"• 📍 *Sede:* Calle 100 # 15-20, Centro Médico Odontológico\n\n"
            f"💡 *Recomendación:* Por favor llega 10 a 15 minutos antes de tu turno para prepararte con calma.\n\n"
            f"¡El equipo de Nexus Odonto te espera con gusto! ¿Hay algo más en lo que te pueda colaborar hoy? 😊🦷"
        )
    except Exception as exc:
        logger.error(f"Error al confirmar cita para cédula {cedula}: {exc}", exc_info=True)
        return "Lo siento, ocurrió un problema al confirmar la cita. Por favor intenta de nuevo o comunícate con recepción."


# ─── Herramientas LangChain Expuestas al LLM ──────────────────────────────────

@tool
def agendar_cita_tool(
    cedula: str,
    nombre_paciente: str,
    profesional_id: str,
    servicio_id: str,
    fecha_hora_inicio: str,
    motivo_consulta: str,
    config: Annotated[Optional[RunnableConfig], InjectedToolArg] = None,
) -> str:
    """
    Registra una cita en el sistema para un paciente identificado por su cédula.
    
    IMPORTANTE: Antes de invocar esta herramienta DEBES tener los siguientes datos del usuario:
    - cedula: Número de cédula o documento del paciente (OBLIGATORIO).
    - nombre_paciente: Nombre completo del paciente (OBLIGATORIO).
    - profesional_id: ID o nombre del odontólogo seleccionado.
    - servicio_id: ID o nombre del servicio odontológico.
    - fecha_hora_inicio: Fecha y hora de inicio en formato ISO 8601 (ej. YYYY-MM-DDTHH:MM:SS).
    - motivo_consulta: Breve descripción de la razón de la consulta.
    
    Usa esta herramienta SOLAMENTE después de presentar la ficha de propuesta y obtener confirmación explícita del usuario.
    El número de WhatsApp del paciente se usa automáticamente como teléfono de contacto.
    Si el paciente no existe en el sistema, se creará automáticamente con los datos básicos.
    """
    return _run_sync(_agendar_cita_impl(cedula, nombre_paciente, profesional_id, servicio_id, fecha_hora_inicio, motivo_consulta, config))


@tool
def consultar_cita_por_cedula_tool(cedula: str) -> str:
    """
    Consulta todas las citas programadas de un paciente usando su número de cédula o documento de identidad.
    Muestra: nombre del paciente, cédula, doctor asignado, tratamiento, fecha, horario y estado de cada cita.
    
    Usa esta herramienta cuando el usuario pregunte por sus citas, quiera ver el estado de su agendamiento
    o necesite el ID de una cita para modificarla o cancelarla.
    Si el usuario no ha proporcionado su cédula, pídesela antes de invocar esta herramienta.
    """
    return _run_sync(_consultar_cita_por_cedula_impl(cedula))


@tool
def cancelar_cita_tool(cedula: str, cita_id: Optional[str] = None) -> str:
    """
    Cancela una cita activa de un paciente verificando su cédula.
    
    Parámetros:
    - cedula: Número de cédula o documento de identidad del paciente (OBLIGATORIO).
    - cita_id: (Opcional) Número de la cita que el paciente desea cancelar (ej: '1', '2', 'primera', 'cita 1') o ID de la cita. Si el paciente tiene solo una cita activa, el sistema la identificará automáticamente sin necesidad de especificar este parámetro.
    
    Usa esta herramienta SOLAMENTE tras haber confirmado con el usuario que realmente desea cancelar su cita.
    """
    return _run_sync(_cancelar_cita_impl(cedula, cita_id))


@tool
def modificar_cita_tool(
    cedula: str,
    nueva_fecha_hora: str,
    cita_id: Optional[str] = None,
    nuevo_profesional_id: Optional[str] = None,
) -> str:
    """
    Reprograma una cita existente a una nueva fecha y horario (y opcionalmente con otro doctor).
    
    Parámetros:
    - cedula: Número de cédula del paciente (OBLIGATORIO).
    - nueva_fecha_hora: Nueva fecha y hora en formato ISO 8601 o 'YYYY-MM-DD HH:MM' (ej: '2026-09-04 14:00').
    - cita_id: (Opcional) Número de cita a modificar (ej: '1', '2', 'primera', 'cita 1') o ID de la cita. Si el paciente solo tiene una cita activa, el sistema la detectará automáticamente.
    - nuevo_profesional_id: (Opcional) ID o nombre del nuevo profesional si desea cambiarlo.
    
    IMPORTANTE: Antes de proponer o confirmar un nuevo horario, consulta SIEMPRE la disponibilidad con consultar_disponibilidad_tool para asegurar que el especialista no esté en horario de almuerzo (ej. 12:00 PM a 2:00 PM) ni fuera de turno.
    Usa esta herramienta SOLAMENTE después de presentar la propuesta de cambio y obtener confirmación explícita del usuario.
    """
    return _run_sync(_modificar_cita_impl(cedula, nueva_fecha_hora, cita_id, nuevo_profesional_id))


@tool
def confirmar_cita_tool(cedula: str, cita_id: Optional[str] = None) -> str:
    """
    Confirma formalmente la asistencia del paciente a una cita activa o recordatorio en el sistema Nexus Odonto.
    Actualiza el estado de la cita en la base de datos a CONFIRMADA (verde).
    
    Parámetros:
    - cedula: Número de cédula o documento de identidad del paciente (OBLIGATORIO).
    - cita_id: (Opcional) Número de la cita (ej: '1', '2', 'primera') o ID de la cita a confirmar. Si se omite, el sistema confirmará automáticamente su cita más próxima activa.
    
    Usa esta herramienta cuando el paciente responda a un recordatorio diciendo 'Confirmo', 'Sí confirmo', 'Confirmo mi cita',
    'Confirmo mi asistencia', 'Allá estaré', o cuando solicite explícitamente confirmar su cita.
    """
    return _run_sync(_confirmar_cita_impl(cedula, cita_id))
