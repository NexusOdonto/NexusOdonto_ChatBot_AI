import unicodedata
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from app.clients.dotnet_client import dotnet_client

logger = logging.getLogger(__name__)

def _normalizar_texto(texto: str) -> str:
    """Normaliza texto eliminando acentos y convirtiendo a minúsculas."""
    if not texto:
        return ""
    texto = texto.lower().strip()
    return "".join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

def _obtener_valor(obj: Dict[str, Any], *keys: str) -> Any:
    """Busca un valor en un diccionario de forma insensible a mayúsculas y minúsculas."""
    for key in keys:
        if key in obj:
            return obj[key]
        for k, v in obj.items():
            if k.lower() == key.lower():
                return v
    return None

def _run_sync(coro) -> Any:
    """Ejecuta una corrutina de forma síncrona, gestionando de forma segura los loops activos de asyncio."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    if loop.is_running():
        # Si el loop de este hilo ya está activo, corremos la corrutina en un hilo secundario
        result = []
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            try:
                res = new_loop.run_until_complete(coro)
                result.append(res)
            finally:
                new_loop.close()
        t = threading.Thread(target=run_in_thread)
        t.start()
        t.join()
        return result[0]
    else:
        return loop.run_until_complete(coro)

def _generar_slots_desde_regla(
    start_time_str: str,
    end_time_str: str,
    lunch_start_str: Optional[str] = None,
    lunch_end_str: Optional[str] = None,
    duracion_min: int = 60,
) -> list[str]:
    try:
        sh, sm = map(int, str(start_time_str).split(":")[:2])
        eh, em = map(int, str(end_time_str).split(":")[:2])
        lsh, lsm = (map(int, str(lunch_start_str).split(":")[:2])) if lunch_start_str else (12, 0)
        leh, lem = (map(int, str(lunch_end_str).split(":")[:2])) if lunch_end_str else (13, 0)

        cur = sh * 60 + sm
        end = eh * 60 + em
        lstart = lsh * 60 + lsm
        lend = leh * 60 + lem

        slots = []
        while cur + duracion_min <= end:
            if not (cur < lend and (cur + duracion_min) > lstart):
                h = cur // 60
                m = cur % 60
                slots.append(f"{h:02d}:{m:02d}")
            cur += duracion_min
        return slots
    except Exception:
        return [str(start_time_str)[:5]]


async def _consultar_disponibilidad_impl(especialidad: str, fecha: str) -> str:
    try:
        norm_esp = _normalizar_texto(especialidad)
        if not norm_esp:
            return "Por favor, indica una especialidad o servicio válido."
        
        # 1. Obtener especialidades desde el backend .NET
        especialidades = await dotnet_client.obtener_especialidades()
        if not especialidades:
            return "En este momento no podemos acceder al catálogo de especialidades. Por favor intenta de nuevo más tarde."
            
        especialidad_encontrada = None
        for esp in especialidades:
            nombre_esp = _obtener_valor(esp, "name", "nombre", "code", "codigo") or ""
            desc_esp = _obtener_valor(esp, "description", "descripcion") or ""
            code_esp = _obtener_valor(esp, "code", "codigo") or ""
            if (
                norm_esp in _normalizar_texto(nombre_esp)
                or _normalizar_texto(nombre_esp) in norm_esp
                or norm_esp in _normalizar_texto(code_esp)
                or norm_esp in _normalizar_texto(desc_esp)
            ):
                especialidad_encontrada = esp
                break
                
        if not especialidad_encontrada:
            nombres = [
                (_obtener_valor(e, "name", "nombre") or _obtener_valor(e, "code", "codigo"))
                for e in especialidades
            ]
            return (
                f"No encontramos la especialidad o servicio '{especialidad}' en nuestro catálogo. "
                "Especialidades disponibles: " + ", ".join(filter(None, nombres))
            )
            
        esp_id = _obtener_valor(especialidad_encontrada, "id", "especialidadId", "specialtyId")
        esp_nombre = _obtener_valor(especialidad_encontrada, "name", "nombre") or especialidad
        
        # 2. Buscar profesionales (si vienen sin filtro o filtrados)
        profesionales = await dotnet_client.obtener_profesionales(especialidad_id=esp_id)
        if not profesionales:
            # Fallback: consultar todos los profesionales
            profesionales = await dotnet_client.obtener_profesionales()
            
        if not profesionales:
            return f"No hay profesionales registrados o disponibles actualmente para la especialidad de {esp_nombre}."
            
        # 3. Buscar el servicio correspondiente para determinar la duración esperada
        servicios = await dotnet_client.obtener_servicios()
        servicio_encontrado = None
        if servicios:
            for ser in servicios:
                nombre_ser = _obtener_valor(ser, "name", "nombre", "code", "codigo") or ""
                desc_ser = _obtener_valor(ser, "description", "descripcion") or ""
                if (
                    norm_esp in _normalizar_texto(nombre_ser)
                    or _normalizar_texto(nombre_ser) in norm_esp
                    or norm_esp in _normalizar_texto(desc_ser)
                ):
                    servicio_encontrado = ser
                    break
            if not servicio_encontrado:
                servicio_encontrado = servicios[0]
                
        servicio_id = _obtener_valor(servicio_encontrado, "id", "servicioId", "serviceId") or 1
        servicio_nombre = _obtener_valor(servicio_encontrado, "name", "nombre") or "Consulta General"
        duracion_servicio = int(_obtener_valor(servicio_encontrado, "durationMinutes", "duracionMinutos") or 60)
        
        # Calcular día de la semana ISO (1=Lunes .. 7=Domingo)
        iso_day = None
        try:
            iso_day = datetime.strptime(fecha[:10], "%Y-%m-%d").isoweekday()
        except Exception:
            pass

        # 4. Consultar disponibilidad para cada profesional
        resultados = []
        for prof in profesionales:
            prof_id = _obtener_valor(prof, "id", "profesionalId", "professionalId")
            prof_nombre = _obtener_valor(prof, "name", "nombre", "nombreCompleto")
            if not prof_nombre:
                prof_nombre = f"Dr. ID {prof_id}"
                
            horarios = await dotnet_client.consultar_disponibilidad(
                profesional_id=prof_id,
                fecha=fecha,
                servicio_id=servicio_id,
            )
            
            if horarios:
                slots = []
                for h in horarios:
                    start_day = _obtener_valor(h, "startDay", "diaInicio")
                    end_day = _obtener_valor(h, "endDay", "diaFin")
                    start_time = _obtener_valor(h, "startTime", "horaInicio")
                    end_time = _obtener_valor(h, "endTime", "horaFin")
                    lunch_start = _obtener_valor(h, "lunchStartTime", "horaAlmuerzoInicio")
                    lunch_end = _obtener_valor(h, "lunchEndTime", "horaAlmuerzoFin")

                    if start_day is not None and end_day is not None and start_time and end_time:
                        if iso_day is None or (int(start_day) <= iso_day <= int(end_day)):
                            gen = _generar_slots_desde_regla(str(start_time), str(end_time), lunch_start, lunch_end, duracion_servicio)
                            slots.extend(gen)
                    else:
                        inicio = _obtener_valor(h, "horaInicio", "fechaHoraInicio", "inicio", "startTime", "startsAt")
                        if inicio:
                            if "T" in str(inicio):
                                inicio = str(inicio).split("T")[1][:5]
                            else:
                                inicio = str(inicio)[:5]
                            slots.append(inicio)

                # Eliminar duplicados manteniendo orden
                unique_slots = list(dict.fromkeys(slots))
                if unique_slots:
                    resultados.append(f"👨‍⚕️ {prof_nombre} (ID: {prof_id}):\n   Horarios disponibles: " + ", ".join(unique_slots))
                else:
                    resultados.append(f"👨‍⚕️ {prof_nombre} (ID: {prof_id}): Sin atención programada para este día.")
            else:
                resultados.append(f"👨‍⚕️ {prof_nombre} (ID: {prof_id}): Sin horarios disponibles.")
                
        if not resultados:
            return f"No se encontraron espacios disponibles para {esp_nombre} el día {fecha}."
            
        return f"Horarios disponibles para {servicio_nombre} (ID de servicio: {servicio_id}) el día {fecha}:\n\n" + "\n".join(resultados)
    except Exception as exc:
        logger.error(f"Error al consultar disponibilidad: {exc}", exc_info=True)
        return "En este momento no podemos acceder a la disponibilidad de la agenda debido a problemas de conexión con el servidor. Por favor intenta de nuevo más tarde."

async def _agendar_cita_impl(
    profesional_id: Any,
    servicio_id: Any,
    fecha_hora_inicio: str,
    motivo_consulta: str,
    config: RunnableConfig,
) -> str:
    try:
        thread_id = config.get("configurable", {}).get("thread_id", "")
        
        # 1. Resolver profesional
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

        # 2. Resolver servicio y duración
        servs = await dotnet_client.obtener_servicios() or []
        resolved_serv = None
        for s in servs:
            if str(s.get("id")).lower() == str(servicio_id).lower():
                resolved_serv = s
                break
            s_name = _normalizar_texto(s.get("name") or "")
            norm_target = _normalizar_texto(str(servicio_id))
            if norm_target and (norm_target in s_name or s_name in norm_target):
                resolved_serv = s
                break
        if not resolved_serv and servs:
            resolved_serv = servs[0]
            
        resolved_serv_id = resolved_serv.get("id") if resolved_serv else servicio_id
        serv_nombre_display = resolved_serv.get("name") if resolved_serv else "Consulta Odontológica"
        duracion_min = int(resolved_serv.get("durationMinutes") or 45) if resolved_serv else 45

        # 3. Obtener o resolver el paciente
        contexto = await dotnet_client.obtener_contexto_conversacion(thread_id) if thread_id else None
        paciente_id = None
        
        if contexto:
            paciente_id = _obtener_valor(contexto, "patientId", "pacienteId")
            if not paciente_id and "paciente" in contexto:
                paciente_id = _obtener_valor(contexto["paciente"], "id", "pacienteId", "patientId")
            if not paciente_id and "patient" in contexto:
                paciente_id = _obtener_valor(contexto["patient"], "id", "patientId", "pacienteId")
                
        if not paciente_id and thread_id:
            clean_phone = "".join(c for c in thread_id.split("@")[0] if c.isdigit())
            pacientes = await dotnet_client.buscar_pacientes(clean_phone)
            if pacientes:
                paciente = pacientes[0]
                paciente_id = _obtener_valor(paciente, "id", "pacienteId", "patientId")
                if paciente_id:
                    await dotnet_client.vincular_paciente(thread_id, paciente_id)

        if not paciente_id:
            # Fallback a pacientes registrados
            pacientes_all = await dotnet_client.buscar_pacientes("")
            if pacientes_all:
                paciente_id = _obtener_valor(pacientes_all[0], "id", "pacienteId", "patientId")

        if not paciente_id:
            return (
                "Lo siento, no pude encontrar tu número registrado como paciente en nuestra base de datos. "
                "Por favor, facilítame tu nombre completo y número de documento para que recepción pueda registrarte."
            )

        # 4. Calcular startsAt y endsAt en formato ISO 8601 UTC
        try:
            raw = str(fecha_hora_inicio).strip()
            if "T" not in raw and " " not in raw and len(raw) == 10:
                raw += "T08:00:00"
            raw = raw.replace(" ", "T")

            try:
                starts_dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except Exception:
                import re
                m = re.search(r"(\d{4}-\d{2}-\d{2})[T\s](\d{1,2}):(\d{2})", raw)
                if m:
                    starts_dt = datetime.strptime(f"{m.group(1)}T{int(m.group(2)):02d}:{m.group(3)}:00", "%Y-%m-%dT%H:%M:%S")
                else:
                    starts_dt = datetime.now()

            ends_dt = starts_dt + timedelta(minutes=duracion_min)
            starts_at_iso = starts_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            ends_at_iso = ends_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception as dt_err:
            logger.warning(f"[Agenda Tools] Error formateando fechas ({fecha_hora_inicio}): {dt_err}")
            now = datetime.now()
            starts_at_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
            ends_at_iso = (now + timedelta(minutes=duracion_min)).strftime("%Y-%m-%dT%H:%M:%SZ")
            starts_dt = now
            ends_dt = now + timedelta(minutes=duracion_min)

        # 5. Obtener IDs de estado y origen
        status_id = await dotnet_client.obtener_appointment_status_id("AGENDADA")
        origin_id = await dotnet_client.obtener_appointment_origin_id("AGENTE_BOT")

        # 6. Agendar la cita en .NET con el DTO completo
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
        if respuesta:
            cita_id = _obtener_valor(respuesta, "id", "citaId", "appointmentId")
            hora_inicio_str = starts_dt.strftime("%H:%M") if "starts_dt" in locals() else fecha_hora_inicio
            hora_fin_str = ends_dt.strftime("%H:%M") if "ends_dt" in locals() else ""
            fecha_str = starts_dt.strftime("%Y-%m-%d") if "starts_dt" in locals() else ""

            return (
                f"¡Cita agendada con éxito! 🎉\n\n"
                f"📋 *Detalles de tu cita:*\n"
                f"- 👨‍⚕️ *Doctor:* {prof_nombre_display}\n"
                f"- 🦷 *Servicio:* {serv_nombre_display}\n"
                f"- 📅 *Fecha:* {fecha_str}\n"
                f"- ⏰ *Horario:* {hora_inicio_str} a {hora_fin_str}\n"
                f"- 🆔 *Código de cita:* `{cita_id}`\n\n"
                f"📍 *Ubicación:* Cr 24 #35-12, Santander (Nexus Odonto).\n"
                f"¡Te esperamos!"
            )
        else:
            return "Lo siento, ocurrió un problema al registrar la cita en el sistema. Es posible que el horario seleccionado ya esté ocupado o fuera del turno de atención. Por favor, elige otro horario disponible."
    except Exception as exc:
        logger.error(f"Error al agendar cita: {exc}", exc_info=True)
        return "Lo siento, ocurrió un problema de conexión al registrar la cita en el sistema. Por favor, intenta de nuevo en unos minutos."


async def _consultar_doctores_impl(especialidad: Optional[str] = None) -> str:
    try:
        especialidades = await dotnet_client.obtener_especialidades() or []
        esp_id = None
        esp_nombre = None
        
        if especialidad:
            norm_esp = _normalizar_texto(especialidad)
            for esp in especialidades:
                nombre = _obtener_valor(esp, "name", "nombre", "code", "codigo") or ""
                if norm_esp in _normalizar_texto(nombre) or _normalizar_texto(nombre) in norm_esp:
                    esp_id = _obtener_valor(esp, "id", "especialidadId")
                    esp_nombre = _obtener_valor(esp, "name", "nombre")
                    break

        profesionales = await dotnet_client.obtener_profesionales(especialidad_id=esp_id) or []
        
        if profesionales:
            nombres_prof = []
            for p in profesionales:
                nombre = _obtener_valor(p, "name", "nombre", "nombreCompleto")
                if not nombre and "empleado" in p:
                    nombre = _obtener_valor(p["empleado"], "nombreCompleto", "nombre", "name")
                if not nombre and "persona" in p:
                    nombre = _obtener_valor(p["persona"], "nombreCompleto", "nombre", "firstName")
                if not nombre:
                    p_id = _obtener_valor(p, "id", "profesionalId")
                    nombre = f"Dr. Especialista (ID: {p_id})"
                nombres_prof.append(f"👨‍⚕️ {nombre}")
            
            if esp_nombre:
                return f"Nuestros profesionales para {esp_nombre} son:\n" + "\n".join(nombres_prof)
            return "Nuestros profesionales disponibles en la clínica son:\n" + "\n".join(nombres_prof)
        else:
            # Si no hay profesionales en BD, mostramos los servicios/especialidades que sí existen
            nombres_esp = [(_obtener_valor(e, "name", "nombre") or _obtener_valor(e, "code", "codigo")) for e in especialidades]
            esp_str = ", ".join(filter(None, nombres_esp)) if nombres_esp else "Ortodoncia, Valoración General, Profilaxis y Cirugía Oral"
            return (
                "Actualmente no tenemos doctores con turnos asignados en el sistema en este momento. "
                f"Sin embargo, nuestra clínica Nexus Odonto cuenta con atención en las siguientes especialidades: {esp_str}. "
                "¿Deseas consultar sobre alguno de nuestros servicios o agendar para una fecha específica?"
            )
    except Exception as exc:
        logger.error(f"Error al consultar doctores: {exc}", exc_info=True)
        return "En este momento no podemos acceder a la lista de doctores. Por favor intenta de nuevo en unos minutos."


async def _consultar_servicios_impl() -> str:
    try:
        servicios = await dotnet_client.obtener_servicios() or []
        if not servicios:
            return "En este momento no podemos acceder a la lista de servicios. Por favor intenta de nuevo más tarde."
            
        lineas = []
        for s in servicios:
            nombre = _obtener_valor(s, "name", "nombre") or "Servicio Odontológico"
            desc = _obtener_valor(s, "description", "descripcion") or ""
            precio = _obtener_valor(s, "price", "precio")
            duracion = _obtener_valor(s, "durationMinutes", "duracionMinutos")
            
            detalles = []
            if duracion:
                detalles.append(f"Duración: {duracion} min")
            if precio:
                detalles.append(f"Precio: ${precio:,.0f}" if isinstance(precio, (int, float)) else f"Precio: ${precio}")
            
            detalle_str = f" ({', '.join(detalles)})" if detalles else ""
            lineas.append(f"🦷 *{nombre}*{detalle_str}: {desc}" if desc else f"🦷 *{nombre}*{detalle_str}")
            
        return "Nuestros servicios odontológicos disponibles en Nexus Odonto son:\n\n" + "\n".join(lineas)
    except Exception as exc:
        logger.error(f"Error al consultar servicios: {exc}", exc_info=True)
        return "En este momento no podemos acceder al catálogo de servicios. Por favor intenta de nuevo más tarde."


@tool
def consultar_disponibilidad_tool(especialidad: str, fecha: str) -> str:
    """
    Consulta los horarios disponibles para una especialidad odontológica en una fecha específica (formato YYYY-MM-DD).
    Usa esta herramienta cuando el usuario pregunte por horarios o citas disponibles para un servicio/especialidad (ej. ortodoncia, limpieza, profilaxis, valoración, resina).
    """
    return _run_sync(_consultar_disponibilidad_impl(especialidad, fecha))


@tool
def agendar_cita_tool(
    profesional_id: str,
    servicio_id: str,
    fecha_hora_inicio: str,
    motivo_consulta: str,
    config: RunnableConfig,
) -> str:
    """
    Registra una cita en el sistema para el paciente de la conversación actual.
    Usa esta herramienta SOLAMENTE después de proponer los detalles de la cita y obtener una confirmación explícita y afirmativa del usuario en el chat.
    
    Parámetros:
    - profesional_id: ID o UUID del odontólogo seleccionado.
    - servicio_id: ID o UUID del servicio odontológico.
    - fecha_hora_inicio: Fecha y hora de inicio de la cita en formato ISO 8601 (ej. YYYY-MM-DDTHH:MM:SS-05:00).
    - motivo_consulta: Breve descripción de la razón de la consulta.
    """
    return _run_sync(_agendar_cita_impl(profesional_id, servicio_id, fecha_hora_inicio, motivo_consulta, config))


@tool
def consultar_doctores_tool(especialidad: Optional[str] = None) -> str:
    """
    Consulta la lista de doctores, odontólogos o profesionales disponibles en la clínica Nexus Odonto, opcionalmente filtrados por especialidad.
    Usa esta herramienta cuando el paciente pregunte quiénes son los doctores, qué odontólogos atienden o qué profesionales hay disponibles.
    """
    return _run_sync(_consultar_doctores_impl(especialidad))


@tool
def consultar_servicios_y_precios_tool() -> str:
    """
    Consulta la lista oficial de servicios odontológicos, especialidades, duración y precios vigentes en Nexus Odonto desde la base de datos.
    Usa esta herramienta cuando el usuario pregunte qué servicios prestan, qué tratamientos hacen, o cuánto cuestan los procedimientos.
    """
    return _run_sync(_consultar_servicios_impl())
