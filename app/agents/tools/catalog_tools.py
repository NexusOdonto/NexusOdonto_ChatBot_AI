"""Herramientas de consulta de Catálogo, Servicios y Disponibilidad de Nexus Odonto.
Totalmente desacopladas, con validación de horarios y protección contra respuestas robóticas.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from langchain_core.tools import tool

from app.clients.dotnet_client import dotnet_client
from app.agents.tools.agenda_helpers import (
    _normalizar_texto,
    _obtener_valor,
    _formatear_hora_ampm,
    _generar_slots_desde_regla,
    _buscar_servicio_por_texto,
    _buscar_especialidad_por_texto,
    _servicios_activos,
    _servicios_relacionados_a_especialidad,
    _etiqueta_servicio,
    _lista_servicios_whatsapp,
    _mensaje_catalogo_no_encontrado,
    _mensaje_especialidad_sin_servicio_unico,
    DESCRIPCIONES_SERVICIO_ES,
)

logger = logging.getLogger(__name__)


async def _consultar_disponibilidad_impl(especialidad: str, fecha: str) -> str:
    """Consulta la disponibilidad real de turnos evitando recesos de almuerzo (12:00 a 14:00) y traslapes."""
    try:
        norm_query = _normalizar_texto(especialidad)
        if not norm_query:
            return "Por favor, indica una especialidad o servicio válido."

        servicios, especialidades = await asyncio.gather(
            dotnet_client.obtener_servicios(),
            dotnet_client.obtener_especialidades(),
        )
        servicios = servicios or []
        especialidades = especialidades or []
        servicio_encontrado = _buscar_servicio_por_texto(norm_query, servicios)
        especialidad_encontrada = None
        esp_id = None
        esp_nombre = None

        if servicio_encontrado:
            cat_servicio = _obtener_valor(servicio_encontrado, "category", "categoria", "Category") or ""
            nombre_ser = _obtener_valor(servicio_encontrado, "name", "nombre") or ""
            for candidato in (cat_servicio, nombre_ser, especialidad):
                if not candidato:
                    continue
                especialidad_encontrada = _buscar_especialidad_por_texto(
                    _normalizar_texto(str(candidato)), especialidades
                )
                if especialidad_encontrada:
                    break
            if especialidad_encontrada:
                esp_id = _obtener_valor(especialidad_encontrada, "id", "especialidadId", "specialtyId")
                esp_nombre = _obtener_valor(especialidad_encontrada, "name", "nombre")
        else:
            if not especialidades:
                return (
                    "En este momento no podemos acceder al catálogo de servicios ni especialidades. "
                    "Por favor intenta de nuevo más tarde."
                )
            especialidad_encontrada = _buscar_especialidad_por_texto(norm_query, especialidades)
            if not especialidad_encontrada:
                return _mensaje_catalogo_no_encontrado(especialidad, servicios)

            esp_id = _obtener_valor(especialidad_encontrada, "id", "especialidadId", "specialtyId")
            esp_nombre = _obtener_valor(especialidad_encontrada, "name", "nombre") or especialidad
            relacionados = _servicios_relacionados_a_especialidad(
                especialidad_encontrada, servicios, norm_query
            )
            if len(relacionados) == 1:
                servicio_encontrado = relacionados[0]
            else:
                return _mensaje_especialidad_sin_servicio_unico(
                    especialidad, str(esp_nombre), relacionados, servicios
                )

        profesionales = []
        if esp_id:
            profesionales = await dotnet_client.obtener_profesionales(especialidad_id=esp_id) or []
        if not profesionales:
            profesionales = await dotnet_client.obtener_profesionales() or []

        etiqueta = (
            (_etiqueta_servicio(servicio_encontrado) if servicio_encontrado else None)
            or esp_nombre
            or especialidad
        )
        if not profesionales:
            return f"No hay profesionales registrados o disponibles actualmente para {etiqueta}."

        if servicio_encontrado:
            servicio_id = _obtener_valor(servicio_encontrado, "id", "servicioId", "serviceId")
            servicio_nombre = _etiqueta_servicio(servicio_encontrado) or etiqueta
            duracion_servicio = int(
                _obtener_valor(servicio_encontrado, "durationMinutes", "duracionMinutos") or 60
            )
        else:
            return _mensaje_catalogo_no_encontrado(especialidad, servicios)

        iso_day = None
        try:
            iso_day = datetime.strptime(fecha[:10], "%Y-%m-%d").isoweekday()
        except Exception:
            pass

        # Una sola consulta de citas del día (antes: por cada profesional)
        fecha_target = str(fecha)[:10]
        try:
            citas_raw_shared = await dotnet_client.consultar_citas(fecha_target)
            if not citas_raw_shared:
                citas_raw_shared = await dotnet_client.consultar_citas("")
        except Exception:
            citas_raw_shared = None

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
                    h_prof = str(_obtener_valor(h, "professionalId", "profesionalId") or "")
                    if h_prof and h_prof.lower() != str(prof_id).lower():
                        continue

                    start_day = _obtener_valor(h, "startDay", "diaInicio")
                    end_day = _obtener_valor(h, "endDay", "diaFin")
                    start_time = _obtener_valor(h, "startTime", "horaInicio")
                    end_time = _obtener_valor(h, "endTime", "horaFin")
                    lunch_start = _obtener_valor(h, "lunchStartTime", "horaAlmuerzoInicio")
                    lunch_end = _obtener_valor(h, "lunchEndTime", "horaAlmuerzoFin")

                    if start_day is not None and end_day is not None and start_time and end_time:
                        if iso_day is None or (int(start_day) <= iso_day <= int(end_day)):
                            gen = _generar_slots_desde_regla(
                                str(start_time), str(end_time), lunch_start, lunch_end, duracion_servicio
                            )
                            slots.extend(gen)
                    else:
                        inicio = _obtener_valor(h, "horaInicio", "fechaHoraInicio", "inicio", "startTime", "startsAt")
                        if inicio:
                            if "T" in str(inicio):
                                inicio = str(inicio).split("T")[1][:5]
                            else:
                                inicio = str(inicio)[:5]
                            slots.append(inicio)

                unique_slots = sorted(list(dict.fromkeys(slots)))

                # Filtrar traslapes con citas existentes
                try:
                    citas_raw = citas_raw_shared
                    if isinstance(citas_raw, dict):
                        citas_existentes = citas_raw.get("items", [])
                    elif isinstance(citas_raw, list):
                        citas_existentes = citas_raw
                    else:
                        citas_existentes = []

                    intervalos_ocupados = []
                    for c in citas_existentes:
                        if not isinstance(c, dict):
                            continue
                        c_prof = str(
                            c.get("professionalId")
                            or c.get("profesionalId")
                            or c.get("employeeId")
                            or c.get("doctorId")
                            or ""
                        ).lower().strip()
                        c_canc = c.get("cancelledAt") or c.get("cancelado")
                        c_st_id = str(c.get("appointmentStatusId") or "").lower().strip()
                        c_st_name = str(c.get("statusName") or c.get("status") or c.get("estado") or "").lower()

                        if prof_id and c_prof and c_prof != str(prof_id).lower().strip():
                            continue

                        if c_canc or c_st_id == "10000000-0000-0000-0000-000000000005" or "cancel" in c_st_name:
                            continue

                        c_start_raw = str(
                            c.get("startsAt")
                            or c.get("fechaHoraInicio")
                            or c.get("startDateTime")
                            or c.get("startTime")
                            or ""
                        ).replace("Z", "").split(".")[0].replace(" ", "T")

                        c_end_raw = str(
                            c.get("endsAt")
                            or c.get("fechaHoraFin")
                            or c.get("endDateTime")
                            or c.get("endTime")
                            or ""
                        ).replace("Z", "").split(".")[0].replace(" ", "T")

                        if len(c_start_raw) <= 8 and ":" in c_start_raw:
                            c_start_raw = f"{fecha_target}T{c_start_raw}"
                        if len(c_end_raw) <= 8 and ":" in c_end_raw:
                            c_end_raw = f"{fecha_target}T{c_end_raw}"

                        if c_start_raw and c_start_raw[:10] == fecha_target:
                            try:
                                dt_c_start = datetime.fromisoformat(c_start_raw)
                                if c_end_raw and len(c_end_raw) >= 16:
                                    dt_c_end = datetime.fromisoformat(c_end_raw)
                                else:
                                    c_dur = int(c.get("durationMinutes") or duracion_servicio or 45)
                                    dt_c_end = dt_c_start + timedelta(minutes=c_dur)
                                intervalos_ocupados.append((dt_c_start, dt_c_end))
                            except Exception:
                                pass

                    dur_eval = max(30, int(duracion_servicio or 30))
                    slots_libres = []
                    for s in unique_slots:
                        try:
                            s_clean = s[:5]
                            slot_start_dt = datetime.strptime(f"{fecha_target}T{s_clean}:00", "%Y-%m-%dT%H:%M:%S")
                            slot_end_dt = slot_start_dt + timedelta(minutes=dur_eval)

                            colision = any(
                                slot_start_dt < c_end and slot_end_dt > c_start
                                for c_start, c_end in intervalos_ocupados
                            )
                            if not colision:
                                slots_libres.append(s_clean)
                        except Exception:
                            slots_libres.append(s[:5])

                    unique_slots = slots_libres
                except Exception as c_err:
                    logger.warning(f"[CatalogTools] Error consultando citas ocupadas: {c_err}")

                try:
                    from zoneinfo import ZoneInfo
                    now_bogota = datetime.now(ZoneInfo("America/Bogota"))
                except Exception:
                    now_bogota = datetime.now()

                if str(fecha)[:10] == now_bogota.strftime("%Y-%m-%d"):
                    min_dt = now_bogota + timedelta(minutes=30)
                    min_hhmm = min_dt.strftime("%H:%M")
                    unique_slots = [s for s in unique_slots if s >= min_hhmm]

                if unique_slots:
                    slots_ampm = [_formatear_hora_ampm(s) for s in unique_slots]
                    bloque_horarios = "   ⏰ " + ", ".join(slots_ampm)
                    resultados.append(f"👨‍⚕️ *{prof_nombre}*:\n{bloque_horarios}")
                else:
                    resultados.append(f"👨‍⚕️ *{prof_nombre}*: Sin turnos disponibles para esta fecha.")
            else:
                resultados.append(f"👨‍⚕️ *{prof_nombre}*: Sin horarios registrados.")

        if not resultados:
            return f"No se encontraron espacios disponibles para *{servicio_nombre}* en la fecha `{fecha}`. ¿Deseas consultar otro día? 😊"

        return (
            f"📅 *Horarios Disponibles para {servicio_nombre}* ✨\n"
            f"🗓️ *Fecha:* {fecha}\n\n"
            + "\n\n".join(resultados)
            + "\n\n💬 *¿Cuál de estos horarios te queda más cómodo para apartar tu cita?* 😊"
        )
    except Exception as exc:
        logger.error(f"Error al consultar disponibilidad: {exc}", exc_info=True)
        return "En este momento no podemos acceder a la disponibilidad de la agenda. Por favor intenta de nuevo en unos minutos."


async def _consultar_doctores_impl(especialidad: Optional[str] = None) -> str:
    """Consulta odontólogos y especialistas en Nexus Odonto."""
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
            tarjetas_prof = []
            for p in profesionales:
                nombre = _obtener_valor(p, "name", "nombre", "nombreCompleto") or "Especialista Odontológico"
                nombre_clean = str(nombre).strip()
                if not nombre_clean.lower().startswith(("dr", "dra")):
                    nombre_clean = f"Dr(a). {nombre_clean}"
                tarjetas_prof.append(f"• *{nombre_clean}* — Odontología Integral y Especializada")

            titulo = f"En *Nexus Odonto* contamos con especialistas en {esp_nombre}:" if esp_nombre else "En *Nexus Odonto* contamos con odontólogos especialistas:"
            return (
                f"{titulo}\n\n"
                + "\n".join(tarjetas_prof)
                + "\n\n¿Te gustaría consultar los horarios de alguno de ellos para agendar tu cita? 😊"
            )
        else:
            servicios = await dotnet_client.obtener_servicios() or []
            lista_serv = _lista_servicios_whatsapp(servicios)
            if lista_serv:
                return (
                    "Actualmente estamos actualizando los turnos de nuestros doctores.\n\n"
                    "Mientras tanto, estos son los *servicios activos* que puedes agendar:\n"
                    f"{lista_serv}\n\n"
                    "¿Deseas consultar disponibilidad de alguno? 😊"
                )
            nombres_esp = [
                (_obtener_valor(e, "name", "nombre") or _obtener_valor(e, "code", "codigo"))
                for e in especialidades
            ]
            esp_str = ", ".join(filter(None, nombres_esp))
            if esp_str:
                return (
                    "Actualmente estamos actualizando los turnos de nuestros doctores. "
                    f"Nuestra clínica cuenta con atención en: *{esp_str}*.\n\n"
                    "¿Deseas consultar sobre alguno de nuestros tratamientos? 😊"
                )
            return (
                "Actualmente estamos actualizando los turnos de nuestros doctores. "
                "¿Deseas intentar de nuevo en unos minutos? 😊"
            )
    except Exception as exc:
        logger.error(f"Error al consultar doctores: {exc}", exc_info=True)
        return "En este momento no podemos acceder a la lista de especialistas. Por favor intenta de nuevo en unos minutos."


async def _consultar_servicios_impl() -> str:
    """Consulta la lista oficial de servicios activos de Nexus Odonto."""
    try:
        servicios = await dotnet_client.obtener_servicios() or []
        activos = _servicios_activos(servicios)
        if not activos:
            return (
                "En este momento no hay servicios activos en el catálogo, "
                "o no podemos acceder a la lista. Por favor intenta de nuevo más tarde."
            )

        items_servicios = []
        for s in activos:
            nombre = _etiqueta_servicio(s)
            desc = _obtener_valor(s, "description", "descripcion")
            desc_str = str(desc).strip() if desc and str(desc).strip() else ""
            desc_es = DESCRIPCIONES_SERVICIO_ES.get(desc_str.lower(), desc_str)
            precio = _obtener_valor(s, "price", "precio")
            duracion = _obtener_valor(s, "durationMinutes", "duracionMinutos")

            partes = []
            if precio is not None and str(precio).strip() != "":
                try:
                    precio_num = float(precio)
                    partes.append(f"${precio_num:,.0f} COP")
                except (TypeError, ValueError):
                    partes.append(f"${precio} COP")
            if duracion:
                partes.append(f"{duracion} min")

            detalles = f" ({', '.join(partes)})" if partes else ""
            items_servicios.append(f"• *{nombre}*{detalles}")

        return (
            "Catálogo de Servicios y Tratamientos Activos en Nexus Odonto:\n"
            + "\n".join(items_servicios)
            + "\n\n[INSTRUCCIÓN CRÍTICA DE COMUNICACIÓN HUMANA: "
            "Responde al paciente como una asesora empática y conversacional por WhatsApp. "
            "NUNCA hagas un volcado copiado de toda la lista de servicios con todas las duraciones y precios a la vez. "
            "Si preguntó de forma general qué servicios tienen, salúdalo con calidez por su nombre, resume en 3 o 4 viñetas limpias las categorías principales (limpieza/profilaxis, resinas/calzas estéticas, blanqueamiento, valoración general) "
            "y pregúntale amablemente si presenta alguna molestia o qué procedimiento en particular le interesa. "
            "Si el paciente preguntó por un servicio puntual, dale directamente su valor y detalles amables.]"
        )
    except Exception as exc:
        logger.error(f"Error al consultar servicios: {exc}", exc_info=True)
        return "En este momento no podemos acceder al catálogo de servicios. Por favor intenta de nuevo más tarde."


# ─── Herramientas LangChain (async nativas — ToolNode.ainvoke sin _run_sync/t.join) ─

@tool
async def consultar_disponibilidad_tool(especialidad: str, fecha: str) -> str:
    """
    Consulta los horarios disponibles para un servicio odontológico (o especialidad) en una fecha (YYYY-MM-DD).
    IMPORTANTE: Usa SOLO nombres de servicios activos del catálogo (consultar_servicios_y_precios_tool).
    No inventes ni ofrezcas ejemplos de tratamientos que no hayan salido de esa herramienta.
    Si el paciente nombra una especialidad (p. ej. ortodoncia), esta herramienta listará los servicios activos relacionados.
    """
    return await _consultar_disponibilidad_impl(especialidad, fecha)


@tool
async def consultar_doctores_tool(especialidad: Optional[str] = None) -> str:
    """
    Consulta la lista de doctores, odontólogos o profesionales disponibles en la clínica Nexus Odonto, opcionalmente filtrados por especialidad.
    Usa esta herramienta cuando el paciente pregunte quiénes son los doctores, qué odontólogos atienden o qué profesionales hay disponibles.
    """
    return await _consultar_doctores_impl(especialidad)


@tool
async def consultar_servicios_y_precios_tool() -> str:
    """
    Consulta la lista oficial y actual de servicios activos de Nexus Odonto (nombres, precios y duraciones).
    Usa esta herramienta como fuente de verdad para responder de forma cálida, conversacional y humana al paciente.
    NUNCA inventes precios ni servicios que no existan en este catálogo.
    """
    return await _consultar_servicios_impl()