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
    fecha_legible,
    formatear_precio_cop,
    hora_corta,
    motivo_dia_cerrado,
    es_consulta_urgencia,
    servicio_para_urgencia,
)

logger = logging.getLogger(__name__)


def _intervalos_ocupados(citas_raw: Any, prof_id: Any, fecha_target: str, duracion_servicio: int) -> list:
    if isinstance(citas_raw, dict):
        citas_existentes = citas_raw.get("items", [])
    elif isinstance(citas_raw, list):
        citas_existentes = citas_raw
    else:
        citas_existentes = []

    intervalos = []
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
                intervalos.append((dt_c_start, dt_c_end))
            except Exception:
                pass
    return intervalos


def _dentro_jornada_clinica(hhmm: str, iso_day: Optional[int], duracion: int) -> bool:
    """Clinic hours win over a professional's rule: Mon-Fri 8-12/14-17, Sat 8-12, Sun closed."""
    try:
        h, m = (int(x) for x in str(hhmm)[:5].split(":"))
    except ValueError:
        return False
    inicio = h * 60 + m
    fin = inicio + max(30, int(duracion or 30))
    if iso_day == 7:
        return False
    if iso_day == 6:
        return 8 * 60 <= inicio and fin <= 12 * 60
    en_manana = 8 * 60 <= inicio and fin <= 12 * 60
    en_tarde = 14 * 60 <= inicio and fin <= 17 * 60
    return en_manana or en_tarde


async def _turnos_por_profesional(
    profesionales: List[Dict[str, Any]],
    fecha: str,
    servicio_id: Any,
    duracion_servicio: int,
) -> List[tuple]:
    """[(prof_id, prof_nombre, slots HH:MM | None si no tiene horario)] libres para la fecha.

    Respeta la regla del profesional, almuerzo 12-14, cierre 17:00, duración del servicio,
    citas existentes y (si es hoy) un margen de 30 min.
    """
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

    try:
        from zoneinfo import ZoneInfo
        now_bogota = datetime.now(ZoneInfo("America/Bogota"))
    except Exception:
        now_bogota = datetime.now()

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
        if not horarios:
            resultados.append((prof_id, prof_nombre, None))
            continue

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

        unique_slots = [
            s for s in sorted(dict.fromkeys(slots)) if _dentro_jornada_clinica(s, iso_day, duracion_servicio)
        ]

        # Filtrar traslapes con citas existentes
        try:
            intervalos_ocupados = _intervalos_ocupados(
                citas_raw_shared, prof_id, fecha_target, duracion_servicio
            )
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

        if str(fecha)[:10] == now_bogota.strftime("%Y-%m-%d"):
            min_hhmm = (now_bogota + timedelta(minutes=30)).strftime("%H:%M")
            unique_slots = [s for s in unique_slots if s >= min_hhmm]

        resultados.append((prof_id, prof_nombre, unique_slots))
    return resultados


async def profesionales_para_servicio(servicio: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Professionals of the service's specialty (all active professionals if none matches)."""
    especialidades = await dotnet_client.obtener_especialidades() or []
    esp_id = None
    cat_servicio = _obtener_valor(servicio, "category", "categoria", "Category") or ""
    nombre_ser = _obtener_valor(servicio, "name", "nombre") or ""
    for candidato in (cat_servicio, nombre_ser):
        if not candidato:
            continue
        esp = _buscar_especialidad_por_texto(_normalizar_texto(str(candidato)), especialidades)
        if esp:
            esp_id = _obtener_valor(esp, "id", "especialidadId", "specialtyId")
            break

    profesionales = []
    if esp_id:
        profesionales = await dotnet_client.obtener_profesionales(especialidad_id=esp_id) or []
    if not profesionales:
        profesionales = await dotnet_client.obtener_profesionales() or []
    return profesionales


async def turnos_disponibles(servicio: Dict[str, Any], fecha: str) -> List[tuple]:
    """Turnos libres de un servicio del catálogo en una fecha YYYY-MM-DD: [(prof_id, prof_nombre, [HH:MM])]."""
    profesionales = await profesionales_para_servicio(servicio)
    if not profesionales:
        return []

    servicio_id = _obtener_valor(servicio, "id", "servicioId", "serviceId")
    duracion = int(_obtener_valor(servicio, "durationMinutes", "duracionMinutos") or 60)
    turnos = await _turnos_por_profesional(profesionales, fecha, servicio_id, duracion)
    return [(pid, nombre, slots) for pid, nombre, slots in turnos if slots]


def _hoy_bogota() -> datetime:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Bogota")).replace(tzinfo=None)
    except Exception:
        return datetime.now()


def _rangos_slots(slots: List[str]) -> str:
    """['08:00','08:30','09:00','14:00'] → 'de 8:00 a 9:00 AM, 2:00 PM' (consecutive 30-min starts merged)."""
    mins = sorted({int(s[:2]) * 60 + int(s[3:5]) for s in slots})
    grupos: List[List[int]] = []
    for m in mins:
        if grupos and m - grupos[-1][-1] == 30:
            grupos[-1].append(m)
        else:
            grupos.append([m])
    partes = []
    for g in grupos:
        ini = hora_corta(f"{g[0] // 60:02d}:{g[0] % 60:02d}")
        if len(g) == 1:
            partes.append(ini)
            continue
        fin = hora_corta(f"{g[-1] // 60:02d}:{g[-1] % 60:02d}")
        if ini[-2:] == fin[-2:]:
            ini = ini[:-3]
        partes.append(f"de {ini} a {fin}")
    return ", ".join(partes)


def _normalizar_hora(hora: Optional[str]) -> Optional[str]:
    """'9', '9:00 AM', '14:00', '2 pm' → 'HH:MM'. Bare 1-6 means afternoon (clinic hours)."""
    import re
    m = re.search(r"(\d{1,2})(?:[:.](\d{2}))?\s*([ap])?\.?\s*m?", str(hora or "").lower())
    if not m:
        return None
    h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ap == "p" and h < 12:
        h += 12
    elif ap == "a" and h == 12:
        h = 0
    elif not ap and 1 <= h <= 6:
        h += 12
    if h > 23 or mi > 59:
        return None
    return f"{h:02d}:{mi:02d}"


def _linea_hora_pedida(hhmm: str, turnos: List[tuple]) -> str:
    libres = [nombre for nombre, slots in turnos if hhmm in slots]
    if libres:
        return f"[HORA PEDIDA] {hora_corta(hhmm)}: LIBRE con {', '.join(libres)}."
    objetivo = int(hhmm[:2]) * 60 + int(hhmm[3:])
    cercanas = sorted(
        {(abs(int(s[:2]) * 60 + int(s[3:]) - objetivo), s, n) for n, slots in turnos for s in slots}
    )[:3]
    alternativas = ", ".join(f"{hora_corta(s)} con {n}" for _d, s, n in cercanas)
    return f"[HORA PEDIDA] {hora_corta(hhmm)}: NO está libre ese día. Más cercanas: {alternativas}."


def _bloque_dia(dia: datetime, turnos: List[tuple]) -> str:
    detalle = "; ".join(f"{nombre} {_rangos_slots(slots)}" for nombre, slots in turnos)
    return f"• {fecha_legible(dia)} ({dia.strftime('%Y-%m-%d')}): {detalle}"


def _sugerencias(dias: List[tuple], maximo: int = 4) -> List[str]:
    """Up to `maximo` options mixing days and morning/afternoon, earliest first."""
    por_dia = []
    uso: Dict[str, int] = {}
    for dia, turnos in dias:
        candidatos = []
        for franja in (lambda s: s < "12:00", lambda s: s >= "14:00"):
            mejor = min(
                ((s, uso.get(n, 0), n) for n, slots in turnos for s in slots if franja(s)),
                default=None,
            )
            if mejor:
                uso[mejor[2]] = uso.get(mejor[2], 0) + 1
                candidatos.append((dia, mejor[0], mejor[2]))
        por_dia.append(candidatos)
    elegidas: List[tuple] = []
    ronda = 0
    while len(elegidas) < maximo and any(len(c) > ronda for c in por_dia):
        for candidatos in por_dia:
            if ronda < len(candidatos) and len(elegidas) < maximo:
                elegidas.append(candidatos[ronda])
        ronda += 1
    return [
        f"{fecha_legible(dia, con_anio=False)} a las {hora_corta(s)} con {nombre}"
        for dia, s, nombre in sorted(elegidas, key=lambda x: (x[0], x[1]))
    ]


async def _dias_con_turnos(
    profesionales: List[Dict[str, Any]],
    servicio_id: Any,
    duracion: int,
    cerca_de: datetime,
    max_dias: int = 3,
    horizonte: int = 14,
) -> List[tuple]:
    """Open days with free slots closest to `cerca_de` (never before today): [(day, [(doctor, slots)])]."""
    hoy = _hoy_bogota().replace(hour=0, minute=0, second=0, microsecond=0)
    objetivo = max(cerca_de.replace(hour=0, minute=0, second=0, microsecond=0), hoy)
    candidatos = [objetivo + timedelta(days=i) for i in range(-3, horizonte + 1)]
    candidatos = [d for d in candidatos if d >= hoy and not motivo_dia_cerrado(d)]
    candidatos.sort(key=lambda d: (abs((d - objetivo).days), d))

    encontrados: List[tuple] = []
    for i in range(0, len(candidatos), 4):
        lote = candidatos[i : i + 4]
        resultados = await asyncio.gather(
            *(_turnos_por_profesional(profesionales, d.strftime("%Y-%m-%d"), servicio_id, duracion) for d in lote)
        )
        for dia, turnos in zip(lote, resultados):
            libres = [(nombre, slots) for _pid, nombre, slots in turnos if slots]
            if libres:
                encontrados.append((dia, libres))
        if len(encontrados) >= max_dias:
            break
    encontrados.sort(key=lambda x: (abs((x[0] - objetivo).days), x[0]))
    return sorted(encontrados[:max_dias], key=lambda x: x[0])


_INSTRUCCION_OPCIONES = (
    "[CÓMO RESPONDER] Ofrece 2-4 opciones (las sugeridas u otras de la lista, mezclando días y horas) "
    "nombrando el día de la semana EXACTAMENTE como aparece aquí, y pregunta cuál prefiere. "
    "No enumeres todos los turnos salvo que lo pida. Los rangos son inicios de cita cada 30 min "
    "(«de 8:00 a 10:00 AM» incluye 8:00, 8:30, 9:00, 9:30 y 10:00); una hora fuera de ellos NO está libre."
)


def _mas_tempranos(dias: List[tuple], por_dia: int = 2, max_dias: int = 2) -> List[str]:
    """Earliest distinct start times across all doctors on the first open days, in order."""
    out = []
    for dia, turnos in dias[:max_dias]:
        vistos: List[str] = []
        for s, nombre in sorted((s, n) for n, slots in turnos for s in slots):
            if s in vistos:
                continue
            vistos.append(s)
            out.append(f"{fecha_legible(dia, con_anio=False)} a las {hora_corta(s)} con {nombre}")
            if len(vistos) >= por_dia:
                break
    return out


def _texto_opciones(dias: List[tuple]) -> str:
    sugeridas = _sugerencias(dias)
    return (
        "\n".join(_bloque_dia(d, t) for d, t in dias)
        + ("\nSugeridas: " + " | ".join(sugeridas) if sugeridas else "")
    )


# Generic words that cover several catalog services ("limpieza dental" = Limpieza profunda or Profilaxis).
_TERMINOS_GENERICOS = {
    "limpieza": ("limpieza", "cleaning", "profilaxis", "prophylaxis", "higiene"),
    "limpiezas": ("limpieza", "cleaning", "profilaxis", "prophylaxis", "higiene"),
}
_PALABRAS_RELLENO = {"dental", "dentales", "de", "la", "el", "una", "un", "los", "dientes"}


def _nota_servicio_ambiguo(norm_query: str, servicios: List[Dict[str, Any]]) -> str:
    tokens = [t for t in norm_query.split() if t not in _PALABRAS_RELLENO]
    if len(tokens) != 1 or tokens[0] not in _TERMINOS_GENERICOS:
        return ""
    aliases = _TERMINOS_GENERICOS[tokens[0]]
    candidatos = []
    for s in _servicios_activos(servicios):
        nombre = str(_obtener_valor(s, "name", "nombre") or "")
        campos = " ".join(
            _normalizar_texto(str(c or ""))
            for c in (nombre, _etiqueta_servicio(s), _obtener_valor(s, "description", "descripcion"))
        )
        if any(a in campos for a in aliases):
            candidatos.append(s)
    if len(candidatos) < 2:
        return ""
    opciones = "; ".join(
        f"{_etiqueta_servicio(s)} ({formatear_precio_cop(_obtener_valor(s, 'price', 'precio'))}, "
        f"{_obtener_valor(s, 'durationMinutes', 'duracionMinutos') or '?'} min)"
        for s in candidatos
    )
    return (
        f"\n[SERVICIO AMBIGUO] «{norm_query}» puede ser: {opciones}. En una línea pregúntale cuál prefiere "
        "mostrando esos precios, y ofrece ya los horarios de abajo (no lo hagas esperar)."
    )


async def _consultar_disponibilidad_impl(
    especialidad: str, fecha: Optional[str] = None, hora: Optional[str] = None
) -> str:
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

        precio = _obtener_valor(servicio_encontrado, "price", "precio")
        cabecera = f"Servicio: {servicio_nombre} ({duracion_servicio} min"
        cabecera += f", {formatear_precio_cop(precio)})" if precio not in (None, "") else ")"
        if es_consulta_urgencia(norm_query):
            _ser, sustituto = servicio_para_urgencia(servicios)
            if sustituto and _ser is servicio_encontrado:
                cabecera += (
                    f"\n[URGENCIA] El catálogo no tiene un servicio de valoración/urgencia: se aparta como "
                    f"«{servicio_nombre}» y el odontólogo evalúa la molestia en la cita (dilo así en una línea). "
                    f"Usa «{servicio_nombre}» como servicio en agendar_cita_tool."
                )
        else:
            cabecera += _nota_servicio_ambiguo(norm_query, servicios)
        hoy = _hoy_bogota()

        dia = None
        if fecha and str(fecha).strip().lower() not in ("none", "null", ""):
            try:
                dia = datetime.strptime(str(fecha).strip()[:10], "%Y-%m-%d")
            except ValueError:
                dia = None
        if dia and dia.date() < hoy.date():
            dia = None

        if dia is None:
            dias = await _dias_con_turnos(profesionales, servicio_id, duracion_servicio, hoy)
            if not dias:
                return f"{cabecera}\nNo hay turnos libres en los próximos 14 días. Sugiere llamar al +57 324 6030217."
            if es_consulta_urgencia(norm_query):
                sin_hoy = (
                    "" if dias[0][0].date() == hoy.date()
                    else " HOY YA NO QUEDAN TURNOS: dilo, da la línea +57 324 6030217 y ofrece estos."
                )
                return (
                    f"{cabecera}\n[MÁS TEMPRANOS] {' | '.join(_mas_tempranos(dias))}\n"
                    "[CÓMO RESPONDER] Es una cita prioritaria: ofrece ESTOS turnos en este orden (el primero es "
                    f"el más pronto), nombrando el día tal como aparece, y pregunta cuál toma.{sin_hoy}"
                )
            return (
                f"{cabecera}\nHoy es {fecha_legible(hoy)}. Próximos días con turnos libres:\n"
                f"{_texto_opciones(dias)}\n{_INSTRUCCION_OPCIONES}"
            )

        cerrado = motivo_dia_cerrado(dia)
        if cerrado:
            dias = await _dias_con_turnos(profesionales, servicio_id, duracion_servicio, dia)
            return (
                f"{cabecera}\n[DÍA CERRADO] {cerrado} Dilo así, con naturalidad (no digas que no hay turnos "
                "con los odontólogos), y ofrece los días más cercanos:\n"
                f"{_texto_opciones(dias)}\n{_INSTRUCCION_OPCIONES}"
            )

        turnos = await _turnos_por_profesional(profesionales, dia.strftime("%Y-%m-%d"), servicio_id, duracion_servicio)
        libres = [(nombre, slots) for _pid, nombre, slots in turnos if slots]
        if libres:
            hhmm = _normalizar_hora(hora)
            return (
                f"{cabecera}\nTurnos libres el {fecha_legible(dia)}:\n"
                f"{_texto_opciones([(dia, libres)])}\n"
                + (f"{_linea_hora_pedida(hhmm, libres)}\n" if hhmm else "")
                + _INSTRUCCION_OPCIONES
            )

        dias = await _dias_con_turnos(profesionales, servicio_id, duracion_servicio, dia)
        return (
            f"{cabecera}\n[SIN CUPOS] El {fecha_legible(dia)} ya no quedan turnos para este servicio. "
            "Días más cercanos con turnos:\n"
            f"{_texto_opciones(dias)}\n{_INSTRUCCION_OPCIONES}"
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
                partes.append(formatear_precio_cop(precio))
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
            "Si el paciente preguntó por un servicio puntual, dale directamente su valor y detalles amables. "
            "Escribe los precios tal cual aparecen aquí (ej. $960.000, sin 'COP'). "
            "Al dar precios NO pidas cédula ni nombre: pregunta si quiere agendar o qué día le sirve.]"
        )
    except Exception as exc:
        logger.error(f"Error al consultar servicios: {exc}", exc_info=True)
        return "En este momento no podemos acceder al catálogo de servicios. Por favor intenta de nuevo más tarde."


# ─── Herramientas LangChain (async nativas — ToolNode.ainvoke sin _run_sync/t.join) ─

@tool
async def consultar_disponibilidad_tool(
    especialidad: str, fecha: Optional[str] = None, hora: Optional[str] = None
) -> str:
    """
    Consulta los horarios disponibles para un servicio odontológico (o especialidad).
    - fecha (YYYY-MM-DD): el día que pidió el paciente. Si el día está cerrado (domingo/festivo)
      o lleno, devuelve el motivo y los días más cercanos con turnos.
    - Sin fecha: cuando pregunta qué días hay o no ha dicho día; devuelve los próximos días con
      turnos empezando desde hoy.
    - hora (HH:MM, opcional): la hora que pidió el paciente; el resultado dice si está LIBRE y con quién.
    Cada fecha del resultado trae su día de la semana correcto: úsalo tal cual.
    IMPORTANTE: Usa SOLO nombres de servicios activos del catálogo (consultar_servicios_y_precios_tool).
    No inventes ni ofrezcas ejemplos de tratamientos que no hayan salido de esa herramienta.
    Si el paciente nombra una especialidad (p. ej. ortodoncia), esta herramienta listará los servicios activos relacionados.
    """
    return await _consultar_disponibilidad_impl(especialidad, fecha, hora)


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