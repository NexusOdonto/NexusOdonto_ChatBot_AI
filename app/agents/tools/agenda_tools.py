import os
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


# Mapeo de términos habituales en español a nombres de catálogo en inglés (especialidades)
SINONIMOS_ESPANOL = {
    "general": "General Dentistry",
    "odontologia": "General Dentistry",
    "odontologia general": "General Dentistry",
    "limpieza": "General Dentistry",
    "valoracion": "General Dentistry",
    "revision": "General Dentistry",
    "consulta": "General Dentistry",
    "ortodoncia": "Orthodontics",
    "brackets": "Orthodontics",
    "frenillos": "Orthodontics",
    "endodoncia": "Endodontics",
    "conducto": "Endodontics",
    "periodoncia": "Periodontics",
    "encias": "Periodontics",
    "odontopediatria": "Pediatric Dentistry",
    "pediatria": "Pediatric Dentistry",
    "ninos": "Pediatric Dentistry",
    "cirugia": "Oral Surgery",
    "cirugia oral": "Oral Surgery",
    "extraccion": "Oral Surgery",
    "cordales": "Oral Surgery",
    "muela del juicio": "Oral Surgery",
}

# Aliases de búsqueda para servicios (español paciente → términos que pueden aparecer en el catálogo)
ALIAS_BUSQUEDA_SERVICIO = {
    "limpieza": ["limpieza", "profilaxis", "prophylaxis", "cleaning", "higiene"],
    "profilaxis": ["profilaxis", "prophylaxis", "limpieza", "cleaning"],
    "valoracion": ["valoracion", "assessment", "evaluacion", "consulta general"],
    "revision": ["revision", "assessment", "valoracion", "chequeo"],
    "resina": ["resina", "composite", "obturation", "relleno", "resin"],
    "ortodoncia": ["ortodoncia", "orthodontics", "brackets", "alineadores", "frenillos"],
    "endodoncia": ["endodoncia", "endodontics", "conducto", "root canal"],
    "periodoncia": ["periodoncia", "periodontics", "encias", "gingival"],
    "extraccion": ["extraccion", "extraction", "cirugia", "surgery", "cordales"],
    "cirugia": ["cirugia", "surgery", "extraccion", "extraction"],
    "blanqueamiento": ["blanqueamiento", "whitening", "bleaching"],
    "implante": ["implante", "implant"],
}

# Etiquetas amigables en español para nombres seed en inglés (sin inventar servicios nuevos)
ETIQUETAS_SERVICIO_ES = {
    "general assessment": "Valoración general",
    "dental assessment": "Valoración dental",
    "dental prophylaxis": "Profilaxis dental",
    "dental cleaning": "Limpieza dental",
    "prophylaxis": "Profilaxis",
    "composite resin": "Resina",
    "composite filling": "Resina / obturación",
    "tooth extraction": "Extracción dental",
    "oral surgery": "Cirugía oral",
    "orthodontics": "Ortodoncia",
    "orthodontic consultation": "Consulta de ortodoncia",
    "endodontics": "Endodoncia",
    "root canal": "Endodoncia / conducto",
    "periodontics": "Periodoncia",
    "gum graft": "Injerto de encía",
    "pediatric dentistry": "Odontopediatría",
    "general dentistry": "Odontología general",
}


def _aplicar_sinonimos_especialidad(norm_texto: str) -> str:
    """Aplica sinónimos de especialidad sobre texto ya normalizado."""
    if not norm_texto:
        return norm_texto
    for k, v in SINONIMOS_ESPANOL.items():
        if k in norm_texto or norm_texto in k:
            return _normalizar_texto(v)
    return norm_texto


def _variantes_busqueda_servicio(norm_query: str) -> List[str]:
    """Expande la consulta con aliases y sinónimos de especialidad para matching de servicios."""
    variantes = [norm_query] if norm_query else []
    if not norm_query:
        return variantes
    for clave, aliases in ALIAS_BUSQUEDA_SERVICIO.items():
        if clave in norm_query or norm_query in clave:
            for alias in aliases:
                na = _normalizar_texto(alias)
                if na and na not in variantes:
                    variantes.append(na)
    sinonimo = _aplicar_sinonimos_especialidad(norm_query)
    if sinonimo and sinonimo not in variantes:
        variantes.append(sinonimo)
    return variantes


def _es_servicio_activo(ser: Dict[str, Any]) -> bool:
    """True si el servicio está activo o si no trae bandera de activo."""
    active = _obtener_valor(ser, "isActive", "IsActive", "active", "activo")
    if active is None:
        return True
    if isinstance(active, bool):
        return active
    if isinstance(active, (int, float)):
        return active != 0
    return str(active).strip().lower() not in ("false", "0", "no", "inactive", "inactivo")


def _texto_coincide(norm_query: str, *candidatos: str) -> bool:
    if not norm_query:
        return False
    for raw in candidatos:
        norm = _normalizar_texto(raw or "")
        if not norm:
            continue
        if norm_query in norm or norm in norm_query:
            return True
    return False


def _etiqueta_servicio(ser: Dict[str, Any]) -> str:
    """Nombre preferido para mostrar al paciente (español amigable si el API trae seed EN)."""
    display = _obtener_valor(ser, "displayName", "DisplayName", "nombreDisplay", "nombreMostrar")
    if display:
        return str(display).strip()
    nombre = str(_obtener_valor(ser, "name", "nombre") or "").strip()
    if not nombre:
        code = _obtener_valor(ser, "code", "codigo")
        return str(code).strip() if code else "Servicio odontológico"
    etiqueta = ETIQUETAS_SERVICIO_ES.get(_normalizar_texto(nombre))
    return etiqueta or nombre


def _servicios_activos(servicios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [s for s in servicios if _es_servicio_activo(s)]


def _buscar_servicio_por_texto(norm_query: str, servicios: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Busca un servicio activo por nombre, código, descripción o categoría (con aliases)."""
    for variante in _variantes_busqueda_servicio(norm_query):
        for ser in servicios:
            if not _es_servicio_activo(ser):
                continue
            nombre = _obtener_valor(ser, "name", "nombre") or ""
            display = _obtener_valor(ser, "displayName", "DisplayName", "nombreDisplay") or ""
            code = _obtener_valor(ser, "code", "codigo") or ""
            desc = _obtener_valor(ser, "description", "descripcion") or ""
            category = _obtener_valor(ser, "category", "categoria", "Category") or ""
            etiqueta_es = ETIQUETAS_SERVICIO_ES.get(_normalizar_texto(nombre), "")
            if _texto_coincide(variante, nombre, display, code, desc, category, etiqueta_es):
                return ser
    return None


def _buscar_especialidad_por_texto(
    norm_query: str, especialidades: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Busca una especialidad por nombre, código o descripción (con sinónimos)."""
    norm_esp = _aplicar_sinonimos_especialidad(norm_query)
    for esp in especialidades:
        nombre = _obtener_valor(esp, "name", "nombre") or ""
        code = _obtener_valor(esp, "code", "codigo") or ""
        desc = _obtener_valor(esp, "description", "descripcion") or ""
        if _texto_coincide(norm_esp, nombre, code, desc) or _texto_coincide(norm_query, nombre, code, desc):
            return esp
    return None


def _servicios_relacionados_a_especialidad(
    especialidad: Dict[str, Any],
    servicios: List[Dict[str, Any]],
    norm_query: str = "",
) -> List[Dict[str, Any]]:
    """Servicios activos cuya categoría/nombre/desc coinciden con la especialidad o la consulta."""
    esp_nombre = _obtener_valor(especialidad, "name", "nombre") or ""
    esp_code = _obtener_valor(especialidad, "code", "codigo") or ""
    candidatos_norm = [
        _normalizar_texto(str(esp_nombre)),
        _normalizar_texto(str(esp_code)),
        _aplicar_sinonimos_especialidad(_normalizar_texto(str(esp_nombre))),
    ]
    if norm_query:
        candidatos_norm.extend(_variantes_busqueda_servicio(norm_query))
    candidatos_norm = [c for c in candidatos_norm if c]

    relacionados: List[Dict[str, Any]] = []
    vistos = set()
    for ser in _servicios_activos(servicios):
        sid = str(_obtener_valor(ser, "id", "servicioId", "serviceId") or id(ser))
        if sid in vistos:
            continue
        nombre = _obtener_valor(ser, "name", "nombre") or ""
        display = _obtener_valor(ser, "displayName", "DisplayName") or ""
        desc = _obtener_valor(ser, "description", "descripcion") or ""
        category = _obtener_valor(ser, "category", "categoria", "Category") or ""
        for cand in candidatos_norm:
            if _texto_coincide(cand, nombre, display, desc, category):
                relacionados.append(ser)
                vistos.add(sid)
                break
    return relacionados


def _lista_servicios_whatsapp(servicios: List[Dict[str, Any]], limite: int = 12) -> str:
    """Lista corta de servicios activos con etiquetas amigables para WhatsApp."""
    activos = _servicios_activos(servicios)
    if not activos:
        return ""
    etiquetas = [_etiqueta_servicio(s) for s in activos[:limite]]
    texto = "\n".join(f"• *{e}*" for e in etiquetas if e)
    if len(activos) > limite:
        texto += f"\n• _…y {len(activos) - limite} más_"
    return texto


def _nombres_servicios_activos(servicios: List[Dict[str, Any]]) -> List[str]:
    return [_etiqueta_servicio(s) for s in _servicios_activos(servicios)]


def _mensaje_especialidad_sin_servicio_unico(
    consulta: str,
    esp_nombre: str,
    relacionados: List[Dict[str, Any]],
    todos_servicios: List[Dict[str, Any]],
) -> str:
    """Cuando el paciente nombra una especialidad: no decir 'no está en catálogo'; ofrecer servicios reales."""
    if relacionados:
        lista = "\n".join(f"• *{_etiqueta_servicio(s)}*" for s in relacionados[:12])
        return (
            f"*{consulta}* corresponde a la especialidad *{esp_nombre}*. "
            "Para agendar necesito el servicio concreto. Estos son los activos relacionados:\n\n"
            f"{lista}\n\n"
            "¿Cuál de estos servicios deseas agendar? 😊"
        )
    lista_todos = _lista_servicios_whatsapp(todos_servicios)
    if lista_todos:
        return (
            f"Tenemos la especialidad *{esp_nombre}*, pero ahora mismo no hay un servicio "
            "activo específicamente asociado para agendar con ese nombre.\n\n"
            "Servicios activos disponibles:\n"
            f"{lista_todos}\n\n"
            "¿Cuál de estos te gustaría agendar? 😊"
        )
    return (
        f"Tenemos la especialidad *{esp_nombre}*, pero no hay servicios activos agendables "
        "en este momento. ¿Deseas que te ayude con otra consulta? 😊"
    )


def _mensaje_catalogo_no_encontrado(consulta: str, servicios: List[Dict[str, Any]]) -> str:
    lista = _lista_servicios_whatsapp(servicios)
    if lista:
        return (
            f"No encontré un servicio agendable con el nombre *{consulta}*. "
            "Estos son los *servicios activos* que sí puedes reservar ahora:\n\n"
            f"{lista}\n\n"
            "Indícame cuál deseas y con gusto reviso disponibilidad. 😊"
        )
    return (
        f"No encontré un servicio agendable con el nombre *{consulta}* "
        "y por ahora no hay servicios activos en el catálogo. "
        "Por favor intenta más tarde o contacta a recepción."
    )

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
        step_min = 30
        dur_min = max(30, min(duracion_min, 60))
        while cur + dur_min <= end:
            if not (cur < lend and (cur + dur_min) > lstart):
                h = cur // 60
                m = cur % 60
                slots.append(f"{h:02d}:{m:02d}")
            cur += step_min
        return slots
    except Exception:
        return [str(start_time_str)[:5]]


async def _consultar_disponibilidad_impl(especialidad: str, fecha: str) -> str:
    try:
        norm_query = _normalizar_texto(especialidad)
        if not norm_query:
            return "Por favor, indica una especialidad o servicio válido."

        # 1. Catálogo de servicios primero (bookable por nombre/código/descripción/categoría)
        servicios = await dotnet_client.obtener_servicios() or []
        servicio_encontrado = _buscar_servicio_por_texto(norm_query, servicios)

        especialidades = await dotnet_client.obtener_especialidades() or []
        especialidad_encontrada = None
        esp_id = None
        esp_nombre = None

        if servicio_encontrado:
            # Resolver especialidad opcional (categoría / nombre del servicio / sinónimos)
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
            # 2. Si no hay servicio único, flujo por especialidades → listar servicios activos reales
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
                # Especialidad conocida: nunca decir "no está en catálogo"; pedir servicio concreto
                return _mensaje_especialidad_sin_servicio_unico(
                    especialidad, str(esp_nombre), relacionados, servicios
                )

        # 3. Profesionales: filtrar por especialidad si hay match; si no, todos los activos
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
            # No debería llegar aquí: especialidad sin servicio único ya retornó lista
            return _mensaje_catalogo_no_encontrado(especialidad, servicios)
        
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

                # Eliminar duplicados y ordenar cronológicamente
                unique_slots = sorted(list(dict.fromkeys(slots)))

                # Filtrar turnos ya ocupados por citas existentes considerando traslapes de intervalos
                try:
                    fecha_target = str(fecha)[:10]
                    citas_existentes = await dotnet_client.consultar_citas(fecha_target) or []
                    if isinstance(citas_existentes, dict):
                        citas_existentes = citas_existentes.get("items", [])

                    # Extraer citas del profesional en la fecha consultada que no estén canceladas
                    intervalos_ocupados = []
                    for c in citas_existentes:
                        c_prof = str(c.get("professionalId") or "").lower()
                        c_canc = c.get("cancelledAt")
                        c_st_id = str(c.get("appointmentStatusId") or "").lower()
                        
                        # Ignorar si es de otro profesional o si está cancelada
                        if c_prof != str(prof_id).lower() or c_canc or c_st_id == "10000000-0000-0000-0000-000000000005":
                            continue

                        c_start_raw = str(c.get("startsAt") or c.get("fechaHoraInicio") or "").replace("Z", "").split(".")[0].replace(" ", "T")
                        c_end_raw = str(c.get("endsAt") or c.get("fechaHoraFin") or "").replace("Z", "").split(".")[0].replace(" ", "T")

                        if c_start_raw and c_start_raw[:10] == fecha_target:
                            try:
                                dt_c_start = datetime.fromisoformat(c_start_raw)
                                if c_end_raw:
                                    dt_c_end = datetime.fromisoformat(c_end_raw)
                                else:
                                    dt_c_end = dt_c_start + timedelta(minutes=duracion_servicio or 45)
                                intervalos_ocupados.append((dt_c_start, dt_c_end))
                            except Exception as parse_err:
                                logger.debug(f"[Agenda] Error parseando cita existente: {parse_err}")

                    # Descartar slots candidatos que colisionen con citas existentes
                    slots_libres = []
                    for s in unique_slots:
                        try:
                            slot_start_dt = datetime.strptime(f"{fecha_target}T{s}:00", "%Y-%m-%dT%H:%M:%S")
                            slot_end_dt = slot_start_dt + timedelta(minutes=duracion_servicio)

                            # Hay colisión si: slot_inicio < cita_fin AND slot_fin > cita_inicio
                            colision = any(slot_start_dt < c_end and slot_end_dt > c_start for c_start, c_end in intervalos_ocupados)
                            if not colision:
                                slots_libres.append(s)
                        except Exception:
                            slots_libres.append(s)

                    unique_slots = slots_libres
                except Exception as c_err:
                    logger.warning(f"[Agenda] Error consultando citas ocupadas: {c_err}")

                # Si la consulta es para hoy, filtrar horarios que ya pasaron en Colombia
                try:
                    from zoneinfo import ZoneInfo
                    now_bogota = datetime.now(ZoneInfo("America/Bogota"))
                except Exception:
                    now_bogota = datetime.now()

                if str(fecha)[:10] == now_bogota.strftime("%Y-%m-%d"):
                    # Mínimo 30 minutos de anticipación para citas del mismo día
                    min_dt = now_bogota + timedelta(minutes=30)
                    min_hhmm = min_dt.strftime("%H:%M")
                    unique_slots = [s for s in unique_slots if s >= min_hhmm]
                if unique_slots:
                    # Agrupar visualmente slots mañana y tarde
                    manana = [s for s in unique_slots if int(s.split(":")[0]) < 12]
                    tarde = [s for s in unique_slots if int(s.split(":")[0]) >= 12]
                    
                    horarios_str_list = []
                    if manana:
                        horarios_str_list.append(f"   🌅 *Mañana:* " + ", ".join(manana))
                    if tarde:
                        horarios_str_list.append(f"   🌇 *Tarde:* " + ", ".join(tarde))
                    
                    bloque_horarios = "\n".join(horarios_str_list) if horarios_str_list else ("   ⏰ " + ", ".join(unique_slots))
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


async def _agendar_cita_impl(
    cedula: str,
    nombre_paciente: str,
    profesional_id: Any,
    servicio_id: Any,
    fecha_hora_inicio: str,
    motivo_consulta: str,
    config: RunnableConfig,
) -> str:
    """Agenda una cita buscando o creando el paciente por su cédula.
    
    Flujo:
    1. Buscar al paciente en el backend por cédula.
    2. Si no existe, crearlo con los datos básicos (cédula + nombre + teléfono WA).
    3. Resolver profesional y servicio.
    4. Crear la cita y retornar el resumen.
    """
    try:
        thread_id = config.get("configurable", {}).get("thread_id", "")
        
        # ── 1. Resolver paciente por cédula ──────────────────────────────────────
        persona = await dotnet_client.buscar_persona_por_documento(cedula.strip())
        paciente_id = None
        nombre_display = nombre_paciente

        if persona:
            person_id = persona.get("id")
            nombre_bd = f"{persona.get('firstName', '')} {persona.get('lastName', '')}".strip()
            if nombre_bd:
                nombre_display = nombre_bd
            if person_id:
                paciente = await dotnet_client.buscar_paciente_por_person_id(str(person_id))
                if paciente:
                    paciente_id = paciente.get("id")
                else:
                    # La persona existe en Persons pero aún no está en Patients: crearlo directamente
                    logger.info(f"[Agenda] Persona {person_id} existe pero no tiene registro en Patients. Creando paciente...")
                    nuevo_pac = await dotnet_client.crear_paciente_para_persona(str(person_id))
                    if nuevo_pac:
                        paciente_id = nuevo_pac.get("id")

        if not paciente_id:
            # El paciente no existe: crearlo con datos básicos usando su número de WA como teléfono
            logger.info(f"[Agenda] Paciente con cédula {cedula} no encontrado. Creando perfil básico...")
            resultado_registro = await dotnet_client.crear_paciente_basico(
                cedula=cedula,
                nombre=nombre_paciente,
                telefono_whatsapp=thread_id,
            )
            if resultado_registro:
                paciente_id = resultado_registro.get("patientId") or resultado_registro.get("id")
                logger.info(f"[Agenda] Paciente creado exitosamente con ID: {paciente_id}")
            else:
                # Si falló (por ejemplo conflicto porque la persona ya existía con otro formato de documento)
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

        # ── 2. Resolver profesional ───────────────────────────────────────────────
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

        # ── 3. Resolver servicio y duración (catálogo primero; sin fallback a servs[0]) ──
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
            # Si nombró una especialidad, listar servicios reales relacionados (no inventar)
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

        # ── 4. Calcular startsAt y endsAt en formato ISO 8601 ───────────────────
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

            # Ajustar ends_dt si cruza el horario de almuerzo (12:00 a 13:00) o fin de jornada (17:00)
            if starts_dt.hour < 12 and ends_dt.hour >= 12 and (ends_dt.hour > 12 or ends_dt.minute > 0):
                ends_dt = starts_dt.replace(hour=12, minute=0, second=0)
            elif starts_dt.hour < 17 and ends_dt.hour >= 17 and (ends_dt.hour > 17 or ends_dt.minute > 0):
                ends_dt = starts_dt.replace(hour=17, minute=0, second=0)

            starts_at_iso = starts_dt.strftime("%Y-%m-%dT%H:%M:%S")
            ends_at_iso = ends_dt.strftime("%Y-%m-%dT%H:%M:%S")

            # Validar que si la cita es para hoy, tenga al menos 20 minutos de margen de anticipación
            try:
                from zoneinfo import ZoneInfo
                now_bogota = datetime.now(ZoneInfo("America/Bogota"))
                if starts_dt.date() == now_bogota.date():
                    dt_check = starts_dt.replace(tzinfo=ZoneInfo("America/Bogota"))
                    if dt_check < now_bogota + timedelta(minutes=20):
                        hora_sol = starts_dt.strftime("%I:%M %p")
                        return (
                            f"⚠️ No es posible agendar una cita para hoy a las *{hora_sol}* con tan poco margen de tiempo (menos de 20-30 minutos). "
                            "Por favor selecciona un turno más adelante para que tengas tiempo suficiente de llegar al consultorio. 😊"
                        )
            except Exception:
                pass
        except Exception as dt_err:
            logger.warning(f"[Agenda Tools] Error formateando fechas ({fecha_hora_inicio}): {dt_err}")
            now = datetime.now()
            starts_at_iso = now.strftime("%Y-%m-%dT%H:%M:%S")
            ends_at_iso = (now + timedelta(minutes=duracion_min)).strftime("%Y-%m-%dT%H:%M:%S")
            starts_dt = now
            ends_dt = now + timedelta(minutes=duracion_min)

        # ── 5. Obtener IDs de estado y origen ────────────────────────────────────
        status_id = await dotnet_client.obtener_appointment_status_id("AGENDADA")
        origin_id = await dotnet_client.obtener_appointment_origin_id("AGENTE_BOT")

        # ── 6. Agendar la cita ───────────────────────────────────────────────────
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
            hora_inicio_str = starts_dt.strftime("%H:%M")
            hora_fin_str = ends_dt.strftime("%H:%M")
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
                f"📍 *Sede:* Nexus Odonto — Cr 24 #35-12, Santander\n"
                f"📞 *Atención:* +57 324 6030217\n\n"
                f"¡Será un placer cuidar de tu sonrisa! 😊✨"
            )
        else:
            return "Lo siento, ocurrió un inconveniente al registrar la cita en el sistema. Es posible que el horario seleccionado ya esté ocupado. ¿Te gustaría intentar con otro horario disponible? 😊"
    except Exception as exc:
        logger.error(f"Error al agendar cita: {exc}", exc_info=True)
        return "Lo siento, ocurrió un problema de conexión al registrar la cita. Por favor intenta de nuevo en unos minutos."


async def _consultar_cita_por_cedula_impl(cedula: str) -> str:
    """Busca y formatea las citas de un paciente discriminando próximas vs historial."""
    try:
        cedula = cedula.strip()
        if not cedula:
            return "Por favor, indícame tu número de cédula para poder consultar tus citas. 🆔"

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
            motivo = c.get("reasonForVisit") or ""
            status_id = str(c.get("appointmentStatusId") or "").lower()

            dt_start = None
            dt_end = None
            fecha_display = ""
            hora_display = ""

            try:
                if starts_at_raw:
                    clean_start = str(starts_at_raw).replace("Z", "").split(".")[0].replace(" ", "T")
                    dt_start = datetime.fromisoformat(clean_start)
                    fecha_display = dt_start.strftime("%d/%m/%Y")
                    hora_start_str = dt_start.strftime("%I:%M %p")

                    if ends_at_raw:
                        clean_end = str(ends_at_raw).replace("Z", "").split(".")[0].replace(" ", "T")
                        dt_end = datetime.fromisoformat(clean_end)
                        hora_end_str = dt_end.strftime("%I:%M %p")
                        hora_display = f"{hora_start_str} - {hora_end_str}"
                    else:
                        dt_end = dt_start + timedelta(minutes=45)
                        hora_display = hora_start_str
            except Exception:
                fecha_display = str(starts_at_raw)[:10]
                hora_display = str(starts_at_raw)[11:16]

            # Clasificación inteligente de estado
            is_cancelled = bool(c.get("cancelledAt")) or status_id == "10000000-0000-0000-0000-000000000005" or "cancel" in estado.lower()
            is_completed = status_id == "10000000-0000-0000-0000-000000000004" or "complet" in estado.lower()
            is_noshow = status_id == "10000000-0000-0000-0000-000000000006" or "no_asist" in estado.lower() or "no asist" in estado.lower()

            if is_noshow:
                estado = "No Asistió"

            # Si la hora programada ya pasó por más de 45 minutos y no se completó, considerar vencida / no asistió
            if dt_start and not is_cancelled and not is_completed and not is_noshow:
                dt_cmp = dt_start.replace(tzinfo=now_colombia.tzinfo) if dt_start.tzinfo is None and now_colombia.tzinfo else dt_start
                if dt_cmp < (now_colombia - timedelta(minutes=45)):
                    estado = "No Asistió (Vencida)"
                    is_noshow = True

            c_info = {
                "id": cita_id,
                "profesional": prof_nom,
                "servicio": serv_nom,
                "estado": estado,
                "fecha": fecha_display,
                "hora": hora_display,
                "motivo": motivo,
                "dt_start": dt_start,
            }

            if is_cancelled or is_completed or is_noshow:
                historial.append(c_info)
            elif dt_start:
                dt_cmp = dt_start.replace(tzinfo=now_colombia.tzinfo) if dt_start.tzinfo is None and now_colombia.tzinfo else dt_start
                if dt_cmp >= (now_colombia - timedelta(minutes=15)):
                    proximas.append(c_info)
                else:
                    historial.append(c_info)
            else:
                proximas.append(c_info)

        # Ordenar próximas cronológicamente
        proximas.sort(key=lambda x: x["dt_start"] or datetime.max)
        # Ordenar historial de más reciente a más antigua
        historial.sort(key=lambda x: x["dt_start"] or datetime.min, reverse=True)

        bloques = []

        if proximas:
            bloques.append(f"📅 *Tus Próximas Citas Programadas:*")
            for i, c in enumerate(proximas, 1):
                motivo_line = f"\n   • 📝 *Motivo:* {c['motivo']}" if c["motivo"] else ""
                bloques.append(
                    f"{i}️⃣ *Cita #{i}*\n"
                    f"   • 🦷 *Tratamiento:* {c['servicio']}\n"
                    f"   • 👨‍⚕️ *Especialista:* {c['profesional']}\n"
                    f"   • 📅 *Fecha:* {c['fecha']}\n"
                    f"   • ⏰ *Horario:* {c['hora']}\n"
                    f"   • 📌 *Estado:* {c['estado']}{motivo_line}\n"
                    f"   • 🔑 *ID de Cita:* `{c['id']}`"
                )
        else:
            bloques.append("✨ *No tienes citas pendientes o próximas por asistir.*")

        if historial:
            bloques.append("📜 *Historial de Citas Anteriores:*")
            for c in historial[:3]:  # Máximo 3 registros
                bloques.append(f"• 🦷 *{c['servicio']}* con {c['profesional']}\n  📅 {c['fecha']} a las {c['hora']} — Estado: _{c['estado']}_")

        cuerpo = "\n\n".join(bloques)

        return (
            f"📋 *Tus Citas en Nexus Odonto* 🦷✨\n\n"
            f"🆔 *Cédula:* {cedula}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{cuerpo}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 *Sede:* Nexus Odonto — Cr 24 #35-12, Santander\n"
            f"📞 *Atención / Cambios:* +57 324 6030217\n\n"
            f"💡 _Si deseas reprogramar o cancelar alguna de tus citas próximas, dime el ID o la fecha y con gusto te ayudo._ 😊"
        )
    except Exception as exc:
        logger.error(f"Error consultando citas por cédula {cedula}: {exc}", exc_info=True)
        return "Lo siento, ocurrió un problema al consultar tus citas. Por favor intenta de nuevo en unos minutos."


async def _cancelar_cita_impl(cedula: str, cita_id: Optional[str] = None) -> str:
    """Cancela una cita verificando primero que la cédula corresponda al paciente dueño de la cita."""
    try:
        cedula = (cedula or "").strip()
        cita_id = (cita_id or "").strip()

        if not cedula:
            return "Para cancelar tu cita, necesito tu *número de cédula* 🆔. ¿Me la puedes indicar? 😊"

        # Obtener citas del paciente
        citas = await dotnet_client.buscar_citas_por_cedula(cedula)
        if not citas:
            return (
                f"📋 *Consulta de Citas* 🦷✨\n\n"
                f"No encontré citas activas registradas para la cédula *{cedula}*.\n\n"
                "¿Deseas agendar una nueva cita? Con gusto te ayudo. 😊"
            )

        # Filtrar citas activas (no canceladas)
        citas_activas = [c for c in citas if str(c.get("statusName", "")).lower() != "cancelada"]
        if not citas_activas:
            return f"Todas las citas registradas para la cédula *{cedula}* ya se encuentran canceladas o atendidas. 😊"

        cita_encontrada = None
        if cita_id and cita_id.lower() not in ("none", "null", "n/a", ""):
            for c in citas_activas:
                c_id = str(c.get("id") or c.get("citaId") or c.get("appointmentId") or "").lower()
                if c_id == cita_id.lower():
                    cita_encontrada = c
                    break

        # Si no se especificó ID o no coincidió pero solo hay 1 cita activa, seleccionarla automáticamente
        if not cita_encontrada:
            if len(citas_activas) == 1:
                cita_encontrada = citas_activas[0]
            else:
                # Mostrar lista para que el paciente elija
                opciones = []
                for i, c in enumerate(citas_activas, 1):
                    cid = c.get("id") or c.get("citaId")
                    s_raw = c.get("startsAt") or ""
                    opciones.append(f"{i}️⃣ Cita del `{str(s_raw)[:10]}` — ID: `{cid}`")
                return (
                    f"Tienes varias citas activas registradas con la cédula *{cedula}*:\n\n"
                    + "\n".join(opciones)
                    + "\n\n¿Cuál de estas citas deseas cancelar? Indícame el ID o la fecha. 😊"
                )

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

        # Realizar la cancelación enviando la info completa
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
                f"• 🔑 *ID de Cita:* `{target_id}`\n"
                f"━━━━━━━━━━━━━━━━━━━━\n\n"
                f"Tu cita ha sido cancelada. Si deseas reagendar en otro horario, con gusto te ayudo. 😊\n"
                f"📞 *Atención:* +57 324 6030217"
            )
        else:
            err = resultado.get("error", "")
            return (
                f"⚠️ No fue posible cancelar la cita `{target_id}` en este momento.\n\n"
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
    """Modifica la fecha/horario (y opcionalmente el profesional) de una cita existente."""
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

        # Buscar citas del paciente
        citas = await dotnet_client.buscar_citas_por_cedula(cedula)
        if not citas:
            return (
                f"❌ No encontré ninguna cita activa registrada para la cédula *{cedula}*.\n\n"
                "¿Deseas agendar una nueva cita? Con gusto te ayudo. 😊"
            )

        citas_activas = [c for c in citas if str(c.get("statusName", "")).lower() != "cancelada"]
        if not citas_activas:
            return f"No tienes citas activas para reprogramar con la cédula *{cedula}*. ¿Deseas agendar una nueva cita? 😊"

        cita_encontrada = None
        if cita_id and cita_id.lower() not in ("none", "null", "n/a", ""):
            for c in citas_activas:
                c_id = str(c.get("id") or c.get("citaId") or c.get("appointmentId") or "").lower()
                if c_id == cita_id.lower():
                    cita_encontrada = c
                    break

        if not cita_encontrada:
            if len(citas_activas) == 1:
                cita_encontrada = citas_activas[0]
            else:
                opciones = []
                for i, c in enumerate(citas_activas, 1):
                    cid = c.get("id") or c.get("citaId")
                    s_raw = c.get("startsAt") or ""
                    opciones.append(f"{i}️⃣ Cita del `{str(s_raw)[:10]}` — ID: `{cid}`")
                return (
                    f"Tienes varias citas activas con la cédula *{cedula}*:\n\n"
                    + "\n".join(opciones)
                    + "\n\n¿Cuál de estas citas deseas reprogramar? Indícame el ID o la fecha. 😊"
                )

        target_id = str(cita_encontrada.get("id") or cita_encontrada.get("citaId") or cita_id)

        # Calcular nuevas fechas
        try:
            raw = str(nueva_fecha_hora).strip().replace(" ", "T")
            if "T" not in raw and len(raw) == 10:
                raw += "T08:00:00"
            starts_dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            return (
                f"❌ No pude interpretar la fecha '{nueva_fecha_hora}'.\n"
                "Por favor usa el formato: *YYYY-MM-DD HH:MM* (ej: 2026-09-04 14:00)"
            )

        # Determinar duración del servicio actual
        serv_id = cita_encontrada.get("serviceId") or cita_encontrada.get("servicioId")
        duracion_min = 45
        if serv_id:
            servs = await dotnet_client.obtener_servicios() or []
            for s in servs:
                if str(s.get("id")).lower() == str(serv_id).lower():
                    duracion_min = int(s.get("durationMinutes") or 45)
                    break

        ends_dt = starts_dt + timedelta(minutes=duracion_min)
        starts_at_iso = starts_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        ends_at_iso = ends_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Determinar profesional
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

        # Construir payload completo de actualización
        datos_actualizacion: Dict[str, Any] = {
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
                f"• 🔑 *ID de Cita:* `{target_id}`\n"
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
                    "💡 Puede deberse al receso de almuerzo (12:00 PM a 1:00 PM) o a que está fuera de su turno de atención.\n"
                    "Por favor consulta los horarios disponibles o elige otra hora (ej. 8:00 AM - 11:30 AM o 1:00 PM - 4:30 PM). 😊"
                )
            elif "overlap" in err_msg or "already has" in err_msg:
                return (
                    f"⚠️ {prof_nombre_display} ya tiene otra cita programada a las *{hora_req}*.\n\n"
                    "Por favor elige otro horario disponible para apartar tu turno. 😊"
                )
            else:
                return (
                    f"⚠️ No fue posible reprogramar la cita `{target_id}` en este momento.\n\n"
                    f"Detalle: {resultado.get('error')}\n"
                    "Por favor intenta de nuevo o comunícate con recepción: *+57 324 6030217* 😊"
                )
    except Exception as exc:
        logger.error(f"Error modificando cita para cédula {cedula}: {exc}", exc_info=True)
        return "Lo siento, ocurrió un problema al modificar la cita. Por favor intenta de nuevo o llama a recepción."


async def _confirmar_cita_impl(cedula: str, cita_id: Optional[str] = None) -> str:
    """Confirma la asistencia del paciente a una cita activa o programada."""
    try:
        cedula = (cedula or "").strip()
        cita_id = (cita_id or "").strip()

        if not cedula:
            return "Para confirmar tu cita, por favor indícame tu *número de cédula* 🆔. 😊"

        # Obtener citas del paciente
        citas = await dotnet_client.buscar_citas_por_cedula(cedula)
        if not citas:
            return (
                f"📋 *Consulta de Citas* 🦷✨\n\n"
                f"No encontré citas registradas para la cédula *{cedula}*.\n\n"
                "Si deseas agendar una nueva cita, ¡con gusto te ayudo! 😊"
            )

        # Filtrar citas que no estén canceladas ni atendidas
        citas_activas = [
            c for c in citas
            if str(c.get("statusName", "")).lower() not in ("cancelada", "completed", "atendida")
        ]
        if not citas_activas:
            return f"No tienes citas pendientes por confirmar para la cédula *{cedula}*. Todas se encuentran completadas o canceladas. 😊"

        cita_a_confirmar = None
        if cita_id and cita_id.lower() not in ("none", "null", "n/a", ""):
            for c in citas_activas:
                c_id = str(c.get("id") or c.get("citaId") or c.get("appointmentId") or "").lower()
                if c_id == cita_id.lower():
                    cita_a_confirmar = c
                    break

        if not cita_a_confirmar:
            if len(citas_activas) == 1:
                cita_a_confirmar = citas_activas[0]
            else:
                cita_a_confirmar = citas_activas[0]

        target_id = str(cita_a_confirmar.get("id") or cita_a_confirmar.get("citaId"))
        res = await dotnet_client.confirmar_estado_cita(target_id)
        if not res.get("success"):
            return (
                "Hubo un pequeño problema al confirmar tu cita en el sistema. "
                "Por favor comunícate directamente con recepción al +57 324 6030217 para asegurar tu asistencia. 😊"
            )

        # Formatear datos para respuesta agradable
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
            f"• 📍 *Sede:* Cr 24 #35-12, Santander\n\n"
            f"💡 *Recomendación:* Por favor llega 10 a 15 minutos antes de tu turno para prepararte con calma.\n\n"
            f"¡El equipo de Nexus Odonto te espera con gusto! ¿Hay algo más en lo que te pueda colaborar hoy? 😊🦷"
        )
    except Exception as exc:
        logger.error(f"Error confirmando cita para cédula {cedula}: {exc}", exc_info=True)
        return "Lo siento, ocurrió un problema al confirmar tu cita. Por favor intenta de nuevo en unos minutos o contacta a recepción."


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
            tarjetas_prof = []
            for p in profesionales:
                nombre = _obtener_valor(p, "name", "nombre", "nombreCompleto") or "Especialista Odontológico"
                licencia = _obtener_valor(p, "professionalLicense", "licencia", "tarjetaProfesional")
                licencia_str = f"\n   • 🎓 *Reg. Profesional:* `{licencia}`" if licencia else ""
                tarjetas_prof.append(f"👨‍⚕️ *{nombre}*{licencia_str}\n   • 🦷 Especialista en Odontología Integral")
            
            titulo = f"✨ *Especialistas en {esp_nombre} en Nexus Odonto* 🦷" if esp_nombre else "✨ *Equipo Médico de Nexus Odonto* 🦷"
            return (
                f"{titulo}\n\n"
                f"Contamos con profesionales de primer nivel para cuidar de tu sonrisa:\n\n"
                + "\n\n".join(tarjetas_prof)
                + "\n\n💡 _¿Te gustaría consultar los horarios disponibles de alguno de nuestros doctores para agendar tu cita?_ 😊"
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
    try:
        servicios = await dotnet_client.obtener_servicios() or []
        activos = _servicios_activos(servicios)
        if not activos:
            return (
                "En este momento no hay servicios activos en el catálogo, "
                "o no podemos acceder a la lista. Por favor intenta de nuevo más tarde."
            )

        tarjetas = []
        for s in activos:
            nombre = _etiqueta_servicio(s)
            desc = _obtener_valor(s, "description", "descripcion")
            # No inventar descripciones: solo mostrar si el API trae texto real
            desc_str = str(desc).strip() if desc and str(desc).strip() else ""
            precio = _obtener_valor(s, "price", "precio")
            duracion = _obtener_valor(s, "durationMinutes", "duracionMinutos")

            detalles = []
            if duracion:
                detalles.append(f"⏱️ *Duración:* {duracion} min")
            if precio is not None and str(precio).strip() != "":
                try:
                    precio_num = float(precio)
                    precio_str = f"${precio_num:,.0f} COP"
                except (TypeError, ValueError):
                    precio_str = f"${precio} COP"
                detalles.append(f"💰 *Inversión:* {precio_str}")

            meta_str = " | ".join(detalles) if detalles else ""
            meta_line = f"\n   • {meta_str}" if meta_str else ""
            desc_line = f"\n   • 📝 {desc_str}" if desc_str else ""
            tarjetas.append(f"✨ *{nombre}*{meta_line}{desc_line}")

        return (
            "🦷 *Tratamientos y Servicios en Nexus Odonto* ✨\n\n"
            "Estos son los *servicios activos* disponibles para agendar:\n\n"
            + "\n\n".join(tarjetas)
            + "\n\n💬 *¿Cuál de estos tratamientos te gustaría realizarte? "
            "Con gusto verifico la disponibilidad para ti.* 😊"
        )
    except Exception as exc:
        logger.error(f"Error al consultar servicios: {exc}", exc_info=True)
        return "En este momento no podemos acceder al catálogo de servicios. Por favor intenta de nuevo más tarde."


# ─────────────────────────────────────────────────────────────────────────────
# Definición de herramientas LangChain (expuestas al LLM)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def consultar_disponibilidad_tool(especialidad: str, fecha: str) -> str:
    """
    Consulta los horarios disponibles para un servicio odontológico (o especialidad) en una fecha (YYYY-MM-DD).
    IMPORTANTE: Usa SOLO nombres de servicios activos del catálogo (consultar_servicios_y_precios_tool).
    No inventes ni ofrezcas ejemplos de tratamientos que no hayan salido de esa herramienta.
    Si el paciente nombra una especialidad (p. ej. ortodoncia), esta herramienta listará los servicios activos relacionados.
    """
    return _run_sync(_consultar_disponibilidad_impl(especialidad, fecha))


@tool
def agendar_cita_tool(
    cedula: str,
    nombre_paciente: str,
    profesional_id: str,
    servicio_id: str,
    fecha_hora_inicio: str,
    motivo_consulta: str,
    config: RunnableConfig,
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
    - cita_id: (Opcional) ID de la cita a cancelar. Si el paciente tiene solo una cita activa, el sistema la identificará automáticamente.
    
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
    - cita_id: (Opcional) ID de la cita a modificar. Si el paciente solo tiene una cita activa, el sistema la detectará automáticamente.
    - nuevo_profesional_id: (Opcional) ID o nombre del nuevo profesional si desea cambiarlo.
    
    IMPORTANTE: Antes de proponer o confirmar un nuevo horario, consulta SIEMPRE la disponibilidad con consultar_disponibilidad_tool para asegurar que el especialista no esté en horario de almuerzo (ej. 12:00 PM a 1:00 PM) ni fuera de turno.
    Usa esta herramienta SOLAMENTE después de presentar la propuesta de cambio y obtener confirmación explícita del usuario.
    """
    return _run_sync(_modificar_cita_impl(cedula, nueva_fecha_hora, cita_id, nuevo_profesional_id))


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
    Consulta la lista OFICIAL y ACTUAL de servicios activos (nombre, duración, precios) desde GET /Services.
    Usa esta herramienta ANTES de mencionar, sugerir o ejemplificar cualquier tratamiento.
    NUNCA inventes nombres de servicios; solo ofrece los que devuelve esta herramienta.
    """
    return _run_sync(_consultar_servicios_impl())


@tool
def confirmar_cita_tool(cedula: str, cita_id: Optional[str] = None) -> str:
    """
    Confirma formalmente la asistencia del paciente a una cita activa o recordatorio en el sistema Nexus Odonto.
    Actualiza el estado de la cita en la base de datos a CONFIRMADA (verde).
    
    Parámetros:
    - cedula: Número de cédula o documento de identidad del paciente (OBLIGATORIO).
    - cita_id: (Opcional) ID de la cita a confirmar si se conoce. Si se omite, el sistema confirmará automáticamente su cita más próxima activa.
    
    Usa esta herramienta cuando el paciente responda a un recordatorio diciendo 'Confirmo', 'Sí confirmo', 'Confirmo mi cita',
    'Confirmo mi asistencia', 'Allá estaré', o cuando solicite explícitamente confirmar su cita.
    """
    return _run_sync(_confirmar_cita_impl(cedula, cita_id))
