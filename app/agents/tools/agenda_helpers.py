"""Funciones auxiliares, formateadores, normalizadores y diccionarios de sinónimos
para las herramientas de catálogo y agenda de Nexus Odonto.
"""

import re
import unicodedata
import logging
import asyncio
import threading
from datetime import datetime, timedelta, time
from typing import Optional, List, Dict, Any, Tuple

from app.domain.clinical.schedule_rules import (
    es_horario_almuerzo,
    es_dia_laboral,
    es_hora_laboral_valida,
    redondear_a_bloque_30_minutos,
)
from app.domain.security.habeas_data import validar_cedula as domain_validar_cedula

logger = logging.getLogger(__name__)


def _normalizar_texto(texto: str) -> str:
    """Normaliza texto eliminando acentos y convirtiendo a minúsculas."""
    if not texto:
        return ""
    texto = str(texto).lower().strip()
    return "".join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )


def _obtener_valor(obj: Dict[str, Any], *keys: str) -> Any:
    """Busca un valor en un diccionario de forma insensible a mayúsculas y minúsculas."""
    if not isinstance(obj, dict):
        return None
    for key in keys:
        if key in obj:
            return obj[key]
        for k, v in obj.items():
            if k.lower() == key.lower():
                return v
    return None


def _validar_cedula(cedula: str) -> Optional[str]:
    """Valida que la cédula sea solo dígitos y tenga al menos 7 caracteres.
    Retorna None si es válida, o un mensaje de error amigable si no lo es.
    """
    is_valid, clean_doc, error_msg = domain_validar_cedula(cedula)
    if not is_valid:
        return (
            f"⚠️ La cédula *{cedula}* no parece ser válida (debe tener al menos 7 dígitos).\n"
            "Por favor verifica el número e inténtalo de nuevo. 🆔"
        )
    return None


def _run_sync(coro) -> Any:
    """Ejecuta una corrutina de forma síncrona desde herramientas sync de LangChain.

    Si ya hay un loop en ejecución, corre la corrutina en un hilo con loop propio.
    Los clientes .NET usan locks por-loop (`loop_safe_asyncio_lock`) para no chocar
    con el lock del loop principal.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: list[Any] = []
    error: list[BaseException] = []

    def run_in_thread() -> None:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            result.append(new_loop.run_until_complete(coro))
        except BaseException as exc:  # noqa: BLE001 — re-raise after join
            error.append(exc)
        finally:
            try:
                new_loop.run_until_complete(new_loop.shutdown_asyncgens())
            except Exception:
                pass
            new_loop.close()
            asyncio.set_event_loop(None)

    t = threading.Thread(target=run_in_thread, name="nexus-tool-sync")
    t.start()
    t.join()
    if error:
        raise error[0]
    return result[0]


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

# Aliases de búsqueda para servicios
ALIAS_BUSQUEDA_SERVICIO = {
    "limpieza": ["limpieza", "profilaxis", "prophylaxis", "cleaning", "higiene", "dental cleaning", "dental prophylaxis"],
    "profilaxis": ["profilaxis", "prophylaxis", "limpieza", "cleaning", "dental prophylaxis", "dental cleaning"],
    "valoracion": ["valoracion", "assessment", "evaluacion", "consulta general", "general assessment", "dental assessment"],
    "revision": ["revision", "assessment", "valoracion", "chequeo", "general assessment"],
    "resina": ["resina", "composite", "obturation", "relleno", "resin", "composite resin", "composite filling"],
    "ortodoncia": ["ortodoncia", "orthodontics", "brackets", "alineadores", "frenillos", "orthodontic"],
    "endodoncia": ["endodoncia", "endodontics", "conducto", "root canal"],
    "periodoncia": ["periodoncia", "periodontics", "encias", "gingival", "gum"],
    "extraccion": ["extraccion", "extraction", "cirugia", "surgery", "cordales", "tooth extraction"],
    "cirugia": ["cirugia", "surgery", "extraccion", "extraction", "oral surgery"],
    "blanqueamiento": ["blanqueamiento", "whitening", "bleaching", "teeth whitening"],
    "implante": ["implante", "implant", "dental implant"],
    "corona": ["corona", "crown", "dental crown", "porcelain crown"],
    "carilla": ["carilla", "veneer", "dental veneer", "porcelain veneer"],
    "injerto": ["injerto", "graft", "gum graft", "injerto de encia", "injerto gingival", "tejido blando"],
    "graft": ["graft", "injerto", "gum graft", "injerto de encia"],
    "puente": ["puente", "bridge", "dental bridge", "fixed bridge"],
    "protesis": ["protesis", "prosthesis", "denture", "dentadura", "dental prosthesis"],
    "radiografia": ["radiografia", "xray", "x-ray", "radiograph", "radiologia"],
    "fluoruro": ["fluoruro", "fluoride", "fluoride treatment", "aplicacion de fluor"],
    "sellante": ["sellante", "sealant", "dental sealant", "fissure sealant"],
    "brackets": ["brackets", "ortodoncia", "orthodontics", "braces", "alineadores"],
    "alineadores": ["alineadores", "aligners", "invisalign", "clear aligners", "brackets"],
    "ninos": ["ninos", "pediatric", "pediatria", "odontopediatria", "odontologia infantil"],
    "infantil": ["infantil", "pediatric", "ninos", "pediatric dentistry"],
}

# Etiquetas amigables en español para nombres seed en inglés
ETIQUETAS_SERVICIO_ES = {
    "general assessment": "Valoración general",
    "dental assessment": "Valoración dental",
    "dental prophylaxis": "Profilaxis dental",
    "dental cleaning": "Limpieza dental",
    "deep cleaning": "Limpieza profunda",
    "prophylaxis": "Profilaxis",
    "composite resin": "Resina / Calza estética",
    "composite resin filling": "Resina / Calza dental estética",
    "composite filling": "Resina / Calza estética",
    "resin filling": "Resina dental",
    "teeth whitening": "Blanqueamiento dental",
    "tooth whitening": "Blanqueamiento dental",
    "whitening": "Blanqueamiento dental",
    "tooth extraction": "Extracción dental",
    "oral surgery": "Cirugía oral",
    "orthodontics": "Ortodoncia",
    "orthodontic consultation": "Consulta de ortodoncia",
    "endodontics": "Endodoncia",
    "root canal": "Endodoncia / tratamiento de conducto",
    "periodontics": "Periodoncia",
    "gum graft": "Injerto de encía",
    "pediatric dentistry": "Odontopediatría",
    "general dentistry": "Odontología general",
}

DESCRIPCIONES_SERVICIO_ES = {
    "professional dental cleaning and plaque removal.": "Limpieza profunda para remover placa bacteriana y sarro.",
    "restoration of a tooth using composite resin material.": "Restauración estética con resina de alta estética del color del diente.",
    "professional treatment to improve tooth shade.": "Tratamiento profesional para aclarar y embellecer el tono de los dientes.",
    "surgical procedure to cover exposed tooth roots.": "Procedimiento para cubrir encías retraídas y proteger la raíz dental.",
}


def _aplicar_sinonimos_especialidad(norm_texto: str) -> str:
    """Aplica sinónimos de especialidad sobre texto ya normalizado."""
    if not norm_texto:
        return norm_texto
    for k, v in SINONIMOS_ESPANOL.items():
        if k in norm_texto or norm_texto in k:
            return _normalizar_texto(v)
    return norm_texto


def _variantes_desde_etiquetas_es(norm_query: str) -> List[str]:
    """Mapeo INVERSO: dado un texto en español, devuelve los nombres en inglés del catálogo."""
    variantes = []
    if not norm_query:
        return variantes
    for nombre_en, etiqueta_es in ETIQUETAS_SERVICIO_ES.items():
        norm_etiqueta = _normalizar_texto(etiqueta_es)
        if not norm_etiqueta:
            continue
        if norm_query in norm_etiqueta or norm_etiqueta in norm_query:
            nv = _normalizar_texto(nombre_en)
            if nv and nv not in variantes:
                variantes.append(nv)
    return variantes


def _variantes_busqueda_servicio(norm_query: str) -> List[str]:
    """Expande la consulta con aliases, sinónimos de especialidad y mapeo inverso."""
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
    for v_en in _variantes_desde_etiquetas_es(norm_query):
        if v_en not in variantes:
            variantes.append(v_en)
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
    """Nombre preferido para mostrar al paciente en español."""
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
    """Busca un servicio activo por nombre, código, descripción o categoría."""
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

    tokens_query = [t for t in norm_query.split() if len(t) >= 4]
    if tokens_query:
        mejor_candidato = None
        mejor_score = 0
        for ser in servicios:
            if not _es_servicio_activo(ser):
                continue
            nombre = _normalizar_texto(_obtener_valor(ser, "name", "nombre") or "")
            display = _normalizar_texto(_obtener_valor(ser, "displayName", "DisplayName", "nombreDisplay") or "")
            desc = _normalizar_texto(_obtener_valor(ser, "description", "descripcion") or "")
            etiqueta_es = _normalizar_texto(ETIQUETAS_SERVICIO_ES.get(nombre, ""))
            campo_total = f"{nombre} {display} {desc} {etiqueta_es}"
            score = sum(1 for t in tokens_query if t in campo_total)
            tokens_servicio = [t for t in nombre.split() if len(t) >= 4]
            score += sum(1 for t in tokens_servicio if t in norm_query)
            if score > mejor_score:
                mejor_score = score
                mejor_candidato = ser
        if mejor_candidato and mejor_score >= max(1, len(tokens_query) // 2):
            return mejor_candidato

    return None


def _buscar_especialidad_por_texto(
    norm_query: str, especialidades: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Busca una especialidad por nombre, código o descripción."""
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
    """Servicios activos relacionados a la especialidad."""
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


def _etiqueta_especialidad(nombre: Any) -> str:
    raw = str(nombre or "").strip()
    if not raw:
        return "especialidad odontológica"
    return ETIQUETAS_SERVICIO_ES.get(_normalizar_texto(raw), raw)


def _mensaje_especialidad_sin_servicio_unico(
    consulta: str,
    esp_nombre: str,
    relacionados: List[Dict[str, Any]],
    todos_servicios: List[Dict[str, Any]],
) -> str:
    esp_label = _etiqueta_especialidad(esp_nombre)
    if relacionados:
        lista = "\n".join(f"• *{_etiqueta_servicio(s)}*" for s in relacionados[:12])
        return (
            f"*{consulta}* corresponde a la especialidad *{esp_label}*. "
            "Para agendar necesito el servicio concreto. Estos son los activos relacionados:\n\n"
            f"{lista}\n\n"
            "¿Cuál de estos servicios deseas agendar? 😊"
        )
    lista_todos = _lista_servicios_whatsapp(todos_servicios)
    if lista_todos:
        return (
            f"Tenemos la especialidad *{esp_label}*, pero ahora mismo no hay un servicio "
            "activo específicamente asociado para agendar con ese nombre.\n\n"
            "Servicios activos disponibles:\n"
            f"{lista_todos}\n\n"
            "¿Cuál de estos te gustaría agendar? 😊"
        )
    return (
        f"Tenemos la especialidad *{esp_label}*, pero no hay servicios activos agendables "
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


def _generar_slots_desde_regla(
    start_time_str: str,
    end_time_str: str,
    lunch_start_str: Optional[str] = None,
    lunch_end_str: Optional[str] = None,
    duracion_min: int = 60,
) -> list[str]:
    """Genera slots bloqueando estrictamente la franja institucional de almuerzo (12:00 a 14:00)."""
    try:
        sh, sm = map(int, str(start_time_str).split(":")[:2])
        eh, em = map(int, str(end_time_str).split(":")[:2])

        lsh, lsm = (map(int, str(lunch_start_str).split(":")[:2])) if lunch_start_str else (12, 0)
        leh, lem = (map(int, str(lunch_end_str).split(":")[:2])) if lunch_end_str else (14, 0)

        cur = sh * 60 + sm
        end = eh * 60 + em

        # Asegurar bloqueo institucional de 12:00 a 14:00 (720 min a 840 min)
        lstart = min(720, lsh * 60 + lsm)
        lend = max(840, leh * 60 + lem)

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


def _formatear_hora_ampm(hora_str: str) -> str:
    """Convierte 24h a 12h AM/PM."""
    if not hora_str:
        return ""
    texto = str(hora_str).strip()
    if "AM" in texto.upper() or "PM" in texto.upper():
        return texto
    try:
        partes = texto.split(":")
        h = int(partes[0])
        m = int(partes[1]) if len(partes) > 1 else 0
        ampm = "AM" if h < 12 else "PM"
        h12 = h % 12
        if h12 == 0:
            h12 = 12
        return f"{h12:02d}:{m:02d} {ampm}"
    except Exception:
        return texto


def _parsear_fecha_hora_flexible(texto: str) -> Optional[datetime]:
    """Parsea fechas en ISO, 'YYYY-MM-DD HH:MM', o con AM/PM."""
    if not texto:
        return None
    raw = str(texto).strip()
    m = re.search(r"(\d{4}-\d{2}-\d{2})[T\s](\d{1,2}):(\d{2})(?::(\d{2}))?(?:\s*(AM|PM))?", raw, re.IGNORECASE)
    if m:
        fecha_p = m.group(1)
        h = int(m.group(2))
        minute = int(m.group(3))
        sec = int(m.group(4) or 0)
        ampm = m.group(5)
        if ampm:
            ampm = ampm.upper()
            if ampm == "PM" and h < 12:
                h += 12
            elif ampm == "AM" and h == 12:
                h = 0
        elif "PM" in raw.upper() and h < 12:
            h += 12
        elif "AM" in raw.upper() and h == 12:
            h = 0
        return datetime.strptime(f"{fecha_p}T{h:02d}:{minute:02d}:{sec:02d}", "%Y-%m-%dT%H:%M:%S")

    if len(raw) == 10 and re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
        return datetime.strptime(f"{raw}T08:00:00", "%Y-%m-%dT%H:%M:%S")

    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00").replace(" ", "T"))
    except Exception:
        pass
    return None


def _validar_horario_cita(
    dt: datetime,
    duracion_min: int = 30,
    prof_nombre: str = "el especialista",
) -> Tuple[bool, Optional[str]]:
    """Valida que la fecha y hora cumpla las políticas clínicas y de almuerzo."""
    if dt.weekday() == 6:
        return (
            False,
            "⚠️ El consultorio *Nexus Odonto* permanece cerrado los domingos 🏥.\n\n"
            "Nuestra jornada de atención es de Lunes a Sábado. ¿Te gustaría agendar para el próximo día hábil o consultar horarios disponibles? 😊",
        )

    if dt.weekday() == 5:
        if dt.hour < 8 or dt.hour >= 12 or (dt.hour == 11 and dt.minute > 30 and duracion_min > 30):
            return (
                False,
                "⚠️ Los sábados nuestro consultorio atiende únicamente en jornada continua de *8:00 AM a 12:00 PM* ⏰.\n\n"
                "¿Te gustaría agendar el sábado en la mañana o para el lunes en la tarde? 😊",
            )

    if 12 <= dt.hour < 14:
        hora_sol = _formatear_hora_ampm(dt.strftime("%H:%M"))
        return (
            False,
            f"⚠️ El horario solicitado (*{hora_sol}*) coincide con el receso de almuerzo de nuestros especialistas (12:00 PM a 2:00 PM) 🍽️.\n\n"
            f"En la jornada de la tarde disponemos de turnos con {prof_nombre} a partir de las *2:00 PM* o *2:30 PM*.\n\n"
            "¿Te gustaría que te reserve a las *2:00 PM*? 😊",
        )

    if dt.hour < 8 or dt.hour >= 17 or (dt.hour == 16 and dt.minute > 30 and duracion_min > 30):
        hora_sol = _formatear_hora_ampm(dt.strftime("%H:%M"))
        return (
            False,
            f"⚠️ El horario solicitado (*{hora_sol}*) se encuentra fuera de nuestra jornada de atención ⏰.\n\n"
            "Nuestros horarios de consulta son:\n"
            "• ☀️ *Mañana:* 8:00 AM a 12:00 PM\n"
            "• 🌤️ *Tarde:* 2:00 PM a 5:00 PM\n"
            "• 📅 *Sábados:* 8:00 AM a 12:00 PM\n\n"
            "¿Deseas consultar los turnos disponibles dentro de este horario? 😊",
        )

    try:
        from zoneinfo import ZoneInfo
        now_bogota = datetime.now(ZoneInfo("America/Bogota"))
    except Exception:
        now_bogota = datetime.now()

    if dt.date() == now_bogota.date():
        dt_check = dt.replace(tzinfo=now_bogota.tzinfo) if dt.tzinfo is None and now_bogota.tzinfo else dt
        if dt_check < now_bogota + timedelta(minutes=15):
            hora_sol = _formatear_hora_ampm(dt.strftime("%H:%M"))
            return (
                False,
                f"⚠️ Para poder prepararte adecuadamente y garantizar que alcances a llegar al consultorio, "
                f"las citas para hoy requieren un margen mínimo de 15 minutos de anticipación.\n\n"
                f"Para hoy a las *{hora_sol}* ya no alcanzamos a prepararte, pero con gusto podemos agendarte en los turnos más cercanos de esta tarde o para mañana. ¿Te gustaría consultar los horarios disponibles? 😊",
            )

    return (True, None)


def _filtrar_citas_proximas_activas(
    citas: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Separa las citas de un paciente en (proximas_activas, historial).

    - proximas_activas: Citas futuras o vigentes hoy (con margen de 45 min), que no
      estén canceladas, completadas ni marcadas como no asistió.
    - historial: Citas pasadas, canceladas, completadas o no asistidas.
    """
    if not citas:
        return [], []

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
        estado = str(c.get("statusName", "Programada"))
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
            # Campos directos para compatibilidad con selector y cancelador
            "startsAt": starts_at_raw,
            "endsAt": ends_at_raw,
            "professionalName": prof_nom,
            "serviceName": serv_nom,
            "statusName": estado,
            "patientId": c.get("patientId") or c.get("pacienteId"),
            "professionalId": c.get("professionalId"),
            "serviceId": c.get("serviceId"),
            "appointmentStatusId": c.get("appointmentStatusId"),
            "appointmentOriginId": c.get("appointmentOriginId"),
            "reasonForVisit": c.get("reasonForVisit"),
            "notes": c.get("notes"),
        }

        if is_cancelled or is_completed or is_noshow or (dt_start and dt_start < (now_colombia - timedelta(minutes=45))):
            historial.append(c_info)
        else:
            proximas.append(c_info)

    max_aware = datetime.max.replace(tzinfo=now_colombia.tzinfo) if now_colombia.tzinfo else datetime.max
    min_aware = datetime.min.replace(tzinfo=now_colombia.tzinfo) if now_colombia.tzinfo else datetime.min
    proximas.sort(key=lambda x: x["dt"] or max_aware)
    historial.sort(key=lambda x: x["dt"] or min_aware, reverse=True)

    return proximas, historial


def _resolver_cita_por_selector(
    citas_activas: List[Dict[str, Any]],
    selector: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Resuelve cuál cita corresponde al selector del usuario (ordinal, número, fecha o automática)."""
    if not citas_activas:
        return None, None

    def _key_dt(c):
        s = c.get("startsAt") or c.get("fechaHoraInicio") or ""
        try:
            return datetime.fromisoformat(str(s).replace("Z", "").split(".")[0])
        except Exception:
            return datetime.max

    citas_ordenadas = sorted(citas_activas, key=_key_dt)
    raw_sel = str(selector or "").strip()
    norm = _normalizar_texto(raw_sel)

    if len(citas_ordenadas) == 1:
        if norm in ("2", "segunda", "segundo", "3", "tercera", "tercero"):
            return None, "Solo tienes una cita activa programada (Cita #1). ¿Deseas gestionar esa cita? 😊"
        return citas_ordenadas[0], None

    if not norm or norm in ("none", "null", "n/a", "cita", "mi cita", "la cita", "cancelar", "reprogramar", "modificar"):
        opciones = []
        for i, c in enumerate(citas_ordenadas, 1):
            prof = c.get("professionalName", "Especialista")
            serv = c.get("serviceName", "Consulta Odontológica")
            s_raw = c.get("startsAt") or ""
            f_display = str(s_raw)[:10]
            try:
                dt = datetime.fromisoformat(str(s_raw).replace("Z", "").split(".")[0])
                f_display = dt.strftime("%d/%m/%Y a las %I:%M %p")
            except Exception:
                pass
            opciones.append(f"{i}️⃣ *Cita #{i}:* {serv} con {prof} — 📅 {f_display}")

        msg_ambiguo = (
            "Tienes varias citas activas programadas:\n\n"
            + "\n".join(opciones)
            + "\n\n¿Cuál de estas citas deseas gestionar? Indícame el número (ej: *1* o *2*) o la fecha. 😊"
        )
        return None, msg_ambiguo

    ORDINALES = {
        "1": 1, "primera": 1, "primero": 1, "uno": 1, "cita 1": 1, "la 1": 1, "la primera": 1,
        "2": 2, "segunda": 2, "segundo": 2, "dos": 2, "cita 2": 2, "la 2": 2, "la segunda": 2,
        "3": 3, "tercera": 3, "tercero": 3, "tres": 3, "cita 3": 3, "la 3": 3, "la tercera": 3,
        "4": 4, "cuarta": 4, "cuarto": 4, "cuatro": 4, "cita 4": 4,
        "5": 5, "quinta": 5, "quinto": 5, "cinco": 5, "cita 5": 5,
        "ultima": len(citas_ordenadas), "la ultima": len(citas_ordenadas), "ultimo": len(citas_ordenadas),
    }
    m = re.search(r"#?(\d+)", norm)
    idx_num = None
    if m:
        try:
            idx_num = int(m.group(1))
        except Exception:
            pass

    if idx_num is None:
        for k, v in ORDINALES.items():
            if k in norm:
                idx_num = v
                break

    if idx_num is not None:
        if 1 <= idx_num <= len(citas_ordenadas):
            return citas_ordenadas[idx_num - 1], None
        else:
            return None, f"El número de cita *#{idx_num}* no existe. Tienes {len(citas_ordenadas)} citas activas. Por favor indica un número del 1 al {len(citas_ordenadas)}. 😊"

    for c in citas_ordenadas:
        cid = str(c.get("id") or c.get("citaId") or "").lower()
        if cid and cid == raw_sel.lower():
            return c, None

    for c in citas_ordenadas:
        s_raw = str(c.get("startsAt") or "")
        if raw_sel in s_raw:
            return c, None

    return None, None
