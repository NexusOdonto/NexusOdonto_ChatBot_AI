"""Respuestas informativas sin LLM (servicios, precios, especialistas, horario, ubicación...).

Se usan como fast path cuando la intención es clara y como respaldo cuando Gemini no
responde. Los datos salen del catálogo real (.NET) con una copia "última buena" para
cuando la API también falle. Nunca mencionan fallas ni piden reintentar.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import unicodedata
from typing import Any, Optional

from app.services import reply_variants as rv

logger = logging.getLogger(__name__)

_CATALOG_TIMEOUT_S = 4.0
_LAST_GOOD_TTL_S = 12 * 3600.0
_LAST_GOOD: dict[str, tuple[list, float]] = {}
_MAX_LIST = 6


def _norm(text: str) -> str:
    t = (text or "").lower()
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


_INTENT_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    (
        "como_agendar",
        re.compile(
            r"\bcomo (puedo |hago para |se puede )?(agendo|agendar|saco|sacar|pido|pedir|separo|separar|reservo|reservar)\b"
        ),
    ),
    (
        "que_es",
        re.compile(
            r"para que sirve|a que se dedica|de que se trata|que es (esto|este|esta|nexus|la clinica|el negocio|ustedes)"
            r"|quienes son|que son ustedes|que clase de|que tipo de (negocio|clinica|lugar)|que hacen ustedes"
        ),
    ),
    (
        "precio",
        re.compile(r"\b(precios?|costos?|tarifas?|cuanto (vale|valen|cuesta|cuestan|cobran|sale|es)|cuesta|cuestan|valor)\b"),
    ),
    (
        "servicios",
        re.compile(r"\b(servicios?|tratamientos?|procedimientos?)\b|que (ofrecen|manejan|hacen)\b"),
    ),
    (
        "especialistas",
        re.compile(
            r"\b(odontolog[oa]s|doctor(es|as)|especialistas?|profesionales|dentistas?)\b|quien(es)? (atiende|atienden)"
        ),
    ),
    (
        "horario",
        re.compile(r"\b(horarios?|a que hora|abren|cierran|que dias atienden|atienden (los )?(sabados|domingos|festivos))\b"),
    ),
    (
        "ubicacion",
        re.compile(r"\b(donde (estan|quedan|queda|es la|son)|direccion|ubicacion|ubicados|como (llego|llegar))\b"),
    ),
    ("contacto", re.compile(r"\b(telefono|numero de contacto|correo|email|llamarlos|llamar)\b")),
)

_BOOKING_OR_MUTATION = re.compile(
    r"\b(agendar|agendo|agende|cita|citas|cancelar|cancela|reprogramar|modificar|cambiar|confirmar|turno|cedula|reservar)\b"
)
_PAIN = re.compile(r"\b(dolor|duele|duelen|molestia|sangra|sangrado|hinchad[oa]|inflamad[oa]|urgencia|emergencia)\b")


def detect_info_intents(text: str) -> list[str]:
    norm = _norm(text)
    if not norm:
        return []
    return [name for name, pattern in _INTENT_PATTERNS if pattern.search(norm)]


def is_fast_path_candidate(text: str) -> bool:
    """Intención informativa clara y sin agenda/dolor/datos: se puede responder sin LLM."""
    raw = (text or "").strip()
    if not raw or len(raw) > 160 or re.search(r"\d{6,}", raw):
        return False
    norm = _norm(raw)
    intents = detect_info_intents(raw)
    if not intents:
        return False
    if "como_agendar" in intents:
        return True
    return not _BOOKING_OR_MUTATION.search(norm) and not _PAIN.search(norm)


def _precio(valor: Any) -> Optional[str]:
    try:
        return "$" + f"{int(round(float(valor))):,}".replace(",", ".")
    except (TypeError, ValueError):
        return None


async def _cached_catalog(key: str, loader) -> list:
    try:
        data = await asyncio.wait_for(loader(), timeout=_CATALOG_TIMEOUT_S)
        if data:
            _LAST_GOOD[key] = (list(data), time.monotonic())
            return list(data)
    except Exception as exc:
        logger.warning("[InfoReplies] catálogo %s no disponible: %s", key, exc)
    entry = _LAST_GOOD.get(key)
    if entry and time.monotonic() - entry[1] < _LAST_GOOD_TTL_S:
        return entry[0]
    return []


async def _servicios() -> list[dict]:
    from app.agents.tools.agenda_helpers import _servicios_activos
    from app.clients.dotnet_client import dotnet_client

    data = await _cached_catalog("services", dotnet_client.obtener_servicios)
    activos = _servicios_activos([s for s in data if isinstance(s, dict)])
    return sorted(activos, key=lambda s: s.get("sortOrder") or 0)


async def _profesionales() -> list[dict]:
    from app.clients.dotnet_client import dotnet_client

    data = await _cached_catalog("professionals", dotnet_client.obtener_profesionales)
    return [p for p in data if isinstance(p, dict) and p.get("isActive", True) is not False]


def _linea_servicio(s: dict, con_precio: bool = True) -> str:
    from app.agents.tools.agenda_helpers import _etiqueta_servicio

    nombre = _etiqueta_servicio(s)
    precio = _precio(s.get("price") if "price" in s else s.get("precio"))
    return f"• *{nombre}* — {precio}" if con_precio and precio else f"• *{nombre}*"


def _nombres_servicios(servicios: list[dict], limite: int = 4) -> str:
    from app.agents.tools.agenda_helpers import _etiqueta_servicio

    nombres = [_etiqueta_servicio(s).split(" / ")[0].lower() for s in servicios[:limite]]
    if not nombres:
        return ""
    return nombres[0] if len(nombres) == 1 else ", ".join(nombres[:-1]) + " y " + nombres[-1]


def _buscar_servicio(text: str, servicios: list[dict]) -> Optional[dict]:
    from app.agents.tools.agenda_helpers import _buscar_servicio_por_texto, _normalizar_texto

    norm = _normalizar_texto(text)
    for filler in ("cuanto", "cuesta", "cuestan", "vale", "valen", "precio", "precios", "costo", "valor", "tarifa", "de", "la", "el", "una", "un", "que", "tiene", "sale", "cobran", "es"):
        norm = re.sub(rf"\b{filler}\b", " ", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    if len(norm) < 4:
        return None
    return _buscar_servicio_por_texto(norm, servicios)


async def _seccion_servicios(phone: Optional[str], con_intro_clinica: bool) -> str:
    servicios = await _servicios()
    partes = []
    if con_intro_clinica:
        partes.append(rv.pick("info_que_es_corto", QUE_ES_CORTO, phone=phone))
    if servicios:
        lista = "\n".join(_linea_servicio(s) for s in servicios[:_MAX_LIST])
        partes.append(rv.pick("info_servicios_intro", SERVICIOS_INTRO, phone=phone) + "\n" + lista)
    else:
        partes.append(rv.pick("info_servicios_sin_datos", SERVICIOS_SIN_DATOS, phone=phone))
    return "\n\n".join(partes)


async def _seccion_que_es(phone: Optional[str]) -> str:
    nombres = _nombres_servicios(await _servicios())
    if nombres:
        return rv.pick("info_que_es", QUE_ES_CON_SERVICIOS, phone=phone, servicios=nombres)
    return rv.pick("info_que_es_corto", QUE_ES_CORTO, phone=phone)


async def _seccion_precio(text: str, phone: Optional[str]) -> str:
    servicios = await _servicios()
    encontrado = _buscar_servicio(text, servicios) if servicios else None
    if encontrado:
        from app.agents.tools.agenda_helpers import _etiqueta_servicio

        precio = _precio(encontrado.get("price"))
        duracion = encontrado.get("durationMinutes")
        if precio:
            pool = PRECIO_SERVICIO if duracion else PRECIO_SERVICIO_SIN_DURACION
            return rv.pick(
                "info_precio",
                pool,
                phone=phone,
                servicio=_etiqueta_servicio(encontrado),
                precio=precio,
                duracion=str(duracion or ""),
            )
    if servicios:
        lista = "\n".join(_linea_servicio(s) for s in servicios[:_MAX_LIST])
        return rv.pick("info_precios_intro", PRECIOS_INTRO, phone=phone) + "\n" + lista
    return rv.pick("info_servicios_sin_datos", SERVICIOS_SIN_DATOS, phone=phone)


async def _seccion_especialistas(phone: Optional[str]) -> str:
    profs = await _profesionales()
    nombres = [str(p.get("name") or p.get("nombreCompleto") or "").strip() for p in profs]
    nombres = [n for n in nombres if n]
    if nombres:
        lista = "\n".join(f"• {n}" for n in nombres[:8])
        return rv.pick("info_especialistas_intro", ESPECIALISTAS_INTRO, phone=phone) + "\n" + lista
    return rv.pick("info_especialistas_sin_datos", ESPECIALISTAS_SIN_DATOS, phone=phone)


async def build_info_reply(
    text: str, *, phone: Optional[str] = None, nombre: Optional[str] = None
) -> Optional[str]:
    """Respuesta informativa con datos reales, o None si el mensaje no es informativo."""
    intents = detect_info_intents(text)
    if not intents:
        return None
    if "como_agendar" in intents:
        return rv.pick("agendar_inicio", rv.AGENDAR_INICIO, phone=phone)

    secciones: list[str] = []
    lista_mostrada = False
    if "servicios" in intents:
        secciones.append(await _seccion_servicios(phone, con_intro_clinica="que_es" in intents))
        lista_mostrada = True
    elif "que_es" in intents:
        secciones.append(await _seccion_que_es(phone))
    if "precio" in intents:
        # The services list already shows prices; only add a section for a specific service.
        if "servicios" not in intents or _buscar_servicio(text, await _servicios()):
            secciones.append(await _seccion_precio(text, phone))
            lista_mostrada = True
    if "especialistas" in intents:
        secciones.append(await _seccion_especialistas(phone))
    if "horario" in intents:
        secciones.append(rv.pick("info_horario", HORARIO_DATO, phone=phone))
    if "ubicacion" in intents:
        secciones.append(rv.pick("info_ubicacion", UBICACION_DATO, phone=phone))
    if "contacto" in intents:
        secciones.append(rv.pick("info_contacto", CONTACTO_DATO, phone=phone))
    if not secciones:
        return None

    cierre_pool = CIERRE_LISTA if lista_mostrada else CIERRE_GENERAL
    secciones.append(rv.pick("info_cierre", cierre_pool, phone=phone, nombre=nombre))
    return "\n\n".join(secciones)


_WANT_OR_BOOK = re.compile(
    r"\b(necesito|necesitaria|quiero|quisiera|me gustaria|deseo|busco|me interesa|hacerme|me hago"
    r"|agendar|agendame|agenda|agendo|sacar|separar|separame|reservar|apartar|programar)\b"
    r"|\b(una )?cita (para|de)\b"
)
_NOT_BOOKING = re.compile(r"\b(cancelar|cancela|reprogramar|modificar|cambiar|cuanto|precio|precios|costo)\b")
# Last bot message offered the catalog or asked which treatment.
_OFFER_MARKERS = (
    "te interesa alguno",
    "llama la atención",
    "te separo una cita",
    "te agende alguno",
    "servicios y precios",
    " — $",
    "qué tratamiento",
    "que tratamiento",
)
_SERVICE_SYNONYMS = {
    "limpieza": ("limpieza", "profilaxis"),
    "limpiar": ("limpieza", "profilaxis"),
    "profilaxis": ("profilaxis",),
    "blanqueamiento": ("blanqueamiento",),
    "blanquear": ("blanqueamiento",),
    "blanqueo": ("blanqueamiento",),
    "resina": ("resina",),
    "resinas": ("resina",),
    "calza": ("calza", "resina"),
    "calzas": ("calza", "resina"),
    "injerto": ("injerto",),
    "encia": ("injerto", "encia"),
}
_GENERIC_TOKENS = {"dental", "dentales", "clinico", "clinica", "estetica", "estetico", "general", "servicio"}


def _service_tokens(servicio: dict) -> set[str]:
    from app.agents.tools.agenda_helpers import _etiqueta_servicio

    raw = f"{_etiqueta_servicio(servicio)} {servicio.get('name') or ''}"
    return {t for t in _norm(raw).split() if len(t) >= 5 and t not in _GENERIC_TOKENS}


def match_service(text: str, servicios: list[dict]) -> Optional[dict]:
    """Catalog service named in the text (accent-insensitive, synonyms), or None."""
    words = set(_norm(text).split())
    wanted = set()
    for w in words:
        wanted.update(_SERVICE_SYNONYMS.get(w, ()))
        if len(w) >= 5 and w not in _GENERIC_TOKENS:
            wanted.add(w[:-1] if w.endswith("s") and len(w) > 5 else w)
    best, best_score = None, 0
    for servicio in servicios:
        tokens = _service_tokens(servicio)
        score = sum(1 for t in tokens if t in wanted or (t.endswith("s") and t[:-1] in wanted))
        # Explicit synonym hits outrank incidental word overlap.
        score += sum(2 for w in words for syn in _SERVICE_SYNONYMS.get(w, ())[:1] if syn in tokens)
        if score > best_score:
            best, best_score = servicio, score
    return best


def is_service_booking_intent(text: str, prev_ai_text: str = "") -> bool:
    norm = _norm(text)
    if not norm or _NOT_BOOKING.search(norm):
        return False
    if _WANT_OR_BOOK.search(norm):
        return True
    if _PAIN.search(norm):
        return False
    prev = (prev_ai_text or "").lower()
    return len(norm.split()) <= 6 and any(m in prev for m in _OFFER_MARKERS)


async def build_service_booking_reply(
    text: str,
    *,
    phone: Optional[str] = None,
    prev_ai_text: str = "",
    nombre: Optional[str] = None,
    identity_known: bool = False,
) -> Optional[str]:
    """Booking reply when the patient names a catalog service; stores the choice for later steps."""
    if not is_service_booking_intent(text, prev_ai_text):
        return None
    servicios = await _servicios()
    servicio = match_service(text, servicios) if servicios else None
    if not servicio:
        return None
    from app.agents.tools.agenda_helpers import _etiqueta_servicio
    from app.services.booking_flow import set_pending_service

    etiqueta = _etiqueta_servicio(servicio)
    if phone:
        set_pending_service(phone, etiqueta)
    precio = _precio(servicio.get("price"))
    duracion = servicio.get("durationMinutes")
    if identity_known:
        pool, key = (SERVICIO_PEDIR_FECHA if precio and duracion else SERVICIO_PEDIR_FECHA_SIMPLE), "servicio_fecha"
    else:
        pool, key = (SERVICIO_PEDIR_DATOS if precio and duracion else SERVICIO_PEDIR_DATOS_SIMPLE), "servicio_datos"
    return rv.pick(
        key,
        pool,
        phone=phone,
        nombre=nombre,
        servicio=etiqueta,
        precio=precio or "",
        duracion=str(duracion or ""),
    )


def build_generic_reply(
    text: str = "", *, phone: Optional[str] = None, nombre: Optional[str] = None
) -> str:
    """Respuesta humana para cualquier otro mensaje cuando no hay LLM: sigue la conversación."""
    norm = _norm(text)
    if norm and _PAIN.search(norm):
        return rv.pick("dolor_sin_llm", DOLOR, phone=phone, nombre=nombre)
    return rv.pick("generico", GENERICO, phone=phone, nombre=nombre)


# ─────────────────────────────────────────────────────────────────────────────
# Pools (tono recepción, 0-1 emoji, sin revelar que es un bot ni hablar de fallas)
# ─────────────────────────────────────────────────────────────────────────────

QUE_ES_CORTO = (
    "Somos *Nexus Odonto*, una clínica odontológica en la Calle 100 # 15-20.",
    "*Nexus Odonto* es una clínica dental; aquí cuidamos tu salud oral y tu sonrisa.",
    "Te cuento: *Nexus Odonto* es un consultorio odontológico, atendemos todo lo de dientes, encías y boca.",
    "En *Nexus Odonto* nos dedicamos a la odontología: prevención, tratamientos y estética dental.",
)

QUE_ES_CON_SERVICIOS = (
    "*Nexus Odonto* es una clínica odontológica: hacemos {servicios}, y por aquí mismo te ayudamos a agendar.",
    "Somos una clínica dental en la Calle 100 # 15-20. Manejamos {servicios}, entre otros tratamientos.",
    "Te cuento: en *Nexus Odonto* cuidamos tu salud oral. Hacemos {servicios}, y te puedo separar la cita por aquí.",
)

SERVICIOS_INTRO = (
    "Estos son los tratamientos que manejamos:",
    "Te cuento lo que hacemos en la clínica:",
    "Claro, en este momento ofrecemos:",
    "Mira, estos son nuestros servicios con su valor:",
)

PRECIOS_INTRO = (
    "Te comparto los precios actuales:",
    "Estos son nuestros valores:",
    "Claro, así están los precios:",
)

SERVICIOS_SIN_DATOS = (
    "Manejamos odontología general y estética: limpiezas, resinas y blanqueamiento, entre otros. "
    "Si me dices qué te interesa, te confirmo el valor.",
    "Hacemos desde limpiezas y resinas hasta blanqueamiento y tratamientos de encías. "
    "Dime cuál te interesa y te doy el detalle.",
)

PRECIO_SERVICIO = (
    "*{servicio}* tiene un valor de *{precio}* y dura unos {duracion} minutos.",
    "El precio de *{servicio}* es *{precio}*; la sesión toma cerca de {duracion} minutos.",
    "*{servicio}* está en *{precio}* (más o menos {duracion} minutos).",
    "Para *{servicio}* manejamos *{precio}*, con una duración aproximada de {duracion} minutos.",
)

PRECIO_SERVICIO_SIN_DURACION = (
    "*{servicio}* tiene un valor de *{precio}*.",
    "El precio de *{servicio}* es *{precio}*.",
    "*{servicio}* está en *{precio}*.",
)

ESPECIALISTAS_INTRO = (
    "Nuestro equipo de odontólogos:",
    "Estos son los profesionales que atienden en la clínica:",
    "Te cuento quiénes atienden aquí:",
    "En *Nexus Odonto* atienden:",
)

ESPECIALISTAS_SIN_DATOS = (
    "Tenemos odontólogos generales y especialistas; según el tratamiento te asignamos quién te atiende.",
    "Contamos con odontología general y especialistas, y al agendar te confirmo con quién quedas.",
)

HORARIO_DATO = (
    "Atendemos de *lunes a sábado* de 8:00 AM a 6:00 PM; domingos y festivos cerramos.",
    "Nuestro horario es de *lunes a sábado*, 8:00 AM a 6:00 PM (domingos y festivos no abrimos).",
    "Estamos abiertos de *lunes a sábado* entre 8:00 AM y 6:00 PM.",
)

UBICACION_DATO = (
    "Estamos en la *Calle 100 # 15-20*, Centro Médico Odontológico.",
    "Nos encuentras en la *Calle 100 # 15-20* (Centro Médico Odontológico).",
    "La clínica queda en la *Calle 100 # 15-20*, dentro del Centro Médico Odontológico.",
)

CONTACTO_DATO = (
    "Puedes escribirnos por aquí o llamar al *+57 324 6030217*; el correo es soporte@nexusodonto.com.",
    "Nuestro teléfono es el *+57 324 6030217* y el correo soporte@nexusodonto.com. Por este chat también te atendemos.",
    "Te dejo los datos: *+57 324 6030217* y soporte@nexusodonto.com.",
)

CIERRE_LISTA = (
    "¿Te interesa alguno en particular?",
    "¿Cuál te llama la atención? Si quieres te ayudo a agendar.",
    "Si alguno te sirve, te separo una cita.",
    "¿Quieres que te agende alguno?",
)

CIERRE_GENERAL = (
    "¿Te ayudo con algo más?",
    "¿Quieres que te separe una cita?",
    "Si quieres, te ayudo a agendar.",
    "¿Hay algo más en lo que te pueda ayudar{nombre}?",
)

# Must keep "número de cédula" + "nombre completo" + "cita": booking_flow detects the identity step by text.
SERVICIO_PEDIR_DATOS = (
    "¡Claro! *{servicio}* tiene un valor de *{precio}* y dura unos {duracion} minutos.\n\n"
    "Para agendarte la cita, ¿me pasas tu *número de cédula* y tu *nombre completo*?",
    "Perfecto, *{servicio}* ({precio}, aprox. {duracion} minutos).\n\n"
    "Para separarte la cita necesito tu *número de cédula* y tu *nombre completo*.",
    "Listo, te ayudo con *{servicio}*. Está en *{precio}* y toma cerca de {duracion} minutos.\n\n"
    "¿Me regalas tu *número de cédula* y tu *nombre completo* para apartar la cita?",
    "Con gusto. *{servicio}* cuesta *{precio}* (unos {duracion} minutos).\n\n"
    "Para la cita, ¿me compartes tu *número de cédula* y tu *nombre completo*?",
)

SERVICIO_PEDIR_DATOS_SIMPLE = (
    "¡Claro! Te ayudo con *{servicio}*.\n\n"
    "Para agendarte la cita, ¿me pasas tu *número de cédula* y tu *nombre completo*?",
    "Perfecto, *{servicio}*. Para separarte la cita necesito tu *número de cédula* y tu *nombre completo*.",
)

SERVICIO_PEDIR_FECHA = (
    "Perfecto{nombre}. *{servicio}* está en *{precio}* (unos {duracion} minutos).\n\n"
    "¿Qué día y a qué hora te gustaría venir?",
    "Listo{nombre}, *{servicio}*: {precio}, cerca de {duracion} minutos.\n\n"
    "¿Para qué fecha y hora te agendo la cita?",
    "Claro{nombre}. *{servicio}* cuesta *{precio}* y dura unos {duracion} minutos.\n\n"
    "¿Qué día te queda cómodo y a qué hora?",
)

SERVICIO_PEDIR_FECHA_SIMPLE = (
    "Perfecto{nombre}, *{servicio}*. ¿Qué día y a qué hora te gustaría venir?",
    "Listo{nombre}. ¿Para qué fecha y hora te agendo *{servicio}*?",
)

GENERICO = (
    "Claro{nombre}, cuéntame. ¿Tu duda es sobre una cita, algún tratamiento o precios?",
    "Con gusto te ayudo. ¿Me das un poquito más de detalle? Por ejemplo si quieres agendar, saber un precio o conocer los servicios.",
    "Te escucho{nombre}. ¿Es para agendar, cambiar una cita o saber de algún tratamiento?",
    "Dime qué necesitas y lo revisamos: puedo ayudarte a agendar, darte precios o contarte de los servicios y horarios.",
    "Cuéntame un poco más para ayudarte bien. Te puedo colaborar con citas, precios, servicios u horarios.",
)

DOLOR = (
    "Qué pena que tengas esa molestia{nombre}. Lo mejor es que te revise un odontólogo; "
    "si quieres te agendo una cita de valoración. ¿Me pasas tu *número de cédula* y tu *nombre completo*?",
    "Uy, entiendo. Para revisarte bien te puedo separar una cita de valoración. "
    "Mientras tanto, una compresa fría por fuera ayuda a aliviar. ¿Me regalas tu *número de cédula* y tu *nombre completo*?",
    "Lamento la molestia. Si quieres te busco cita pronto con uno de nuestros odontólogos; "
    "solo necesito tu *número de cédula* y tu *nombre completo*. Si es muy fuerte, llámanos al *+57 324 6030217*.",
)
