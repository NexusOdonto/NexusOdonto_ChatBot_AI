"""Agenda sin LLM: continúa la cita en curso (datos → día/hora → horario → confirmación → crear).

Usa la misma disponibilidad real que el LLM (`turnos_disponibles`) y crea la cita con
`_agendar_cita_impl`. Si el mensaje no responde a la pregunta pendiente devuelve None y el
resto del pipeline (info / LLM / respaldo) decide, sin perder la sesión.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import unicodedata
from datetime import date, datetime, time, timedelta
from typing import Awaitable, Callable, Optional

from app.services import reply_variants as rv
from app.services.booking_flow import (
    BookingSession,
    Slot,
    build_ask_service_response,
    clear_awaiting_booking_identity,
    clear_pending_service,
    get_session,
    parse_identity_from_text,
    prev_asked_for_booking_identity,
    save_session,
)
from app.services.natural_datetime import (
    WEEKDAY_NAMES,
    DateTimeRequest,
    dia_humano,
    hora_humana,
    parse_datetime_request,
)

logger = logging.getLogger(__name__)

_MAX_OPCIONES_DIA = 4
_MAX_ALTERNATIVAS = 3
_DIAS_BUSQUEDA = 8
_DAY_LOOKUP_TIMEOUT_S = 8.0

_EXIT_RE = re.compile(
    r"\b(cancelar|cancela|cancelo|anular|anula|reprogramar|reprograma|modificar|mis citas|asesor|humano"
    r"|duele|duelen|dolor|sangra|sangrado|hinchad[oa]|urgencia|emergencia)\b"
)
_YES_RE = re.compile(
    r"^(si+|sip|claro|dale|listo|ok|okay|oka|va|vale|de una|hagale|perfecto|confirmo|confirmado|confirmala"
    r"|agendala|apartala|agendamela|esa|ese|esa misma|ese mismo|correcto|asi es|me sirve|me parece|bueno"
    r"|bien|soy yo|exacto|por favor|porfa|obvio|sisas|hecho)\b"
)
_NO_RE = re.compile(r"^(no|nop|nel|mejor no|todavia no|aun no|espera|ninguno|ninguna|no me sirve|no puedo|no soy)\b")
_WHATEVER_RE = re.compile(
    r"^(no se|nose|ni idea|cualquiera|como sea|cuando sea|me da igual|da igual|el que tengas|la que tengas)\b"
)
_ORDINAL_WORDS = (
    (re.compile(r"\b(primer[oa]?|1ra|1ro|1era)\b"), 0),
    (re.compile(r"\bsegund[oa]\b"), 1),
    (re.compile(r"\btercer[oa]?\b"), 2),
    (re.compile(r"\bcuart[oa]\b"), 3),
    (re.compile(r"\bultim[oa]\b"), -1),
)
_OPTION_ONLY_RE = re.compile(r"^(?:la |el |opcion |numero |la opcion |el numero )?(\d|uno|dos|tres|cuatro)$")
_OPTION_WORDS = {"uno": 0, "dos": 1, "tres": 2, "cuatro": 3}


def _plain(text: str) -> str:
    t = (text or "").lower()
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _a_veces(session: BookingSession, prob: float = 0.4) -> Optional[str]:
    """Use the first name now and then, like a person would."""
    return session.nombre if session.nombre and random.random() < prob else None


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _dia_lista(d: date, today: date) -> str:
    return _cap(dia_humano(d, today).removeprefix("el "))


def _dia_resumen(d: date, today: date) -> str:
    if d == today:
        return f"Hoy, {WEEKDAY_NAMES[d.weekday()]} {d.day}"
    return _dia_lista(d, today)


def _minutos(hhmm: str) -> int:
    h, m = (int(x) for x in hhmm.split(":")[:2])
    return h * 60 + m


def _lista(opciones: list[Slot], today: date) -> str:
    mismo_dia = len({s.fecha for s in opciones}) == 1
    lineas = []
    for s in opciones:
        if mismo_dia:
            lineas.append(f"• {hora_humana(s.hhmm)} con {s.prof_nombre}")
        else:
            lineas.append(f"• {_dia_lista(s.fecha, today)}, {hora_humana(s.hhmm)} con {s.prof_nombre}")
    return "\n".join(lineas)


def _resumen(session: BookingSession, slot: Slot, today: date) -> str:
    return (
        f"• *Tratamiento:* {session.servicio}\n"
        f"• *Día:* {_dia_resumen(slot.fecha, today)}\n"
        f"• *Hora:* {hora_humana(slot.hhmm)}\n"
        f"• *Te atiende:* {slot.prof_nombre}"
    )


async def _servicio_dict(etiqueta: str) -> Optional[dict]:
    from app.agents.tools.agenda_helpers import _etiqueta_servicio
    from app.services.info_replies import _servicios

    for servicio in await _servicios():
        if _etiqueta_servicio(servicio) == etiqueta:
            return servicio
    return None


async def _slots_dia(servicio: dict, d: date, today: date) -> list[Slot]:
    if d < today or d.weekday() == 6:
        return []
    from app.agents.tools.catalog_tools import turnos_disponibles

    try:
        turnos = await asyncio.wait_for(turnos_disponibles(servicio, d.isoformat()), _DAY_LOOKUP_TIMEOUT_S)
    except Exception as exc:
        logger.warning("[BookingAssistant] disponibilidad %s no disponible: %s", d, exc)
        return []
    por_hora: dict[str, Slot] = {}
    for prof_id, prof_nombre, horas in turnos:
        for hhmm in horas:
            por_hora.setdefault(hhmm, Slot(d, hhmm, str(prof_id), str(prof_nombre)))
    return [por_hora[h] for h in sorted(por_hora)]


def _repartir(slots: list[Slot], n: int) -> list[Slot]:
    if len(slots) <= n:
        return list(slots)
    return [slots[round(i * (len(slots) - 1) / (n - 1))] for i in range(n)]


def _en_periodo(slot: Slot, periodo: Optional[str]) -> bool:
    if periodo == "manana":
        return _minutos(slot.hhmm) < 12 * 60
    if periodo == "tarde":
        return _minutos(slot.hhmm) >= 12 * 60
    return True


def _motivo_hora(d: date, hora: tuple[int, int], duracion: int, now: datetime, dia: str, phone: str) -> str:
    hhmm = f"{hora[0]:02d}:{hora[1]:02d}"
    fmt = {"hora": hora_humana(hhmm), "dia": dia, "dia_cap": _cap(dia)}
    inicio = datetime.combine(d, time(*hora))
    fin = inicio + timedelta(minutes=max(30, duracion))
    if d == now.date() and inicio <= now.replace(tzinfo=None) + timedelta(minutes=15):
        return rv.pick("motivo_hora_pasada", rv.MOTIVO_HORA_PASADA, phone=phone, **fmt)
    if d.weekday() == 5 and inicio.time() >= time(12, 0):
        return rv.pick("motivo_sabado", rv.MOTIVO_SABADO_TARDE, phone=phone)
    if time(12, 0) <= inicio.time() < time(14, 0):
        return rv.pick("motivo_almuerzo", rv.MOTIVO_ALMUERZO, phone=phone, **fmt)
    if inicio.time() < time(8, 0) or inicio.time() >= time(17, 0):
        return rv.pick("motivo_fuera", rv.MOTIVO_FUERA_JORNADA, phone=phone, **fmt)
    cierre = time(12, 0) if d.weekday() == 5 else time(17, 0)
    cruza_almuerzo = inicio.time() < time(12, 0) and fin.time() > time(12, 0)
    if fin.time() > cierre or cruza_almuerzo or fin.date() > d:
        return rv.pick("motivo_no_cabe", rv.MOTIVO_NO_CABE, phone=phone, **fmt)
    return rv.pick("motivo_ocupado", rv.MOTIVO_OCUPADO, phone=phone, **fmt)


async def _alternativas(
    base: date,
    req: DateTimeRequest,
    today: date,
    slots_de: Callable[[date], Awaitable[list[Slot]]],
    excluir: Optional[Slot] = None,
) -> list[Slot]:
    """Nearest real free slots around `base` (same day first, then following open days)."""
    base = max(base, today)
    dias: list[date] = [base]
    if base.weekday() == 6 and base - timedelta(days=1) >= today:
        dias.insert(0, base - timedelta(days=1))
    dias += [base + timedelta(days=i) for i in range(1, _DIAS_BUSQUEDA)]
    objetivo = (
        req.hora[0] * 60 + req.hora[1] if req.hora
        else 9 * 60 if req.periodo == "manana"
        else 15 * 60 if req.periodo == "tarde"
        else None
    )
    elegidos: list[Slot] = []
    dias_con_cupo = 0
    for d in dias:
        libres = [
            s for s in await slots_de(d)
            if not (excluir and s.fecha == excluir.fecha and s.hhmm == excluir.hhmm)
        ]
        if not libres:
            continue
        dias_con_cupo += 1
        if objetivo is not None:
            libres.sort(key=lambda s: abs(_minutos(s.hhmm) - objetivo))
        elegidos += libres[: (3 if d == base else 2)]
        if len(elegidos) >= _MAX_ALTERNATIVAS or dias_con_cupo >= 2:
            break
    elegidos = elegidos[:_MAX_ALTERNATIVAS + 1]
    return sorted(elegidos, key=lambda s: (s.fecha, s.hhmm))


def _ofrecer(phone: str, session: BookingSession, opciones: list[Slot], intro: str, today: date) -> str:
    session.offered, session.proposed, session.stage = opciones, None, "slot"
    save_session(phone, session)
    cierre = rv.pick("elegir_cierre", rv.ELEGIR_CIERRE, phone=phone)
    return f"{intro}\n{_lista(opciones, today)}\n\n{cierre}"


def _proponer(phone: str, session: BookingSession, slot: Slot, today: date) -> str:
    session.proposed, session.stage = slot, "confirm"
    save_session(phone, session)
    return rv.pick(
        "confirmar_cita", rv.CONFIRMAR_CITA, phone=phone,
        nombre=_a_veces(session), resumen=_resumen(session, slot, today),
    )


async def _responder_fecha(
    phone: str, session: BookingSession, req: DateTimeRequest, now: datetime
) -> Optional[str]:
    servicio = await _servicio_dict(session.servicio)
    if not servicio:
        return None
    today = now.date()
    duracion = int(servicio.get("durationMinutes") or 30)
    cache: dict[date, list[Slot]] = {}

    async def slots_de(d: date) -> list[Slot]:
        if d not in cache:
            cache[d] = await _slots_dia(servicio, d, today)
        return cache[d]

    fecha = req.fecha
    motivo = ""
    if fecha and fecha < today:
        motivo = rv.pick("motivo_fecha_pasada", rv.MOTIVO_FECHA_PASADA, phone=phone)
        base = today
    elif fecha and fecha.weekday() == 6:
        motivo = rv.pick("motivo_domingo", rv.MOTIVO_DOMINGO, phone=phone)
        base = fecha
    elif not fecha and not req.hora and not req.periodo:
        base = today
    else:
        d = fecha or today
        if not fecha and not await slots_de(d):
            d = next(today + timedelta(days=i) for i in range(1, 3) if (today + timedelta(days=i)).weekday() != 6)
        dia = dia_humano(d, today)
        dia_fmt = {"dia": dia, "dia_cap": _cap(dia)}
        libres = await slots_de(d)
        if req.hora:
            exacto = next((s for s in libres if s.hhmm == req.hhmm), None)
            if exacto:
                return _proponer(phone, session, exacto, today)
            motivo = _motivo_hora(d, req.hora, duracion, now, dia, phone)
        elif libres:
            preferidos = [s for s in libres if _en_periodo(s, req.periodo)]
            if preferidos:
                intro = rv.pick(
                    "opciones_dia", rv.OPCIONES_DIA, phone=phone, nombre=_a_veces(session), **dia_fmt
                )
                return _ofrecer(phone, session, _repartir(preferidos, _MAX_OPCIONES_DIA), intro, today)
            periodo = "mañana" if req.periodo == "manana" else "tarde"
            motivo = rv.pick(
                "motivo_sin_cupo_periodo", rv.MOTIVO_SIN_CUPO_PERIODO, phone=phone, periodo=periodo, **dia_fmt
            )
        elif d.weekday() == 5 and req.periodo == "tarde":
            motivo = rv.pick("motivo_sabado", rv.MOTIVO_SABADO_TARDE, phone=phone)
        else:
            motivo = rv.pick("motivo_sin_cupo_dia", rv.MOTIVO_SIN_CUPO_DIA, phone=phone, **dia_fmt)
        base = d

    opciones = await _alternativas(base, req, today, slots_de)
    if not opciones:
        session.stage, session.offered, session.proposed = "date", [], None
        save_session(phone, session)
        sin = rv.pick("sin_cupos", rv.SIN_CUPOS, phone=phone, servicio=session.servicio)
        return f"{motivo} {sin}".strip()
    intro = rv.pick("opciones_cercanas", rv.OPCIONES_CERCANAS, phone=phone, servicio=session.servicio)
    return _ofrecer(phone, session, opciones, f"{motivo} {intro}".strip(), today)


def _elegir_ofrecido(plain: str, req: DateTimeRequest, ofrecidos: list[Slot]) -> Optional[Slot]:
    m = _OPTION_ONLY_RE.match(plain)
    if m:
        raw = m.group(1)
        idx = int(raw) - 1 if raw.isdigit() else _OPTION_WORDS[raw]
        if 0 <= idx < len(ofrecidos):
            return ofrecidos[idx]
    if not req.hora:
        for rx, idx in _ORDINAL_WORDS:
            if rx.search(plain) and (idx == -1 or idx < len(ofrecidos)):
                return ofrecidos[idx]
    candidatos = ofrecidos
    if req.fecha:
        candidatos = [s for s in candidatos if s.fecha == req.fecha]
    if req.hora:
        candidatos = [s for s in candidatos if s.hhmm == req.hhmm]
    elif req.periodo and req.fecha:
        candidatos = [s for s in candidatos if _en_periodo(s, req.periodo)]
    if (req.hora or req.fecha) and len(candidatos) == 1:
        return candidatos[0]
    if req.hora and candidatos and not req.fecha:
        return candidatos[0]
    return None


async def _crear(phone: str, session: BookingSession, now: datetime, confirmar_misma_persona: bool = False) -> str:
    from app.agents.tools.agenda_helpers import _obtener_valor
    from app.agents.tools.appointment_tools import _agendar_cita_impl, _web_portal_url

    slot = session.proposed
    today = now.date()
    servicio = await _servicio_dict(session.servicio)
    servicio_ref = (_obtener_valor(servicio, "id", "servicioId", "serviceId") if servicio else None) or session.servicio
    resultado = await _agendar_cita_impl(
        session.cedula or "",
        session.nombre or "",
        slot.prof_id,
        servicio_ref,
        f"{slot.fecha.isoformat()}T{slot.hhmm}:00",
        f"Cita de {session.servicio}",
        config={"configurable": {"thread_id": phone}},
        confirmar_misma_persona=confirmar_misma_persona,
    )
    low = resultado.lower()
    if "confirmada con éxito" in low:
        logger.info("[BookingAssistant] cita creada sin LLM para %s: %s %s", phone, slot.fecha, slot.hhmm)
        resumen = _resumen(session, slot, today)
        primera_vez = "[primera_vez_portal]" in low
        portal = rv.pick(
            "portal_primera_vez" if primera_vez else "portal_recordatorio",
            rv.PORTAL_PRIMERA_VEZ if primera_vez else rv.PORTAL_RECORDATORIO,
            phone=phone,
            url=_web_portal_url(),
        )
        cabecera = rv.pick("cita_agendada", rv.CITA_AGENDADA, phone=phone, nombre=session.nombre)
        despedida = rv.pick("cita_despedida", rv.CITA_DESPEDIDA, phone=phone)
        clear_pending_service(phone)
        clear_awaiting_booking_identity(phone)
        return f"{cabecera}\n\n{resumen}\n\n{portal}\n\n{despedida}"

    if "misma persona" in low:
        m = re.search(r"a nombre de \*([^*]+)\*", resultado)
        session.registered_name = m.group(1) if m else ""
        session.stage = "confirm_identity"
        save_session(phone, session)
        return rv.pick(
            "misma_persona", rv.MISMA_PERSONA, phone=phone,
            cedula=session.cedula or "", registrado=session.registered_name or "otra persona",
        )

    if "nombre completo" in low and "necesito" in low:
        session.nombre, session.stage = None, "identity"
        save_session(phone, session)
        return rv.pick("pedir_nombre_cita", rv.PEDIR_NOMBRE_CITA, phone=phone)

    logger.warning("[BookingAssistant] no se pudo crear la cita (%s): %s", phone, resultado[:200])
    cache: dict[date, list[Slot]] = {}

    async def slots_de(d: date) -> list[Slot]:
        if d not in cache:
            cache[d] = await _slots_dia(servicio, d, today) if servicio else []
        return cache[d]

    dia = dia_humano(slot.fecha, today)
    motivo = rv.pick(
        "motivo_ocupado", rv.MOTIVO_OCUPADO, phone=phone,
        hora=hora_humana(slot.hhmm), dia=dia, dia_cap=_cap(dia),
    )
    req = DateTimeRequest(fecha=slot.fecha, hora=(int(slot.hhmm[:2]), int(slot.hhmm[3:5])))
    opciones = await _alternativas(slot.fecha, req, today, slots_de, excluir=slot)
    if not opciones:
        session.stage, session.offered, session.proposed = "date", [], None
        save_session(phone, session)
        return f"{motivo} " + rv.pick("sin_cupos", rv.SIN_CUPOS, phone=phone, servicio=session.servicio)
    intro = rv.pick("opciones_cercanas", rv.OPCIONES_CERCANAS, phone=phone, servicio=session.servicio)
    return _ofrecer(phone, session, opciones, f"{motivo} {intro}", today)


async def _is_service_switch(text: str, session: BookingSession) -> bool:
    from app.agents.tools.agenda_helpers import _etiqueta_servicio
    from app.services.info_replies import _WANT_OR_BOOK, _norm, _servicios, match_service

    if not _WANT_OR_BOOK.search(_norm(text)) and "mejor" not in _norm(text):
        return False
    nuevo = match_service(text, await _servicios())
    return bool(nuevo) and _etiqueta_servicio(nuevo) != session.servicio


async def _identity_step(phone: str, session: BookingSession, text: str, now: datetime) -> Optional[str]:
    ident = parse_identity_from_text(text, allow_name_only=True)
    if ident.cedula:
        if ident.cedula != session.cedula:
            session.cedula, session.nombre = ident.cedula, ident.nombre or None
        elif ident.nombre:
            session.nombre = ident.nombre
    elif ident.nombre:
        session.nombre = ident.nombre
    else:
        return None
    if not session.cedula:
        save_session(phone, session)
        return rv.pick("pedir_cedula_cita", rv.PEDIR_CEDULA_CITA, phone=phone, nombre=session.nombre)
    if not session.nombre:
        save_session(phone, session)
        return rv.pick("pedir_nombre_cita", rv.PEDIR_NOMBRE_CITA, phone=phone)
    session.stage = "date"
    save_session(phone, session)
    clear_awaiting_booking_identity(phone)
    req = parse_datetime_request(text, now.date())
    if not req.empty:
        return await _responder_fecha(phone, session, req, now)
    return build_ask_service_response(nombre=session.nombre, phone=phone)


async def handle_booking_turn(phone: str, text: str, *, now: datetime) -> Optional[str]:
    """Reply for the pending booking question, or None when the message is about something else."""
    session = get_session(phone)
    if not session:
        return None
    plain = _plain(text)
    if not plain or _EXIT_RE.search(plain):
        return None
    if await _is_service_switch(text, session):
        return None
    if session.stage == "identity":
        return await _identity_step(phone, session, text, now)

    today = now.date()
    req = parse_datetime_request(text, today)
    yes = bool(_YES_RE.match(plain))
    no = bool(_NO_RE.match(plain))

    if session.stage == "confirm_identity" and session.proposed:
        if yes:
            return await _crear(phone, session, now, confirmar_misma_persona=True)
        if no:
            session.cedula, session.nombre, session.stage = None, None, "identity"
            save_session(phone, session)
            return rv.pick("misma_persona_no", rv.MISMA_PERSONA_NO, phone=phone)
        return None

    if session.stage == "confirm" and session.proposed and req.empty:
        if yes:
            return await _crear(phone, session, now)
        if no:
            session.stage, session.proposed = "date", None
            save_session(phone, session)
            return rv.pick("no_confirma", rv.NO_CONFIRMA, phone=phone, nombre=_a_veces(session))

    if session.stage in ("slot", "confirm") and session.offered:
        elegido = _elegir_ofrecido(plain, req, session.offered)
        if elegido:
            return _proponer(phone, session, elegido, today)
        if session.stage == "slot" and req.empty and (yes or _WHATEVER_RE.match(plain)):
            if len(session.offered) == 1 or _WHATEVER_RE.match(plain):
                return _proponer(phone, session, session.offered[0], today)
            return rv.pick(
                "elegir_slot", rv.ELEGIR_SLOT_REPROMPT, phone=phone,
                nombre=_a_veces(session), opciones=_lista(session.offered, today),
            )

    if not req.empty:
        return await _responder_fecha(phone, session, req, now)
    if session.stage == "date" and (yes or _WHATEVER_RE.match(plain)):
        return await _responder_fecha(phone, session, DateTimeRequest(lo_antes_posible=True), now)
    return None


def reprompt_pending_question(phone: str, prev_ai_text: str = "") -> Optional[str]:
    """Re-ask whatever the conversation is waiting for (used when nothing else understood the message)."""
    session = get_session(phone)
    if session:
        today = rv._now_bogota().date()
        nombre = _a_veces(session)
        if session.stage == "identity":
            return rv.pick("repedir_identidad", rv.REPEDIR_IDENTIDAD, phone=phone, servicio=session.servicio)
        if session.stage == "slot" and session.offered:
            return rv.pick(
                "elegir_slot", rv.ELEGIR_SLOT_REPROMPT, phone=phone,
                nombre=nombre, opciones=_lista(session.offered, today),
            )
        if session.stage in ("confirm", "confirm_identity") and session.proposed:
            return rv.pick(
                "repedir_confirmacion", rv.REPEDIR_CONFIRMACION, phone=phone,
                nombre=nombre, resumen=_resumen(session, session.proposed, today),
            )
        return rv.pick("repedir_fecha", rv.REPEDIR_FECHA, phone=phone, nombre=nombre, servicio=session.servicio)
    prev = (prev_ai_text or "").lower()
    if prev_asked_for_booking_identity(prev):
        return rv.pick("repedir_identidad_general", rv.REPEDIR_IDENTIDAD_GENERAL, phone=phone)
    if "?" in prev and any(k in prev for k in ("qué día", "que dia", "qué fecha", "que fecha", "a qué hora", "qué hora")):
        return rv.pick("repedir_fecha_general", rv.REPEDIR_FECHA_GENERAL, phone=phone)
    return None


def nudge_for_active_booking(phone: str) -> Optional[str]:
    """Short line to bring the conversation back to the booking after answering a side question."""
    session = get_session(phone)
    if not session or session.stage == "identity":
        return None
    return rv.pick("seguimos_cita", rv.SEGUIMOS_CITA, phone=phone, servicio=session.servicio)
