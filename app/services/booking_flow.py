"""Booking context helpers for the LLM (no replies are produced here).

- Parse cédula / full name from what the patient wrote, so Gemini never re-asks data already given.
- Keep a light per-phone booking state (service being booked, identity) that is injected into
  Gemini's prompt, built from the tools Gemini itself called.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# Booking in progress per phone (service + identity), fed to the LLM prompt.
_SESSIONS: dict[str, "BookingSession"] = {}
_SESSION_TTL_SECONDS = 15 * 60.0

_BOOKING_START_RE = re.compile(
    r"^(quiero|deseo|necesito|quisiera|me gustaria)?\s*"
    r"(una\s+|sacar\s+|apartar\s+|programar\s+)?"
    r"(cita|agendar|agendar\s+(una\s+)?cita|apartar\s+(una\s+)?cita|"
    r"programar\s+(una\s+)?cita|reservar\s+(una\s+)?cita)$",
    re.IGNORECASE,
)

_CEDULA_RE = re.compile(r"\b(\d{7,12})\b")
_IDENTITY_LINE_RE = re.compile(
    r"^\s*(\d{7,12})\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ\s.'-]{2,80})\s*$"
)


@dataclass(frozen=True)
class PatientIdentity:
    cedula: Optional[str] = None
    nombre: Optional[str] = None

    @property
    def complete(self) -> bool:
        return bool(self.cedula and self.nombre and len(self.nombre.split()) >= 2)

    @property
    def primer_nombre(self) -> str:
        if not self.nombre:
            return ""
        return self.nombre.strip().split()[0].title()


def _normalize_key(text: str) -> str:
    import unicodedata

    t = (text or "").lower().strip()
    t = "".join(
        c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn"
    )
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def is_booking_start_intent(text: str) -> bool:
    """True when the user is starting a booking (agendar / quiero cita)."""
    norm = _normalize_key(text)
    if not norm:
        return False
    if _BOOKING_START_RE.match(norm):
        return True
    # Broader phrases that still mean "start booking"
    if any(
        p in norm
        for p in (
            "quiero agendar",
            "quiero una cita",
            "necesito una cita",
            "agendar una cita",
            "agendar cita",
            "apartar cita",
            "programar cita",
            "reservar cita",
        )
    ) and not any(
        p in norm
        for p in ("cancelar", "modificar", "reprogramar", "consultar", "ver mis")
    ):
        return True
    return False


_HOY_RE = re.compile(r"\bhoy\b")
_HORA_RE = re.compile(
    r"(?:\ba\s+las?\s+|\blas\s+)?\b(\d{1,2})(?:[:.h](\d{2}))?\s*"
    r"(a\s*\.?\s*m\b\.?|p\s*\.?\s*m\b\.?|de\s+la\s+manana|de\s+la\s+tarde|del\s+mediodia)?"
)


def parse_same_day_time(text: str) -> Optional[tuple[int, int]]:
    """(hour24, minute) when the user asks for a specific time *today*, e.g. 'hoy a las 9AM'."""
    import unicodedata

    norm = "".join(
        c for c in unicodedata.normalize("NFD", (text or "").lower()) if unicodedata.category(c) != "Mn"
    )
    if not norm or not _HOY_RE.search(norm):
        return None
    for m in _HORA_RE.finditer(norm):
        prefix = m.group(0).lstrip().startswith(("a la", "las"))
        suffix = (m.group(3) or "").replace(" ", "").replace(".", "")
        if not prefix and not suffix:
            continue
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        if hour > 23 or minute > 59:
            continue
        if suffix in ("pm", "delatarde") and hour < 12:
            hour += 12
        elif suffix in ("am", "delamanana") and hour == 12:
            hour = 0
        elif not suffix and 1 <= hour <= 6:
            hour += 12
        return hour, minute
    return None


def parse_identity_from_text(text: str, *, allow_name_only: bool = False) -> PatientIdentity:
    """Parse cédula and/or full name from a user message.

    A name without a cédula is only taken when `allow_name_only` (the bot just asked for it);
    otherwise casual replies like "pues bueno el domingo" would become the patient's name.
    """
    raw = (text or "").strip()
    if not raw:
        return PatientIdentity()

    m = _IDENTITY_LINE_RE.match(raw)
    if m:
        return PatientIdentity(cedula=m.group(1), nombre=_clean_name(m.group(2)))

    ced = None
    m_ced = _CEDULA_RE.search(raw)
    if m_ced:
        ced = m_ced.group(1)

    nombre = None
    if ced:
        remainder = (raw[: m_ced.start()] + " " + raw[m_ced.end() :]).strip()
        remainder = re.sub(r"[^\wÁÉÍÓÚÜÑáéíóúüñ\s.'-]", " ", remainder, flags=re.UNICODE)
        remainder = re.sub(r"\s+", " ", remainder).strip()
        if remainder and _looks_like_person_name(remainder):
            nombre = _clean_name(remainder)
    elif (
        allow_name_only
        and _looks_like_person_name(raw)
        and len(raw.split()) >= 2
        and not any(ch.isdigit() for ch in raw)
    ):
        nombre = _clean_name(raw)

    return PatientIdentity(cedula=ced, nombre=nombre)


def _clean_name(name: str) -> str:
    parts = [p for p in re.split(r"\s+", (name or "").strip()) if p]
    return " ".join(p.title() for p in parts)


_NAME_STOPWORDS = frozenset(
    """
    pues bueno buena buenas buenos listo lista ok okay oka vale dale si no sip nop hola gracias
    claro perfecto perfecta mejor seria sería entonces porfa favor por please gusto igual tal vez
    quiero quisiera necesito deseo agendar agenda cita citas una uno un para hoy manana mañana
    pasado proximo próximo proxima próxima siguiente tarde noche temprano semana dia día hora horas
    lunes martes miercoles miércoles jueves viernes sabado sábado domingo el en a al
    am pm tipo como sobre despues después antes que qué cual cuál cuando cuándo donde dónde
    limpieza profilaxis resina calza blanqueamiento injerto encia encía valoracion valoración
    doctor doctora dra dr odontologo odontólogo precio precios cuanto cuánto cuesta vale
    """.split()
)


def _looks_like_person_name(text: str) -> bool:
    parts = [p for p in re.split(r"\s+", (text or "").strip()) if p]
    if len(parts) < 2 or len(parts) > 6:
        return False
    if any(p.lower().strip(".'-") in _NAME_STOPWORDS for p in parts):
        return False
    return all(re.fullmatch(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ.'-]+", p) for p in parts)


def _asked_for_name(ai_text: str) -> bool:
    low = (ai_text or "").lower()
    return "nombre" in low and ("cédula" in low or "cedula" in low or "completo" in low or "llamas" in low)


def _human_turns_with_prompt(messages: Iterable[Any]) -> list[tuple[str, str]]:
    """[(human text, previous bot text)] in chronological order."""
    turns: list[tuple[str, str]] = []
    prev_ai = ""
    for msg in messages:
        cls = type(msg).__name__
        content = getattr(msg, "content", None)
        if cls == "AIMessage" or getattr(msg, "type", None) == "ai":
            if content:
                prev_ai = str(content)
            continue
        if cls != "HumanMessage" and getattr(msg, "type", None) != "human":
            continue
        if content:
            turns.append((str(content), prev_ai))
    return turns


def extract_identity_from_history_newest_first(messages: Iterable[Any]) -> PatientIdentity:
    """Extract identity preferring the newest human message that has data.

    Name-only messages count only when the bot had just asked for the name.
    """
    parsed_turns = [
        parse_identity_from_text(text, allow_name_only=_asked_for_name(prev_ai))
        for text, prev_ai in _human_turns_with_prompt(messages)
    ]
    for parsed in reversed(parsed_turns):
        if parsed.complete:
            return parsed
    # Fallback: merge fragments across recent human turns
    cedula = None
    nombre = None
    for parsed in reversed(parsed_turns):
        if parsed.cedula and not cedula:
            cedula = parsed.cedula
        if parsed.nombre and not nombre:
            nombre = parsed.nombre
        if cedula and nombre:
            break
    return PatientIdentity(cedula=cedula, nombre=nombre)


def prev_asked_for_booking_identity(prev_ai: str) -> bool:
    """True if the last bot message asked for cédula/nombre in a booking context."""
    prev = (prev_ai or "").lower()
    if not prev:
        return False
    asks_id = any(
        k in prev
        for k in (
            "número de cédula",
            "numero de cedula",
            "cédula",
            "cedula",
            "nombre completo",
        )
    )
    bookingish = any(
        k in prev
        for k in (
            "agendar",
            "cita",
            "tratamiento",
            "servicio",
            "horarios disponibles",
            "para continuar",
        )
    )
    return asks_id and bookingish


_CLAIM_RE = re.compile(
    r"\b(?:queda|qued[oó]|ha\s+quedado|est[aá]|fue|ha\s+sido)\s+(?:\w+\s+){0,2}?"
    r"(?:agendad|reservad|programad|confirmad|reprogramad|cancelad|apartad)[oa]s?\b"
    r"|\b(?:agend|reserv|apart|cancel|reprogram)é\b"
    r"|\b(?:he|hemos)\s+(?:agendado|reservado|apartado|cancelado|reprogramado)\b",
    re.IGNORECASE,
)
_CONDITIONAL_RE = re.compile(
    r"\b(?:si|cuando|una\s+vez|apenas|en\s+cuanto|para\s+que|quieres|deseas|puedo|podemos|confirmas|confirmes)\b",
    re.IGNORECASE,
)
_MUTATION_TOOLS = frozenset({"agendar_cita_tool", "modificar_cita_tool", "cancelar_cita_tool", "confirmar_cita_tool"})
_MUTATION_SUCCESS = (
    "confirmada con éxito",
    "reprogramada exitosamente",
    "cancelada exitosamente",
    "confirmada exitosamente",
)


def claims_agenda_change(text: str) -> bool:
    """True if the reply states (not asks) that an appointment was booked/moved/cancelled."""
    for sentence in re.split(r"(?<=[.!?\n])", text or ""):
        s = sentence.strip()
        if not s or s.startswith("¿") or s.endswith("?") or _CONDITIONAL_RE.search(s):
            continue
        if _CLAIM_RE.search(s):
            return True
    return False


def agenda_changed_this_turn(tool_msgs: Iterable[Any]) -> bool:
    """A mutation tool succeeded, or the patient's existing appointments were just listed."""
    for msg in tool_msgs:
        name = (getattr(msg, "name", None) or "").strip()
        content = str(getattr(msg, "content", "") or "").lower()
        if name == "consultar_cita_por_cedula_tool" and "próximas citas programadas" in content:
            return True
        if name in _MUTATION_TOOLS and any(m in content for m in _MUTATION_SUCCESS):
            return True
    return False


def is_unbacked_confirmation(text: str, tool_msgs: Iterable[Any]) -> bool:
    return claims_agenda_change(text) and not agenda_changed_this_turn(tool_msgs)


@dataclass
class BookingSession:
    servicio: str
    cedula: Optional[str] = None
    nombre: Optional[str] = None
    expires: float = 0.0

    @property
    def identity_complete(self) -> bool:
        return bool(self.cedula and self.nombre)


def _session_key(phone: str) -> str:
    return (phone or "").strip()


def get_session(phone: str) -> Optional[BookingSession]:
    key = _session_key(phone)
    session = _SESSIONS.get(key) if key else None
    if session and time.monotonic() > session.expires:
        _SESSIONS.pop(key, None)
        return None
    return session


def _save_session(phone: str, session: BookingSession) -> None:
    key = _session_key(phone)
    if key:
        session.expires = time.monotonic() + _SESSION_TTL_SECONDS
        _SESSIONS[key] = session


def set_pending_service(phone: str, servicio: str) -> None:
    """Start (or switch the service of) a booking; identity already captured is kept."""
    if not _session_key(phone) or not servicio:
        return
    session = get_session(phone) or BookingSession(servicio=servicio)
    session.servicio = servicio
    _save_session(phone, session)
    logger.info("[BookingFlow] servicio en curso para %s: %s", phone, servicio)


def clear_pending_service(phone: str) -> None:
    key = _session_key(phone)
    if key:
        _SESSIONS.pop(key, None)


def sync_session_identity(phone: str, cedula: Optional[str], nombre: Optional[str]) -> None:
    """Fill identity gaps of an active booking; a name already captured is never replaced by a guess."""
    session = get_session(phone)
    if not session:
        return
    if cedula and cedula != session.cedula:
        session.cedula = cedula
        session.nombre = nombre or None
    elif nombre and not session.nombre:
        session.nombre = nombre
    _save_session(phone, session)


def describe_session_for_llm(phone: str) -> str:
    """One-line booking state for the LLM prompt ('' when no booking is in progress)."""
    session = get_session(phone)
    if not session:
        return ""
    pendiente = (
        "falta elegir día/hora y confirmar la cita"
        if session.identity_complete
        else "faltan cédula y nombre completo"
    )
    parts = [f"servicio={session.servicio}", f"pendiente: {pendiente}"]
    if session.cedula:
        parts.append(f"cédula={session.cedula}")
    if session.nombre:
        parts.append(f"nombre={session.nombre}")
    return " | ".join(parts)
