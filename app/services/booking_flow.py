"""Booking identity continuity helpers.

Fixes two WhatsApp loops:
A) "Quiero agendar" after cédula+nombre already in history must not re-ask identity.
B) After the bot asks for cédula+nombre for agendar, capturing them must continue
   to service selection — never reset to the welcome/menu greeting.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from app.services import reply_variants

logger = logging.getLogger(__name__)

# In-memory pending booking identity capture (phone → expiry monotonic).
_PENDING_BOOKING_IDENTITY: dict[str, float] = {}
_PENDING_TTL_SECONDS = 15 * 60.0

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


def parse_identity_from_text(text: str) -> PatientIdentity:
    """Parse cédula and/or full name from a user message."""
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
    elif _looks_like_person_name(raw) and len(raw.split()) >= 2 and not any(ch.isdigit() for ch in raw):
        nombre = _clean_name(raw)

    return PatientIdentity(cedula=ced, nombre=nombre)


def _clean_name(name: str) -> str:
    parts = [p for p in re.split(r"\s+", (name or "").strip()) if p]
    return " ".join(p.title() for p in parts)


def _looks_like_person_name(text: str) -> bool:
    parts = [p for p in re.split(r"\s+", (text or "").strip()) if p]
    if len(parts) < 2 or len(parts) > 6:
        return False
    skip = {
        "quiero",
        "agendar",
        "cita",
        "una",
        "para",
        "hoy",
        "manana",
        "mañana",
        "limpieza",
        "profilaxis",
        "resina",
        "blanqueamiento",
        "doctor",
        "doctora",
    }
    if any(p.lower() in skip for p in parts):
        return False
    return all(re.fullmatch(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ.'-]+", p) for p in parts)


def extract_identity_from_messages(messages: Iterable[Any]) -> PatientIdentity:
    """Scan recent human messages for the latest cédula + full name."""
    cedula = None
    nombre = None
    for msg in messages:
        content = getattr(msg, "content", None)
        # LangChain HumanMessage / plain str
        role_ok = False
        cls = type(msg).__name__
        if cls == "HumanMessage" or getattr(msg, "type", None) == "human":
            role_ok = True
        if isinstance(msg, str):
            content = msg
            role_ok = True
        if not role_ok or not content:
            continue
        parsed = parse_identity_from_text(str(content))
        if parsed.cedula and not cedula:
            cedula = parsed.cedula
        if parsed.nombre and (not nombre or len(parsed.nombre) > len(nombre)):
            nombre = parsed.nombre
        if cedula and nombre:
            # Prefer the most recent complete pair; keep scanning newer→older
            # callers should pass messages newest-first or we accumulate best.
            pass
    return PatientIdentity(cedula=cedula, nombre=nombre)


def extract_identity_from_history_newest_first(messages: Iterable[Any]) -> PatientIdentity:
    """Extract identity preferring the newest human message that has data."""
    msgs = list(messages)
    # Prefer newest complete identity
    for msg in reversed(msgs):
        cls = type(msg).__name__
        if cls != "HumanMessage" and getattr(msg, "type", None) != "human":
            continue
        content = getattr(msg, "content", None)
        if not content:
            continue
        parsed = parse_identity_from_text(str(content))
        if parsed.complete:
            return parsed
    # Fallback: merge fragments across recent human turns
    cedula = None
    nombre = None
    for msg in reversed(msgs):
        cls = type(msg).__name__
        if cls != "HumanMessage" and getattr(msg, "type", None) != "human":
            continue
        content = getattr(msg, "content", None)
        if not content:
            continue
        parsed = parse_identity_from_text(str(content))
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


def mark_awaiting_booking_identity(phone: str) -> None:
    key = (phone or "").strip()
    if not key:
        return
    _PENDING_BOOKING_IDENTITY[key] = time.monotonic() + _PENDING_TTL_SECONDS
    logger.info("[BookingFlow] awaiting identity for %s", key)


def clear_awaiting_booking_identity(phone: str) -> None:
    key = (phone or "").strip()
    if key:
        _PENDING_BOOKING_IDENTITY.pop(key, None)


def is_awaiting_booking_identity(phone: str) -> bool:
    key = (phone or "").strip()
    if not key:
        return False
    exp = _PENDING_BOOKING_IDENTITY.get(key)
    if exp is None:
        return False
    if time.monotonic() > exp:
        _PENDING_BOOKING_IDENTITY.pop(key, None)
        return False
    return True


def build_ask_service_response(*, nombre: Optional[str] = None, phone: Optional[str] = None) -> str:
    """Deterministic next booking step after identity is known — never invents slots."""
    return reply_variants.pick(
        "pedir_servicio", reply_variants.PEDIR_SERVICIO, phone=phone, nombre=nombre
    )


def build_booking_reply_without_llm(
    text: str, now: Any, cedula_conocida: bool = False, phone: Optional[str] = None
) -> Optional[str]:
    """Deterministic booking reply for when the LLM is unavailable; None if the message isn't booking-related."""
    from app.agents.tools.agenda_helpers import _formatear_hora_ampm, franjas_restantes_hoy

    hm = parse_same_day_time(text)
    if hm:
        pedir = (
            reply_variants.pick("pedir_tratamiento", reply_variants.PEDIR_TRATAMIENTO_CORTO, phone=phone)
            if cedula_conocida
            else reply_variants.pick("pedir_datos", reply_variants.PEDIR_DATOS_Y_TRATAMIENTO, phone=phone)
        )
        req = now.replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)
        hora = _formatear_hora_ampm(req.strftime("%H:%M")).lstrip("0")
        if req.timestamp() < now.timestamp() + 15 * 60:
            restantes = franjas_restantes_hoy(now)
            if restantes:
                return reply_variants.pick(
                    "hora_pasada",
                    reply_variants.HORA_PASADA_CON_JORNADA,
                    phone=phone,
                    hora=hora,
                    restantes=restantes,
                    pedir=pedir,
                )
            return reply_variants.pick(
                "hora_pasada_cerrado",
                reply_variants.HORA_PASADA_SIN_JORNADA,
                phone=phone,
                hora=hora,
                pedir=pedir,
            )
        return reply_variants.pick(
            "hora_hoy", reply_variants.HORA_HOY_VALIDA, phone=phone, hora=hora, pedir=pedir
        )
    if is_booking_start_intent(text):
        return build_agendar_inicio_response(phone=phone)
    return None


def build_agendar_inicio_response(phone: Optional[str] = None) -> str:
    return reply_variants.pick("agendar_inicio", reply_variants.AGENDAR_INICIO, phone=phone)
