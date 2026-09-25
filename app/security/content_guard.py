"""Guards de contenido para recepción WhatsApp: respeto y alcance odontológico.

Detecta (0 tokens) bromas vulgares / falta de respeto y dolores fuera de odontología
para responder con tono humano de recepción, sin agendar ni escalar urgencias.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional, Tuple

MENSAJE_RESPETO = (
    "Te pedimos por favor que nos escribas con respeto. "
    "En *Nexus Odonto* atendemos temas odontológicos de forma seria y profesional. "
    "Si de verdad tienes una molestia dental o quieres agendar una cita, "
    "cuéntanos con claridad y con gusto te orientamos."
)

MENSAJE_FUERA_DE_ALCANCE = (
    "En *Nexus Odonto* solo atendemos *salud oral* (dientes, encías y boca). "
    "Por un dolor en otra parte del cuerpo te conviene consultar un médico general "
    "o el especialista correspondiente. "
    "Si tienes una molestia dental o quieres una cita odontológica, aquí te ayudamos."
)

# Partes del cuerpo claramente no odontológicas
_NON_DENTAL_PARTS = (
    "rodilla", "rodillas", "tobillo", "tobillos", "pie", "pies", "talon", "talones",
    "espalda", "lumbar", "cadera", "caderas", "brazo", "brazos", "antebrazo",
    "mano", "manos", "dedo", "dedos", "hombro", "hombros", "codo", "codos",
    "muneca", "munecas", "pierna", "piernas", "muslo", "muslos", "pantorrilla",
    "estomago", "barriga", "abdomen", "higado", "rinon", "rinones", "oido",
    "oidos", "oreja", "orejas", "ojo", "ojos", "pecho", "torax", "pulmon",
    "corazon", "cabeza",  # cefalea genérica sin contexto oral
)

_DENTAL_CONTEXT = (
    "diente", "dientes", "muela", "muelas", "encia", "encias", "boca", "dental",
    "odontolog", "odontologia", "mandibula", "maxilar", "labio", "labios",
    "lengua", "paladar", "cordal", "cordales", "brackets", "ortodoncia",
    "implante", "implantes", "caries", "juicio", "molar", "molares",
    "canino", "incisivo", "corona", "endodoncia", "protesis",
)

# Bromas vulgares / ofensivas que no deben tratarse como caso clínico
_DISRESPECT_PATTERNS = (
    r"\bmuela\s+del\s+ano\b",
    r"\bdiente\s+del\s+ano\b",
    r"\bmuela\s+del\s+culo\b",
    r"\bdiente\s+del\s+culo\b",
    r"\bmuela\s+del\s+trasero\b",
    r"\b(dolor|duele).{0,40}\b(ano|culo|trasero)\b",
    r"\b(ano|culo)\b.{0,40}\b(muela|diente|dental)\b",
    r"\bchupa(me|r)?\b",
    r"\bverga\b",
    r"\bpene\b",
    r"\bvagina\b",
    r"\bcono\b",
    r"\bconcha\b",
    r"\bhijueputa\b",
    r"\bhijo\s*de\s*puta\b",
    r"\bmalparid[oa]\b",
    r"\bgonorrea\b",
    r"\bputa\b",
    r"\bmarica\b",
    r"\bputo\b",
)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    return text


def has_dental_context(text: str) -> bool:
    norm = normalize_text(text)
    return any(term in norm for term in _DENTAL_CONTEXT)


def has_non_dental_body_part(text: str) -> bool:
    norm = normalize_text(text)
    # word-boundary-ish: avoid matching substrings inside unrelated words
    for part in _NON_DENTAL_PARTS:
        if re.search(rf"\b{re.escape(part)}\b", norm):
            return True
    return False


def is_non_dental_only(text: str) -> bool:
    """True when the message mentions a non-dental body part and no oral/dental cue."""
    return has_non_dental_body_part(text) and not has_dental_context(text)


def detect_disrespect(text: str) -> bool:
    """Detecta vulgaridad / broma ofensiva que no debe tratarse como urgencia ni cita."""
    if not text or not text.strip():
        return False
    norm = normalize_text(text)
    for pattern in _DISRESPECT_PATTERNS:
        if re.search(pattern, norm, re.IGNORECASE):
            return True
    return False


def detect_off_topic_non_dental(text: str) -> bool:
    """Dolor / cita sobre zona no odontológica (ej. rodilla)."""
    if not text or not text.strip():
        return False
    if not is_non_dental_only(text):
        return False
    norm = normalize_text(text)
    pain_or_booking = any(
        k in norm
        for k in (
            "duele", "dolor", "cita", "cits", "agendar", "urgencia", "emergencia",
            "ayuda", "medico",
        )
    )
    return pain_or_booking


def evaluate_content_guard(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Evalúa guards en orden de prioridad.

    Returns:
        (kind, reply_message) where kind is 'disrespect' | 'off_topic' | None.
    """
    if detect_disrespect(text):
        return "disrespect", MENSAJE_RESPETO
    if detect_off_topic_non_dental(text):
        return "off_topic", MENSAJE_FUERA_DE_ALCANCE
    return None, None
