"""Guards de contenido para recepción WhatsApp: respeto y alcance odontológico.

Detecta (0 tokens) bromas vulgares / falta de respeto y dolores fuera de odontología
para responder con tono humano de recepción, sin agendar ni escalar urgencias.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Optional, Sequence, Tuple

# Fallback legacy (tests / imports externos). Preferir build_respect_reply.
MENSAJE_RESPETO = (
    "Oye, te pedimos escribirnos con respeto. "
    "Si de verdad tienes una molestia dental o quieres cita, cuéntanos con claridad."
)

MENSAJE_FUERA_DE_ALCANCE = (
    "En *Nexus Odonto* solo atendemos *salud oral* (dientes, encías y boca). "
    "Por un dolor en otra parte del cuerpo te conviene consultar un médico general "
    "o el especialista correspondiente. "
    "Si tienes una molestia dental o quieres una cita odontológica, aquí te ayudamos."
)

# Respuestas cortas y naturales (recepción humana). Una idea, sin sermón largo.
_RESPETO_INSULTO = (
    "Oye, te pedimos escribirnos con respeto, por favor.",
    "Tranqui, pero aquí atendemos con respeto. ¿En qué te puedo ayudar de verdad?",
    "Por favor háblanos con respeto. Si necesitas algo de la clínica, dime claro.",
    "Hey, respetemos el chat. Si tienes una duda dental o quieres cita, aquí estoy.",
)

_RESPETO_BROMA = (
    "Jaja, pero por aquí solo atendemos temas de dientes y boca de verdad.",
    "Esa broma no aplica para cita. Si te duele una muela de verdad, cuéntame bien.",
    "Con respeto: eso no es un caso dental. Si tienes molestia oral real, te oriento.",
    "Ok, pero en Nexus Odonto solo salud oral. ¿Te duele algo en la boca de verdad?",
)

_RESPETO_DISCULPA_BROMA = (
    "Tranqui por el perdón. Igual eso no es dental; si te duele una muela de verdad, cuéntame.",
    "Listo, sin problema. Aquí solo salud oral: si hay dolor dental real, dime qué sientes.",
    "Dale, gracias. Para ayudarte necesito un caso dental de verdad, no la broma.",
    "Ok, gracias. Si es molestia en dientes o encías, te ayudo; si no, no aplica cita aquí.",
)

_RESPETO_SEGUIDO = (
    "Sigamos en serio, porfa. ¿Tienes alguna molestia dental o quieres agendar?",
    "Mejor vamos al grano: ¿dolor dental real o quieres una cita?",
    "Con gusto te ayudo con odontología. Dime claro qué necesitas.",
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

_VULGAR_JOKE_PATTERNS = (
    r"\bmuela\s+del\s+ano\b",
    r"\bdiente\s+del\s+ano\b",
    r"\bmuela\s+del\s+culo\b",
    r"\bdiente\s+del\s+culo\b",
    r"\bmuela\s+del\s+trasero\b",
    r"\b(dolor|duele).{0,40}\b(ano|culo|trasero)\b",
    r"\b(ano|culo)\b.{0,40}\b(muela|diente|dental)\b",
)

_APOLOGY_PATTERNS = (
    r"\bperdon\b",
    r"\bperdona\b",
    r"\bperdoname\b",
    r"\bdisculpa\b",
    r"\bdisculpame\b",
    r"\blo\s+siento\b",
    r"\bsorry\b",
    r"\bmea\s+culpa\b",
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


def has_apology(text: str) -> bool:
    if not text or not text.strip():
        return False
    norm = normalize_text(text)
    return any(re.search(p, norm) for p in _APOLOGY_PATTERNS)


def is_vulgar_joke(text: str) -> bool:
    if not text or not text.strip():
        return False
    norm = normalize_text(text)
    return any(re.search(p, norm) for p in _VULGAR_JOKE_PATTERNS)


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


def _pick_variant(
    pool: Sequence[str],
    *,
    seed_text: str,
    avoid: Optional[str] = None,
    salt: str = "",
) -> str:
    """Elige una variante estable por hash; evita repetir `avoid` si hay otras opciones."""
    candidates = [m for m in pool if m != avoid] or list(pool)
    digest = hashlib.sha1(f"{salt}|{seed_text}".encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(candidates)
    return candidates[idx]


def classify_disrespect_tone(text: str) -> str:
    """Subtipo de falta de respeto para tono conversacional.

    Returns:
        'apology_joke' | 'vulgar_joke' | 'insult'
    """
    if has_apology(text) and detect_disrespect(text):
        return "apology_joke"
    if is_vulgar_joke(text):
        return "vulgar_joke"
    return "insult"


def build_respect_reply(
    text: str,
    *,
    avoid_reply: Optional[str] = None,
    prior_respect: bool = False,
) -> str:
    """Arma una respuesta corta y natural de recepción (sin el mismo párrafo fijo)."""
    tone = classify_disrespect_tone(text)
    if tone == "apology_joke":
        pool = _RESPETO_DISCULPA_BROMA
    elif tone == "vulgar_joke":
        pool = _RESPETO_BROMA
    elif prior_respect:
        pool = _RESPETO_SEGUIDO
    else:
        pool = _RESPETO_INSULTO

    return _pick_variant(
        pool,
        seed_text=normalize_text(text),
        avoid=avoid_reply,
        salt=tone,
    )


def evaluate_content_guard(
    text: str,
    *,
    avoid_reply: Optional[str] = None,
    prior_respect: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """Evalúa guards en orden de prioridad.

    Returns:
        (kind, reply_message) where kind is 'disrespect' | 'off_topic' | None.
    """
    if detect_disrespect(text):
        return "disrespect", build_respect_reply(
            text,
            avoid_reply=avoid_reply,
            prior_respect=prior_respect,
        )
    if detect_off_topic_non_dental(text):
        return "off_topic", MENSAJE_FUERA_DE_ALCANCE
    return None, None
