"""Fast-path determinista para el inicio del flujo de agendamiento (sin LLM).

Cubre: intencion de agendar -> pedir cedula -> pedir nombre -> pedir tratamiento.
Cualquier mensaje ambiguo o de cancelar/reprogramar/clinica cae al grafo completo.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_BOOKING_STATE: Dict[str, dict] = {}

MSG_ASK_CEDULA = (
    "¡Con gusto te ayudo a *agendar tu cita* en *Nexus Odonto* 📅🦷!\n\n"
    "Para continuar, por favor escríbeme tu *número de cédula* 🆔."
)

MSG_ASK_NOMBRE = (
    "¡Muchas gracias por tu cédula! ✅\n\n"
    "Para registrar tu cita y crear tu ficha clínica, ¿me indicas tu "
    "*nombre completo* (nombre y apellido), por favor? 👤"
)

MSG_ASK_TRATAMIENTO = (
    "¡Perfecto, *{nombre}*! 😊\n\n"
    "¿Qué tratamiento o consulta necesitas? "
    "(por ejemplo: limpieza, valoración, resina, blanqueamiento…) 🦷✨"
)

_AGENDAR_RE = re.compile(
    r"\b("
    r"quiero\s+agendar|"
    r"deseo\s+agendar|"
    r"necesito\s+agendar|"
    r"me\s+(gustaria|gustaría)\s+agendar|"
    r"agendar(\s+una)?\s+cita|"
    r"sacar(\s+una)?\s+cita|"
    r"apartar(\s+una)?\s+cita|"
    r"programar(\s+una)?\s+cita|"
    r"reservar(\s+una)?\s+cita|"
    r"pedir(\s+una)?\s+cita|"
    r"cita\s+nueva|"
    r"nueva\s+cita"
    r")\b",
    re.IGNORECASE,
)

_CANCEL_MODIFY_RE = re.compile(
    r"\b(cancelar|anular|modificar|reprogramar|cambiar\s+(mi\s+)?cita|ver\s+mis\s+citas|"
    r"consultar\s+(mis\s+)?citas)\b",
    re.IGNORECASE,
)

_NAME_BLOCK_RE = re.compile(
    r"\b(cancelar|modificar|reprogramar|asesor|humano|precio|servicio|horario|"
    r"doctor|cita|agendar|cédula|cedula|urgencia|dolor)\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    t = (text or "").lower().strip()
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def is_cedula_only(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    digits = re.sub(r"\D", "", raw)
    chars = re.sub(r"[\s.\-]", "", raw)
    return bool(digits) and len(digits) == len(chars) and 6 <= len(digits) <= 12


def extract_cedula(text: str) -> Optional[str]:
    if not is_cedula_only(text):
        return None
    return re.sub(r"\D", "", text.strip())


def is_agendar_intent(text: str) -> bool:
    norm = _normalize(text)
    if not norm:
        return False
    if _CANCEL_MODIFY_RE.search(norm):
        return False
    return bool(_AGENDAR_RE.search(norm))


def is_booking_skip_embed(text: str) -> bool:
    if is_cedula_only(text):
        return True
    norm = _normalize(text)
    if not norm:
        return False
    if is_agendar_intent(text):
        return True
    return bool(re.search(r"\b(agendar|cita|citas|apartar|programar|reservar)\b", norm))


def is_name_like(text: str) -> bool:
    raw = (text or "").strip()
    if not raw or is_cedula_only(raw):
        return False
    if _NAME_BLOCK_RE.search(raw):
        return False
    words = [w for w in re.split(r"\s+", raw) if w]
    if not (1 <= len(words) <= 5):
        return False
    if len(raw) > 60:
        return False
    letters = re.sub(r"[^a-záéíóúñA-ZÁÉÍÓÚÑ\s]", "", raw)
    if len(letters.replace(" ", "")) < 3:
        return False
    return bool(re.search(r"[a-záéíóúñA-ZÁÉÍÓÚÑ]", raw))


def get_booking_state(phone: str) -> Optional[dict]:
    return _BOOKING_STATE.get(phone)


def set_booking_state(phone: str, **kwargs) -> dict:
    state = _BOOKING_STATE.get(phone) or {}
    state.update(kwargs)
    _BOOKING_STATE[phone] = state
    return state


def clear_booking_state(phone: str) -> None:
    _BOOKING_STATE.pop(phone, None)


def infer_awaiting_from_bot_text(last_bot: str) -> Optional[str]:
    low = (last_bot or "").lower()
    if not low:
        return None
    if "nombre completo" in low or ("nombre" in low and "apellido" in low):
        return "awaiting_nombre"
    if "cédula" in low or "cedula" in low:
        if "gracias por tu cédula" in low or "gracias por tu cedula" in low:
            return "awaiting_nombre"
        return "awaiting_cedula"
    return None


def try_booking_fastpath(
    phone: str,
    message: str,
    last_bot_text: str = "",
) -> Optional[Tuple[str, dict]]:
    text = (message or "").strip()
    if not text:
        return None

    if _CANCEL_MODIFY_RE.search(_normalize(text)):
        clear_booking_state(phone)
        return None

    state = get_booking_state(phone) or {}
    step = state.get("step")
    if not step and last_bot_text:
        inferred = infer_awaiting_from_bot_text(last_bot_text)
        if inferred:
            step = inferred
            state = set_booking_state(phone, step=inferred)

    if is_agendar_intent(text) and step not in (
        "awaiting_cedula",
        "awaiting_nombre",
        "awaiting_tratamiento",
    ):
        new_state = set_booking_state(phone, step="awaiting_cedula", cedula=None, nombre=None)
        logger.info(f"[BookingFastPath] agendar->pedir_cedula phone={phone}")
        return MSG_ASK_CEDULA, new_state

    if step == "awaiting_cedula" or (
        step is None
        and is_cedula_only(text)
        and infer_awaiting_from_bot_text(last_bot_text) == "awaiting_cedula"
    ):
        cedula = extract_cedula(text)
        if cedula:
            new_state = set_booking_state(
                phone, step="awaiting_nombre", cedula=cedula, nombre=None
            )
            logger.info(f"[BookingFastPath] cedula->pedir_nombre phone={phone}")
            return MSG_ASK_NOMBRE, new_state
        if step == "awaiting_cedula":
            return (
                "⚠️ Ese valor no parece una cédula válida (debe tener entre 6 y 12 dígitos). "
                "Por favor escríbeme solo tu *número de cédula* 🆔.",
                state,
            )

    if step == "awaiting_nombre":
        if is_name_like(text):
            nombre = " ".join(text.split()).title()
            new_state = set_booking_state(
                phone, step="awaiting_tratamiento", nombre=nombre
            )
            logger.info(f"[BookingFastPath] nombre->pedir_tratamiento phone={phone}")
            return MSG_ASK_TRATAMIENTO.format(nombre=nombre), new_state
        if is_cedula_only(text):
            cedula = extract_cedula(text)
            new_state = set_booking_state(phone, step="awaiting_nombre", cedula=cedula)
            return MSG_ASK_NOMBRE, new_state
        return None

    return None
