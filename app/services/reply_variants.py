"""Variantes naturales para los avisos fijos que no son conversación (reinicio, escalamiento,
medios/audio, spam, inactividad, falta de respeto, portal). La conversación la redacta el LLM.

Cada aviso tiene varias redacciones con el mismo significado; `pick`
elige una al azar evitando repetir las usadas recientemente con el mismo teléfono.

Reglas de redacción: español de Colombia, tono de recepción humana, sin emojis,
nunca revelar que es un bot, sin formas "o/a".
"""

from __future__ import annotations

import random
import threading
from collections import OrderedDict, deque
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

_MAX_PHONES = 5000
_HISTORY: "OrderedDict[str, dict[str, deque[int]]]" = OrderedDict()
_LOCK = threading.Lock()
_GLOBAL_KEY = "_global"


def _now_bogota() -> datetime:
    try:
        from app.core.config import settings

        tz = ZoneInfo(settings.reminder_timezone or "America/Bogota")
    except Exception:
        try:
            tz = ZoneInfo("America/Bogota")
        except Exception:
            tz = timezone(timedelta(hours=-5))
    return datetime.now(tz)


def momento_del_dia(now: Optional[datetime] = None) -> str:
    hour = (now or _now_bogota()).hour
    if 5 <= hour < 12:
        return "manana"
    if 12 <= hour < 19:
        return "tarde"
    return "noche"


def saludo_del_momento(now: Optional[datetime] = None) -> str:
    return {"manana": "Buenos días", "tarde": "Buenas tardes", "noche": "Buenas noches"}[
        momento_del_dia(now)
    ]


def despedida_del_momento(now: Optional[datetime] = None) -> str:
    return {"manana": "un buen día", "tarde": "una buena tarde", "noche": "una buena noche"}[
        momento_del_dia(now)
    ]


def primer_nombre(nombre: Optional[str]) -> str:
    parts = (nombre or "").strip().split()
    return parts[0].title() if parts else ""


def _recent_indexes(phone: str, key: str, window: int) -> deque:
    with _LOCK:
        per_phone = _HISTORY.get(phone)
        if per_phone is None:
            per_phone = {}
            _HISTORY[phone] = per_phone
            while len(_HISTORY) > _MAX_PHONES:
                _HISTORY.popitem(last=False)
        else:
            _HISTORY.move_to_end(phone)
        recent = per_phone.get(key)
        if recent is None or recent.maxlen != window:
            recent = deque(recent or (), maxlen=window)
            per_phone[key] = recent
        return recent


def pick(
    key: str,
    variants: Sequence[str],
    *,
    phone: Optional[str] = None,
    nombre: Optional[str] = None,
    avoid: Optional[str] = None,
    **fmt: str,
) -> str:
    """Elige una variante de `variants` sin repetir las últimas usadas para `phone`.

    Placeholders disponibles en cada variante: {nombre} (", Ana" o ""),
    {saludo} ("Buenos días"...), {saludo_min}, {despedida} ("un buen día"...) y
    cualquier kwarg extra de `fmt`. `avoid` excluye una variante literal (p. ej. el último
    mensaje del hilo tras un reinicio del proceso).
    """
    if not variants:
        return ""
    window = max(0, min(len(variants) - 1, len(variants) // 2 + 1))
    phone_key = (phone or "").strip() or _GLOBAL_KEY
    recent = _recent_indexes(phone_key, key, window) if window else deque()
    all_idx = [i for i in range(len(variants)) if variants[i] != avoid] or list(range(len(variants)))
    candidates = [i for i in all_idx if i not in recent] or all_idx
    idx = random.choice(candidates)
    if window:
        with _LOCK:
            recent.append(idx)

    first = primer_nombre(nombre)
    now = _now_bogota()
    saludo = saludo_del_momento(now)
    values = {
        "nombre": f", {first}" if first else "",
        "saludo": saludo,
        "saludo_min": saludo.lower(),
        "despedida": despedida_del_momento(now),
        **fmt,
    }
    return variants[idx].format(**values)


def reset_history(phone: Optional[str] = None) -> None:
    with _LOCK:
        if phone is None:
            _HISTORY.clear()
        else:
            _HISTORY.pop(phone, None)


# ─────────────────────────────────────────────────────────────────────────────
# Pools
# ─────────────────────────────────────────────────────────────────────────────

ESCALAMIENTO = (
    "Entiendo. Un asesor de la clínica revisará tu solicitud y te contactará pronto.",
    "Listo, ya le paso tu caso a un asesor de la clínica; te escribe en un momento.",
    "Claro que sí. Un asesor de la clínica va a revisar tu solicitud y te responde pronto.",
    "Entendido, dejo tu caso con un asesor de la clínica para que te contacte en breve.",
)

MEDIOS_NO_SOPORTADOS = (
    "Por aquí no alcanzo a ver fotos, videos ni documentos.\n\n"
    "¿Me cuentas por texto o nota de voz qué necesitas (síntoma, tratamiento u orden médica)? "
    "Así te ayudo mejor.",
    "Qué pena, por este chat no puedo abrir archivos ni fotos.\n\n"
    "¿Me escribes o me mandas una nota de voz contándome qué necesitas?",
    "No me carga el archivo por aquí.\n\n"
    "Si me cuentas por texto o audio qué necesitas, te ayudo de una.",
)

REINICIO = (
    "Listo, empezamos de nuevo. Hola, estás hablando con *Nexus Odonto*. ¿En qué te ayudo?",
    "Perfecto, arrancamos desde cero. ¿Qué necesitas?",
    "Listo, conversación nueva. Cuéntame en qué te puedo ayudar.",
    "Dale, empezamos otra vez. ¿Qué se te ofrece?",
)

AUDIO_NO_ENTENDIDO = (
    "No alcancé a escuchar bien tu nota de voz.\n\n"
    "¿La grabas otra vez en un lugar más silencioso, o me lo escribes por texto?",
    "Qué pena, el audio no se escuchó bien.\n\n¿Me lo repites o me escribes?",
    "No logré entender la nota de voz.\n\n¿Me la mandas de nuevo o me cuentas por texto?",
)

AUDIO_ERROR = (
    "Se nos complicó procesar tu nota de voz.\n\n"
    "¿La intentas de nuevo o me cuentas por texto lo que necesitas?",
    "No pude abrir tu audio.\n\n¿Me lo envías otra vez o me escribes?",
    "Tu nota de voz no me cargó bien.\n\n¿La reenvías o me lo escribes por aquí?",
)

INACTIVIDAD_CITA_INCOMPLETA = (
    "Como no alcanzamos a completar los datos, dejamos el agendamiento por ahora.\n\n"
    "Cuando quieras retomarlo o consultar algo de la clínica, escríbeme y con gusto te ayudo.",
    "Dejamos la cita en pausa porque no alcanzamos a terminar.\n\n"
    "Cuando quieras la retomamos, solo escríbeme.",
    "Veo que quedó pendiente el agendamiento; lo dejamos por ahora.\n\n"
    "Si quieres terminarlo más tarde, me escribes y seguimos.",
)

INACTIVIDAD_GENERAL = (
    "Te dejo por ahora por si estás ocupado.\n\n"
    "Cuando quieras agendar, consultar un servicio o resolver una duda, aquí estoy.",
    "Parece que andas con muchas cosas, te dejo por ahora.\n\n"
    "Cuando necesites algo de la clínica, me escribes.",
    "Por ahora cierro la conversación.\n\n"
    "Si luego necesitas una cita o tienes alguna pregunta, aquí te atendemos.",
)

SPAM = (
    "Estás enviando muchos mensajes seguidos. "
    "Espera un momento y escríbeme tu consulta en un solo mensaje para atenderte bien.",
    "Me están llegando muchos mensajes muy rápido. "
    "Dame un momentico y cuéntame todo en un solo mensaje, porfa.",
    "Uy, van muchos mensajes seguidos. Mejor escríbeme lo que necesitas en uno solo y te ayudo.",
)

SOLO_ODONTOLOGIA = (
    "Solo puedo ayudarte con temas odontológicos de Nexus Odonto.",
    "Por aquí te ayudo solo con temas de la clínica: citas, tratamientos y dudas dentales.",
    "Eso no lo manejo; te puedo ayudar con citas, precios o dudas odontológicas.",
)

PORTAL_RECORDATORIO = (
    "También puedes consultar tu cita en la *plataforma virtual*:\n{url}",
    "Si quieres revisarla, está también en la *plataforma virtual*:\n{url}",
    "Recuerda que tus citas las ves en la *plataforma virtual*:\n{url}",
)

PORTAL_PRIMERA_VEZ = (
    "También puedes ver tu cita en la *plataforma virtual*:\n{url}\n"
    "Tu *usuario* es tu número de cédula y la *contraseña* también es tu número de cédula "
    "(acceso temporal). Al entrar, cámbiala por tu seguridad; desde aquí no podemos modificar contraseñas.",
    "Tu cita también queda en la *plataforma virtual*:\n{url}\n"
    "Para entrar, el *usuario* y la *contraseña* son tu número de cédula (acceso temporal). "
    "Apenas ingreses, cámbiala; desde este chat no podemos cambiar contraseñas.",
    "Puedes revisarla cuando quieras en la *plataforma virtual*:\n{url}\n"
    "Entras con tu número de cédula como *usuario* y también como *contraseña* temporal. "
    "Te recomiendo cambiarla al ingresar; por aquí no manejamos contraseñas.",
)

PORTAL_SOLO_CREDENCIALES = (
    "Tu *usuario* es tu número de cédula y la *contraseña* también es tu número de cédula "
    "(acceso temporal). Al entrar, cámbiala por tu seguridad; desde aquí no podemos modificar contraseñas.",
    "Para entrar, el *usuario* y la *contraseña* son tu número de cédula (acceso temporal). "
    "Apenas ingreses, cámbiala; desde este chat no podemos cambiar contraseñas.",
    "Entras con tu número de cédula como *usuario* y también como *contraseña* temporal. "
    "Te recomiendo cambiarla al ingresar; por aquí no manejamos contraseñas.",
)
