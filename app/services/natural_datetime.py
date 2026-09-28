"""Fecha y hora en español coloquial (con errores de tipeo) para agendar sin LLM.

Ejemplos: "hoy", "mañana a las 3", "pasado mañana", "el domingo", "el próximo lunes",
"15/10", "15 de octubre", "9am", "3 pm", "en la tarde", "a las 10 y media".
Funciones puras: no consultan agenda ni validan jornada.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

WEEKDAYS = {"lunes": 0, "martes": 1, "miercoles": 2, "jueves": 3, "viernes": 4, "sabado": 5, "domingo": 6}
WEEKDAY_NAMES = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
    "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}
MONTH_NAMES = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
               "septiembre", "octubre", "noviembre", "diciembre")

# Words worth typo-correcting; everything else is left untouched.
_VOCAB = tuple(WEEKDAYS) + tuple(MONTHS) + (
    "manana", "tarde", "noche", "pasado", "proximo", "proxima", "siguiente", "mediodia",
    "media", "cuarto", "semana", "temprano",
)
_NUM_WORDS = {
    "una": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
    "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12,
}

_ASAP_RE = re.compile(
    r"lo (mas|antes) pronto|lo antes posible|cuanto antes|lo mas rapido|apenas (haya|tengan)"
    r"|cuando (haya|tengan|puedan|sea|me den)|el que sea|la que sea|cualquier (dia|hora|horario|turno)"
    r"|primer (turno|espacio|cupo)|mas cercan|disponibilidad|disponibles?"
    r"|que (dias|horarios?|horas|turnos|espacios|cupos) (hay|tienen|tienes|tiene|quedan|manejan)"
)
_PERIOD_MANANA_RE = re.compile(r"\b(en|por|de|a|para) (la|las) mananas?\b|\btemprano\b|\ba primera hora\b")
_PERIOD_TARDE_RE = re.compile(r"\b(en|por|de|a|para) (la|las) (tardes?|noches?)\b|\bdespues de almuerzo\b")
_NEXT_WEEK_RE = re.compile(r"\b(la )?(proxima|siguiente|otra) semana\b|\bsemana que viene\b")
_WEEKDAY_RE = re.compile(
    r"\b(?:(?:el|este|esta|para el|pal)\s+)?(?P<pre>proximo|proxima|siguiente)?\s*"
    r"(?P<wd>lunes|martes|miercoles|jueves|viernes|sabado|domingo)"
    r"(?:\s+(?P<post>proximo|que viene|siguiente|de la otra semana))?\b"
)
_DMY_RE = re.compile(r"(?<![\d:])(\d{1,2})\s*[/-]\s*(\d{1,2})(?:\s*[/-]\s*(\d{2,4}))?(?![\d:])")
_DAY_MONTH_RE = re.compile(r"\b(\d{1,2})\s+(?:de\s+)?(" + "|".join(MONTHS) + r")\b")
_MONTH_DAY_RE = re.compile(r"\b(" + "|".join(MONTHS) + r")\s+(\d{1,2})\b")
_DAY_ONLY_RE = re.compile(
    r"\b(?:el|para el|pal)\s+(?:dia\s+)?(\d{1,2})\b(?!\s*(?:[:.h]\d|am\b|pm\b|a\s*m\b|p\s*m\b|de la|y media|y cuarto))"
)
_TIME_RE = re.compile(
    r"(?P<pre>\b(?:a\s+las?|las|la|tipo|tipo\s+las?|como\s+a\s+las?|sobre\s+las?|desde\s+las?"
    r"|despues\s+de\s+las?|antes\s+de\s+las?|a\s+eso\s+de\s+las?)\s+)?"
    r"\b(?P<h>\d{1,2}|" + "|".join(_NUM_WORDS) + r")"
    r"(?:\s*[:.h]\s*(?P<m>\d{2}))?"
    r"(?:\s+y\s+(?P<frac>media|cuarto|\d{1,2}))?"
    r"\s*(?P<suf>a\.?\s*m\b\.?|p\.?\s*m\b\.?|de\s+la\s+manana|de\s+la\s+tarde|de\s+la\s+noche|del\s+mediodia|en\s+punto)?"
    r"(?![\d/\-])"
)


@dataclass(frozen=True)
class DateTimeRequest:
    fecha: Optional[date] = None
    hora: Optional[tuple[int, int]] = None
    periodo: Optional[str] = None  # "manana" | "tarde"
    lo_antes_posible: bool = False

    @property
    def empty(self) -> bool:
        return not (self.fecha or self.hora or self.periodo or self.lo_antes_posible)

    @property
    def hhmm(self) -> Optional[str]:
        return f"{self.hora[0]:02d}:{self.hora[1]:02d}" if self.hora else None


def normalize(text: str) -> str:
    t = (text or "").lower()
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^\w\s:/.\-]", " ", t)
    t = re.sub(r"(?<!\d)[.\-](?!\d)", " ", t)
    words = []
    for w in t.split():
        if len(w) >= 4 and w.isalpha() and w not in _VOCAB:
            close = difflib.get_close_matches(w, _VOCAB, n=1, cutoff=0.8)
            if close:
                w = close[0]
        words.append(w)
    return " ".join(words)


def _safe_date(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _future_day_month(day: int, month: int, today: date, year: Optional[int] = None) -> Optional[date]:
    if year is not None:
        return _safe_date(year if year > 99 else 2000 + year, month, day)
    found = _safe_date(today.year, month, day)
    if found and found < today:
        found = _safe_date(today.year + 1, month, day)
    return found


def _parse_date(norm: str, today: date) -> tuple[Optional[date], str]:
    """(fecha, texto sin la parte de fecha) para no confundir días con horas."""
    m = re.search(r"\bpasado\s+manana\b", norm)
    if m:
        return today + timedelta(days=2), norm[: m.start()] + " " + norm[m.end():]
    m = _DMY_RE.search(norm)
    if m:
        d, mo = int(m.group(1)), int(m.group(2))
        y = int(m.group(3)) if m.group(3) else None
        found = _future_day_month(d, mo, today, y) if 1 <= mo <= 12 else None
        if found:
            return found, norm[: m.start()] + " " + norm[m.end():]
    for rx, day_first in ((_DAY_MONTH_RE, True), (_MONTH_DAY_RE, False)):
        m = rx.search(norm)
        if m:
            day = int(m.group(1) if day_first else m.group(2))
            month = MONTHS[m.group(2) if day_first else m.group(1)]
            found = _future_day_month(day, month, today)
            if found:
                return found, norm[: m.start()] + " " + norm[m.end():]
    m = _WEEKDAY_RE.search(norm)
    if m:
        delta = (WEEKDAYS[m.group("wd")] - today.weekday()) % 7
        if delta == 0 or (m.group("post") == "de la otra semana" and delta < 7):
            delta = 7 if delta == 0 else delta + 7
        return today + timedelta(days=delta), norm[: m.start()] + " " + norm[m.end():]
    if re.search(r"\bmanana\b", norm):
        stripped = re.sub(r"\bmanana\b", " ", norm, count=1)
        return today + timedelta(days=1), stripped
    if re.search(r"\bhoy\b", norm):
        return today, re.sub(r"\bhoy\b", " ", norm)
    m = _NEXT_WEEK_RE.search(norm)
    if m:
        return today + timedelta(days=7 - today.weekday()), norm[: m.start()] + " " + norm[m.end():]
    m = _DAY_ONLY_RE.search(norm)
    if m:
        day = int(m.group(1))
        if 1 <= day <= 31:
            month, year = today.month, today.year
            found = _safe_date(year, month, day)
            if not found or found < today:
                month, year = (1, year + 1) if month == 12 else (month + 1, year)
                found = _safe_date(year, month, day)
            if found:
                return found, norm[: m.start()] + " " + norm[m.end():]
    return None, norm


def _parse_time(norm: str, periodo: Optional[str], whole: str) -> Optional[tuple[int, int]]:
    if re.search(r"\bmediodia\b", norm) and not re.search(r"\d", norm):
        return 12, 0
    for m in _TIME_RE.finditer(norm):
        raw_h = m.group("h")
        pre, suf, mins, frac = m.group("pre"), (m.group("suf") or ""), m.group("m"), m.group("frac")
        is_word = not raw_h.isdigit()
        if is_word and not pre:
            continue
        bare_message = re.fullmatch(r"\s*(a las?\s+)?\d{1,2}\s*", whole) is not None
        if not (pre or suf or mins or frac or bare_message):
            continue
        hour = _NUM_WORDS[raw_h] if is_word else int(raw_h)
        minute = int(mins) if mins else 0
        if frac == "media":
            minute = 30
        elif frac == "cuarto":
            minute = 15
        elif frac and frac.isdigit():
            minute = int(frac)
        if hour > 23 or minute > 59:
            continue
        suf_c = suf.replace(" ", "").replace(".", "")
        if suf_c in ("pm", "delatarde", "delanoche") and hour < 12:
            hour += 12
        elif suf_c in ("am", "delamanana"):
            if hour == 12:
                hour = 0
        elif suf_c == "delmediodia":
            hour = 12 if hour in (12, 0) else hour
        elif hour < 12:
            if periodo == "tarde" or (periodo is None and 1 <= hour <= 6):
                hour += 12
        return hour, minute
    return None


def parse_datetime_request(text: str, today: date) -> DateTimeRequest:
    norm = normalize(text)
    if not norm:
        return DateTimeRequest()
    periodo = None
    if _PERIOD_MANANA_RE.search(norm):
        periodo = "manana"
        norm_for_date = _PERIOD_MANANA_RE.sub(" ", norm)
    else:
        norm_for_date = norm
    if _PERIOD_TARDE_RE.search(norm):
        periodo = "tarde"
    fecha, rest = _parse_date(norm_for_date, today)
    hora = _parse_time(rest, periodo, norm)
    if hora and periodo is None:
        periodo = "manana" if hora[0] < 12 else "tarde"
    asap = bool(_ASAP_RE.search(norm)) and not (fecha or hora)
    return DateTimeRequest(fecha=fecha, hora=hora, periodo=periodo, lo_antes_posible=asap)


def dia_humano(d: date, today: date) -> str:
    """'hoy', 'mañana martes 29', 'el sábado 3 de octubre'."""
    if d == today:
        return "hoy"
    nombre = WEEKDAY_NAMES[d.weekday()]
    if d == today + timedelta(days=1):
        return f"mañana {nombre} {d.day}"
    return f"el {nombre} {d.day} de {MONTH_NAMES[d.month - 1]}"


def hora_humana(hhmm: str) -> str:
    h, m = (int(x) for x in hhmm.split(":")[:2])
    suf = "AM" if h < 12 else "PM"
    return f"{h % 12 or 12}:{m:02d} {suf}"
