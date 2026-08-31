import re
import unicodedata
import logging
from typing import Optional, Tuple
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)

MENSAJE_EMERGENCIA_URGENCIAS = (
    "🚨 *ATENCIÓN DE URGENCIA ODONTOLÓGICA / MÉDICA* 🚨\n\n"
    "Hemos detectado que puedes estar presentando una situación de emergencia que requiere atención presencial inmediata.\n\n"
    "⚠️ *Por tu seguridad, suspende el uso de este chat y acude de inmediato a un centro médico de urgencias o servicio de emergencias odontológicas más cercano.*\n\n"
    "📍 *Consultorio Nexus Odonto:* Cr 24 #35-12, Santander\n"
    "📞 *Línea directa / Urgencias:* +57 324 6030217\n\n"
    "Hemos escalado tu caso inmediatamente a nuestro equipo humano con prioridad *CRÍTICA*."
)

# Patrones regex normalizados para detección determinista de emergencias severas
EMERGENCY_PATTERNS = [
    # Criterios explícitos del DoD
    (r"\bdolor\s+insoportable\b", "DOLOR_INSOPORTABLE"),
    (r"\bsangrado\s+que\s+no\s+para\b", "SANGRADO_QUE_NO_PARA"),
    (r"\binfecci[oó]n\s+gigante\b", "INFECCION_GIGANTE"),

    # Variaciones de hemorragia / sangrado severo
    (r"\bno\s+para\s+de\s+sangrar\b", "SANGRADO_INCONTROLABLE"),
    (r"\bsangrado\s+(abundante|incontrolable|excesivo|profuso)\b", "SANGRADO_ABUNDANTE"),
    (r"\b(hemorragia|hemorragias)\b", "HEMORRAGIA"),
    (r"\bmucha\s+sangre\s+y\s+no\s+(para|se\s+detiene)\b", "SANGRADO_INCONTROLABLE"),

    # Variaciones de dolor insoportable / extremo
    (r"\bdolor\s+(inaguantable|extremo|intenso\s+e\s+insoportable)\b", "DOLOR_EXTREMO"),
    (r"\bno\s+(aguanto|soporto)\s+(el|este)\s+dolor\b", "DOLOR_INSOPORTABLE"),

    # Variaciones de infecciones o inflamación crítica
    (r"\bflem[oó]n\s+gigante\b", "FLEMON_GIGANTE"),
    (r"\bcelulitis\s+facial\b", "CELULITIS_FACIAL"),
    (r"\babsceso\s+(gigante|grave|severo)\b", "ABSCESO_GRAVE"),
    (r"\binfecci[oó]n\s+(grave|severa|muy\s+avanzada)\b", "INFECCION_SEVERA"),
    (r"\b(cara|cuello|mejilla)\s+(muy\s+hinchada|deforme)\s+y\s+(fiebre|asfixia|no\s+puedo\s+respirar)\b", "COMPROMISO_VIA_AEREA"),

    # Traumatismos maxilofaciales severos / avulsión
    (r"\bmand[ií]bula\s+(fracturada|rota|desencajada)\b", "FRACTURA_MANDIBULA"),
    (r"\b(diente|dientes)\s+(arrancado|arrancados)\s+de\s+ra[ií]z\b", "AVULSION_DENTAL"),
    (r"\bavulsi[oó]n\s+dental\b", "AVULSION_DENTAL"),
]


def normalize_text(text: str) -> str:
    """Normaliza texto eliminando tildes y caracteres diacríticos, convirtiendo a minúsculas."""
    if not text:
        return ""
    text = text.lower().strip()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    return text


def detect_emergency_keywords(text: str) -> Tuple[bool, Optional[str]]:
    """Detección rápida y determinista de palabras clave de emergencia severa."""
    if not text:
        return False, None

    # Verificación directa con texto original
    for pattern, reason in EMERGENCY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True, reason

    # Verificación sobre texto normalizado (sin acentos)
    norm_text = normalize_text(text)
    for pattern, reason in EMERGENCY_PATTERNS:
        clean_pattern = normalize_text(pattern)
        if re.search(clean_pattern, norm_text, re.IGNORECASE):
            return True, reason

    return False, None


async def classify_emergency_llm(text: str) -> Tuple[bool, Optional[str]]:
    """Clasificador LLM rápido (gpt-4o-mini) para evaluar triage de emergencia severa cuando no coincide por regex."""
    if not settings.openai_api_key or not text or len(text.strip()) < 8:
        return False, None

    evaluator_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=settings.openai_api_key,
    )

    prompt = (
        "Eres un clasificador de triage clínico para un consultorio odontológico. "
        "Tu única tarea es determinar si el mensaje del paciente describe una EMERGENCIA ODONTOLÓGICA O MÉDICA SEVERA "
        "que ponga en riesgo su integridad física inmediata (ejemplos: hemorragia incesante, dolor extremo/insoportable, "
        "infección facial masiva/flemón gigante con dificultad respiratoria o fiebre alta, traumatismo severo/fractura mandibular).\n\n"
        "NO clasifiques como EMERGENCIA consultas comunes como agendar citas, dolor leve/moderado tratable con cita previa, "
        "presupuestos, limpieza dental, preguntas sobre ortodoncia o dudas generales.\n\n"
        f"Mensaje del paciente:\n\"\"\"\n{text}\n\"\"\"\n\n"
        "Responde ESTRICTAMENTE con una sola palabra: EMERGENCIA o REGULAR."
    )

    try:
        response = await evaluator_llm.ainvoke([SystemMessage(content=prompt)])
        result = response.content.strip().upper()
        if "EMERGENCIA" in result:
            return True, "TRIAGE_CLINICO_LLM"
    except Exception as e:
        logger.warning(f"[Emergency Detector] Error en clasificador LLM: {e}")

    return False, None


async def detect_severe_emergency(text: str) -> Tuple[bool, Optional[str]]:
    """Función principal híbrida: evalúa primero por reglas deterministas y luego por LLM si es necesario."""
    is_emergency, reason = detect_emergency_keywords(text)
    if is_emergency:
        return True, reason
    return await classify_emergency_llm(text)
