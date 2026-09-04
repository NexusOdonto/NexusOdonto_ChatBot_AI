import re
import unicodedata
import logging
from typing import Optional, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from app.core.llm_factory import get_evaluator_llm, extract_text_content
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
    (r"\bfractura\s+mandibular\b", "FRACTURA_MANDIBULAR"),
    (r"\btraumatismo\s+severo\b", "TRAUMATISMO_SEVERO"),
    (r"\bflemon\s+con\s+fiebre\b", "FLEMON_CON_FIEBRE"),
    (r"\bdificultad\s+para\s+respirar\b", "DIFICULTAD_RESPIRATORIA"),
    (r"\basfixia\b", "ASFIXIA"),
    (r"\bno\s+puedo\s+abrir\s+la\s+boca\s+del\s+dolor\b", "TRISMUS_SEVERO"),
    (r"\bhinchaz[oó]n\s+en\s+el\s+cuello\b", "EDEMA_CERVICAL"),
    (r"\bpus\s+abundante\b", "SUPURACION_SEVERA"),

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
    """Clasificador LLM rápido para evaluar triage de emergencia severa cuando no coincide por regex."""
    has_api_key = (
        (settings.llm_provider.lower() == "gemini" and bool(settings.gemini_api_key))
        or (settings.llm_provider.lower() == "openai" and bool(settings.openai_api_key))
    )
    if not has_api_key or not text or len(text.strip()) < 8:
        return False, None

    try:
        evaluator_llm = get_evaluator_llm()
    except Exception as exc:
        logger.warning(f"[Emergency Detector] No se pudo instanciar LLM: {exc}")
        return False, None

    sys_prompt = (
        "Eres un clasificador de triage clínico para un consultorio odontológico. "
        "Tu única tarea es determinar si el mensaje del paciente describe una EMERGENCIA ODONTOLÓGICA O MÉDICA SEVERA "
        "que ponga en riesgo su integridad física inmediata (ejemplos: hemorragia incesante, dolor extremo/insoportable, "
        "infección facial masiva/flemón gigante con dificultad respiratoria o fiebre alta, traumatismo severo/fractura mandibular).\n\n"
        "NO clasifiques como EMERGENCIA consultas comunes como agendar citas, dolor leve/moderado tratable con cita previa, "
        "presupuestos, limpieza dental, preguntas sobre ortodoncia o dudas generales.\n\n"
        "Responde ESTRICTAMENTE con una sola palabra: EMERGENCIA o REGULAR."
    )

    try:
        response = await evaluator_llm.ainvoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=f"Mensaje del paciente:\n\"\"\"\n{text}\n\"\"\"")
        ])
        result = extract_text_content(response.content).strip().upper()
        if "EMERGENCIA" in result:
            return True, "TRIAGE_CLINICO_LLM"
    except Exception as e:
        logger.warning(f"[Emergency Detector] Error en clasificador LLM: {e}")

    return False, None


# Términos sospechosos que justifican activar el triage LLM si el regex directo no hizo match
SUSPICIOUS_EMERGENCY_TERMS = {
    "dolor", "sangr", "infect", "urgenc", "emerg", "hinch", "asfix", "respir",
    "fractur", "trauma", "grave", "absces", "flemon", "arranc", "desmay",
    "fiebre", "morir", "auxilio", "ayuda", "insoportable", "inaguantable",
    "hemorrag", "accidente", "golpe", "partio", "parti"
}


async def detect_severe_emergency(text: str) -> Tuple[bool, Optional[str]]:
    """Función principal híbrida de alta eficiencia:
    1. Evalúa primero por reglas deterministas (0 tokens).
    2. Si no coincide, solo llama al LLM clasificador si el texto contiene términos clínicos o de sospecha de urgencia.
    """
    is_emergency, reason = detect_emergency_keywords(text)
    if is_emergency:
        return True, reason

    # Pre-filtro: Si el mensaje no contiene ningún indicio de urgencia (ej. "quiero una cita", "cuál es el horario"),
    # evitamos gastar tokens llamando a OpenAI.
    norm_text = normalize_text(text)
    has_suspicious_term = any(term in norm_text for term in SUSPICIOUS_EMERGENCY_TERMS)
    if not has_suspicious_term:
        return False, None

    return await classify_emergency_llm(text)

