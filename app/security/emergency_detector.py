import re
import unicodedata
import logging
from typing import Optional, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from app.core.llm_factory import get_evaluator_llm, extract_text_content
from app.core.config import settings
from app.security.content_guard import has_dental_context, is_non_dental_only, normalize_text

logger = logging.getLogger(__name__)

MENSAJE_EMERGENCIA_URGENCIAS = (
    "*Urgencia odontológica*\n\n"
    "Detectamos una posible emergencia que requiere atención presencial inmediata.\n\n"
    "Por tu seguridad, suspende este chat y acude de inmediato a un centro de urgencias "
    "o al servicio de emergencias odontológicas más cercano.\n\n"
    "*Consultorio Nexus Odonto:* Calle 100 # 15-20, Centro Médico Odontológico\n"
    "*Línea / Urgencias:* +57 324 6030217\n\n"
    "Escalamos tu caso al equipo humano con prioridad crítica."
)

# Solo emergencias odontológicas / maxilofaciales reales.
# El dolor genérico ("me duele mucho") NO es emergencia por sí solo.
EMERGENCY_PATTERNS = [
    # Alerta / urgencia explícita con ancla dental o médica oral
    (r"\b(alerta\s+odontologica|urgencia\s+odontologica|emergencia\s+odontologica)\b", "URGENCIA_ODONTOLOGICA"),
    (r"\b(alerta\s+medica|estado\s+de\s+alerta|emergencia\s+medica)\b", "ESTADO_DE_ALERTA"),

    # Dolor extremo anclado a cavidad oral / dental
    (r"\b(me\s+)?duele\s+(mucho|demasiado|bastante)\s+(el|la|los|las)?\s*(diente|dientes|muela|muelas|encia|encias|boca|mandibula)\b", "DOLOR_AGUDO_DENTAL"),
    (r"\bdolor\s+(fuerte|severo|agudo|terrible|horrible|inaguantable|muy\s+fuerte|insoportable|extremo)\s+(de|en|en\s+la|en\s+el|de\s+la|de\s+el)?\s*(diente|dientes|muela|muelas|encia|encias|boca|mandibula|dental)\b", "DOLOR_SEVERO_DENTAL"),
    (r"\bno\s+(aguanto|soporto)\s+(el|este)\s+dolor\s+(de|en|en\s+la|en\s+el)?\s*(diente|muela|encia|boca|mandibula)\b", "DOLOR_INSOPORTABLE_DENTAL"),
    (r"\bno\s+puedo\s+abrir\s+la\s+boca\s+del\s+dolor\b", "TRISMUS_SEVERO"),

    # Sangrado / infección / inflamación oral
    (r"\b(sangrado|sangre)\s+(en\s+la\s+boca|en\s+la\s+encia|en\s+las\s+encias|de\s+la\s+muela|de\s+la\s+boca)\b", "SANGRADO_DENTAL"),
    (r"\b(pus|mucha\s+pus|supuraci[oó]n)\s+(en\s+la\s+boca|en\s+la\s+encia|de\s+la\s+muela|dental)?\b", "SUPURACION_DENTAL"),
    (r"\b(cara|mejilla|enc[ií]a|labio)\s+(hinchada|inflamada|deforme)\b", "INFLAMACION_SEVERA"),
    (r"\b(diente|muela)\s+(rot[oa]|partid[oa]|quebrad[oa]|floj[oa]\s+por\s+golpe)\b", "TRAUMA_DENTAL"),

    # Criterios de riesgo vital / maxilofacial
    (r"\bsangrado\s+que\s+no\s+para\b", "SANGRADO_QUE_NO_PARA"),
    (r"\binfecci[oó]n\s+gigante\b", "INFECCION_GIGANTE"),
    (r"\bfractura\s+mandibular\b", "FRACTURA_MANDIBULAR"),
    (r"\btraumatismo\s+severo\b", "TRAUMATISMO_SEVERO"),
    (r"\bflemon\s+con\s+fiebre\b", "FLEMON_CON_FIEBRE"),
    (r"\bdificultad\s+para\s+respirar\b", "DIFICULTAD_RESPIRATORIA"),
    (r"\basfixia\b", "ASFIXIA"),
    (r"\bhinchaz[oó]n\s+en\s+el\s+cuello\b", "EDEMA_CERVICAL"),
    (r"\bpus\s+abundante\b", "SUPURACION_SEVERA"),
    (r"\bno\s+para\s+de\s+sangrar\b", "SANGRADO_INCONTROLABLE"),
    (r"\bsangrado\s+(abundante|incontrolable|excesivo|profuso)\b", "SANGRADO_ABUNDANTE"),
    (r"\b(hemorragia|hemorragias)\b", "HEMORRAGIA"),
    (r"\bmucha\s+sangre\s+y\s+no\s+(para|se\s+detiene)\b", "SANGRADO_INCONTROLABLE"),
    (r"\bflem[oó]n\s+gigante\b", "FLEMON_GIGANTE"),
    (r"\bcelulitis\s+facial\b", "CELULITIS_FACIAL"),
    (r"\babsceso\s+(gigante|grave|severo)\b", "ABSCESO_GRAVE"),
    (r"\binfecci[oó]n\s+(grave|severa|muy\s+avanzada)\b", "INFECCION_SEVERA"),
    (r"\b(cara|cuello|mejilla)\s+(muy\s+hinchada|deforme)\s+y\s+(fiebre|asfixia|no\s+puedo\s+respirar)\b", "COMPROMISO_VIA_AEREA"),
    (r"\bmand[ií]bula\s+(fracturada|rota|desencajada)\b", "FRACTURA_MANDIBULA"),
    (r"\b(diente|dientes)\s+(arrancado|arrancados)\s+de\s+ra[ií]z\b", "AVULSION_DENTAL"),
    (r"\bavulsi[oó]n\s+dental\b", "AVULSION_DENTAL"),
]

def detect_emergency_keywords(text: str) -> Tuple[bool, Optional[str]]:
    """Detección rápida y determinista de emergencias odontológicas severas."""
    if not text:
        return False, None

    # Rodilla / espalda / etc. nunca son urgencia odontológica
    if is_non_dental_only(text):
        return False, None

    norm_text = normalize_text(text)

    for pattern, reason in EMERGENCY_PATTERNS:
        clean_pattern = normalize_text(pattern)
        if re.search(clean_pattern, norm_text, re.IGNORECASE):
            return True, reason

    return False, None


async def classify_emergency_llm(text: str) -> Tuple[bool, Optional[str]]:
    """Clasificador LLM rápido para triage de emergencia severa cuando no coincide por regex."""
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
        "Eres un clasificador de triage para el consultorio odontológico Nexus Odonto. "
        "Decide si el mensaje describe una EMERGENCIA ODONTOLÓGICA O MAXILOFACIAL SEVERA "
        "(hemorragia oral que no para, traumatismo dental/mandibular, flemón/absceso con "
        "fiebre o dificultad respiratoria, hinchazón facial/cuello que compromete vía aérea, "
        "dolor dental extremo e intolerable claramente de diente/muela/encía/boca).\n\n"
        "Clasifica como REGULAR (NO emergencia) si:\n"
        "- Es dolor o cita de una zona NO dental (rodilla, espalda, brazo, pie, etc.).\n"
        "- Solo dice 'me duele mucho' sin anclar a diente, muela, encía, boca o mandíbula.\n"
        "- Es broma, vulgaridad o falta de respeto.\n"
        "- Es agendar cita, precios, ortodoncia u consulta rutinaria.\n\n"
        "Responde ESTRICTAMENTE con una sola palabra: EMERGENCIA o REGULAR."
    )

    try:
        response = await evaluator_llm.ainvoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=f"Mensaje del paciente:\n\"\"\"\n{text}\n\"\"\"")
        ])
        result = extract_text_content(response.content).strip().upper()
        token = result.split()[0] if result.split() else ""
        if token == "EMERGENCIA":
            return True, "TRIAGE_CLINICO_LLM"
    except Exception as e:
        logger.warning(f"[Emergency Detector] Error en clasificador LLM: {e}")

    return False, None


# Términos que justifican triage LLM (nunca por "duele" solo en zona no dental)
SUSPICIOUS_EMERGENCY_TERMS = {
    "alerta", "sangr", "infect", "urgenc", "emerg", "hinch", "asfix", "respir",
    "fractur", "trauma", "grave", "absces", "flemon", "arranc", "desmay",
    "fiebre", "morir", "auxilio", "insoportable", "inaguantable",
    "hemorrag", "accidente", "golpe", "partio", "parti",
}


async def detect_severe_emergency(text: str) -> Tuple[bool, Optional[str]]:
    """Función principal híbrida:
    1. Excluye mensajes solo no dentales.
    2. Reglas deterministas (0 tokens).
    3. LLM solo con indicios clínicos y contexto oral/dental o riesgo vital.
    """
    if not text or not text.strip():
        return False, None

    if is_non_dental_only(text):
        return False, None

    is_emergency, reason = detect_emergency_keywords(text)
    if is_emergency:
        return True, reason

    norm_text = normalize_text(text)
    has_suspicious_term = any(term in norm_text for term in SUSPICIOUS_EMERGENCY_TERMS)
    # Dolor fuerte solo dispara LLM si hay ancla dental
    has_severe_pain_word = any(
        w in norm_text
        for w in ("insoportable", "inaguantable", "no aguanto", "no soporto", "hemorrag")
    )
    if not has_suspicious_term and not (has_severe_pain_word and has_dental_context(text)):
        return False, None

    # Sin contexto dental ni síntoma vital, no gastar tokens
    vital_terms = ("asfix", "respir", "sangr", "hemorrag", "desmay", "fiebre", "fractur", "trauma")
    has_vital = any(t in norm_text for t in vital_terms)
    if not has_dental_context(text) and not has_vital:
        return False, None

    return await classify_emergency_llm(text)
