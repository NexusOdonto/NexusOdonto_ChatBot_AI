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

# Solo red flags odontológicas / maxilofaciales reales: sangrado que no para, hinchazón con
# fiebre o compromiso para respirar/tragar, trauma. El dolor dental (aunque sea fuerte), la
# sensibilidad, un diente partido o un sangrado leve son una CITA PRIORITARIA que agenda el bot.
EMERGENCY_PATTERNS = [
    (r"\b(alerta\s+odontologica|emergencia\s+odontologica)\b", "URGENCIA_ODONTOLOGICA"),
    (r"\b(alerta\s+medica|estado\s+de\s+alerta|emergencia\s+medica)\b", "ESTADO_DE_ALERTA"),
    (r"\b(diente|dientes|muela|muelas)\s+(floj[oa]s?|rot[oa]s?|partid[oa]s?)\s+por\s+(un\s+)?golpe\b", "TRAUMA_DENTAL"),
    (r"\b(no\s+puedo|dificultad\s+para)\s+tragar\b", "DIFICULTAD_DEGLUCION"),

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
        "Decide si el mensaje describe una EMERGENCIA ODONTOLÓGICA O MAXILOFACIAL SEVERA: "
        "hemorragia oral que no para, traumatismo por golpe/caída/accidente en dientes o mandíbula, "
        "flemón/absceso o hinchazón de cara/cuello con fiebre o dificultad para respirar o tragar.\n\n"
        "Clasifica como REGULAR (NO emergencia; es una cita prioritaria que se agenda) si:\n"
        "- Es dolor de diente, muela, encía o boca, aunque sea muy fuerte o insoportable, "
        "o sensibilidad, o pide que lo atiendan pronto/'con urgencia'.\n"
        "- Se le partió, despicó o aflojó un diente sin golpe, con o sin un sangrado leve.\n"
        "- Encía inflamada o que sangra un poco, sin fiebre ni dificultad para respirar/tragar.\n"
        "- Es dolor o cita de una zona NO dental (rodilla, espalda, brazo, pie, etc.).\n"
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
    "alerta", "sangr", "infect", "emerg", "hinch", "asfix", "respir",
    "fractur", "trauma", "grave", "absces", "flemon", "arranc", "desmay",
    "fiebre", "morir", "auxilio", "hemorrag", "accidente", "golpe", "caida", "tragar",
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
    if not any(term in norm_text for term in SUSPICIOUS_EMERGENCY_TERMS):
        return False, None

    # Sin contexto dental ni síntoma vital, no gastar tokens
    vital_terms = ("asfix", "respir", "sangr", "hemorrag", "desmay", "fiebre", "fractur", "trauma")
    has_vital = any(t in norm_text for t in vital_terms)
    if not has_dental_context(text) and not has_vital:
        return False, None

    return await classify_emergency_llm(text)
