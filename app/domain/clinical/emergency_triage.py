"""Triage médico y odontológico de emergencias para Nexus Odonto.
Identifica situaciones que ponen en riesgo la vida o la integridad del paciente
para suspender el bot y derivar de inmediato a urgencias presenciales.
"""

import re
import unicodedata
from typing import Tuple, Optional
from app.domain.models.emergency import EmergencySeverity, EmergencyTriageResult

MENSAJE_EMERGENCIA_URGENCIAS = (
    "🚨 *ATENCIÓN DE URGENCIA ODONTOLÓGICA / MÉDICA* 🚨\n\n"
    "Hemos detectado que puedes estar presentando una situación de emergencia que requiere atención presencial inmediata.\n\n"
    "⚠️ *Por tu seguridad, suspende el uso de este chat y acude de inmediato a un centro médico de urgencias o servicio de emergencias odontológicas más cercano.*\n\n"
    "📍 *Consultorio Nexus Odonto:* Calle 100 # 15-20, Centro Médico Odontológico\n"
    "📞 *Línea directa / Urgencias:* +57 324 6030217\n\n"
    "Hemos escalado tu caso inmediatamente a nuestro equipo humano con prioridad *CRÍTICA*."
)

# Patrones regex normalizados para detección determinista de emergencias severas
EMERGENCY_PATTERNS = [
    # Alerta y urgencias médicas/odontológicas explícitas
    (r"\b(alerta|estado\s+de\s+alerta|alerta\s+medica|alerta\s+odontologica)\b", "ESTADO_DE_ALERTA", EmergencySeverity.CRITICO),
    (r"\b(urgencia|urgencia\s+odontologica|emergencia|emergencia\s+medica)\b", "URGENCIA_ODONTOLOGICA", EmergencySeverity.CRITICO),
    (r"\bdolor\s+(fuerte|severo|agudo|terrible|horrible|inaguantable|muy\s+fuerte)\b", "DOLOR_SEVERO", EmergencySeverity.CRITICO),
    (r"\bme\s+duele\s+(mucho|demasiado|bastante)\b", "DOLOR_AGUDO", EmergencySeverity.ALTO),
    (r"\b(sangrado|sangre)\s+(en\s+la\s+boca|en\s+la\s+encia|en\s+las\s+encias|de\s+la\s+muela)\b", "SANGRADO_DENTAL", EmergencySeverity.ALTO),
    (r"\b(pus|mucha\s+pus|supuraci[oó]n)\b", "SUPURACION_DENTAL", EmergencySeverity.ALTO),
    (r"\b(cara|mejilla|enc[ií]a|labio)\s+(hinchada|inflamada|deforme)\b", "INFLAMACION_SEVERA", EmergencySeverity.CRITICO),
    (r"\b(diente|muela)\s+(rot[oa]|partid[oa]|quebrad[oa]|floj[oa]\s+por\s+golpe)\b", "TRAUMA_DENTAL", EmergencySeverity.ALTO),

    # Criterios explícitos de riesgo vital
    (r"\bdolor\s+insoportable\b", "DOLOR_INSOPORTABLE", EmergencySeverity.CRITICO),
    (r"\bsangrado\s+que\s+no\s+para\b", "SANGRADO_QUE_NO_PARA", EmergencySeverity.CRITICO),
    (r"\binfecci[oó]n\s+gigante\b", "INFECCION_GIGANTE", EmergencySeverity.CRITICO),
    (r"\bfractura\s+mandibular\b", "FRACTURA_MANDIBULAR", EmergencySeverity.CRITICO),
    (r"\btraumatismo\s+severo\b", "TRAUMATISMO_SEVERO", EmergencySeverity.CRITICO),
    (r"\bflemon\s+con\s+fiebre\b", "FLEMON_CON_FIEBRE", EmergencySeverity.CRITICO),
    (r"\bdificultad\s+para\s+respirar\b", "DIFICULTAD_RESPIRATORIA", EmergencySeverity.CRITICO),
    (r"\basfixia\b", "ASFIXIA", EmergencySeverity.CRITICO),
    (r"\bno\s+puedo\s+abrir\s+la\s+boca\s+del\s+dolor\b", "TRISMUS_SEVERO", EmergencySeverity.CRITICO),
    (r"\bhinchaz[oó]n\s+en\s+el\s+cuello\b", "EDEMA_CERVICAL", EmergencySeverity.CRITICO),
    (r"\bpus\s+abundante\b", "SUPURACION_SEVERA", EmergencySeverity.CRITICO),

    # Hemorragia / sangrado severo
    (r"\bno\s+para\s+de\s+sangrar\b", "SANGRADO_INCONTROLABLE", EmergencySeverity.CRITICO),
    (r"\bsangrado\s+(abundante|incontrolable|excesivo|profuso)\b", "SANGRADO_ABUNDANTE", EmergencySeverity.CRITICO),
    (r"\b(hemorragia|hemorragias)\b", "HEMORRAGIA", EmergencySeverity.CRITICO),
    (r"\bmucha\s+sangre\s+y\s+no\s+(para|se\s+detiene)\b", "SANGRADO_INCONTROLABLE", EmergencySeverity.CRITICO),

    # Dolor extremo
    (r"\bdolor\s+(inaguantable|extremo|intenso\s+e\s+insoportable)\b", "DOLOR_EXTREMO", EmergencySeverity.CRITICO),
    (r"\bno\s+(aguanto|soporto)\s+(el|este)\s+dolor\b", "DOLOR_INSOPORTABLE", EmergencySeverity.CRITICO),

    # Infecciones con compromiso sistémico
    (r"\bflem[oó]n\s+gigante\b", "FLEMON_GIGANTE", EmergencySeverity.CRITICO),
    (r"\bcelulitis\s+facial\b", "CELULITIS_FACIAL", EmergencySeverity.CRITICO),
    (r"\babsceso\s+(gigante|grave|severo)\b", "ABSCESO_GRAVE", EmergencySeverity.CRITICO),
    (r"\binfecci[oó]n\s+(grave|severa|muy\s+avanzada)\b", "INFECCION_SEVERA", EmergencySeverity.CRITICO),
    (r"\b(cara|cuello|mejilla)\s+(muy\s+hinchada|deforme)\s+y\s+(fiebre|asfixia|no\s+puedo\s+respirar)\b", "COMPROMISO_VIA_AEREA", EmergencySeverity.CRITICO),

    # Traumatismos maxilofaciales severos / avulsión
    (r"\bmand[ií]bula\s+(fracturada|rota|desencajada)\b", "FRACTURA_MANDIBULA", EmergencySeverity.CRITICO),
    (r"\b(diente|dientes)\s+(arrancado|arrancados)\s+de\s+ra[ií]z\b", "AVULSION_DENTAL", EmergencySeverity.CRITICO),
    (r"\bavulsi[oó]n\s+dental\b", "AVULSION_DENTAL", EmergencySeverity.CRITICO),
]


def normalizar_texto(texto: str) -> str:
    """Normaliza texto eliminando tildes y caracteres diacríticos, convirtiendo a minúsculas."""
    if not texto:
        return ""
    texto = texto.lower().strip()
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def evaluar_emergencia_determinista(texto: str) -> EmergencyTriageResult:
    """Evalúa de forma puramente determinista (0 tokens) si el mensaje describe una urgencia médica u odontológica severa."""
    texto_norm = normalizar_texto(texto)
    if not texto_norm:
        return EmergencyTriageResult(is_emergency=False, severity=EmergencySeverity.NORMAL)

    for pattern, reason, severity in EMERGENCY_PATTERNS:
        if re.search(pattern, texto_norm, re.IGNORECASE):
            return EmergencyTriageResult(
                is_emergency=True,
                severity=severity,
                reason=reason,
                trigger_pattern=pattern,
                action_required="ESCALAR_URGENCIAS",
            )

    return EmergencyTriageResult(
        is_emergency=False,
        severity=EmergencySeverity.NORMAL,
        reason=None,
        action_required="CONTINUAR",
    )
