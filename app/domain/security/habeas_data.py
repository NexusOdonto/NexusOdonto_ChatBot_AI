"""Políticas de Habeas Data, Privacidad y Validación de Identidad (Ley 1581).
Garantiza el aislamiento absoluto de datos personales y la estricta validación de cédulas.
"""

import re
from typing import Optional, Tuple
from app.domain.models.patient import SafePatientContext


MENSAJE_SEGURIDAD_CONTRASENAS = (
    "Por seguridad no cambiamos ni restablecemos contraseñas por este chat.\n\n"
    "Puedes actualizarla en https://nexusodonto.chatcampuslands.com/login "
    "con tu cédula como usuario y contraseña temporal; ahí te pedirá cambiarla al entrar."
)

MENSAJE_FUERA_DE_DOMINIO = (
    "Hola, soy de recepción de *Nexus Odonto*. "
    "Te puedo ayudar con temas odontológicos, servicios, horarios y citas. "
    "¿En qué te oriento?"
)


def validar_cedula(cedula: Optional[str]) -> Tuple[bool, Optional[str], Optional[str]]:
    """Valida que una cédula esté compuesta por dígitos numéricos y tenga al menos 7 caracteres.
    
    Retorna:
        Tuple[es_valida, cedula_limpia, mensaje_error]
    """
    if not cedula:
        return False, None, "Por favor indícame tu *número de cédula* 🆔 para poder ayudarte."

    raw = str(cedula).strip()
    solo_digitos = re.sub(r"\D", "", raw)

    if len(solo_digitos) < 7:
        return (
            False,
            None,
            f"⚠️ El número *{raw}* no parece ser una cédula válida (debe tener al menos 7 dígitos).\n"
            "Por favor verifica el número e inténtalo de nuevo. 🆔"
        )

    if len(solo_digitos) > 12:
        return (
            False,
            None,
            f"⚠️ El número *{raw}* contiene demasiados dígitos para ser una cédula válida.\n"
            "Por favor verifica el número e inténtalo de nuevo. 🆔"
        )

    return True, solo_digitos, None


def crear_contexto_seguro(phone: str, push_name: str = "") -> SafePatientContext:
    """Crea un contexto seguro para el hilo de conversación.
    POLÍTICA INVIOLABLE: NUNCA se asume ni se consulta la identidad o cédula por el teléfono.
    """
    return SafePatientContext(
        phone=phone.strip(),
        nombre="",
        primer_nombre="",
        cedula=None,
        is_registered=False,
        push_name=push_name.strip() if push_name else "",
    )


def es_solicitud_contrasena(texto: str) -> bool:
    """Detecta si el usuario está solicitando cambiar, recuperar o restablecer contraseñas."""
    if not texto:
        return False
    lower = texto.lower()
    patrones = [
        r"\b(cambiar|restablecer|recuperar|olvide|olvid[eé]|reseteo)\b.*\b(contrase[nñ]a|clave|password)\b",
        r"\b(contrase[nñ]a|clave|password)\b.*\b(olvidada|perdida|bloqueada)\b",
        r"\bno\s+puedo\s+entrar\s+a\s+la\s+(plataforma|web|pagina)\b",
    ]
    return any(re.search(p, lower) for p in patrones)
