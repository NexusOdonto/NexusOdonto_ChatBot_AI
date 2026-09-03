"""Utilidades compartidas de sesión del chatbot.

Este módulo reemplaza la antigua máquina de estados de registro/login.
La autenticación fue eliminada: el bot atiende a cualquier usuario sin requerir registro previo.

Solo se conserva `_normalize_phone` como utilidad compartida por otros módulos.
"""

import logging

logger = logging.getLogger(__name__)


def _normalize_phone(phone: str) -> str:
    """Extrae solo dígitos del número de WhatsApp, quitando el sufijo @s.whatsapp.net."""
    clean = phone.replace("@s.whatsapp.net", "").replace("@g.us", "").strip()
    return clean
