import logging
import re
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Registros dinámicos en memoria para mapear bidireccionalmente LID <-> Teléfono
# Un LID (Linked Device Identifier) de WhatsApp tiene el formato "...@lid" y típicamente 14-15 dígitos.
# Un JID estándar de usuario tiene el formato "57XXXXXXXXXX@s.whatsapp.net" o solo los dígitos.
_LID_TO_PHONE: dict[str, str] = {}
_PHONE_TO_LID: dict[str, str] = {}

def limpiar_digitos(texto: str) -> str:
    """Extrae únicamente los dígitos de una cadena."""
    if not texto:
        return ""
    return re.sub(r"\D", "", str(texto))


def es_identificador_lid(identificador: str) -> bool:
    """Determina si un identificador de WhatsApp corresponde a un LID."""
    raw = str(identificador or "").strip().lower()
    if "@lid" in raw:
        return True
    digits = limpiar_digitos(raw)
    # LIDs de WhatsApp típicamente poseen 14 o más dígitos
    return len(digits) >= 14


def registrar_asociacion_lid(telefono: str, lid: str) -> None:
    """Registra la asociación bidireccional entre un número de teléfono y un LID de WhatsApp."""
    clean_tel = str(telefono or "").strip()
    clean_lid = str(lid or "").strip()

    if not clean_tel or not clean_lid:
        return

    # Normalizar teléfono con sufijo @s.whatsapp.net si es numérico puro
    tel_jid = clean_tel if "@" in clean_tel else f"{clean_tel}@s.whatsapp.net"
    tel_digits = limpiar_digitos(clean_tel)

    # Normalizar LID con sufijo @lid si no lo tiene
    lid_jid = clean_lid if "@" in clean_lid else f"{clean_lid}@lid"

    _LID_TO_PHONE[lid_jid] = tel_jid
    _LID_TO_PHONE[clean_lid] = tel_jid
    _LID_TO_PHONE[limpiar_digitos(clean_lid)] = tel_jid

    _PHONE_TO_LID[tel_jid] = lid_jid
    _PHONE_TO_LID[clean_tel] = lid_jid
    if tel_digits:
        # Registro estricto 1:1 por dígitos completos únicamente.
        # NO se registra por los últimos 10 dígitos para evitar colisiones entre
        # números de distintos usuarios que compartan el mismo sufijo.
        _PHONE_TO_LID[tel_digits] = lid_jid

    logger.debug(f"[WhatsApp Identity] Asociación registrada: LID {lid_jid} <-> Tel {tel_jid}")


def obtener_telefono_canonico(identificador: str) -> str:
    """Retorna el número de teléfono canónico para un identificador (resolviendo si es un LID).

    Este es el identificador único y consistente que DEBE usarse internamente en:
    - LangGraph thread_id
    - PostgreSQL checkpointer
    - .NET ChatbotConversations chatIdentifier
    - Búsqueda de paciente por teléfono
    """
    raw = str(identificador or "").strip()
    if not raw:
        return ""

    # 1. Buscar coincidencia directa en el registro LID -> Teléfono
    if raw in _LID_TO_PHONE:
        return _LID_TO_PHONE[raw]

    # 2. Buscar por dígitos si es un LID
    digits = limpiar_digitos(raw)
    if digits in _LID_TO_PHONE:
        return _LID_TO_PHONE[digits]

    # 3. Si es un JID móvil estándar (@s.whatsapp.net), asegurar formato limpio
    if "@s.whatsapp.net" in raw:
        return raw

    # 4. Si son dígitos telefónicos (entre 10 y 12 dígitos, ej. Colombia 573... o 3...)
    if 10 <= len(digits) <= 12:
        return f"{digits}@s.whatsapp.net"

    # 5. Si es un LID no conocido aún, devolverlo mientras se resuelve dinámicamente
    return raw


def obtener_destino_envio(numero_o_jid: str) -> str:
    """Determina el destinatario adecuado para Evolution API al enviar un mensaje.

    Si el usuario inició o mantiene su conversación activa a través de un LID vinculado,
    Evolution API requiere que el mensaje se entregue a ese LID para que no falle el socket
    ni se abra un chat paralelo en el dispositivo del usuario.
    """
    raw = str(numero_o_jid or "").strip()
    if not raw:
        return ""

    # Si empieza con +, removerlo
    if raw.startswith("+"):
        raw = raw[1:].strip()

    # Si ya termina en @lid, mantenerlo
    if raw.endswith("@lid"):
        return raw

    # Verificar si existe un LID mapeado para este número de teléfono
    if raw in _PHONE_TO_LID:
        return _PHONE_TO_LID[raw]

    digits = limpiar_digitos(raw)
    if digits in _PHONE_TO_LID:
        return _PHONE_TO_LID[digits]
    if len(digits) >= 10 and digits[-10:] in _PHONE_TO_LID:
        return _PHONE_TO_LID[digits[-10:]]

    # Si tiene 14 o más dígitos y no tiene dominio, es un LID numérico
    if len(digits) >= 14:
        return f"{digits}@lid"

    # En caso contrario, enviar al número/JID estándar
    return raw


def extraer_identidad_webhook(raw_payload: dict, data_obj: Any = None) -> Tuple[str, str]:
    """Extrae y sincroniza la identidad del remitente desde un payload de Evolution API.

    Retorna:
        Tuple[str, str]: (numero_canonico, destino_envio)
        - numero_canonico: Para uso interno (LangGraph, PostgreSQL, .NET DB).
        - destino_envio: Para enviar mensajes de vuelta a través de Evolution API.
    """
    data = (raw_payload or {}).get("data", {}) if isinstance(raw_payload, dict) else {}
    key = data.get("key", {}) if isinstance(data, dict) else {}

    # Extraer remoteJid
    remote_jid = ""
    if data_obj and hasattr(data_obj, "key") and data_obj.key:
        remote_jid = getattr(data_obj.key, "remoteJid", None) or ""
    if not remote_jid and isinstance(key, dict):
        remote_jid = key.get("remoteJid") or ""

    remote_jid = str(remote_jid).strip()

    # Buscar candidatos a número de teléfono real
    candidate_sources = []

    # 1. Objeto Pydantic data_obj si fue provisto
    if data_obj:
        part_key = getattr(data_obj.key, "participant", None) if hasattr(data_obj, "key") and data_obj.key else None
        if part_key:
            candidate_sources.append(str(part_key))
        part_data = getattr(data_obj, "participant", None)
        if part_data:
            candidate_sources.append(str(part_data))
        sender_data = getattr(data_obj, "sender", None)
        if sender_data:
            candidate_sources.append(str(sender_data))

    # 2. Diccionario raw JSON
    if isinstance(key, dict):
        if key.get("participant"):
            candidate_sources.append(str(key["participant"]))
    if isinstance(data, dict):
        if data.get("participant"):
            candidate_sources.append(str(data["participant"]))
        # NOTA DE SEGURIDAD: data.get("sender") es el número del bot/instancia,
        # NO del paciente remitente. Se excluye intencionalmente para evitar que
        # todos los usuarios queden mapeados al mismo número canónico del bot.
        # raw_payload.get("sender") tiene el mismo problema y también se excluye.

    # Encontrar si alguno de los candidatos es un teléfono real (@s.whatsapp.net o móvil)
    phone_candidato = ""
    for cand in candidate_sources:
        cand_clean = cand.strip()
        if not cand_clean:
            continue
        if "@s.whatsapp.net" in cand_clean:
            phone_candidato = cand_clean
            break
        digits = limpiar_digitos(cand_clean)
        if 10 <= len(digits) <= 12:
            phone_candidato = f"{digits}@s.whatsapp.net"
            break

    # Si remoteJid es un LID y encontramos el teléfono real del remitente, registrar asociación
    if es_identificador_lid(remote_jid) and phone_candidato:
        registrar_asociacion_lid(phone_candidato, remote_jid)

    # Determinar el número canónico interno
    if es_identificador_lid(remote_jid):
        # Intentar resolverlo desde el registro
        numero_canonico = obtener_telefono_canonico(remote_jid)
        if es_identificador_lid(numero_canonico) and phone_candidato:
            numero_canonico = phone_candidato
        # Si sigue siendo LID, se usa el LID como fallback
        destino_envio = remote_jid
    else:
        numero_canonico = remote_jid
        # Si remoteJid ya es el número, revisar si tiene LID asociado para el transporte
        destino_envio = obtener_destino_envio(remote_jid)

    return numero_canonico, destino_envio
