import logging
import re
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Registros dinámicos en memoria para mapear bidireccionalmente LID <-> Teléfono
# Un LID (Linked Device Identifier) de WhatsApp tiene el formato "...@lid" y típicamente 14-15 dígitos.
# Un JID estándar de usuario tiene el formato "57XXXXXXXXXX@s.whatsapp.net" o solo los dígitos.
_LID_TO_PHONE: dict[str, str] = {}
_PHONE_TO_LID: dict[str, str] = {}

# Placeholders que NUNCA deben guardarse ni usarse como teléfono real.
PLACEHOLDER_PHONE_DIGITS = frozenset({
    "3000000000",
    "0000000000",
    "1111111111",
    "1234567890",
    "3001234567",
    "573000000000",
    "573001234567",
})


def limpiar_digitos(texto: str) -> str:
    """Extrae únicamente los dígitos de una cadena."""
    if not texto:
        return ""
    return re.sub(r"\D", "", str(texto))


def es_identificador_lid(identificador: str) -> bool:
    """Determina si un identificador de WhatsApp corresponde a un LID."""
    raw = str(identificador or "").strip().lower()
    if not raw:
        return False
    if "@lid" in raw:
        return True
    # Dominios de teléfono real nunca son LID
    if "@s.whatsapp.net" in raw or "@c.us" in raw:
        return False
    digits = limpiar_digitos(raw)
    # LIDs de WhatsApp típicamente poseen 14 o más dígitos (opacos, no E.164)
    return len(digits) >= 14


def es_telefono_movil_valido(identificador: str) -> bool:
    """True si el identificador es un teléfono WhatsApp usable (no LID / no placeholder)."""
    return normalizar_e164(identificador) is not None


def es_telefono_placeholder(identificador: str) -> bool:
    digits = limpiar_digitos(identificador)
    if not digits:
        return True
    if digits in PLACEHOLDER_PHONE_DIGITS:
        return True
    # Sufijos locales de placeholders conocidos
    local = digits[2:] if digits.startswith("57") and len(digits) > 2 else digits
    return local in PLACEHOLDER_PHONE_DIGITS


def normalizar_e164(identificador: str) -> Optional[str]:
    """Normaliza a E.164 (+57…) desde JID/dígitos WhatsApp. None si es LID o inválido.

    Preferencia clínica Colombia:
    - 10 dígitos móviles 3XXXXXXXXX → +573XXXXXXXXX
    - 12 dígitos 573XXXXXXXXX → +573XXXXXXXXX
    - Otros internacionales 10–13 dígitos (no LID) → +digits
    """
    raw = str(identificador or "").strip()
    if not raw or es_identificador_lid(raw):
        return None

    # Solo parte de usuario si viene como JID
    user_part = raw.split("@", 1)[0].strip()
    digits = limpiar_digitos(user_part)
    if not digits or es_telefono_placeholder(digits):
        return None

    # Celular CO 10 dígitos
    if len(digits) == 10 and digits.startswith("3"):
        return f"+57{digits}"

    # Celular CO con código país
    if len(digits) == 12 and digits.startswith("573"):
        return f"+{digits}"

    # Fijo CO 60x / 601
    if len(digits) == 10 and digits.startswith("60"):
        return f"+57{digits}"
    if len(digits) == 12 and digits.startswith("5760"):
        return f"+{digits}"

    # Internacional razonable: E.164 max 15, pero >=14 sin @s.whatsapp suele ser LID
    if 10 <= len(digits) <= 13:
        return f"+{digits}"

    return None


def telefono_para_almacenar(identificador: str) -> Optional[str]:
    """E.164 listo para Person.Phone / chatIdentifier canónico. None si no hay teléfono real."""
    return normalizar_e164(identificador)


def jid_desde_telefono(identificador: str) -> str:
    """Convierte teléfono/JID a forma @s.whatsapp.net para identidad canónica interna."""
    e164 = normalizar_e164(identificador)
    if not e164:
        raw = str(identificador or "").strip()
        return raw
    digits = limpiar_digitos(e164)
    return f"{digits}@s.whatsapp.net"


def registrar_asociacion_lid(telefono: str, lid: str) -> None:
    """Registra la asociación bidireccional entre un número de teléfono y un LID de WhatsApp."""
    clean_tel = str(telefono or "").strip()
    clean_lid = str(lid or "").strip()

    if not clean_tel or not clean_lid:
        return
    if es_identificador_lid(clean_tel) or not es_telefono_movil_valido(clean_tel):
        return

    tel_jid = jid_desde_telefono(clean_tel)
    tel_digits = limpiar_digitos(tel_jid)

    lid_jid = clean_lid if "@" in clean_lid else f"{clean_lid}@lid"

    _LID_TO_PHONE[lid_jid] = tel_jid
    _LID_TO_PHONE[clean_lid] = tel_jid
    _LID_TO_PHONE[limpiar_digitos(clean_lid)] = tel_jid

    _PHONE_TO_LID[tel_jid] = lid_jid
    _PHONE_TO_LID[clean_tel] = lid_jid
    if tel_digits:
        _PHONE_TO_LID[tel_digits] = lid_jid
        e164 = normalizar_e164(tel_digits)
        if e164:
            _PHONE_TO_LID[e164] = lid_jid
            _PHONE_TO_LID[limpiar_digitos(e164)] = lid_jid

    logger.info(f"[WhatsApp Identity] Asociación registrada: LID {lid_jid} <-> Tel {tel_jid}")


def obtener_telefono_canonico(identificador: str) -> str:
    """Retorna el identificador canónico interno (preferir JID teléfono; resolver LID si hay mapa).

    Usado en LangGraph thread_id, checkpointer, ChatbotConversations y búsqueda de paciente.
    Si el input es LID sin mapa, se devuelve el LID (destino de envío sigue funcionando).
    """
    raw = str(identificador or "").strip()
    if not raw:
        return ""

    if raw in _LID_TO_PHONE:
        return _LID_TO_PHONE[raw]

    digits = limpiar_digitos(raw)
    if digits in _LID_TO_PHONE:
        return _LID_TO_PHONE[digits]

    if es_identificador_lid(raw):
        return raw if "@" in raw else f"{digits}@lid"

    if "@s.whatsapp.net" in raw or "@c.us" in raw:
        e164 = normalizar_e164(raw)
        return jid_desde_telefono(e164) if e164 else raw

    e164 = normalizar_e164(raw)
    if e164:
        return jid_desde_telefono(e164)

    return raw


def obtener_destino_envio(numero_o_jid: str) -> str:
    """Destinatario Evolution API (preferir LID vinculado si existe)."""
    raw = str(numero_o_jid or "").strip()
    if not raw:
        return ""

    if raw.startswith("+"):
        raw = raw[1:].strip()

    if raw.endswith("@lid") or "@lid" in raw.lower():
        return raw if "@" in raw else f"{limpiar_digitos(raw)}@lid"

    if raw in _PHONE_TO_LID:
        return _PHONE_TO_LID[raw]

    digits = limpiar_digitos(raw)
    if digits in _PHONE_TO_LID:
        return _PHONE_TO_LID[digits]

    e164 = normalizar_e164(raw)
    if e164:
        e_digits = limpiar_digitos(e164)
        if e_digits in _PHONE_TO_LID:
            return _PHONE_TO_LID[e_digits]

    if len(digits) >= 14:
        return f"{digits}@lid"

    return raw


def _candidato_es_telefono(cand: str) -> Optional[str]:
    """Devuelve JID @s.whatsapp.net si el candidato es teléfono real; None si LID/basura."""
    cand_clean = str(cand or "").strip()
    if not cand_clean:
        return None
    if es_identificador_lid(cand_clean):
        return None
    if "@g.us" in cand_clean or "status@broadcast" in cand_clean:
        return None
    e164 = normalizar_e164(cand_clean)
    if not e164:
        return None
    return jid_desde_telefono(e164)


def _recoger_candidatos_payload(raw_payload: dict, data_obj: Any, key: dict, data: dict) -> list[str]:
    """Campos Evolution/Baileys donde puede venir el teléfono real junto a un LID."""
    candidates: list[str] = []

    def add(val: Any) -> None:
        if val is None:
            return
        s = str(val).strip()
        if s:
            candidates.append(s)

    # key.remoteJidAlt / participantAlt / senderPn (Baileys LID addressing)
    if data_obj and hasattr(data_obj, "key") and data_obj.key:
        key_obj = data_obj.key
        for attr in ("remoteJidAlt", "participantAlt", "senderPn", "participant", "remoteJid"):
            add(getattr(key_obj, attr, None))
        # Modelo con extra=allow: dict-like
        if isinstance(key_obj, dict):
            for k in ("remoteJidAlt", "participantAlt", "senderPn", "participant", "remoteJid"):
                add(key_obj.get(k))

    if isinstance(key, dict):
        for k in ("remoteJidAlt", "participantAlt", "senderPn", "participant", "remoteJid"):
            add(key.get(k))

    if data_obj:
        for attr in ("participant", "senderPn", "remoteJidAlt"):
            add(getattr(data_obj, attr, None))

    if isinstance(data, dict):
        for k in ("participant", "senderPn", "remoteJidAlt", "participantAlt"):
            add(data.get(k))
        # Algunos forks anidan under key / contextInfo
        ctx = data.get("contextInfo") or {}
        if isinstance(ctx, dict):
            add(ctx.get("participant"))
            add(ctx.get("remoteJid"))

    # NO usar data.sender / payload.sender: en Evolution suele ser el bot/instancia.
    return candidates


def extraer_identidad_webhook(raw_payload: dict, data_obj: Any = None) -> Tuple[str, str]:
    """Extrae y sincroniza la identidad del remitente desde un payload de Evolution API.

    Retorna:
        Tuple[str, str]: (numero_canonico, destino_envio)
        - numero_canonico: Para uso interno (LangGraph, PostgreSQL, .NET DB). Preferir teléfono.
        - destino_envio: Para enviar mensajes de vuelta (puede ser LID).
    """
    data = (raw_payload or {}).get("data", {}) if isinstance(raw_payload, dict) else {}
    key = data.get("key", {}) if isinstance(data, dict) else {}

    remote_jid = ""
    if data_obj and hasattr(data_obj, "key") and data_obj.key:
        remote_jid = getattr(data_obj.key, "remoteJid", None) or ""
    if not remote_jid and isinstance(key, dict):
        remote_jid = key.get("remoteJid") or ""
    remote_jid = str(remote_jid).strip()

    phone_candidato = ""
    for cand in _recoger_candidatos_payload(raw_payload, data_obj, key if isinstance(key, dict) else {}, data if isinstance(data, dict) else {}):
        # Si remoteJid ya es teléfono, úsalo; si es LID, busca alt/pn
        resolved = _candidato_es_telefono(cand)
        if not resolved:
            continue
        # Preferir candidatos que no sean el mismo LID
        if es_identificador_lid(remote_jid) and limpiar_digitos(cand) == limpiar_digitos(remote_jid):
            continue
        phone_candidato = resolved
        # remoteJidAlt / senderPn tienen prioridad sobre remoteJid LID
        if any(tag in str(cand).lower() for tag in ()) or "@s.whatsapp.net" in cand.lower() or "@c.us" in cand.lower():
            break
        if not es_identificador_lid(cand):
            break

    # Segunda pasada: priorizar explícitamente remoteJidAlt / senderPn
    priority_vals: list[str] = []
    if isinstance(key, dict):
        for k in ("remoteJidAlt", "senderPn", "participantAlt"):
            if key.get(k):
                priority_vals.append(str(key[k]))
    if data_obj and hasattr(data_obj, "key") and data_obj.key:
        for attr in ("remoteJidAlt", "senderPn", "participantAlt"):
            v = getattr(data_obj.key, attr, None)
            if v:
                priority_vals.append(str(v))
    for cand in priority_vals:
        resolved = _candidato_es_telefono(cand)
        if resolved:
            phone_candidato = resolved
            break

    if es_identificador_lid(remote_jid) and phone_candidato:
        registrar_asociacion_lid(phone_candidato, remote_jid)

    if es_identificador_lid(remote_jid):
        numero_canonico = obtener_telefono_canonico(remote_jid)
        if es_identificador_lid(numero_canonico) and phone_candidato:
            numero_canonico = phone_candidato
        destino_envio = remote_jid if "@" in remote_jid else f"{limpiar_digitos(remote_jid)}@lid"
    else:
        # remoteJid ya es teléfono (o grupo — se filtra antes en webhook)
        numero_canonico = obtener_telefono_canonico(remote_jid) or remote_jid
        if phone_candidato and es_identificador_lid(numero_canonico):
            numero_canonico = phone_candidato
        destino_envio = obtener_destino_envio(numero_canonico)

    logger.debug(
        "[WhatsApp Identity] remote=%s phone=%s canon=%s dest=%s",
        remote_jid,
        phone_candidato,
        numero_canonico,
        destino_envio,
    )
    return numero_canonico, destino_envio
