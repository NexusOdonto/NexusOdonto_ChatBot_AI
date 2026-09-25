import os
import json
import logging
import time
from collections import OrderedDict
from typing import Optional, Dict, Any
import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─── IDs y textos de mensajes enviados por el bot ────────────────────────────────────
# Guardamos el key.id y el snippet de texto de cada mensaje saliente del bot.
# Esto previene que si Baileys emite el webhook fromMe antes de que termine el POST HTTP,
# el webhook lo clasifique erróneamente como mensaje manual de asesor humano.
_BOT_SENT_IDS: OrderedDict[str, float] = OrderedDict()
# (dest_digits_or_empty, norm_text, monotonic_ts)
_BOT_RECENT_OUTGOING: list[tuple[str, str, float]] = []
_BOT_SENT_TTL = 180  # segundos hasta descartar


def _normalize_msg_snippet(text: str) -> str:
    if not text:
        return ""
    import re
    import unicodedata
    norm = unicodedata.normalize("NFKD", text)
    norm = "".join(c for c in norm if not unicodedata.combining(c)).lower()
    norm = re.sub(r"[^a-z0-9\s]", " ", norm)
    return re.sub(r"\s+", " ", norm).strip()


def _dest_fingerprints(destinatario: str) -> list[str]:
    """Huellas de destino tolerantes a LID vs @s.whatsapp.net (últimos 10 dígitos + full)."""
    from app.services.whatsapp_identity import limpiar_digitos

    digits = limpiar_digitos(destinatario or "")
    if not digits:
        return []
    fps = {digits}
    if len(digits) > 10:
        fps.add(digits[-10:])
    if len(digits) > 12:
        fps.add(digits[-12:])
    return list(fps)


def _texts_match(norm_a: str, norm_b: str) -> bool:
    """Coincide prefijo/subcadena con umbral mínimo para no marcar ecos cortos ambiguos."""
    if not norm_a or not norm_b:
        return False
    if norm_a == norm_b:
        return True
    # Mensajes cortos: solo igualdad o uno es prefijo exacto del otro (min 12 chars)
    shorter, longer = (norm_a, norm_b) if len(norm_a) <= len(norm_b) else (norm_b, norm_a)
    if len(shorter) < 24:
        return len(shorter) >= 12 and longer.startswith(shorter)
    return (
        norm_a.startswith(norm_b[:24])
        or norm_b.startswith(norm_a[:24])
        or norm_a[:32] in norm_b
        or norm_b[:32] in norm_a
    )


def register_outgoing_bot_message(destinatario: str, texto: str) -> None:
    """Registra preventivamente el mensaje antes de enviarlo por HTTP para evitar condiciones de carrera.

    Guarda varias huellas del destinatario (LID y teléfono) para que el eco fromMe
    coincida aunque Evolution entregue otro JID en el webhook.
    """
    if not texto:
        return
    now = time.monotonic()
    norm_text = _normalize_msg_snippet(texto)[:120]
    if not norm_text:
        return

    fingerprints = _dest_fingerprints(destinatario) or [""]

    global _BOT_RECENT_OUTGOING
    _BOT_RECENT_OUTGOING = [
        item for item in _BOT_RECENT_OUTGOING if now - item[2] <= _BOT_SENT_TTL
    ]
    for fp in fingerprints:
        _BOT_RECENT_OUTGOING.append((fp, norm_text, now))
    # Entrada sin destino: fallback de coincidencia solo por texto (fromMe ya prueba origen local)
    _BOT_RECENT_OUTGOING.append(("", norm_text, now))
    logger.debug(
        "[Evolution API] Outgoing registrado dest=%s fps=%s snippet=%r",
        destinatario,
        fingerprints,
        norm_text[:40],
    )


def is_recent_bot_text(destinatario: str, texto: str) -> bool:
    """Retorna True si un mensaje con texto coincidente fue enviado recientemente por el bot.

    Si el destinatario no coincide (LID vs teléfono), aún puede hacer match por texto
    contra entradas registradas sin destino — necesario porque fromMe ya garantiza
    que el mensaje salió del dispositivo vinculado.
    """
    if not texto:
        return False
    now = time.monotonic()
    norm_text = _normalize_msg_snippet(texto)[:120]
    if not norm_text:
        return False

    dest_fps = set(_dest_fingerprints(destinatario))

    for item_dest, item_text, ts in _BOT_RECENT_OUTGOING:
        if now - ts > _BOT_SENT_TTL:
            continue
        if not item_text or not _texts_match(norm_text, item_text):
            continue
        # Match por destino (huellas) o entrada comodín sin destino
        if not item_dest or not dest_fps or item_dest in dest_fps:
            return True
    return False


def extract_evolution_message_id(resp_data: Any) -> Optional[str]:
    """Extrae key.id de respuestas Evolution (formas anidadas variables entre versiones)."""
    if not isinstance(resp_data, dict):
        return None

    candidates: list[Any] = [
        (resp_data.get("key") or {}).get("id") if isinstance(resp_data.get("key"), dict) else None,
        resp_data.get("id"),
        resp_data.get("messageId"),
        resp_data.get("message_id"),
    ]
    data = resp_data.get("data")
    if isinstance(data, dict):
        key = data.get("key")
        if isinstance(key, dict):
            candidates.append(key.get("id"))
        candidates.append(data.get("id"))
        candidates.append(data.get("messageId"))
        msg = data.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("key"), dict):
            candidates.append(msg["key"].get("id"))

    key = resp_data.get("key")
    if isinstance(key, dict):
        candidates.append(key.get("id"))

    for cand in candidates:
        if cand is not None and str(cand).strip():
            return str(cand).strip()
    return None


def register_bot_message_id(msg_id: str) -> None:
    """Registra el ID de un mensaje enviado por el bot para ignorar su eco fromMe."""
    if not msg_id:
        return
    now = time.monotonic()
    to_remove = [k for k, ts in _BOT_SENT_IDS.items() if now - ts > _BOT_SENT_TTL]
    for k in to_remove:
        del _BOT_SENT_IDS[k]
    _BOT_SENT_IDS[msg_id] = now


def is_bot_message_id(msg_id: str) -> bool:
    """Retorna True si el ID corresponde a un mensaje enviado por el bot."""
    return bool(msg_id) and msg_id in _BOT_SENT_IDS


class EvolutionClient:
    def __init__(self):
        # Lectura de variables de entorno 
        self.base_url: str = os.getenv("EVOLUTION_API_URL", "http://localhost:8080").rstrip("/")
        self.instance_name: str = os.getenv("EVOLUTION_INSTANCE_NAME", os.getenv("INSTANCE_NAME", "clinica_odonto"))
        self.api_key: str = os.getenv("EVOLUTION_API_KEY", "")
        self.timeout: float = float(os.getenv("EVOLUTION_API_TIMEOUT", "15.0"))
        self._default_delay_ms: int = int(os.getenv("EVOLUTION_SEND_DELAY_MS", "400"))

    def _get_headers(self) -> Dict[str, str]:
        """Encabezados requeridos por Evolution API."""
        return {
            "Content-Type": "application/json; charset=utf-8",
            "apikey": self.api_key
        }

    def _normalize_destination(self, numero: str) -> str:
        from app.services.whatsapp_identity import obtener_destino_envio
        return obtener_destino_envio(numero)

    async def resolver_telefono_desde_lid(self, lid_o_jid: str) -> Optional[str]:
        """Resuelve LID → E.164 vía mensajes/chats Evolution (remoteJidAlt / senderPn).

        Nunca inventa +233 ni truncamientos. None si WhatsApp no envió teléfono real.
        """
        from app.services.whatsapp_identity import (
            es_identificador_lid,
            limpiar_digitos,
            registrar_asociacion_lid,
            telefono_para_almacenar,
        )

        raw = str(lid_o_jid or "").strip()
        if not raw or not es_identificador_lid(raw):
            return telefono_para_almacenar(raw)

        lid_jid = raw if "@" in raw else f"{limpiar_digitos(raw)}@lid"
        digits = limpiar_digitos(lid_jid)

        def _phone_from_blob(obj: Any) -> Optional[str]:
            if not isinstance(obj, dict):
                return None
            key = obj.get("key") if isinstance(obj.get("key"), dict) else {}
            for cand in (
                key.get("remoteJidAlt"),
                key.get("senderPn"),
                key.get("participantAlt"),
                obj.get("senderPn"),
                obj.get("remoteJidAlt"),
            ):
                phone = telefono_para_almacenar(str(cand or ""))
                if phone:
                    return phone
            # lastMessage anidado (findChats)
            lm = obj.get("lastMessage")
            if isinstance(lm, dict):
                return _phone_from_blob(lm)
            return None

        async with httpx.AsyncClient(timeout=min(self.timeout, 12.0)) as client:
            # 1) Mensajes históricos con remoteJidAlt
            try:
                url = f"{self.base_url}/chat/findMessages/{self.instance_name}"
                resp = await client.post(
                    url,
                    json={"where": {"key": {"remoteJid": lid_jid}}, "limit": 40},
                    headers=self._get_headers(),
                )
                if resp.status_code in (200, 201):
                    data = resp.json()
                    msgs: list = []
                    if isinstance(data, list):
                        msgs = data
                    elif isinstance(data, dict):
                        block = data.get("messages") or data.get("data") or data
                        if isinstance(block, dict):
                            msgs = block.get("records") or block.get("rows") or block.get("items") or []
                        elif isinstance(block, list):
                            msgs = block
                    for item in msgs if isinstance(msgs, list) else []:
                        phone = _phone_from_blob(item)
                        if phone:
                            registrar_asociacion_lid(phone, lid_jid)
                            logger.info(
                                "[Evolution API] LID %s → %s vía findMessages",
                                lid_jid,
                                phone,
                            )
                            return phone
            except Exception as exc:
                logger.debug("[Evolution API] findMessages LID resolve falló: %s", exc)

            # 2) Chat store (última key.remoteJidAlt)
            try:
                url = f"{self.base_url}/chat/findChats/{self.instance_name}"
                resp = await client.post(
                    url,
                    json={"where": {"remoteJid": lid_jid}},
                    headers=self._get_headers(),
                )
                if resp.status_code in (200, 201):
                    data = resp.json()
                    items = data if isinstance(data, list) else (data.get("chats") or data.get("data") or [])
                    for item in items if isinstance(items, list) else []:
                        phone = _phone_from_blob(item)
                        if phone:
                            registrar_asociacion_lid(phone, lid_jid)
                            logger.info(
                                "[Evolution API] LID %s → %s vía findChats",
                                lid_jid,
                                phone,
                            )
                            return phone
            except Exception as exc:
                logger.debug("[Evolution API] findChats LID resolve falló: %s", exc)

            # 3) whatsappNumbers: solo si Evolution devolvió @s.whatsapp.net real (no el propio LID)
            try:
                url = f"{self.base_url}/chat/whatsappNumbers/{self.instance_name}"
                resp = await client.post(
                    url,
                    json={"numbers": [digits]},
                    headers=self._get_headers(),
                )
                if resp.status_code in (200, 201):
                    data = resp.json()
                    items = data if isinstance(data, list) else []
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        jid = str(item.get("jid") or item.get("number") or "")
                        if es_identificador_lid(jid):
                            continue
                        phone = telefono_para_almacenar(jid)
                        if phone and limpiar_digitos(phone) != digits:
                            registrar_asociacion_lid(phone, lid_jid)
                            logger.info(
                                "[Evolution API] LID %s → %s vía whatsappNumbers",
                                lid_jid,
                                phone,
                            )
                            return phone
            except Exception as exc:
                logger.debug("[Evolution API] whatsappNumbers LID resolve falló: %s", exc)

        return None

    async def enviar_presencia(
        self,
        numero: str,
        presencia: str = "composing",
        delay: int = 1200,
    ) -> bool:
        """Envía el estado de presencia ('composing', 'recording', 'paused')
        en tiempo real a través de Evolution API v2 para feedback visual inmediato en WhatsApp.
        Endpoint: POST /chat/sendPresence/{instance_name}
        """
        url = f"{self.base_url}/chat/sendPresence/{self.instance_name}"
        target_number = self._normalize_destination(numero)
        payload = {
            "number": target_number,
            "presence": presencia,
            "delay": delay,
        }

        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                response = await client.post(url, content=body_bytes, headers=self._get_headers())
                return response.status_code in (200, 201)
            except Exception as e:
                logger.debug(f"[Evolution API] No se pudo enviar presencia '{presencia}' a {numero}: {e}")
                return False

    async def enviar_mensaje(self, numero: str, texto: str, delay: int | None = None) -> Optional[Dict[str, Any]]:
        """
        Envía un mensaje de texto a través de Evolution API v2.
        Endpoint: POST /message/sendText/{instance_name}
        Payload: {"number": numero, "text": texto, "delay": delay}
        """
        if delay is None:
            delay = self._default_delay_ms
        url = f"{self.base_url}/message/sendText/{self.instance_name}"
        target_number = self._normalize_destination(numero)

        from app.domain.formatters.whatsapp_formatter import formatear_para_whatsapp
        texto_formateado = formatear_para_whatsapp(texto)

        # Registrar preventivamente (teléfono original + destino Evolution/LID) para el eco fromMe
        register_outgoing_bot_message(numero, texto_formateado)
        if target_number and target_number != numero:
            register_outgoing_bot_message(target_number, texto_formateado)

        payload = {
            "number": target_number,
            "options": {
                "delay": delay,
                "presence": "composing"
            },
            "text": texto_formateado
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                response = await client.post(url, content=body_bytes, headers=self._get_headers())
                response.raise_for_status()
                logger.info(f"[Evolution API] Mensaje enviado exitosamente a {numero}")
                resp_data = response.json()
                sent_id = extract_evolution_message_id(resp_data)
                if sent_id:
                    register_bot_message_id(sent_id)
                else:
                    logger.warning(
                        "[Evolution API] sendText OK pero sin key.id parseable; eco fromMe dependerá de match por texto. keys=%s",
                        list(resp_data.keys()) if isinstance(resp_data, dict) else type(resp_data),
                    )
                return resp_data
            except httpx.HTTPStatusError as e:
                logger.error(f"[Evolution API] Error HTTP {e.response.status_code} al enviar mensaje: {e.response.text}")
                return None
            except httpx.RequestError as e:
                logger.error(f"[Evolution API] Error de conexión/timeout con Evolution API (¿contenedor apagado?): {str(e)}")
                return None

    async def enviar_botones(
        self,
        numero: str,
        titulo: str,
        descripcion: str,
        botones: list[Dict[str, str]],
        pie: str = "",
        delay: int | None = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Envía un mensaje interactivo con botones de respuesta rápida vía Evolution API v2.
        Endpoint: POST /message/sendButtons/{instance_name}

        Cada botón es un dict con:
          - "buttonId": identificador técnico (ej. "CC")
          - "buttonText": {"displayText": "Cédula de Ciudadanía"}

        WhatsApp permite máximo 3 botones por mensaje.
        """
        if delay is None:
            delay = self._default_delay_ms
        url = f"{self.base_url}/message/sendButtons/{self.instance_name}"
        payload = {
            "number": numero,
            "options": {"delay": delay, "presence": "composing"},
            "buttonMessage": {
                "title": titulo,
                "description": descripcion,
                "footerText": pie,
                "buttons": botones,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                response = await client.post(url, content=body_bytes, headers=self._get_headers())
                response.raise_for_status()
                logger.info(f"[Evolution API] Botones enviados exitosamente a {numero}")
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"[Evolution API] Error HTTP {e.response.status_code} al enviar botones: {e.response.text}"
                )
                # Fallback: enviar como texto plano con opciones numeradas
                texto_fallback = f"*{titulo}*\n\n{descripcion}\n\n"
                for i, b in enumerate(botones, 1):
                    display = b.get("buttonText", {}).get("displayText", b.get("buttonId", ""))
                    texto_fallback += f"{i}. {display}\n"
                if pie:
                    texto_fallback += f"\n_{pie}_"
                return await self.enviar_mensaje(numero, texto_fallback, delay)
            except httpx.RequestError as e:
                logger.error(f"[Evolution API] Error de conexión al enviar botones: {str(e)}")
                return None

    async def enviar_lista(
        self,
        numero: str,
        titulo: str,
        descripcion: str,
        texto_boton: str,
        secciones: list[Dict[str, Any]],
        pie: str = "",
        delay: int | None = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Envía un mensaje interactivo tipo lista de selección vía Evolution API v2.
        Endpoint: POST /message/sendList/{instance_name}

        Cada sección tiene:
          - "title": nombre de la sección
          - "rows": [{"title": "Opción", "description": "Detalle", "rowId": "ID_TECNICO"}]
        """
        if delay is None:
            delay = self._default_delay_ms
        url = f"{self.base_url}/message/sendList/{self.instance_name}"
        payload = {
            "number": numero,
            "options": {"delay": delay, "presence": "composing"},
            "listMessage": {
                "title": titulo,
                "description": descripcion,
                "footerText": pie,
                "buttonText": texto_boton,
                "sections": secciones,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                response = await client.post(url, content=body_bytes, headers=self._get_headers())
                response.raise_for_status()
                logger.info(f"[Evolution API] Lista enviada exitosamente a {numero}")
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"[Evolution API] Error HTTP {e.response.status_code} al enviar lista: {e.response.text}"
                )
                # Fallback: enviar como texto plano con opciones numeradas
                texto_fallback = f"*{titulo}*\n\n{descripcion}\n\n"
                for sec in secciones:
                    for row in sec.get("rows", []):
                        texto_fallback += f"• {row.get('title', '')} — {row.get('description', '')}\n"
                if pie:
                    texto_fallback += f"\n_{pie}_"
                texto_fallback += f"\n\nResponde con el nombre de la opción que deseas."
                return await self.enviar_mensaje(numero, texto_fallback, delay)
            except httpx.RequestError as e:
                logger.error(f"[Evolution API] Error de conexión al enviar lista: {str(e)}")
                return None

    async def obtener_base64_media(self, message: Dict[str, Any], convert_to_mp4: bool = False) -> Optional[Dict[str, Any]]:
        """
        Obtiene el contenido binario/base64 de un mensaje multimedia desde Evolution API v2.
        Endpoint: POST /chat/getBase64FromMediaMessage/{instance_name}
        Payload: {"message": message, "convertToMp4": convert_to_mp4}
        """
        url = f"{self.base_url}/chat/getBase64FromMediaMessage/{self.instance_name}"
        payload = {
            "message": message,
            "convertToMp4": convert_to_mp4
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                response = await client.post(url, content=body_bytes, headers=self._get_headers())
                response.raise_for_status()
                data = response.json()
                logger.info("[Evolution API] Base64 multimedia obtenido exitosamente")
                return data
            except httpx.HTTPStatusError as e:
                logger.error(f"[Evolution API] Error HTTP {e.response.status_code} al obtener base64: {e.response.text}")
                return None
    async def ensure_webhook_configured(self) -> bool:
        """Asegura de forma idempotente que Evolution API tenga configurado el webhook para la instancia."""
        webhook_url = os.getenv("EVOLUTION_WEBHOOK_URL", "http://agente-python:8000/webhook/whatsapp")
        webhook_secret = os.getenv("WEBHOOK_SECRET", "SECRETO_COMPARTIDO_CON_EVOLUTION")
        url_find = f"{self.base_url}/webhook/find/{self.instance_name}"
        url_set = f"{self.base_url}/webhook/set/{self.instance_name}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.get(url_find, headers=self._get_headers())
                if resp.status_code == 200:
                    data = resp.json()
                    wh = data.get("webhook") if isinstance(data, dict) and isinstance(data.get("webhook"), dict) else data
                    events = []
                    if isinstance(wh, dict):
                        events = wh.get("events") or []
                    # Reconfigurar si falta URL o si aún incluye MESSAGES_UPDATE (ecos fromMe / flood)
                    events_upper = {str(e).upper() for e in events}
                    upsert_only_ok = (
                        isinstance(wh, dict)
                        and wh.get("url") == webhook_url
                        and wh.get("enabled")
                        and "MESSAGES_UPSERT" in events_upper
                        and "MESSAGES_UPDATE" not in events_upper
                    )
                    if upsert_only_ok:
                        logger.debug("[Evolution API] Webhook ya configurado correctamente para %s", self.instance_name)
                        return True

                headers_dict = {"apikey": self.api_key}
                if webhook_secret:
                    headers_dict["Authorization"] = f"Bearer {webhook_secret}"

                payload = {
                    "webhook": {
                        "enabled": True,
                        "url": webhook_url,
                        "headers": headers_dict,
                        "byEvents": False,
                        "base64": True,
                        # Solo UPSERT: UPDATE reenvía fromMe y provoca spam de ecos
                        "events": [
                            "MESSAGES_UPSERT",
                            "CONNECTION_UPDATE",
                        ],
                    }
                }
                resp_set = await client.post(url_set, json=payload, headers=self._get_headers())
                if resp_set.status_code in (200, 201):
                    logger.info("[Evolution API] Webhook configurado exitosamente para instancia %s", self.instance_name)
                    return True
                logger.warning(
                    "[Evolution API] Falló configuración de webhook (HTTP %s): %s",
                    resp_set.status_code,
                    resp_set.text,
                )
            except Exception as e:
                logger.warning("[Evolution API] Error verificando/configurando webhook: %s", e)
            return False


# Instancia singleton para reutilizar en el agente
evolution_client = EvolutionClient()