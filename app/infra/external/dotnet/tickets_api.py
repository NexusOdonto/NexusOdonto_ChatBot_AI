"""Endpoints de Tickets de Soporte, Notificaciones y Auditoría de Mensajes en .NET."""

import logging
import asyncio
import time
import re
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
import threading

from app.infra.external.dotnet.http_transport import (
    dotnet_transport,
    DotNetHttpTransport,
    loop_safe_asyncio_lock,
)
from app.infra.external.dotnet.catalog_api import catalog_api
from app.infra.external.dotnet.patients_api import patients_api

logger = logging.getLogger(__name__)


class DotNetTicketsApi:
    CHANNEL_WHATSAPP: str = "90000000-0000-0000-0000-000000000001"
    CHANNEL_TELEGRAM: str = "90000000-0000-0000-0000-000000000002"
    CHANNEL_WEBCHAT: str = "90000000-0000-0000-0000-000000000003"

    ROLE_USUARIO: str = "a0000000-0000-0000-0000-000000000001"
    ROLE_CHATBOT: str = "a0000000-0000-0000-0000-000000000002"
    ROLE_AGENTE_HUMANO: str = "a0000000-0000-0000-0000-000000000003"
    ROLE_SISTEMA: str = "a0000000-0000-0000-0000-000000000004"

    STATUS_ACTIVA: str = "b0000000-0000-0000-0000-000000000001"
    STATUS_ESCALADA: str = "b0000000-0000-0000-0000-000000000002"
    STATUS_ATENDIDA_HUMANO: str = "b0000000-0000-0000-0000-000000000003"
    STATUS_CERRADA: str = "b0000000-0000-0000-0000-000000000004"

    def __init__(self, transport: Optional[DotNetHttpTransport] = None):
        self.transport = transport or dotnet_transport
        self._conversations_cache: Dict[str, Tuple[str, float]] = {}
        self._conversations_list_cache: Optional[Tuple[List[Dict[str, Any]], float]] = None
        self._conversations_list_ttl: float = 10.0
        self._conversations_locks: Dict[int, asyncio.Lock] = {}
        self._conversations_locks_guard = threading.Lock()
        self._reasons_cache: Optional[List[Dict[str, Any]]] = None
        self._priorities_cache: Optional[List[Dict[str, Any]]] = None

    def _conversations_lock(self) -> asyncio.Lock:
        return loop_safe_asyncio_lock(self._conversations_locks, self._conversations_locks_guard)

    async def _obtener_ticket_reasons(self) -> List[Dict[str, Any]]:
        """Obtiene y cachea los motivos de ticket de soporte disponibles en .NET."""
        if self._reasons_cache:
            return self._reasons_cache
        response = await self.transport.request("GET", "SupportTicketReasons")
        if response and response.status_code == 200:
            self._reasons_cache = response.json() if isinstance(response.json(), list) else []
            return self._reasons_cache
        return []

    async def _obtener_notification_priorities(self) -> List[Dict[str, Any]]:
        """Obtiene y cachea las prioridades de notificación disponibles en .NET."""
        if self._priorities_cache:
            return self._priorities_cache
        response = await self.transport.request("GET", "NotificationPriorities")
        if response and response.status_code == 200:
            self._priorities_cache = response.json() if isinstance(response.json(), list) else []
            return self._priorities_cache
        return []

    async def actualizar_estado_conversacion(
        self, conversacion_id: str, status_id: str, patient_id: Optional[str] = None
    ) -> bool:
        """Actualiza el estado de una conversación en el backend .NET / Oracle."""
        try:
            payload: Dict[str, Any] = {
                "conversationStatusId": status_id,
                "patientId": patient_id,
            }
            if status_id.lower() == self.STATUS_ACTIVA.lower():
                payload["assignedEmployeeId"] = None
                payload["employeeId"] = None
                payload["assignedUserId"] = None
                payload["closedAt"] = None

            resp = await self.transport.request("PUT", f"ChatbotConversations/{conversacion_id}", json=payload)
            if resp and resp.status_code in (200, 204):
                logger.info(f"[TicketsApi] Estado de conversación {conversacion_id} actualizado a {status_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"[TicketsApi] Error actualizando estado de conversación {conversacion_id}: {e}")
            return False

    async def resolver_tickets_conversacion(self, conversacion_id: str) -> int:
        """Cierra o resuelve cualquier ticket de soporte abierto para una conversación."""
        if not conversacion_id:
            return 0
        try:
            tickets = await catalog_api.obtener_catalogo("SupportTickets") or []
            STATUS_RESUELTO = "c0000000-0000-0000-0000-000000000004"
            STATUS_CERRADO = "c0000000-0000-0000-0000-000000000005"
            conv_id_clean = str(conversacion_id).strip().lower()

            cerrados = 0
            for t in tickets:
                if not isinstance(t, dict):
                    continue
                t_conv = str(t.get("chatbotConversationId", "")).strip().lower()
                t_status = str(t.get("ticketStatusId", "")).strip().lower()
                if t_conv == conv_id_clean and t_status not in (STATUS_RESUELTO, STATUS_CERRADO):
                    ticket_id = t.get("id")
                    if ticket_id:
                        payload = dict(t)
                        payload["ticketStatusId"] = STATUS_RESUELTO
                        payload["resolvedAt"] = datetime.now(timezone.utc).isoformat()
                        payload["resolutionNotes"] = "Atención finalizada por asesor. Conversación reactivada con NexusBot."
                        resp = await self.transport.request("PUT", f"SupportTickets/{ticket_id}", json=payload)
                        if resp and resp.status_code in (200, 204):
                            cerrados += 1
            return cerrados
        except Exception as e:
            logger.warning(f"[TicketsApi] Error resolviendo tickets para conversación {conversacion_id}: {e}")
            return 0

    async def crear_ticket_soporte(
        self, telefono: str, motivo: str, prioridad: str = "MEDIA"
    ) -> Optional[Dict[str, Any]]:
        """Crea un ticket en la API de .NET vinculado a la conversación."""
        reasons = await self._obtener_ticket_reasons()
        priorities = await self._obtener_notification_priorities()

        reason_id = None
        motivo_upper = motivo.upper()
        for r in reasons:
            code = (r.get("code") or "").upper()
            if "EMERGENCIA" in motivo_upper and code in ("CONSULTA_COMPLEJA", "SOLICITUD_USUARIO"):
                reason_id = r.get("id")
                break
            if "SOLICITUD" in motivo_upper and "SOLICITUD" in code:
                reason_id = r.get("id")
                break
            if "RAG" in motivo_upper and "RAG" in code:
                reason_id = r.get("id")
                break
        if not reason_id and reasons:
            reason_id = reasons[0].get("id")
        if not reason_id:
            reason_id = "d0000000-0000-0000-0000-000000000002" if "RAG" in motivo_upper else "d0000000-0000-0000-0000-000000000001"

        priority_id = None
        prio_upper = prioridad.upper()
        for p in priorities:
            code = (p.get("code") or "").upper()
            if prio_upper in ("CRITICO", "CRÍTICO", "URGENTE") and code == "URGENTE":
                priority_id = p.get("id")
                break
            if prio_upper in ("ALTA", "HIGH") and code == "ALTA":
                priority_id = p.get("id")
                break
            if prio_upper in ("MEDIA", "NORMAL") and code == "NORMAL":
                priority_id = p.get("id")
                break
        if not priority_id and priorities:
            priority_id = priorities[0].get("id")
        if not priority_id:
            priority_id = "50000000-0000-0000-0000-000000000004" if prio_upper in ("CRITICO", "URGENTE") else "50000000-0000-0000-0000-000000000002"

        conv_id = await self.obtener_o_crear_conversacion(telefono)
        patient_id = None
        try:
            persona = await patients_api.buscar_persona_por_telefono(telefono)
            if persona and persona.get("id"):
                paciente = await patients_api.buscar_paciente_por_person_id(str(persona["id"]))
                if paciente:
                    patient_id = paciente.get("id")
        except Exception:
            pass

        STATUS_ABIERTO = "c0000000-0000-0000-0000-000000000001"
        payload = {
            "ticketReasonId": str(reason_id),
            "priorityId": str(priority_id),
            "ticketStatusId": STATUS_ABIERTO,
            "chatbotConversationId": conv_id,
            "patientId": patient_id,
            "title": f"Escalamiento WhatsApp: {telefono}",
            "description": motivo,
        }

        response = await self.transport.request("POST", "SupportTickets", json=payload)
        if response and response.status_code in (200, 201):
            logger.info(f"[TicketsApi] Ticket de soporte creado exitosamente para {telefono}")
            return response.json()
        return None

    async def crear_notificacion(
        self,
        titulo: str,
        mensaje: str,
        prioridad: str = "MEDIA",
        conversation_id: Optional[str] = None,
        telefono: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Genera una notificación en el panel de recepción de .NET."""
        priorities = await self._obtener_notification_priorities()
        priority_id = None
        prio_upper = prioridad.upper()
        for p in priorities:
            code = (p.get("code") or "").upper()
            if prio_upper in ("CRITICO", "CRÍTICO", "URGENTE") and code == "URGENTE":
                priority_id = p.get("id")
                break
            if prio_upper in ("ALTA", "HIGH") and code == "ALTA":
                priority_id = p.get("id")
                break
            if prio_upper in ("MEDIA", "NORMAL") and code == "NORMAL":
                priority_id = p.get("id")
                break
        if not priority_id and priorities:
            priority_id = priorities[0].get("id")
        if not priority_id:
            priority_id = "50000000-0000-0000-0000-000000000004" if prio_upper in ("CRITICO", "URGENTE") else "50000000-0000-0000-0000-000000000002"

        payload = {
            "title": titulo,
            "message": mensaje,
            "priorityId": str(priority_id),
            "chatbotConversationId": conversation_id,
            "isRead": False,
        }
        resp = await self.transport.request("POST", "Notifications", json=payload)
        if resp and resp.status_code in (200, 201):
            return resp.json()
        return None


    def peek_cached_conversation_id(self, chat_identifier: str) -> Optional[str]:
        """Devuelve conversationId en caché local si aún es válido (<15s)."""
        from app.services.whatsapp_identity import obtener_telefono_canonico
        raw_tid = str(chat_identifier or "").strip()
        ident = obtener_telefono_canonico(raw_tid).strip()
        now = time.monotonic()
        for key in (ident, raw_tid):
            cached = self._conversations_cache.get(key)
            if cached:
                conv_id, cached_at = cached
                if (now - cached_at) < 15.0:
                    return conv_id
        return None

    async def _list_conversations(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """GET ChatbotConversations con TTL corto (evita duplicar el listado en el mismo turno)."""
        now = time.monotonic()
        if (
            not force_refresh
            and self._conversations_list_cache is not None
            and (now - self._conversations_list_cache[1]) < self._conversations_list_ttl
        ):
            return self._conversations_list_cache[0]
        resp = await self.transport.request("GET", "ChatbotConversations")
        items: List[Dict[str, Any]] = []
        if resp and resp.status_code == 200:
            try:
                payload = resp.json()
                if isinstance(payload, dict):
                    items = payload.get("items", []) or []
                elif isinstance(payload, list):
                    items = payload
            except Exception as e:
                logger.debug(f"[TicketsApi] Error parseando lista de conversaciones: {e}")
        self._conversations_list_cache = (items, now)
        return items

    async def obtener_contexto_conversacion(
        self, conversacion_chatbot_id: str
    ) -> Optional[Dict[str, Any]]:
        """Obtiene el contexto completo de una conversación por su ID."""
        response = await self.transport.request("GET", f"ChatbotConversations/{conversacion_chatbot_id}")
        if response and response.status_code == 200:
            return response.json()
        return None

    def _set_cached_conv(self, ident: str, raw_tid: str, clean_tid: str, conv_id: str, now: float) -> None:
        val = (conv_id, now)
        if ident:
            self._conversations_cache[ident] = val
        if raw_tid:
            self._conversations_cache[raw_tid] = val
        if clean_tid:
            self._conversations_cache[clean_tid] = val

    def _peek_fresh_cached(
        self, ident: str, raw_tid: str, now: float
    ) -> Optional[str]:
        cached = self._conversations_cache.get(ident) or self._conversations_cache.get(raw_tid)
        if cached:
            conv_id, cached_at = cached
            if (now - cached_at) < 15.0:
                return conv_id
        return None

    async def obtener_o_crear_conversacion(
        self,
        chat_identifier: str,
        channel_id: Optional[str] = None,
        patient_id: Optional[str] = None,
    ) -> Optional[str]:
        """Obtiene el ID de una conversación activa o crea una nueva en el backend .NET.

        El lock solo protege el caché local / claim de create; list/create/PUT HTTP
        corren fuera del lock para que 5 chats no se serialicen en un único candado.
        """
        from app.services.whatsapp_identity import (
            es_identificador_lid,
            obtener_telefono_canonico,
            telefono_para_almacenar,
        )
        raw_tid = str(chat_identifier or "").strip()
        canon = obtener_telefono_canonico(raw_tid).strip()
        if not canon:
            return None

        # Preferir E.164 real para chatIdentifier; nunca truncar LID a 10 dígitos "teléfono"
        phone_e164 = telefono_para_almacenar(canon)
        if not phone_e164 and es_identificador_lid(canon):
            # Antes de crear/usar LID: intentar resolver teléfono real vía Evolution
            try:
                from app.clients.evolution_client import evolution_client

                resolved = await evolution_client.resolver_telefono_desde_lid(canon)
                if resolved:
                    phone_e164 = resolved
                    logger.info(
                        "[TicketsApi] Identidad LID resuelta antes de obtener/crear conv: %s → %s",
                        canon,
                        phone_e164,
                    )
            except Exception as resolve_err:
                logger.debug("[TicketsApi] Resolve LID previo a create falló: %s", resolve_err)

        if phone_e164:
            ident = phone_e164
        elif es_identificador_lid(canon):
            digits = re.sub(r"\D", "", canon)
            ident = canon if "@" in canon else f"{digits}@lid"
        else:
            ident = canon

        clean_tid = re.sub(r"\D", "", phone_e164 or "")
        if phone_e164 and len(clean_tid) > 10:
            clean_tid = clean_tid[-10:]
        elif not phone_e164:
            # LID: no usar últimos 10 como clave de matching telefónico
            clean_tid = ""

        now = time.monotonic()
        hit = self._peek_fresh_cached(ident, raw_tid, now)
        if hit:
            return hit

        # HTTP list fuera del lock
        items = await self._list_conversations()
        try:
            matching_convs = []
            for conv in items:
                c_chat = str(conv.get("chatIdentifier", "")).strip()
                c_canon = obtener_telefono_canonico(c_chat)
                c_phone = telefono_para_almacenar(c_canon)
                c_digits = re.sub(r"\D", "", c_phone or "")
                if c_phone and len(c_digits) > 10:
                    c_digits = c_digits[-10:]
                elif not c_phone:
                    c_digits = ""

                matches = (
                    c_chat == raw_tid
                    or c_chat == ident
                    or c_canon == canon
                    or (phone_e164 and c_phone and phone_e164 == c_phone)
                    or (clean_tid and c_digits and clean_tid == c_digits)
                )
                if matches:
                    matching_convs.append(conv)

            # LID-only: reutilizar hilos con el mismo @lid (asesor debe ver el chat).
            # Si hay E.164, matching ya favorece teléfono; no descartar coincidencias LID.

            if matching_convs:
                def _sort_key(c):
                    return str(c.get("lastInteractionAt") or c.get("startedAt") or "")

                # 1. Prioridad: Conversación en atención humana (ATENDIDA_HUMANO) o ESCALADA
                human_escalated = [
                    c for c in matching_convs
                    if str(c.get("conversationStatusId", "")).lower() in (
                        self.STATUS_ATENDIDA_HUMANO.lower(),
                        self.STATUS_ESCALADA.lower(),
                    ) and not c.get("closedAt")
                ]
                if human_escalated:
                    human_escalated.sort(key=_sort_key, reverse=True)
                    chosen = human_escalated[0]
                    chosen_id = str(chosen["id"])
                    await self._maybe_repair_chat_identifier(chosen, ident, phone_e164)
                    async with self._conversations_lock():
                        now = time.monotonic()
                        hit = self._peek_fresh_cached(ident, raw_tid, now)
                        if hit:
                            return hit
                        self._set_cached_conv(ident, raw_tid, clean_tid, chosen_id, now)
                    return chosen_id

                # 2. Prioridad: Conversación ACTIVA (abierta y sin cerrar)
                active_convs = [
                    c for c in matching_convs
                    if str(c.get("conversationStatusId", "")).lower() == self.STATUS_ACTIVA.lower()
                    and not c.get("closedAt")
                ]
                if active_convs:
                    active_convs.sort(key=_sort_key, reverse=True)
                    chosen = active_convs[0]
                    chosen_id = str(chosen["id"])
                    await self._maybe_repair_chat_identifier(chosen, ident, phone_e164)
                    async with self._conversations_lock():
                        now = time.monotonic()
                        hit = self._peek_fresh_cached(ident, raw_tid, now)
                        if hit:
                            return hit
                        self._set_cached_conv(ident, raw_tid, clean_tid, chosen_id, now)
                    return chosen_id

                # 3. Si todas están cerradas, tomar la MÁS RECIENTE para reabrirla
                matching_convs.sort(key=_sort_key, reverse=True)
                most_recent = matching_convs[0]
                chosen_id = str(most_recent["id"])
                try:
                    payload_put = {
                        "conversationStatusId": self.STATUS_ACTIVA,
                        "patientId": most_recent.get("patientId") or patient_id,
                        "closedAt": None,
                    }
                    if phone_e164 and (
                        es_identificador_lid(str(most_recent.get("chatIdentifier") or ""))
                        or not telefono_para_almacenar(str(most_recent.get("chatIdentifier") or ""))
                    ):
                        payload_put["chatIdentifier"] = phone_e164
                    await self.transport.request("PUT", f"ChatbotConversations/{chosen_id}", json=payload_put)
                    logger.info(f"[TicketsApi] Conversación {chosen_id} reabierta como ACTIVA para {ident}")
                except Exception as e_put:
                    logger.warning(f"[TicketsApi] Error al reabrir conversación {chosen_id}: {e_put}")

                async with self._conversations_lock():
                    now = time.monotonic()
                    hit = self._peek_fresh_cached(ident, raw_tid, now)
                    if hit:
                        return hit
                    self._set_cached_conv(ident, raw_tid, clean_tid, chosen_id, now)
                return chosen_id
        except Exception as e:
            logger.debug(f"[TicketsApi] Error parseando conversaciones: {e}")

        # Autovincular paciente — HTTP fuera del lock
        if not patient_id:
            try:
                persona = await patients_api.buscar_persona_por_telefono(ident)
                if persona and persona.get("id"):
                    paciente = await patients_api.buscar_paciente_por_person_id(str(persona["id"]))
                    if paciente and paciente.get("id"):
                        patient_id = str(paciente["id"])
            except Exception as pat_err:
                logger.debug(f"[TicketsApi] No se pudo autovincular paciente: {pat_err}")

        # Claim de create: solo actualizar caché bajo lock; POST fuera
        async with self._conversations_lock():
            now = time.monotonic()
            hit = self._peek_fresh_cached(ident, raw_tid, now)
            if hit:
                return hit

        # Siempre crear hilo de asesor, incluso con LID temporal.
        # Preferimos E.164 cuando Evolution resuelve remoteJidAlt/senderPn;
        # si solo hay LID, se guarda el @lid y se repara a teléfono al resolver.
        if not phone_e164 and es_identificador_lid(ident):
            logger.info(
                "[TicketsApi] Creando conversación con identidad LID temporal (%s). "
                "Se actualizará a E.164 cuando Evolution resuelva remoteJidAlt/senderPn.",
                ident,
            )

        payload = {
            "chatIdentifier": ident,
            "chatChannelId": channel_id or self.CHANNEL_WHATSAPP,
            "patientId": patient_id,
        }
        resp_post = await self.transport.request("POST", "ChatbotConversations", json=payload)
        if resp_post and resp_post.status_code in (200, 201):
            try:
                data = resp_post.json()
                conv_id = str(data.get("id"))
                if conv_id:
                    async with self._conversations_lock():
                        now = time.monotonic()
                        hit = self._peek_fresh_cached(ident, raw_tid, now)
                        if hit:
                            return hit
                        self._set_cached_conv(ident, raw_tid, clean_tid, conv_id, now)
                    return conv_id
            except Exception as e:
                logger.error(f"[TicketsApi] Error parseando creación de conversación: {e}")

        return None

    async def _maybe_repair_chat_identifier(
        self,
        conv: dict,
        ident: str,
        phone_e164: Optional[str],
    ) -> None:
        """Si la conversación tiene LID/basura y ya resolvimos E.164, actualizar chatIdentifier."""
        from app.services.whatsapp_identity import es_identificador_lid, telefono_para_almacenar

        if not phone_e164:
            return
        current = str(conv.get("chatIdentifier") or "")
        if telefono_para_almacenar(current) == phone_e164:
            return
        if not (es_identificador_lid(current) or not telefono_para_almacenar(current)):
            return
        try:
            payload_put = {
                "conversationStatusId": conv.get("conversationStatusId"),
                "patientId": conv.get("patientId"),
                "closedAt": conv.get("closedAt"),
                "chatIdentifier": phone_e164,
            }
            await self.transport.request("PUT", f"ChatbotConversations/{conv['id']}", json=payload_put)
            logger.info(
                "[TicketsApi] chatIdentifier reparado conv=%s %s → %s",
                conv.get("id"),
                current,
                phone_e164,
            )
            conv["chatIdentifier"] = phone_e164
        except Exception as err:
            logger.warning("[TicketsApi] No se pudo reparar chatIdentifier: %s", err)

    async def guardar_mensaje_conversacion(
        self,
        conversation_id: str,
        rol: str,
        contenido: str,
        rag_confidence: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Guarda un mensaje en la tabla chatbot_messages vía .NET."""
        if not conversation_id or not contenido:
            return None

        rol_upper = str(rol).upper().strip()
        if rol_upper in ("USUARIO", "USER", "HUMAN"):
            role_id = self.ROLE_USUARIO
        elif rol_upper in ("CHATBOT", "ASSISTANT", "BOT"):
            role_id = self.ROLE_CHATBOT
        elif rol_upper in ("AGENTE_HUMANO", "ASESOR", "HUMAN_AGENT"):
            role_id = self.ROLE_AGENTE_HUMANO
        else:
            role_id = self.ROLE_SISTEMA

        payload = {
            "chatbotConversationId": conversation_id,
            "messageRoleId": role_id,
            "content": str(contenido).strip(),
            "ragConfidence": float(rag_confidence) if rag_confidence is not None else None,
        }

        resp = await self.transport.request("POST", "ChatbotMessages", json=payload)
        if resp and resp.status_code in (200, 201):
            try:
                return resp.json()
            except Exception:
                return payload
        return None

    async def registrar_mensaje(
        self,
        chat_identifier: str,
        rol: str,
        contenido: str,
        rag_confidence: Optional[float] = None,
        patient_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Registra un mensaje asegurando que la conversación exista en base de datos.
        No sobreescribe el estado de la conversación para no desarmar la atención de asesor humano.
        """
        try:
            if not contenido or not str(contenido).strip():
                return None
            conv_id = await self.obtener_o_crear_conversacion(
                chat_identifier=chat_identifier,
                patient_id=patient_id,
            )
            if not conv_id:
                return None
            res = await self.guardar_mensaje_conversacion(
                conversation_id=conv_id,
                rol=rol,
                contenido=contenido,
                rag_confidence=rag_confidence,
            )
            return res
        except Exception as e:
            logger.error(f"[TicketsApi] Error registrando mensaje para {chat_identifier}: {e}")
            return None

    def limpiar_cache_conversacion(self, chat_identifier: str) -> None:
        """Limpia el ID en caché cuando la conversación se reinicia."""
        from app.services.whatsapp_identity import (
            es_identificador_lid,
            obtener_telefono_canonico,
            telefono_para_almacenar,
        )
        raw_tid = str(chat_identifier or "").strip()
        canon = obtener_telefono_canonico(raw_tid).strip()
        phone_e164 = telefono_para_almacenar(canon) if canon else None
        if phone_e164:
            ident = phone_e164
        elif canon and es_identificador_lid(canon):
            digits = re.sub(r"\D", "", canon)
            ident = canon if "@" in canon else f"{digits}@lid"
        else:
            ident = canon

        clean_tid = re.sub(r"\D", "", phone_e164 or "")
        if phone_e164 and len(clean_tid) > 10:
            clean_tid = clean_tid[-10:]
        elif not phone_e164:
            clean_tid = ""

        self._conversations_cache.pop(ident, None)
        self._conversations_cache.pop(raw_tid, None)
        if clean_tid:
            self._conversations_cache.pop(clean_tid, None)


tickets_api = DotNetTicketsApi()
