"""Endpoints de Tickets de Soporte, Notificaciones y Auditoría de Mensajes en .NET."""

import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
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
        self._conversations_cache: Dict[str, str] = {}
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

    async def obtener_contexto_conversacion(
        self, conversacion_chatbot_id: str
    ) -> Optional[Dict[str, Any]]:
        """Obtiene el contexto completo de una conversación por su ID."""
        response = await self.transport.request("GET", f"ChatbotConversations/{conversacion_chatbot_id}")
        if response and response.status_code == 200:
            return response.json()
        return None

    async def obtener_o_crear_conversacion(
        self,
        chat_identifier: str,
        channel_id: Optional[str] = None,
        patient_id: Optional[str] = None,
    ) -> Optional[str]:
        """Obtiene el ID de una conversación activa o crea una nueva en el backend .NET."""
        from app.services.whatsapp_identity import obtener_telefono_canonico
        ident = obtener_telefono_canonico(str(chat_identifier or "")).strip()
        if not ident:
            return None

        if ident in self._conversations_cache:
            return self._conversations_cache[ident]

        async with self._conversations_lock():
            if ident in self._conversations_cache:
                return self._conversations_cache[ident]

            raw_tid = str(chat_identifier or "").strip()
            clean_tid = re.sub(r"\D", "", ident)
            if len(clean_tid) > 10:
                clean_tid = clean_tid[-10:]

            resp = await self.transport.request("GET", "ChatbotConversations")
            if resp and resp.status_code == 200:
                try:
                    items = resp.json()
                    if isinstance(items, dict):
                        items = items.get("items", [])
                    for conv in items:
                        c_chat = str(conv.get("chatIdentifier", "")).strip()
                        c_canon = obtener_telefono_canonico(c_chat)
                        c_digits = re.sub(r"\D", "", c_canon)
                        if len(c_digits) > 10:
                            c_digits = c_digits[-10:]

                        matches = (
                            c_chat == raw_tid
                            or c_chat == ident
                            or c_canon == ident
                            or (clean_tid and c_digits and clean_tid == c_digits)
                        )
                        if matches:
                            conv_id = str(conv.get("id"))
                            conv_status = str(conv.get("conversationStatusId", "")).lower()
                            is_human_or_escalated = conv_status in (
                                self.STATUS_ESCALADA.lower(),
                                self.STATUS_ATENDIDA_HUMANO.lower(),
                            )
                            if (conv.get("closedAt") or conv_status == self.STATUS_CERRADA.lower()) and not is_human_or_escalated:
                                try:
                                    payload_put = {
                                        "conversationStatusId": self.STATUS_ACTIVA,
                                        "patientId": conv.get("patientId") or patient_id,
                                        "closedAt": None,
                                    }
                                    await self.transport.request("PUT", f"ChatbotConversations/{conv_id}", json=payload_put)
                                    logger.info(f"[TicketsApi] Conversación {conv_id} reabierta como ACTIVA para {ident}")
                                except Exception as e_put:
                                    logger.warning(f"[TicketsApi] Error al reabrir conversación {conv_id}: {e_put}")

                            self._conversations_cache[ident] = conv_id
                            self._conversations_cache[raw_tid] = conv_id
                            if clean_tid:
                                self._conversations_cache[clean_tid] = conv_id
                            return conv_id
                except Exception as e:
                    logger.debug(f"[TicketsApi] Error parseando conversaciones: {e}")

            if not patient_id:
                try:
                    persona = await patients_api.buscar_persona_por_telefono(ident)
                    if persona and persona.get("id"):
                        paciente = await patients_api.buscar_paciente_por_person_id(str(persona["id"]))
                        if paciente and paciente.get("id"):
                            patient_id = str(paciente["id"])
                except Exception as pat_err:
                    logger.debug(f"[TicketsApi] No se pudo autovincular paciente: {pat_err}")

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
                        self._conversations_cache[ident] = conv_id
                        return conv_id
                except Exception as e:
                    logger.error(f"[TicketsApi] Error parseando creación de conversación: {e}")

            return None

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
        """Registra un mensaje asegurando que la conversación exista en base de datos."""
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

            # Si el usuario escribió, asegurar que la conversación quede activa y con timestamp al día en recepción
            if str(rol).upper().strip() in ("USUARIO", "USER", "HUMAN"):
                try:
                    payload_touch = {
                        "conversationStatusId": self.STATUS_ACTIVA,
                        "closedAt": None,
                    }
                    asyncio.create_task(
                        self.transport.request("PUT", f"ChatbotConversations/{conv_id}", json=payload_touch)
                    )
                except Exception as e_touch:
                    logger.debug(f"[TicketsApi] No se pudo refrescar timestamp de conversación {conv_id}: {e_touch}")

            return res
        except Exception as e:
            logger.error(f"[TicketsApi] Error registrando mensaje para {chat_identifier}: {e}")
            return None

    def limpiar_cache_conversacion(self, chat_identifier: str) -> None:
        """Limpia el ID en caché cuando la conversación se reinicia."""
        from app.services.whatsapp_identity import obtener_telefono_canonico
        ident = obtener_telefono_canonico(str(chat_identifier or "")).strip()
        self._conversations_cache.pop(ident, None)
        self._conversations_cache.pop(str(chat_identifier).strip(), None)


tickets_api = DotNetTicketsApi()
