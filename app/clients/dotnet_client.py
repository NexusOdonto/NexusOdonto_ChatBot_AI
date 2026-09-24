"""Fachada unificada para la comunicación con la API .NET de Nexus Odonto.
Delega en los clientes de infraestructura especializados bajo app/infra/external/dotnet/.
Garantiza 100% de compatibilidad hacia atrás con todos los servicios y agentes existentes.
"""

import logging
from typing import Optional, Dict, Any, List
from app.infra.external.dotnet.http_transport import dotnet_transport
from app.infra.external.dotnet.catalog_api import catalog_api
from app.infra.external.dotnet.patients_api import patients_api
from app.infra.external.dotnet.appointments_api import appointments_api
from app.infra.external.dotnet.tickets_api import tickets_api
from app.services.appointment.appointment_service import appointment_service

logger = logging.getLogger(__name__)


class DotNetClient:
    """Fachada que centraliza los servicios de .NET de Nexus Odonto."""

    CHANNEL_WHATSAPP: str = tickets_api.CHANNEL_WHATSAPP
    CHANNEL_TELEGRAM: str = tickets_api.CHANNEL_TELEGRAM
    CHANNEL_WEBCHAT: str = tickets_api.CHANNEL_WEBCHAT

    ROLE_USUARIO: str = tickets_api.ROLE_USUARIO
    ROLE_CHATBOT: str = tickets_api.ROLE_CHATBOT
    ROLE_AGENTE_HUMANO: str = tickets_api.ROLE_AGENTE_HUMANO
    ROLE_SISTEMA: str = tickets_api.ROLE_SISTEMA

    STATUS_ACTIVA: str = tickets_api.STATUS_ACTIVA
    STATUS_ESCALADA: str = tickets_api.STATUS_ESCALADA
    STATUS_ATENDIDA_HUMANO: str = tickets_api.STATUS_ATENDIDA_HUMANO
    STATUS_CERRADA: str = tickets_api.STATUS_CERRADA

    def __init__(self):
        self.transport = dotnet_transport
        self.catalog = catalog_api
        self.patients = patients_api
        self.appointments = appointments_api
        self.tickets = tickets_api
        self.appointment_svc = appointment_service

    @property
    def base_url(self) -> str:
        return self.transport.base_url

    @property
    def timeout(self) -> float:
        return self.transport.timeout

    async def _request_with_retry(
        self, method: str, url: str, params: Optional[Dict[str, Any]] = None, json: Optional[Any] = None
    ):
        return await self.transport.request(method, url, params=params, json=json)

    # ─────────────────────────────────────────────────────────────
    # Catálogo, Especialidades y Disponibilidad
    # ─────────────────────────────────────────────────────────────
    async def obtener_servicios(self) -> Optional[List[Dict[str, Any]]]:
        return await self.catalog.obtener_servicios()

    async def obtener_especialidades(self) -> Optional[List[Dict[str, Any]]]:
        return await self.catalog.obtener_especialidades()

    async def obtener_empleados(self) -> List[Dict[str, Any]]:
        return await self.catalog.obtener_empleados()

    async def obtener_personas(self) -> List[Dict[str, Any]]:
        return await self.catalog.obtener_personas()

    async def obtener_profesionales(self, especialidad_id: Optional[Any] = None) -> Optional[List[Dict[str, Any]]]:
        return await self.catalog.obtener_profesionales(especialidad_id=especialidad_id)

    async def consultar_disponibilidad(
        self,
        profesional_id: Optional[Any] = None,
        fecha: Optional[str] = None,
        servicio_id: Optional[Any] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        return await self.catalog.consultar_disponibilidad(
            profesional_id=profesional_id, fecha=fecha, servicio_id=servicio_id
        )

    async def obtener_catalogo(self, nombre_catalogo: str) -> List[Dict[str, Any]]:
        return await self.catalog.obtener_catalogo(nombre_catalogo)

    async def obtener_appointment_status_id(self, code: str = "AGENDADA") -> str:
        return await self.catalog.obtener_appointment_status_id(code)

    async def obtener_appointment_origin_id(self, code: str = "AGENTE_BOT") -> str:
        return await self.catalog.obtener_appointment_origin_id(code)

    async def obtener_tipos_documento(self) -> List[Dict[str, Any]]:
        return await self.catalog.obtener_tipos_documento()

    async def obtener_sexos(self) -> List[Dict[str, Any]]:
        return await self.catalog.obtener_sexos()

    # ─────────────────────────────────────────────────────────────
    # Gestión de Citas
    # ─────────────────────────────────────────────────────────────
    async def agendar_cita(self, datos_cita: Dict[str, Any]) -> Dict[str, Any]:
        return await self.appointments.agendar_cita(datos_cita)

    async def consultar_citas(self, fecha: str = "") -> Optional[Any]:
        return await self.appointments.consultar_citas(fecha)

    async def obtener_citas_paciente(self, patient_id: str) -> List[Dict[str, Any]]:
        return await self.appointments.obtener_citas_paciente(patient_id)

    async def buscar_citas_por_cedula(self, cedula: str) -> List[Dict[str, Any]]:
        return await self.appointments.buscar_citas_por_cedula(cedula)

    async def cancelar_cita(self, cita_id: str, datos_cita: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return await self.appointments.cancelar_cita(cita_id, datos_cita)

    async def modificar_cita(self, cita_id: str, datos_actualizacion: Dict[str, Any]) -> Dict[str, Any]:
        return await self.appointments.modificar_cita(cita_id, datos_actualizacion)

    async def confirmar_estado_cita(self, cita_id: str) -> Dict[str, Any]:
        return await self.appointments.confirmar_estado_cita(cita_id)

    async def obtener_citas_agendadas_para_recordatorio(self, fecha: str) -> List[Dict[str, Any]]:
        return await self.appointment_svc.enriquecer_citas_para_recordatorio(fecha)

    # ─────────────────────────────────────────────────────────────
    # Gestión de Pacientes y Personas
    # ─────────────────────────────────────────────────────────────
    async def buscar_pacientes(self, search: str = "") -> Optional[List[Dict[str, Any]]]:
        return await self.patients.buscar_pacientes(search)

    async def vincular_paciente(self, conversacion_chatbot_id: str, paciente_id: Any) -> bool:
        return await self.patients.vincular_paciente(conversacion_chatbot_id, paciente_id)

    async def registrar_paciente(self, datos_onboard: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return await self.patients.registrar_paciente(datos_onboard)

    async def crear_paciente_basico(
        self,
        cedula: str,
        nombre: str,
        telefono_whatsapp: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        return await self.patients.crear_paciente_basico(cedula, nombre, telefono_whatsapp)

    async def login_paciente(self, document_number: str, password: str) -> Optional[Dict[str, Any]]:
        return await self.patients.login_paciente(document_number, password)

    async def buscar_persona_por_telefono(self, telefono: str) -> Optional[Dict[str, Any]]:
        return await self.patients.buscar_persona_por_telefono(telefono)

    async def buscar_persona_por_documento(self, document_number: str) -> Optional[Dict[str, Any]]:
        return await self.patients.buscar_persona_por_documento(document_number)

    async def resolver_paciente_por_documento(self, document_number: str) -> Optional[Dict[str, Any]]:
        return await self.patients.resolver_paciente_por_documento(document_number)

    async def crear_paciente_para_persona(
        self,
        person_id: str,
        contacto_emergencia: str = "Recepción Nexus",
        telefono_emergencia: str = "+573246030217",
    ) -> Optional[Dict[str, Any]]:
        return await self.patients.crear_paciente_para_persona(person_id, contacto_emergencia, telefono_emergencia)

    async def buscar_paciente_por_person_id(self, person_id: str) -> Optional[Dict[str, Any]]:
        return await self.patients.buscar_paciente_por_person_id(person_id)

    # ─────────────────────────────────────────────────────────────
    # Tickets, Notificaciones y Conversaciones
    # ─────────────────────────────────────────────────────────────
    async def _obtener_ticket_reasons(self) -> List[Dict[str, Any]]:
        return await self.tickets._obtener_ticket_reasons()

    async def _obtener_notification_priorities(self) -> List[Dict[str, Any]]:
        return await self.tickets._obtener_notification_priorities()

    async def actualizar_estado_conversacion(
        self, conversacion_id: str, status_id: str, patient_id: Optional[str] = None
    ) -> bool:
        return await self.tickets.actualizar_estado_conversacion(conversacion_id, status_id, patient_id)

    async def resolver_tickets_conversacion(self, conversacion_id: str) -> int:
        return await self.tickets.resolver_tickets_conversacion(conversacion_id)

    async def crear_ticket_soporte(
        self, telefono: str, motivo: str, prioridad: str = "MEDIA"
    ) -> Optional[Dict[str, Any]]:
        return await self.tickets.crear_ticket_soporte(telefono, motivo, prioridad)

    async def crear_notificacion(
        self,
        titulo: str,
        mensaje: str,
        prioridad: str = "MEDIA",
        conversation_id: Optional[str] = None,
        telefono: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        return await self.tickets.crear_notificacion(titulo, mensaje, prioridad, conversation_id, telefono)

    async def obtener_contexto_conversacion(self, conversacion_chatbot_id: str) -> Optional[Dict[str, Any]]:
        return await self.tickets.obtener_contexto_conversacion(conversacion_chatbot_id)

    async def obtener_o_crear_conversacion(
        self,
        chat_identifier: str,
        channel_id: Optional[str] = None,
        patient_id: Optional[str] = None,
    ) -> Optional[str]:
        return await self.tickets.obtener_o_crear_conversacion(chat_identifier, channel_id, patient_id)

    async def guardar_mensaje_conversacion(
        self,
        conversation_id: str,
        rol: str,
        contenido: str,
        rag_confidence: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        return await self.tickets.guardar_mensaje_conversacion(conversation_id, rol, contenido, rag_confidence)

    async def registrar_mensaje(
        self,
        chat_identifier: str,
        rol: str,
        contenido: str,
        rag_confidence: Optional[float] = None,
        patient_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        return await self.tickets.registrar_mensaje(chat_identifier, rol, contenido, rag_confidence, patient_id)

    def limpiar_cache_conversacion(self, chat_identifier: str) -> None:
        self.tickets.limpiar_cache_conversacion(chat_identifier)


# Instancia única para backward-compatibility
dotnet_client = DotNetClient()
