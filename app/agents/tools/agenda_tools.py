"""Módulo Facade para Herramientas de Agenda y Catálogo de Nexus Odonto.
Re-exporta de manera transparente las herramientas atómicas de:
- catalog_tools: disponibilidad, catálogo de doctores y servicios.
- appointment_tools: agendamiento, consulta, reprogramación, cancelación y confirmación.
- agenda_helpers: funciones auxiliares, validaciones y diccionarios.

Garantiza 100% de compatibilidad hacia atrás con el código existente.
"""

# Re-exportar funciones y diccionarios auxiliares
from app.agents.tools.agenda_helpers import (
    SINONIMOS_ESPANOL,
    ALIAS_BUSQUEDA_SERVICIO,
    ETIQUETAS_SERVICIO_ES,
    DESCRIPCIONES_SERVICIO_ES,
    _normalizar_texto,
    _obtener_valor,
    _validar_cedula,
    _run_sync,
    _formatear_hora_ampm,
    _parsear_fecha_hora_flexible,
    _validar_horario_cita,
    _resolver_cita_por_selector,
    _filtrar_citas_proximas_activas,
    _generar_slots_desde_regla,
    _buscar_servicio_por_texto,
    _buscar_especialidad_por_texto,
    _servicios_activos,
    _servicios_relacionados_a_especialidad,
    _etiqueta_servicio,
    _lista_servicios_whatsapp,
    _nombres_servicios_activos,
    _etiqueta_especialidad,
    _mensaje_catalogo_no_encontrado,
    _mensaje_especialidad_sin_servicio_unico,
    _es_servicio_activo,
    _texto_coincide,
    _aplicar_sinonimos_especialidad,
    _variantes_busqueda_servicio,
    _variantes_desde_etiquetas_es,
)

# Re-exportar herramientas de catálogo
from app.agents.tools.catalog_tools import (
    consultar_disponibilidad_tool,
    consultar_doctores_tool,
    consultar_servicios_y_precios_tool,
    _consultar_disponibilidad_impl,
    _consultar_doctores_impl,
    _consultar_servicios_impl,
)

# Re-exportar herramientas de citas
from app.agents.tools.appointment_tools import (
    agendar_cita_tool,
    consultar_cita_por_cedula_tool,
    cancelar_cita_tool,
    modificar_cita_tool,
    confirmar_cita_tool,
    _agendar_cita_impl,
    _consultar_cita_por_cedula_impl,
    _cancelar_cita_impl,
    _modificar_cita_impl,
    _confirmar_cita_impl,
)

__all__ = [
    # Herramientas LangChain
    "consultar_disponibilidad_tool",
    "consultar_doctores_tool",
    "consultar_servicios_y_precios_tool",
    "agendar_cita_tool",
    "consultar_cita_por_cedula_tool",
    "cancelar_cita_tool",
    "modificar_cita_tool",
    "confirmar_cita_tool",
    # Implementaciones asíncronas directas
    "_consultar_disponibilidad_impl",
    "_consultar_doctores_impl",
    "_consultar_servicios_impl",
    "_agendar_cita_impl",
    "_consultar_cita_por_cedula_impl",
    "_cancelar_cita_impl",
    "_modificar_cita_impl",
    "_confirmar_cita_impl",
    # Helpers y constantes
    "SINONIMOS_ESPANOL",
    "ALIAS_BUSQUEDA_SERVICIO",
    "ETIQUETAS_SERVICIO_ES",
    "DESCRIPCIONES_SERVICIO_ES",
    "_normalizar_texto",
    "_obtener_valor",
    "_validar_cedula",
    "_run_sync",
    "_formatear_hora_ampm",
    "_parsear_fecha_hora_flexible",
    "_validar_horario_cita",
    "_resolver_cita_por_selector",
    "_generar_slots_desde_regla",
]
