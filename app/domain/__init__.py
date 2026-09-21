"""Capa de Dominio: Contiene entidades, modelos tipados y reglas puras del negocio odontológico de Nexus Odonto.
No depende de FastAPI, LangChain, bases de datos ni clientes HTTP externos.
"""

from app.domain.formatters.whatsapp_formatter import formatear_para_whatsapp

__all__ = ["formatear_para_whatsapp"]

