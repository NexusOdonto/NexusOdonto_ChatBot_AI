from typing import Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    from dataclasses import dataclass, field

    def Field(default=None, description=None, **kwargs):
        return default

    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


class Patient(BaseModel):
    id: str = ""
    person_id: Optional[str] = None
    document_number: str = ""
    document_type_id: Optional[str] = None
    full_name: str = ""
    phone: Optional[str] = None
    email: Optional[str] = None
    is_active: bool = True


class SafePatientContext(BaseModel):
    """Contexto seguro del paciente en la sesión actual de chat.
    Garantiza el cumplimiento estricto de Habeas Data (Ley 1581):
    NUNCA precargar cédula ni nombre automáticamente por el número de teléfono.
    """
    phone: str = ""
    nombre: str = ""
    primer_nombre: str = ""
    cedula: Optional[str] = None
    is_registered: bool = False
    push_name: str = ""
