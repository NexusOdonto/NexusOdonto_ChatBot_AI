from enum import Enum
from typing import Optional

try:
    from pydantic import BaseModel
except ImportError:
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


class EmergencySeverity(str, Enum):
    NORMAL = "NORMAL"
    ALTO = "ALTO"
    CRITICO = "CRITICO"


class EmergencyTriageResult(BaseModel):
    is_emergency: bool = False
    severity: EmergencySeverity = EmergencySeverity.NORMAL
    reason: Optional[str] = None
    trigger_pattern: Optional[str] = None
    action_required: str = "CONTINUAR"
