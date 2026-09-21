from typing import Optional, List
from pydantic import BaseModel, Field


class DentalService(BaseModel):
    id: str
    name: str
    price: float = 0.0
    duration_minutes: int = 30
    description: Optional[str] = None
    is_active: bool = True
    specialty_name: Optional[str] = None


class DentalProfessional(BaseModel):
    id: str
    name: str
    specialty_name: Optional[str] = None
    license_number: Optional[str] = None
    is_active: bool = True


class AvailabilitySlot(BaseModel):
    start_time: str
    end_time: str
    professional_id: str
    professional_name: Optional[str] = None
    is_available: bool = True
