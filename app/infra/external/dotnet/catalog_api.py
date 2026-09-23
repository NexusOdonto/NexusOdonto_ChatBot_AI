"""Endpoints de Catálogo, Servicios y Profesionales en la API .NET."""

import asyncio
import logging
import time
from typing import Optional, List, Dict, Any
from app.infra.external.dotnet.http_transport import dotnet_transport, DotNetHttpTransport

logger = logging.getLogger(__name__)

_CATALOG_TTL_SECONDS = 60.0
_catalog_cache = {}


def _cache_get(key: str):
    entry = _catalog_cache.get(key)
    if not entry:
        return None
    value, ts = entry
    if (time.monotonic() - ts) > _CATALOG_TTL_SECONDS:
        return None
    return value


def _cache_set(key: str, value):
    _catalog_cache[key] = (value, time.monotonic())
    return value



class DotNetCatalogApi:
    def __init__(self, transport: Optional[DotNetHttpTransport] = None):
        self.transport = transport or dotnet_transport

    async def obtener_servicios(self) -> Optional[List[Dict[str, Any]]]:
        """Obtiene la lista de servicios activos de la clínica."""
        cached = _cache_get("Services")
        if cached is not None:
            return cached
        response = await self.transport.request("GET", "Services")
        if response and response.status_code == 200:
            payload = response.json()
            if isinstance(payload, list):
                return _cache_set("Services", payload)
            if isinstance(payload, dict):
                return _cache_set("Services", payload.get("items", []))
        return None

    async def obtener_especialidades(self) -> Optional[List[Dict[str, Any]]]:
        """Obtiene la lista de especialidades de la clínica."""
        cached = _cache_get("Specialties")
        if cached is not None:
            return cached
        response = await self.transport.request("GET", "Specialties")
        if response and response.status_code == 200:
            payload = response.json()
            if isinstance(payload, list):
                return _cache_set("Specialties", payload)
            if isinstance(payload, dict):
                return _cache_set("Specialties", payload.get("items", []))
        return None

    async def obtener_empleados(self) -> List[Dict[str, Any]]:
        """Obtiene la lista de empleados activos en el backend .NET."""
        cached = _cache_get("Employees")
        if cached is not None:
            return cached
        response = await self.transport.request("GET", "Employees")
        if response and response.status_code == 200:
            payload = response.json()
            items = payload if isinstance(payload, list) else payload.get("items", [])
            return _cache_set("Employees", items)
        return []

    async def obtener_personas(self) -> List[Dict[str, Any]]:
        """Obtiene el listado general de personas registradas."""
        cached = _cache_get("Persons")
        if cached is not None:
            return cached
        response = await self.transport.request("GET", "Persons")
        if response and response.status_code == 200:
            payload = response.json()
            items = payload if isinstance(payload, list) else payload.get("items", [])
            return _cache_set("Persons", items)
        return []

    async def obtener_profesionales(
        self, especialidad_id: Optional[Any] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Obtiene la lista de profesionales enriquecidos con su nombre completo."""
        cache_key = f"Professionals:{especialidad_id or 'all'}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
        params: Dict[str, Any] = {}
        if especialidad_id is not None:
            params["especialidadId"] = str(especialidad_id)
            params["specialtyId"] = str(especialidad_id)

        response = await self.transport.request("GET", "Professionals", params=params if params else None)
        if response and response.status_code == 200:
            payload = response.json()
            profs = payload if isinstance(payload, list) else payload.get("items", [])
            if not profs:
                return _cache_set(cache_key, [])

            # Enriquecer con nombres reales (Employees + Persons en paralelo, con caché)
            try:
                empleados, personas = await asyncio.gather(
                    self.obtener_empleados(),
                    self.obtener_personas(),
                )
                emp_to_person = {e.get("id"): e.get("personId") for e in empleados if isinstance(e, dict)}
                person_names = {
                    p.get("id"): f"{p.get('firstName', '')} {p.get('lastName', '')}".strip()
                    for p in personas if isinstance(p, dict)
                }

                for p in profs:
                    if isinstance(p, dict):
                        emp_id = p.get("employeeId")
                        per_id = emp_to_person.get(emp_id)
                        nombre_raw = person_names.get(per_id)
                        if nombre_raw:
                            prefijo = (
                                "Dra."
                                if any(n in nombre_raw.lower() for n in ["laura", "maria", "ana", "camila", "valentina", "sofia"])
                                else "Dr."
                            )
                            p["name"] = f"{prefijo} {nombre_raw}"
                            p["nombre"] = f"{prefijo} {nombre_raw}"
                            p["nombreCompleto"] = f"{prefijo} {nombre_raw}"
            except Exception as enh_err:
                logger.debug(f"[CatalogApi] No se pudieron enriquecer nombres de profesionales: {enh_err}")

            return _cache_set(cache_key, profs)
        return None

    async def consultar_disponibilidad(
        self,
        profesional_id: Optional[Any] = None,
        fecha: Optional[str] = None,
        servicio_id: Optional[Any] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Consulta horarios disponibles en el backend .NET."""
        params: Dict[str, Any] = {}
        if profesional_id:
            params["profesionalId"] = str(profesional_id)
            params["professionalId"] = str(profesional_id)
        if fecha:
            params["fecha"] = str(fecha)
            params["date"] = str(fecha)
        if servicio_id:
            params["servicioId"] = str(servicio_id)
            params["serviceId"] = str(servicio_id)

        response = await self.transport.request("GET", "Availabilities", params=params)
        if response and response.status_code == 200:
            payload = response.json()
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict):
                return payload.get("items", [])
        return None

    async def obtener_catalogo(self, nombre_catalogo: str) -> List[Dict[str, Any]]:
        """Obtiene un catálogo general de la API de .NET."""
        response = await self.transport.request("GET", f"{nombre_catalogo}")
        if response and response.status_code == 200:
            payload = response.json()
            return payload if isinstance(payload, list) else payload.get("items", [])
        return []

    async def obtener_appointment_status_id(self, code: str = "AGENDADA") -> str:
        """Obtiene el ID del estado de cita por código o nombre con fallback seguro."""
        c_upper = str(code).strip().upper().replace(" ", "_")
        fallback_map = {
            "PENDIENTE": "10000000-0000-0000-0000-000000000001",
            "AGENDADA": "10000000-0000-0000-0000-000000000001",
            "PROGRAMADA": "10000000-0000-0000-0000-000000000001",
            "SCHEDULED": "10000000-0000-0000-0000-000000000001",
            "CONFIRMADA": "10000000-0000-0000-0000-000000000002",
            "CONFIRMED": "10000000-0000-0000-0000-000000000002",
            "EN_ATENCION": "10000000-0000-0000-0000-000000000003",
            "ATTENDING": "10000000-0000-0000-0000-000000000003",
            "COMPLETADA": "10000000-0000-0000-0000-000000000004",
            "COMPLETED": "10000000-0000-0000-0000-000000000004",
            "CANCELADA": "10000000-0000-0000-0000-000000000005",
            "CANCELLED": "10000000-0000-0000-0000-000000000005",
            "NO_ASISTIO": "10000000-0000-0000-0000-000000000006",
            "NO_SHOW": "10000000-0000-0000-0000-000000000006",
            "NOSHOW": "10000000-0000-0000-0000-000000000006",
        }
        response = await self.transport.request("GET", "AppointmentStatuses")
        if response and response.status_code == 200:
            statuses = response.json() if isinstance(response.json(), list) else []
            for s in statuses:
                s_code = str(s.get("code") or "").upper()
                s_name = str(s.get("name") or "").upper().replace(" ", "_")
                if s_code == c_upper or s_name == c_upper:
                    return str(s.get("id"))
        return fallback_map.get(c_upper, "10000000-0000-0000-0000-000000000001")

    async def obtener_appointment_origin_id(self, code: str = "AGENTE_BOT") -> str:
        """Obtiene el ID del origen de cita por código."""
        response = await self.transport.request("GET", "AppointmentOrigins")
        if response and response.status_code == 200:
            origins = response.json() if isinstance(response.json(), list) else []
            for o in origins:
                if str(o.get("code")).upper() == code.upper():
                    return str(o.get("id"))
        return "20000000-0000-0000-0000-000000000002"

    async def obtener_tipos_documento(self) -> List[Dict[str, Any]]:
        """Obtiene el catálogo de tipos de documento (CC, TI, CE, PP, etc.) desde .NET."""
        response = await self.transport.request("GET", "DocumentTypes")
        if response and response.status_code == 200:
            payload = response.json()
            return payload if isinstance(payload, list) else payload.get("items", [])
        return []

    async def obtener_sexos(self) -> List[Dict[str, Any]]:
        """Obtiene el catálogo de sexos (Masculino, Femenino, Otro) desde .NET."""
        response = await self.transport.request("GET", "Sexes")
        if response and response.status_code == 200:
            payload = response.json()
            return payload if isinstance(payload, list) else payload.get("items", [])
        return []


catalog_api = DotNetCatalogApi()
