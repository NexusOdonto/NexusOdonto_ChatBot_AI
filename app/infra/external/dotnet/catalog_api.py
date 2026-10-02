"""Endpoints de Catálogo, Servicios y Profesionales en la API .NET."""

import asyncio
import logging
import time
from typing import Optional, List, Dict, Any
from app.infra.external.dotnet.http_transport import dotnet_transport, DotNetHttpTransport

logger = logging.getLogger(__name__)

_CATALOG_TTL_SECONDS = 60.0  # 1 min — catálogos estáticos para refresco rápido
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
        """Obtiene la lista de servicios activos de la clínica desde el backend .NET."""
        cached = _cache_get("Services")
        if cached is not None:
            return cached
        response = await self.transport.request("GET", "Services", params={"pageSize": 100})
        if response and response.status_code == 200:
            payload = response.json()
            items = payload if isinstance(payload, list) else payload.get("items", [])
            from app.agents.tools.agenda_helpers import _es_servicio_activo
            servicios_validos = [s for s in items if isinstance(s, dict) and _es_servicio_activo(s)]
            return _cache_set("Services", servicios_validos)
        return None

    async def obtener_especialidades(self) -> Optional[List[Dict[str, Any]]]:
        """Obtiene la lista de especialidades de la clínica."""
        cached = _cache_get("Specialties")
        if cached is not None:
            return cached
        response = await self.transport.request("GET", "Specialties", params={"pageSize": 100})
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
        response = await self.transport.request("GET", "Employees", params={"pageSize": 100})
        if response and response.status_code == 200:
            payload = response.json()
            items = payload if isinstance(payload, list) else payload.get("items", [])
            return _cache_set("Employees", items)
        return []

    async def obtener_personas(self) -> List[Dict[str, Any]]:
        """Obtiene el listado general de personas registradas.

        NOT cached: person identity (cédula) must never be served from a stale list,
        or the chatbot can miss an existing patient and orphan appointments.
        """
        response = await self.transport.request("GET", "Persons", params={"pageSize": 100})
        if response and response.status_code == 200:
            payload = response.json()
            return payload if isinstance(payload, list) else payload.get("items", [])
        return []

    async def obtener_profesionales(
        self, especialidad_id: Optional[Any] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Obtiene la lista de profesionales enriquecidos con su nombre completo, especialidad y consultorio."""
        cache_key = f"Professionals:{especialidad_id or 'all'}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
        params: Dict[str, Any] = {"pageSize": 100}
        if especialidad_id is not None:
            params["especialidadId"] = str(especialidad_id)
            params["specialtyId"] = str(especialidad_id)

        response = await self.transport.request("GET", "Professionals", params=params)
        if response and response.status_code == 200:
            payload = response.json()
            profs = payload if isinstance(payload, list) else payload.get("items", [])
            if not profs:
                return _cache_set(cache_key, [])

            # Filtrar profesionales inactivos o eliminados
            profs_filtrados = []
            for p in profs:
                if not isinstance(p, dict):
                    continue
                is_del = p.get("isDeleted") or p.get("IsDeleted") or p.get("is_deleted")
                is_act = p.get("isActive") if "isActive" in p else p.get("IsActive")
                if is_del is True or str(is_del).strip().lower() in ("true", "1", "yes"):
                    continue
                if is_act is False or str(is_act).strip().lower() in ("false", "0", "no"):
                    continue
                profs_filtrados.append(p)

            profs = profs_filtrados or profs

            # Enriquecer con nombres reales, especialidades y consultorios en paralelo
            try:
                empleados, personas, especialidades = await asyncio.gather(
                    self.obtener_empleados(),
                    self.obtener_personas(),
                    self.obtener_especialidades(),
                )
                emp_to_person = {e.get("id"): e.get("personId") for e in (empleados or []) if isinstance(e, dict)}
                emp_to_office = {
                    e.get("id"): (e.get("consultorio") or e.get("office") or e.get("cubicle") or e.get("location"))
                    for e in (empleados or [])
                    if isinstance(e, dict)
                }
                emp_to_reg = {
                    e.get("id"): (e.get("medicalLicenseNumber") or e.get("registrationNumber") or e.get("licenseNumber") or e.get("reg"))
                    for e in (empleados or [])
                    if isinstance(e, dict)
                }
                person_names = {
                    p.get("id"): f"{p.get('firstName', '')} {p.get('lastName', '')}".strip()
                    for p in (personas or []) if isinstance(p, dict)
                }
                esp_map = {
                    str(e.get("id")).lower(): (e.get("name") or e.get("nombre"))
                    for e in (especialidades or [])
                    if isinstance(e, dict)
                }

                for p in profs:
                    if isinstance(p, dict):
                        emp_id = p.get("employeeId")
                        per_id = emp_to_person.get(emp_id)
                        nombre_raw = person_names.get(per_id)
                        if not nombre_raw:
                            nombre_raw = p.get("name") or p.get("nombre") or p.get("professionalName") or ""

                        if nombre_raw:
                            nombre_clean = str(nombre_raw).strip()
                            if not nombre_clean.lower().startswith(("dr", "dra")):
                                prefijo = (
                                    "Dra."
                                    if any(n in nombre_clean.lower() for n in ["laura", "maria", "ana", "camila", "valentina", "sofia", "andrea"])
                                    else "Dr."
                                )
                                nombre_clean = f"{prefijo} {nombre_clean}"
                            p["name"] = nombre_clean
                            p["nombre"] = nombre_clean
                            p["nombreCompleto"] = nombre_clean

                        # Especialidad real del profesional
                        p_esp_id = str(p.get("specialtyId") or p.get("especialidadId") or "").lower()
                        if p_esp_id in esp_map:
                            p["specialtyName"] = esp_map[p_esp_id]
                            p["especialidad"] = esp_map[p_esp_id]
                        elif not p.get("specialtyName") and not p.get("especialidad"):
                            p["specialtyName"] = "Odontología General"
                            p["especialidad"] = "Odontología General"

                        # Consultorio / Ubicación
                        consultorio = (
                            p.get("consultorio")
                            or p.get("office")
                            or p.get("cubicle")
                            or emp_to_office.get(emp_id)
                        )
                        if consultorio:
                            p["consultorio"] = consultorio
                            p["office"] = consultorio

                        # Registro médico profesional
                        reg = (
                            p.get("medicalLicenseNumber")
                            or p.get("licenseNumber")
                            or p.get("registrationNumber")
                            or p.get("reg")
                            or emp_to_reg.get(emp_id)
                        )
                        if reg:
                            p["medicalLicenseNumber"] = reg
                            p["registrationNumber"] = reg
            except Exception as enh_err:
                logger.debug(f"[CatalogApi] No se pudieron enriquecer profesionales: {enh_err}")

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
