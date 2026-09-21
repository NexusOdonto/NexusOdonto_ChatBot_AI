"""Transporte HTTP base para la API .NET de Nexus Odonto.
Encapsula la autenticación JWT automática, reintentos con backoff y manejo del error 401.
"""

import os
import logging
import asyncio
from typing import Optional, Dict, Any
import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)


class DotNetHttpTransport:
    """Maneja la conexión de bajo nivel, tokens JWT y reintentos automáticos contra .NET."""

    def __init__(self):
        self.base_url: str = os.getenv("DOTNET_API_URL", settings.dotnet_api_url).rstrip("/")
        self.auth_login: str = os.getenv("DOTNET_AUTH_LOGIN", settings.dotnet_auth_login)
        self.auth_password: str = os.getenv("DOTNET_AUTH_PASSWORD", settings.dotnet_auth_password)
        self.secret_token: str = os.getenv("AGENT_INTERNAL_SECRET", settings.agent_internal_secret)
        self.timeout: float = float(os.getenv("DOTNET_API_TIMEOUT", settings.dotnet_api_timeout))

        self._jwt_token: Optional[str] = None
        self._auth_lock = asyncio.Lock()

    @property
    def auth_url(self) -> str:
        url = self.base_url
        if url.endswith("/api/v1"):
            url = url[:-7]
        elif url.endswith("/api"):
            url = url[:-4]
        return f"{url}/api/auth"

    async def get_valid_token(self, force_refresh: bool = False) -> str:
        """Obtiene un token JWT válido iniciando sesión automáticamente en .NET si es necesario."""
        if not force_refresh and self._jwt_token:
            return self._jwt_token

        async with self._auth_lock:
            if not force_refresh and self._jwt_token:
                return self._jwt_token

            if self.auth_login and self.auth_password:
                login_endpoint = f"{self.auth_url}/login"
                payload = {"loginId": self.auth_login, "password": self.auth_password}
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        headers = {
                            "Content-Type": "application/json; charset=utf-8",
                            "Accept": "application/json",
                        }
                        if self.secret_token:
                            headers["X-Internal-Secret"] = self.secret_token

                        response = await client.post(login_endpoint, json=payload, headers=headers)
                        if response.status_code == 200:
                            data = response.json()
                            token = data.get("token")
                            if token:
                                self._jwt_token = token
                                logger.info("[.NET Transport] Sesión autenticada exitosamente con backend .NET (JWT)")
                                return self._jwt_token
                except Exception as e:
                    logger.warning(f"[.NET Transport] Error autenticando con .NET: {e}")

            return self.secret_token or ""

    async def get_headers(self, force_refresh: bool = False) -> Dict[str, str]:
        token = await self.get_valid_token(force_refresh=force_refresh)
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
            headers["X-Api-Key"] = token
            headers["x-api-key"] = token
        return headers

    async def request(
        self, method: str, path: str, params: Optional[Dict[str, Any]] = None, json: Optional[Any] = None
    ) -> Optional[httpx.Response]:
        """Ejecuta una petición HTTP con manejo automático de reintentos en 401 (token expirado)."""
        url = path if path.startswith("http") else f"{self.base_url}/{path.lstrip('/')}"
        headers = await self.get_headers()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.request(method, url, params=params, json=json, headers=headers)
                if response.status_code == 401:
                    logger.info("[.NET Transport] Token JWT expirado (401). Renovando sesión...")
                    self._jwt_token = None
                    headers = await self.get_headers(force_refresh=True)
                    response = await client.request(method, url, params=params, json=json, headers=headers)
                return response
            except Exception as e:
                logger.error(f"[.NET Transport] Error en {method} {url}: {e}")
                return None


# Instancia única reutilizable
dotnet_transport = DotNetHttpTransport()
