"""Transporte HTTP base para la API .NET de Nexus Odonto.
Encapsula la autenticación JWT automática, reintentos con backoff y manejo del error 401.
Usa un httpx.AsyncClient compartido (connection pool) cerrado en shutdown.
"""

import os
import logging
import asyncio
import threading
from typing import Optional, Dict, Any
import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)


def loop_safe_asyncio_lock(
    locks_by_loop: Dict[int, asyncio.Lock],
    guard: threading.Lock,
) -> asyncio.Lock:
    """Return an asyncio.Lock bound to the *current* running loop.

    LangChain tools call `_run_sync`, which may spawn a side thread with a new
    event loop. A singleton `asyncio.Lock()` created at import time is bound to
    the main loop and raises "is bound to a different event loop" there.
    """
    loop = asyncio.get_running_loop()
    key = id(loop)
    with guard:
        lock = locks_by_loop.get(key)
        if lock is None:
            lock = asyncio.Lock()
            locks_by_loop[key] = lock
        return lock


class DotNetHttpTransport:
    """Maneja la conexión de bajo nivel, tokens JWT y reintentos automáticos contra .NET."""

    def __init__(self):
        self.base_url: str = os.getenv("DOTNET_API_URL", settings.dotnet_api_url).rstrip("/")
        self.auth_login: str = os.getenv("DOTNET_AUTH_LOGIN", settings.dotnet_auth_login)
        self.auth_password: str = os.getenv("DOTNET_AUTH_PASSWORD", settings.dotnet_auth_password)
        self.secret_token: str = os.getenv("AGENT_INTERNAL_SECRET", settings.agent_internal_secret)
        self.timeout: float = float(os.getenv("DOTNET_API_TIMEOUT", settings.dotnet_api_timeout))

        self._jwt_token: Optional[str] = None
        self._auth_locks: Dict[int, asyncio.Lock] = {}
        self._auth_locks_guard = threading.Lock()
        self._client: Optional[httpx.AsyncClient] = None
        self._client_locks: Dict[int, asyncio.Lock] = {}
        self._client_locks_guard = threading.Lock()

    def _auth_lock(self) -> asyncio.Lock:
        return loop_safe_asyncio_lock(self._auth_locks, self._auth_locks_guard)

    def _client_lock(self) -> asyncio.Lock:
        return loop_safe_asyncio_lock(self._client_locks, self._client_locks_guard)

    async def get_client(self) -> httpx.AsyncClient:
        """Cliente httpx compartido con pool de conexiones (reutilizado entre requests)."""
        async with self._client_lock():
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(
                    timeout=self.timeout,
                    limits=httpx.Limits(
                        max_connections=20,
                        max_keepalive_connections=10,
                        keepalive_expiry=30.0,
                    ),
                )
                logger.info("[.NET Transport] AsyncClient pool inicializado")
            return self._client

    async def aclose(self) -> None:
        """Cierra el cliente compartido (llamar en shutdown de FastAPI)."""
        async with self._client_lock():
            if self._client is not None and not self._client.is_closed:
                await self._client.aclose()
                logger.info("[.NET Transport] AsyncClient pool cerrado")
            self._client = None

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

        async with self._auth_lock():
            if not force_refresh and self._jwt_token:
                return self._jwt_token

            if self.auth_login and self.auth_password:
                login_endpoint = f"{self.auth_url}/login"
                payload = {"loginId": self.auth_login, "password": self.auth_password}
                try:
                    client = await self.get_client()
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
        client = await self.get_client()

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
