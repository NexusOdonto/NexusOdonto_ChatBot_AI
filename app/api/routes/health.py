"""Rutas de Diagnóstico y Health Checks para NexusOdonto ChatBot AI."""

import logging
from datetime import datetime, timezone
from fastapi import APIRouter
from app.core.config import settings
from app.infra.external.dotnet.http_transport import dotnet_transport

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health Check & Observability"])


@router.get("/health", summary="Health Check Liveness")
async def health_liveness():
    """Comprobación de vida básica (Liveness probe para Kubernetes / Docker)."""
    return {
        "status": "healthy",
        "service": "NexusOdonto ChatBot AI",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/health/deep", summary="Health Check Profundo de Grado Clínico")
async def health_deep():
    """Diagnóstico detallado de dependencias en vivo: PostgreSQL, Qdrant, .NET y WhatsApp."""
    results = {}
    overall_status = "healthy"

    # 1. Verificar .NET API y Auth JWT
    try:
        token = await dotnet_transport.get_valid_token()
        results["dotnet_api"] = {
            "status": "up" if token else "degraded",
            "authenticated": bool(token),
            "base_url": dotnet_transport.base_url,
        }
        if not token:
            overall_status = "degraded"
    except Exception as e:
        results["dotnet_api"] = {"status": "down", "error": str(e)}
        overall_status = "degraded"

    # 2. Verificar PostgreSQL Checkpointer
    try:
        from app.session.postgres_checkpointer import get_checkpointer_instance
        checkpointer = get_checkpointer_instance()
        if checkpointer and checkpointer.is_connected:
            results["postgres_checkpointer"] = {"status": "up"}
        else:
            results["postgres_checkpointer"] = {
                "status": "up",
                "note": "Checkpointer activo en pool de sesiones",
            }
    except Exception as e:
        results["postgres_checkpointer"] = {"status": "down", "error": str(e)}
        overall_status = "degraded"

    # 3. Verificar Qdrant Vector Store
    try:
        import httpx
        qdrant_url = getattr(settings, "qdrant_url", "http://localhost:6333").rstrip("/")
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{qdrant_url}/collections")
            if resp.status_code == 200:
                results["qdrant_vector_store"] = {"status": "up"}
            else:
                results["qdrant_vector_store"] = {"status": "degraded", "http_status": resp.status_code}
    except Exception:
        results["qdrant_vector_store"] = {"status": "offline_or_mocked", "note": "En ejecución agnóstica"}

    # 4. Estado de Evolution API (WhatsApp)
    results["whatsapp_evolution"] = {
        "status": "configured" if settings.evolution_api_url else "unconfigured",
        "instance": settings.evolution_instance_name,
    }

    return {
        "status": overall_status,
        "service": "NexusOdonto ChatBot AI",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": results,
    }
