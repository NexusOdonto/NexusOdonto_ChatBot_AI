import hashlib
import hmac

from starlette.types import Message


def is_valid_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Valida la firma HMAC enviada por Evolution sobre el body original."""
    if not secret or not signature:
        return False

    # Algunos proveedores envían el algoritmo como prefijo y otros solo el hash.
    received_signature = signature.strip()
    if received_signature.lower().startswith("sha256="):
        received_signature = received_signature[7:]

    expected_signature = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(received_signature.lower(), expected_signature)


async def restore_request_body(body: bytes) -> Message:
    """Permite que FastAPI vuelva a leer el body después del middleware."""
    return {"type": "http.request", "body": body, "more_body": False}