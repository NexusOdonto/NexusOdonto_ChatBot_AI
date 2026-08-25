import logging
from fastapi import APIRouter, Request, HTTPException

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/webhook/whatsapp")
async def receive_whatsapp_message(request: Request):
    try:
        payload = await request.json()
        
        # Evitar bucles infinitos ignorando mensajes salientes del propio bot
        if payload.get("data", {}).get("key", {}).get("fromMe", False):
            return {"status": "ignored", "detail": "Bot message ignored"}

        # Validar la estructura estándar de Evolution API v2
        if "data" in payload and "message" in payload["data"]:
            data = payload["data"]
            
            # Extraer el número de teléfono del paciente (thread_id para LangGraph)
            remote_jid = data.get("key", {}).get("remoteJid", "")
            numero_paciente = remote_jid.split("@")[0] if "@" in remote_jid else remote_jid
            
            # Extraer el texto según el formato que envíe Evolution API
            mensaje_texto = ""
            if "conversation" in data["message"]:
                mensaje_texto = data["message"]["conversation"]
            elif "extendedTextMessage" in data["message"]:
                mensaje_texto = data["message"]["extendedTextMessage"].get("text", "")
                
            if numero_paciente and mensaje_texto:
                logger.info(f"Mensaje recibido de [{numero_paciente}]: {mensaje_texto}")
                # TODO: Enviar 'numero_paciente' y 'mensaje_texto' al grafo de LangGraph
                
        return {"status": "success"}

    except Exception as e:
        logger.error(f"Error procesando webhook de WhatsApp: {str(e)}")
        raise HTTPException(status_code=500, detail="Error procesando webhook")