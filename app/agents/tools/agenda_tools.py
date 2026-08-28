import unicodedata
import logging
import asyncio
import threading
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from app.clients.dotnet_client import dotnet_client

logger = logging.getLogger(__name__)

def _normalizar_texto(texto: str) -> str:
    """Normaliza texto eliminando acentos y convirtiendo a minúsculas."""
    if not texto:
        return ""
    texto = texto.lower().strip()
    return "".join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

def _obtener_valor(obj: Dict[str, Any], *keys: str) -> Any:
    """Busca un valor en un diccionario de forma insensible a mayúsculas y minúsculas."""
    for key in keys:
        if key in obj:
            return obj[key]
        for k, v in obj.items():
            if k.lower() == key.lower():
                return v
    return None

def _run_sync(coro) -> Any:
    """Ejecuta una corrutina de forma síncrona, gestionando de forma segura los loops activos de asyncio."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    if loop.is_running():
        # Si el loop de este hilo ya está activo, corremos la corrutina en un hilo secundario
        result = []
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            try:
                res = new_loop.run_until_complete(coro)
                result.append(res)
            finally:
                new_loop.close()
        t = threading.Thread(target=run_in_thread)
        t.start()
        t.join()
        return result[0]
    else:
        return loop.run_until_complete(coro)

async def _consultar_disponibilidad_impl(especialidad: str, fecha: str) -> str:
    try:
        norm_esp = _normalizar_texto(especialidad)
        if not norm_esp:
            return "Por favor, indica una especialidad o servicio válido."
        
        # 1. Obtener especialidades desde el backend .NET
        especialidades = await dotnet_client.obtener_especialidades()
        if not especialidades:
            return "En este momento no podemos acceder al catálogo de especialidades. Por favor intenta de nuevo más tarde."
            
        especialidad_encontrada = None
        for esp in especialidades:
            nombre_esp = esp.get("nombre", "")
            if norm_esp in _normalizar_texto(nombre_esp) or _normalizar_texto(nombre_esp) in norm_esp:
                especialidad_encontrada = esp
                break
                
        if not especialidad_encontrada:
            return f"No encontramos la especialidad o servicio '{especialidad}' en nuestro catálogo. Especialidades disponibles: " + ", ".join(esp.get("nombre", "") for esp in especialidades)
            
        esp_id = _obtener_valor(especialidad_encontrada, "especialidadId", "id")
        esp_nombre = especialidad_encontrada.get("nombre", especialidad)
        
        # 2. Buscar profesionales para esa especialidad
        profesionales = await dotnet_client.obtener_profesionales(especialidad_id=esp_id)
        if not profesionales:
            return f"No hay profesionales registrados o disponibles actualmente para la especialidad de {esp_nombre}."
            
        # 3. Buscar el servicio correspondiente para determinar la duración esperada
        servicios = await dotnet_client.obtener_servicios()
        servicio_encontrado = None
        if servicios:
            # Buscamos un servicio que coincida con el nombre de la especialidad
            for ser in servicios:
                nombre_ser = ser.get("nombre", "")
                if norm_esp in _normalizar_texto(nombre_ser) or _normalizar_texto(nombre_ser) in norm_esp:
                    servicio_encontrado = ser
                    break
            if not servicio_encontrado:
                # Fallback: tomamos el primer servicio de la lista
                servicio_encontrado = servicios[0]
                
        servicio_id = _obtener_valor(servicio_encontrado, "servicioId", "id") if servicio_encontrado else 1
        servicio_nombre = servicio_encontrado.get("nombre", "Consulta General") if servicio_encontrado else "Consulta General"
        
        # 4. Consultar disponibilidad para cada profesional
        resultados = []
        for prof in profesionales:
            prof_id = _obtener_valor(prof, "profesionalId", "id")
            prof_nombre = prof.get("nombre", "")
            if not prof_nombre and "empleado" in prof:
                prof_nombre = _obtener_valor(prof["empleado"], "nombreCompleto", "nombre")
            if not prof_nombre and "persona" in prof:
                prof_nombre = _obtener_valor(prof["persona"], "nombreCompleto", "nombre")
            if not prof_nombre:
                prof_nombre = f"Dr. ID {prof_id}"
                
            horarios = await dotnet_client.consultar_disponibilidad(
                profesional_id=prof_id,
                fecha=fecha,
                servicio_id=servicio_id
            )
            
            if horarios:
                slots = []
                for h in horarios:
                    inicio = _obtener_valor(h, "horaInicio", "fechaHoraInicio", "inicio")
                    if inicio:
                        if "T" in str(inicio):
                            inicio = str(inicio).split("T")[1][:5]
                        else:
                            inicio = str(inicio)[:5]
                        slots.append(inicio)
                if slots:
                    resultados.append(f"👨‍⚕️ {prof_nombre} (ID: {prof_id}):\n   Horarios: " + ", ".join(slots))
                else:
                    resultados.append(f"👨‍⚕️ {prof_nombre} (ID: {prof_id}): Sin horarios disponibles para esta fecha.")
            else:
                resultados.append(f"👨‍⚕️ {prof_nombre} (ID: {prof_id}): Sin horarios disponibles.")
                
        if not resultados:
            return f"No se encontraron espacios disponibles para {esp_nombre} el día {fecha}."
            
        return f"Horarios disponibles para {servicio_nombre} (ID de servicio: {servicio_id}) el día {fecha}:\n\n" + "\n".join(resultados)
    except Exception as exc:
        logger.error(f"Error al consultar disponibilidad: {exc}", exc_info=True)
        return "En este momento no podemos acceder a la disponibilidad de la agenda debido a problemas de conexión con el servidor. Por favor intenta de nuevo más tarde."

async def _agendar_cita_impl(
    profesional_id: int,
    servicio_id: int,
    fecha_hora_inicio: str,
    motivo_consulta: str,
    config: RunnableConfig
) -> str:
    try:
        thread_id = config["configurable"]["thread_id"]
        
        # 1. Obtener contexto de la conversación
        contexto = await dotnet_client.obtener_contexto_conversacion(thread_id)
        paciente_id = None
        
        if contexto:
            paciente_id = _obtener_valor(contexto, "pacienteId")
            if not paciente_id and "paciente" in contexto:
                paciente_id = _obtener_valor(contexto["paciente"], "pacienteId", "id")
                
        # 2. Si no está vinculado, buscamos el paciente por el número de teléfono
        if not paciente_id:
            clean_phone = thread_id.split("@")[0]
            pacientes = await dotnet_client.buscar_pacientes(clean_phone)
            
            if pacientes:
                paciente = pacientes[0]
                paciente_id = _obtener_valor(paciente, "pacienteId", "id")
                if paciente_id:
                    # Vinculamos el paciente de forma persistente
                    await dotnet_client.vincular_paciente(thread_id, paciente_id)
                    
        # 3. Si no existe en la BD relacional, solicitamos datos de registro
        if not paciente_id:
            return (
                "Lo siento, no pude encontrar tu número de teléfono registrado como paciente en nuestra base de datos. "
                "Por favor, facilítame tu nombre completo y número de documento para que nuestro personal de recepción pueda registrarte."
            )
            
        # 4. Agendar la cita en .NET
        payload = {
            "pacienteId": paciente_id,
            "profesionalId": profesional_id,
            "servicioId": servicio_id,
            "fechaHoraInicio": fecha_hora_inicio,
            "motivoConsulta": motivo_consulta,
            "origen": "CHATBOT"
        }
        
        respuesta = await dotnet_client.agendar_cita(payload)
        if respuesta:
            cita_id = _obtener_valor(respuesta, "citaId", "id")
            return f"¡Cita agendada con éxito! 🎉 Tu cita ha sido registrada. ID de cita: {cita_id}."
        else:
            return "Lo siento, ocurrió un problema al registrar la cita en el sistema. Es posible que el horario ya esté ocupado. Por favor, intenta con otro espacio."
    except Exception as exc:
        logger.error(f"Error al agendar cita: {exc}", exc_info=True)
        return "Lo siento, ocurrió un problema de conexión al registrar la cita en el sistema. Por favor, intenta de nuevo en unos minutos."


@tool
def consultar_disponibilidad_tool(especialidad: str, fecha: str) -> str:
    """
    Consulta los horarios disponibles para una especialidad odontológica en una fecha específica (formato YYYY-MM-DD).
    Usa esta herramienta cuando el usuario pregunte por horarios o citas disponibles para un servicio/especialidad (ej. ortodoncia, limpieza, profilaxis, valoración, resina).
    """
    return _run_sync(_consultar_disponibilidad_impl(especialidad, fecha))

@tool
def agendar_cita_tool(
    profesional_id: int,
    servicio_id: int,
    fecha_hora_inicio: str,
    motivo_consulta: str,
    config: RunnableConfig
) -> str:
    """
    Registra una cita en el sistema para el paciente de la conversación actual.
    Usa esta herramienta SOLAMENTE después de proponer los detalles de la cita y obtener una confirmación explícita y afirmativa del usuario en el chat.
    
    Parámetros:
    - profesional_id: ID del odontólogo seleccionado.
    - servicio_id: ID del servicio odontológico.
    - fecha_hora_inicio: Fecha y hora de inicio de la cita en formato ISO 8601 (ej. YYYY-MM-DDTHH:MM:SS-05:00).
    - motivo_consulta: Breve descripción de la razón de la consulta.
    """
    return _run_sync(_agendar_cita_impl(profesional_id, servicio_id, fecha_hora_inicio, motivo_consulta, config))
