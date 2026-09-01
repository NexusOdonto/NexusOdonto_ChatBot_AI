"""Máquina de estados para el flujo de registro/login de pacientes vía WhatsApp.

Este módulo gestiona todo el proceso de onboarding de un usuario nuevo:
1. Bienvenida con botones (Registrarse / Iniciar Sesión)
2. Recolección paso a paso de datos personales
3. Envío al endpoint de onboarding del backend
4. Autenticación del usuario registrado

El flujo es independiente del grafo LangGraph y se ejecuta antes de que
el mensaje llegue al agente conversacional.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional, List

from app.clients.dotnet_client import dotnet_client
from app.clients.evolution_client import evolution_client

logger = logging.getLogger(__name__)

# Tiempo máximo que un registro incompleto permanece en memoria (30 minutos).
REGISTRATION_TTL_SECONDS = 1800


class RegistrationStep(str, Enum):
    """Pasos del flujo de registro de paciente."""
    WELCOME = "WELCOME"
    DOC_TYPE = "DOC_TYPE"
    DOC_NUMBER = "DOC_NUMBER"
    FIRST_NAME = "FIRST_NAME"
    LAST_NAME = "LAST_NAME"
    PHONE = "PHONE"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    SEX = "SEX"
    EMAIL = "EMAIL"
    ADDRESS = "ADDRESS"
    PASSWORD = "PASSWORD"
    EMERGENCY_CONTACT = "EMERGENCY_CONTACT"
    EMERGENCY_PHONE = "EMERGENCY_PHONE"
    COMPLETED = "COMPLETED"
    # Pasos de login
    LOGIN_DOC_NUMBER = "LOGIN_DOC_NUMBER"
    LOGIN_PASSWORD = "LOGIN_PASSWORD"
    LOGIN_COMPLETED = "LOGIN_COMPLETED"


class FlowMode(str, Enum):
    """Modo del flujo activo."""
    NONE = "NONE"
    REGISTER = "REGISTER"
    LOGIN = "LOGIN"


@dataclass
class RegistrationState:
    """Estado de registro en curso para un usuario."""
    step: RegistrationStep = RegistrationStep.WELCOME
    mode: FlowMode = FlowMode.NONE
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Cachés de catálogos para no hacer múltiples requests
    _doc_types_cache: Optional[List[Dict[str, Any]]] = field(default=None, repr=False)
    _sexes_cache: Optional[List[Dict[str, Any]]] = field(default=None, repr=False)

    @property
    def is_expired(self) -> bool:
        return (datetime.utcnow() - self.updated_at).total_seconds() > REGISTRATION_TTL_SECONDS

    def touch(self) -> None:
        self.updated_at = datetime.utcnow()


# Almacén en memoria de estados de registro activos (phone → state).
_active_registrations: Dict[str, RegistrationState] = {}

# Contextos de usuario autenticados (phone → user_context).
_authenticated_users: Dict[str, Dict[str, Any]] = {}


def _normalize_phone(phone: str) -> str:
    """Extrae solo dígitos del número de WhatsApp, quitando el sufijo @s.whatsapp.net."""
    clean = phone.replace("@s.whatsapp.net", "").replace("@g.us", "").strip()
    return clean


def get_user_context(phone: str) -> Optional[Dict[str, Any]]:
    """Retorna el contexto del usuario autenticado si existe."""
    return _authenticated_users.get(_normalize_phone(phone))


def set_user_context(phone: str, context: Dict[str, Any]) -> None:
    """Guarda el contexto del usuario autenticado."""
    _authenticated_users[_normalize_phone(phone)] = context


def is_in_registration_flow(phone: str) -> bool:
    """Verifica si el usuario está en un flujo de registro/login activo."""
    normalized = _normalize_phone(phone)
    state = _active_registrations.get(normalized)
    if state is None:
        return False
    if state.is_expired:
        _active_registrations.pop(normalized, None)
        return False
    return state.step not in (RegistrationStep.COMPLETED, RegistrationStep.LOGIN_COMPLETED)


def is_user_authenticated(phone: str) -> bool:
    """Verifica si el usuario ya está autenticado en esta sesión."""
    return _normalize_phone(phone) in _authenticated_users


async def logout_user(phone: str) -> None:
    """Cierra la sesión del usuario, elimina el contexto y envía mensaje de despedida."""
    normalized = _normalize_phone(phone)
    user_ctx = _authenticated_users.pop(normalized, None)
    _active_registrations.pop(normalized, None)
    
    nombre = ""
    if user_ctx and isinstance(user_ctx, dict):
        nombre = user_ctx.get("firstName") or user_ctx.get("fullName") or ""
    
    saludo = f", *{nombre}*" if nombre else ""

    await evolution_client.enviar_mensaje(
        phone,
        f"👋 *¡Has cerrado sesión exitosamente{saludo}!* 🦷✨\n\n"
        "Tu sesión en *Nexus Odonto* ha finalizado de forma segura.\n\n"
        "Cuando desees volver a ingresar, agendar una cita o consultar nuestros servicios, escríbenos nuevamente. ¡Hasta pronto! 😊"
    )
    logger.info(f"[Registration] Sesión cerrada para {normalized}")




async def check_user_registered(phone: str) -> bool:
    """Consulta al backend si el teléfono ya tiene un paciente asociado."""
    try:
        persona = await dotnet_client.buscar_persona_por_telefono(phone)
        if persona:
            person_id = persona.get("id")
            if person_id:
                paciente = await dotnet_client.buscar_paciente_por_person_id(str(person_id))
                if paciente:
                    return True
        return False
    except Exception as e:
        logger.error(f"[Registration] Error verificando registro de {phone}: {e}")
        return False


async def start_welcome_flow(phone: str) -> None:
    """Inicia el flujo de bienvenida enviando botones de Registrarse / Iniciar Sesión."""
    normalized = _normalize_phone(phone)

    # Crear estado nuevo
    state = RegistrationState()
    _active_registrations[normalized] = state

    botones = [
        {
            "buttonId": "REGISTRARSE",
            "buttonText": {"displayText": "📝 Registrarme"},
        },
        {
            "buttonId": "INICIAR_SESION",
            "buttonText": {"displayText": "🔑 Iniciar Sesión"},
        },
    ]

    await evolution_client.enviar_botones(
        numero=phone,
        titulo="¡Bienvenido a Nexus Odonto! 🦷✨",
        descripcion=(
            "Soy tu asistente virtual. Para brindarte la mejor atención personalizada, "
            "necesito que te registres o inicies sesión.\n\n"
            "¿Qué deseas hacer?"
        ),
        botones=botones,
        pie="Nexus Odonto — Tu sonrisa, nuestra prioridad",
    )

    logger.info(f"[Registration] Flujo de bienvenida iniciado para {normalized}")


async def process_registration_message(phone: str, message: str) -> bool:
    """Procesa un mensaje dentro del flujo de registro/login.

    Returns:
        True si el mensaje fue procesado por el flujo de registro.
        False si el flujo ya completó y el mensaje debe ir al grafo normal.
    """
    normalized = _normalize_phone(phone)
    state = _active_registrations.get(normalized)

    if state is None:
        return False

    if state.is_expired:
        _active_registrations.pop(normalized, None)
        return False

    state.touch()
    texto = message.strip()

    try:
        if state.step == RegistrationStep.WELCOME:
            await _handle_welcome_response(phone, state, texto)
        elif state.step == RegistrationStep.DOC_TYPE:
            await _handle_doc_type(phone, state, texto)
        elif state.step == RegistrationStep.DOC_NUMBER:
            await _handle_doc_number(phone, state, texto)
        elif state.step == RegistrationStep.FIRST_NAME:
            await _handle_first_name(phone, state, texto)
        elif state.step == RegistrationStep.LAST_NAME:
            await _handle_last_name(phone, state, texto)
        elif state.step == RegistrationStep.PHONE:
            await _handle_phone(phone, state, texto)
        elif state.step == RegistrationStep.DATE_OF_BIRTH:
            await _handle_date_of_birth(phone, state, texto)
        elif state.step == RegistrationStep.SEX:
            await _handle_sex(phone, state, texto)
        elif state.step == RegistrationStep.EMAIL:
            await _handle_email(phone, state, texto)
        elif state.step == RegistrationStep.ADDRESS:
            await _handle_address(phone, state, texto)
        elif state.step == RegistrationStep.PASSWORD:
            await _handle_password(phone, state, texto)
        elif state.step == RegistrationStep.EMERGENCY_CONTACT:
            await _handle_emergency_contact(phone, state, texto)
        elif state.step == RegistrationStep.EMERGENCY_PHONE:
            await _handle_emergency_phone(phone, state, texto)
        # Login steps
        elif state.step == RegistrationStep.LOGIN_DOC_NUMBER:
            await _handle_login_doc_number(phone, state, texto)
        elif state.step == RegistrationStep.LOGIN_PASSWORD:
            await _handle_login_password(phone, state, texto)
        elif state.step in (RegistrationStep.COMPLETED, RegistrationStep.LOGIN_COMPLETED):
            return False
        else:
            return False
    except Exception as e:
        logger.error(f"[Registration] Error procesando paso {state.step} para {normalized}: {e}", exc_info=True)
        await evolution_client.enviar_mensaje(
            phone,
            "⚠️ Ocurrió un error procesando tu respuesta. Por favor, intenta de nuevo."
        )

    return True


# ─────────────────────────────────────────────────────────
# Handlers para cada paso del flujo de REGISTRO
# ─────────────────────────────────────────────────────────

async def _handle_welcome_response(phone: str, state: RegistrationState, texto: str) -> None:
    """Procesa la selección de Registrarse o Iniciar Sesión."""
    texto_upper = texto.upper().strip()

    # Detectar selección de botón o texto libre
    is_register = any(k in texto_upper for k in ["REGISTR", "1", "📝"])
    is_login = any(k in texto_upper for k in ["INICIAR", "SESION", "SESIÓN", "LOGIN", "2", "🔑"])

    if is_register:
        state.mode = FlowMode.REGISTER
        state.step = RegistrationStep.DOC_TYPE
        await _ask_doc_type(phone, state)
    elif is_login:
        state.mode = FlowMode.LOGIN
        state.step = RegistrationStep.LOGIN_DOC_NUMBER
        await evolution_client.enviar_mensaje(
            phone,
            "🔑 *Iniciar Sesión*\n\n"
            "Por favor, ingresa tu *número de documento* con el que te registraste:"
        )
    else:
        # Repetir la pregunta
        await evolution_client.enviar_mensaje(
            phone,
            "No entendí tu selección. Por favor, elige una opción:\n\n"
            "1️⃣ *Registrarme*\n"
            "2️⃣ *Iniciar Sesión*\n\n"
            "Puedes escribir el número o el nombre de la opción."
        )


async def _ask_doc_type(phone: str, state: RegistrationState) -> None:
    """Envía los botones/lista con los tipos de documento disponibles."""
    # Obtener catálogo del backend
    if state._doc_types_cache is None:
        state._doc_types_cache = await dotnet_client.obtener_tipos_documento()

    doc_types = state._doc_types_cache

    if not doc_types:
        # Fallback con tipos de documento comunes
        doc_types = [
            {"id": None, "code": "CC", "name": "Cédula de Ciudadanía"},
            {"id": None, "code": "TI", "name": "Tarjeta de Identidad"},
            {"id": None, "code": "CE", "name": "Cédula de Extranjería"},
            {"id": None, "code": "PP", "name": "Pasaporte"},
        ]

    if len(doc_types) <= 3:
        # Usar botones (máximo 3)
        botones = [
            {
                "buttonId": dt.get("code", dt.get("id", "")),
                "buttonText": {"displayText": dt.get("name", dt.get("code", ""))},
            }
            for dt in doc_types
        ]
        await evolution_client.enviar_botones(
            numero=phone,
            titulo="📋 Tipo de Documento",
            descripcion="Selecciona tu tipo de documento de identidad:",
            botones=botones,
            pie="Paso 1 de 11",
        )
    else:
        # Usar lista para más de 3 opciones
        rows = [
            {
                "title": dt.get("name", dt.get("code", "")),
                "description": dt.get("code", ""),
                "rowId": dt.get("code", dt.get("id", "")),
            }
            for dt in doc_types
        ]
        await evolution_client.enviar_lista(
            numero=phone,
            titulo="📋 Tipo de Documento",
            descripcion="Selecciona tu tipo de documento de identidad:",
            texto_boton="Ver opciones",
            secciones=[{"title": "Tipos de Documento", "rows": rows}],
            pie="Paso 1 de 11",
        )


async def _handle_doc_type(phone: str, state: RegistrationState, texto: str) -> None:
    """Procesa la selección del tipo de documento."""
    if state._doc_types_cache is None:
        state._doc_types_cache = await dotnet_client.obtener_tipos_documento()

    doc_types = state._doc_types_cache
    texto_upper = texto.upper().strip()

    selected = None
    for dt in doc_types:
        code = (dt.get("code") or "").upper()
        name = (dt.get("name") or "").upper()
        if texto_upper == code or texto_upper == name or code in texto_upper or texto_upper in name:
            selected = dt
            break

    # También intentar por número si escribieron "1", "2", etc.
    if not selected and texto.isdigit():
        idx = int(texto) - 1
        if 0 <= idx < len(doc_types):
            selected = doc_types[idx]

    if selected:
        state.data["documentTypeId"] = str(selected.get("id", ""))
        state.data["documentTypeCode"] = selected.get("code", "")
        state.data["documentTypeName"] = selected.get("name", "")
        state.step = RegistrationStep.DOC_NUMBER
        await evolution_client.enviar_mensaje(
            phone,
            f"✅ Tipo de documento: *{selected.get('name', selected.get('code', ''))}*\n\n"
            "Ahora ingresa tu *número de documento*:"
        )
    else:
        await evolution_client.enviar_mensaje(
            phone,
            "❌ No reconocí esa opción. Por favor selecciona un tipo de documento válido de la lista."
        )
        await _ask_doc_type(phone, state)


async def _handle_doc_number(phone: str, state: RegistrationState, texto: str) -> None:
    """Procesa el número de documento."""
    # Validar que no esté vacío y tenga formato razonable
    doc_number = texto.strip()
    if len(doc_number) < 3 or len(doc_number) > 30:
        await evolution_client.enviar_mensaje(
            phone,
            "❌ El número de documento debe tener entre 3 y 30 caracteres. "
            "Por favor, ingrésalo nuevamente:"
        )
        return

    state.data["documentNumber"] = doc_number
    state.step = RegistrationStep.FIRST_NAME
    await evolution_client.enviar_mensaje(
        phone,
        f"✅ Documento: *{doc_number}*\n\n"
        "¿Cuáles son tus *nombres*? (Ej: Juan Carlos)"
    )


async def _handle_first_name(phone: str, state: RegistrationState, texto: str) -> None:
    """Procesa los nombres del paciente."""
    if len(texto) < 2 or len(texto) > 100:
        await evolution_client.enviar_mensaje(
            phone,
            "❌ El nombre debe tener entre 2 y 100 caracteres. Inténtalo de nuevo:"
        )
        return

    state.data["firstName"] = texto.strip().title()
    state.step = RegistrationStep.LAST_NAME
    await evolution_client.enviar_mensaje(
        phone,
        f"✅ Nombres: *{state.data['firstName']}*\n\n"
        "¿Cuáles son tus *apellidos*? (Ej: Pérez García)"
    )


async def _handle_last_name(phone: str, state: RegistrationState, texto: str) -> None:
    """Procesa los apellidos del paciente."""
    if len(texto) < 2 or len(texto) > 100:
        await evolution_client.enviar_mensaje(
            phone,
            "❌ Los apellidos deben tener entre 2 y 100 caracteres. Inténtalo de nuevo:"
        )
        return

    state.data["lastName"] = texto.strip().title()
    state.step = RegistrationStep.PHONE
    await evolution_client.enviar_mensaje(
        phone,
        f"✅ Apellidos: *{state.data['lastName']}*\n\n"
        "¿Cuál es tu *número de teléfono o celular* personal de contacto?\n"
        "(Ej: 3101234567)"
    )


async def _handle_phone(phone: str, state: RegistrationState, texto: str) -> None:
    """Procesa el número de teléfono personal ingresado manualmente."""
    phone_digits = "".join(c for c in texto if c.isdigit())
    if len(phone_digits) < 7 or len(phone_digits) > 15:
        await evolution_client.enviar_mensaje(
            phone,
            "❌ Por favor ingresa un número de teléfono válido (entre 7 y 15 dígitos numéricos).\n"
            "Ejemplo: *3101234567*"
        )
        return

    state.data["phone"] = phone_digits
    state.step = RegistrationStep.DATE_OF_BIRTH
    await evolution_client.enviar_mensaje(
        phone,
        f"✅ Teléfono: *{phone_digits}*\n\n"
        "¿Cuál es tu *fecha de nacimiento*?\n"
        "Escríbela en formato *DD/MM/AAAA* (Ej: 15/03/1990)\n\n"
        "Si deseas omitir este campo, escribe *omitir*."
    )



async def _handle_date_of_birth(phone: str, state: RegistrationState, texto: str) -> None:
    """Procesa la fecha de nacimiento."""
    if texto.lower() in ("omitir", "skip", "no", "pasar"):
        state.data["dateOfBirth"] = None
        state.step = RegistrationStep.SEX
        await _ask_sex(phone, state)
        return

    # Intentar parsear la fecha en múltiples formatos
    date_formats = ["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%Y"]
    parsed_date = None
    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(texto.strip(), fmt)
            break
        except ValueError:
            continue

    if parsed_date is None:
        await evolution_client.enviar_mensaje(
            phone,
            "❌ No pude entender esa fecha. Por favor, usa el formato *DD/MM/AAAA*\n"
            "(Ej: 15/03/1990)\n\n"
            "O escribe *omitir* para saltar este campo."
        )
        return

    # Validar que sea una fecha razonable
    if parsed_date.year < 1900 or parsed_date > datetime.now():
        await evolution_client.enviar_mensaje(
            phone,
            "❌ La fecha no parece válida. Por favor, ingresa tu fecha de nacimiento real."
        )
        return

    # Formato ISO para el backend: YYYY-MM-DD
    state.data["dateOfBirth"] = parsed_date.strftime("%Y-%m-%d")
    state.step = RegistrationStep.SEX
    await _ask_sex(phone, state)


async def _ask_sex(phone: str, state: RegistrationState) -> None:
    """Envía botones con las opciones de sexo."""
    if state._sexes_cache is None:
        state._sexes_cache = await dotnet_client.obtener_sexos()

    sexes = state._sexes_cache

    if not sexes:
        sexes = [
            {"id": None, "code": "M", "name": "Masculino"},
            {"id": None, "code": "F", "name": "Femenino"},
            {"id": None, "code": "O", "name": "Otro"},
        ]

    fecha_msg = ""
    if state.data.get("dateOfBirth"):
        fecha_msg = f"✅ Fecha de nacimiento: *{state.data['dateOfBirth']}*\n\n"
    else:
        fecha_msg = "✅ Fecha de nacimiento: *Omitida*\n\n"

    # Máximo 3 botones en WhatsApp
    botones = [
        {
            "buttonId": s.get("code", s.get("id", "")),
            "buttonText": {"displayText": s.get("name", s.get("code", ""))},
        }
        for s in sexes[:3]
    ]

    await evolution_client.enviar_mensaje(phone, fecha_msg + "Selecciona tu *sexo*:")
    await evolution_client.enviar_botones(
        numero=phone,
        titulo="👤 Sexo",
        descripcion="Selecciona una opción:",
        botones=botones,
        pie="Paso 6 de 11",
    )


async def _handle_sex(phone: str, state: RegistrationState, texto: str) -> None:
    """Procesa la selección de sexo."""
    if state._sexes_cache is None:
        state._sexes_cache = await dotnet_client.obtener_sexos()

    sexes = state._sexes_cache
    texto_upper = texto.upper().strip()

    selected = None
    for s in sexes:
        code = (s.get("code") or "").upper()
        name = (s.get("name") or "").upper()
        if texto_upper == code or texto_upper == name or code in texto_upper or texto_upper in name:
            selected = s
            break

    if not selected and texto.isdigit():
        idx = int(texto) - 1
        if 0 <= idx < len(sexes):
            selected = sexes[idx]

    # Búsqueda flexible
    if not selected:
        if any(k in texto_upper for k in ["MASC", "HOMBRE", "M"]):
            for s in sexes:
                if (s.get("code") or "").upper() == "M" or "MASC" in (s.get("name") or "").upper():
                    selected = s
                    break
        elif any(k in texto_upper for k in ["FEM", "MUJER", "F"]):
            for s in sexes:
                if (s.get("code") or "").upper() == "F" or "FEM" in (s.get("name") or "").upper():
                    selected = s
                    break

    if selected:
        state.data["sexId"] = str(selected.get("id", "")) if selected.get("id") else None
        state.data["sexName"] = selected.get("name", "")
        state.step = RegistrationStep.EMAIL
        await evolution_client.enviar_mensaje(
            phone,
            f"✅ Sexo: *{selected.get('name', '')}*\n\n"
            "¿Cuál es tu *correo electrónico*?\n"
            "(Ej: ejemplo@email.com)\n\n"
            "Si deseas omitir, escribe *omitir*."
        )
    else:
        await evolution_client.enviar_mensaje(
            phone,
            "❌ No reconocí esa opción. Por favor selecciona un sexo válido."
        )
        await _ask_sex(phone, state)


async def _handle_email(phone: str, state: RegistrationState, texto: str) -> None:
    """Procesa el correo electrónico."""
    if texto.lower() in ("omitir", "skip", "no", "pasar"):
        state.data["email"] = None
        state.step = RegistrationStep.ADDRESS
        await evolution_client.enviar_mensaje(
            phone,
            "✅ Email: *Omitido*\n\n"
            "¿Cuál es tu *dirección de residencia*?\n"
            "(Ej: Cra 24 #35-12, Barrio Centro, Bucaramanga)\n\n"
            "Si deseas omitir, escribe *omitir*."
        )
        return

    # Validar formato de email básico
    email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    if not email_pattern.match(texto.strip()):
        await evolution_client.enviar_mensaje(
            phone,
            "❌ El correo no parece válido. Por favor, ingresa un email válido\n"
            "(Ej: ejemplo@email.com)\n\n"
            "O escribe *omitir* para saltar."
        )
        return

    state.data["email"] = texto.strip().lower()
    state.step = RegistrationStep.ADDRESS
    await evolution_client.enviar_mensaje(
        phone,
        f"✅ Email: *{state.data['email']}*\n\n"
        "¿Cuál es tu *dirección de residencia*?\n"
        "(Ej: Cra 24 #35-12, Barrio Centro, Bucaramanga)\n\n"
        "Si deseas omitir, escribe *omitir*."
    )


async def _handle_address(phone: str, state: RegistrationState, texto: str) -> None:
    """Procesa la dirección."""
    if texto.lower() in ("omitir", "skip", "no", "pasar"):
        state.data["address"] = None
    else:
        state.data["address"] = texto.strip()

    addr_display = state.data["address"] or "Omitida"
    state.step = RegistrationStep.PASSWORD
    await evolution_client.enviar_mensaje(
        phone,
        f"✅ Dirección: *{addr_display}*\n\n"
        "🔐 Ahora crea una *contraseña* para tu cuenta.\n\n"
        "La contraseña debe tener *mínimo 6 caracteres*.\n"
        "Esta contraseña la usarás para iniciar sesión en el futuro."
    )


async def _handle_password(phone: str, state: RegistrationState, texto: str) -> None:
    """Procesa la contraseña del usuario."""
    if len(texto) < 6:
        await evolution_client.enviar_mensaje(
            phone,
            "❌ La contraseña debe tener *mínimo 6 caracteres*.\n"
            "Por favor, ingresa una contraseña más segura:"
        )
        return

    state.data["password"] = texto
    state.step = RegistrationStep.EMERGENCY_CONTACT
    await evolution_client.enviar_mensaje(
        phone,
        "✅ Contraseña configurada correctamente 🔒\n\n"
        "Ahora necesitamos los datos de tu *contacto de emergencia*.\n\n"
        "¿Cuál es el *nombre completo* de tu contacto de emergencia?\n"
        "(Ej: María García López)"
    )


async def _handle_emergency_contact(phone: str, state: RegistrationState, texto: str) -> None:
    """Procesa el nombre del contacto de emergencia."""
    if len(texto) < 2:
        await evolution_client.enviar_mensaje(
            phone,
            "❌ El nombre del contacto debe tener al menos 2 caracteres. Inténtalo de nuevo:"
        )
        return

    state.data["emergencyContact"] = texto.strip().title()
    state.step = RegistrationStep.EMERGENCY_PHONE
    await evolution_client.enviar_mensaje(
        phone,
        f"✅ Contacto de emergencia: *{state.data['emergencyContact']}*\n\n"
        "¿Cuál es el *número de teléfono* de tu contacto de emergencia?\n"
        "(Ej: 3001234567)"
    )


async def _handle_emergency_phone(phone: str, state: RegistrationState, texto: str) -> None:
    """Procesa el teléfono de emergencia y finaliza el registro."""
    # Limpiar y validar número
    phone_digits = "".join(c for c in texto if c.isdigit())
    if len(phone_digits) < 7 or len(phone_digits) > 15:
        await evolution_client.enviar_mensaje(
            phone,
            "❌ El número de teléfono no parece válido. Debe tener entre 7 y 15 dígitos.\n"
            "Por favor, ingrésalo de nuevo:"
        )
        return

    state.data["emergencyPhone"] = phone_digits
    state.step = RegistrationStep.COMPLETED

    # Mostrar resumen antes de registrar
    await _show_registration_summary(phone, state)


async def _show_registration_summary(phone: str, state: RegistrationState) -> None:
    """Muestra un resumen de los datos y procede con el registro."""
    data = state.data
    summary = (
        "📋 *Resumen de tu Registro*\n\n"
        f"• 📄 *Tipo de documento:* {data.get('documentTypeName', 'N/A')}\n"
        f"• 🆔 *Número de documento:* {data.get('documentNumber', 'N/A')}\n"
        f"• 👤 *Nombres:* {data.get('firstName', 'N/A')}\n"
        f"• 👤 *Apellidos:* {data.get('lastName', 'N/A')}\n"
        f"• 📱 *Teléfono de contacto:* {data.get('phone', 'N/A')}\n"
        f"• 📅 *Fecha de nacimiento:* {data.get('dateOfBirth', 'No especificada')}\n"
        f"• ⚧ *Sexo:* {data.get('sexName', 'No especificado')}\n"
        f"• 📧 *Email:* {data.get('email', 'No especificado')}\n"
        f"• 📍 *Dirección:* {data.get('address', 'No especificada')}\n"
        f"• 🚨 *Contacto emergencia:* {data.get('emergencyContact', 'N/A')}\n"
        f"• 📞 *Teléfono emergencia:* {data.get('emergencyPhone', 'N/A')}\n"
    )
    await evolution_client.enviar_mensaje(phone, summary)

    # Proceder con el registro en el backend
    await _complete_registration(phone, state)


async def _complete_registration(phone: str, state: RegistrationState) -> None:
    """Envía los datos al backend para completar el registro."""
    normalized = _normalize_phone(phone)
    data = state.data
    contacto_phone = data.get("phone") or normalized

    # Construir el payload para OnboardPatientDto
    onboard_payload = {
        "documentTypeId": data.get("documentTypeId", ""),
        "documentNumber": data.get("documentNumber", ""),
        "firstName": data.get("firstName", ""),
        "lastName": data.get("lastName", ""),
        "phone": contacto_phone,
        "password": data.get("password", ""),
        "emergencyContact": data.get("emergencyContact"),
        "emergencyPhone": data.get("emergencyPhone"),
    }

    # Campos opcionales
    if data.get("dateOfBirth"):
        onboard_payload["dateOfBirth"] = data["dateOfBirth"]
    if data.get("sexId"):
        onboard_payload["sexId"] = data["sexId"]
    if data.get("email"):
        onboard_payload["email"] = data["email"]
    if data.get("address"):
        onboard_payload["address"] = data["address"]

    await evolution_client.enviar_mensaje(
        phone,
        "⏳ Registrando tu perfil en nuestro sistema, por favor espera un momento..."
    )

    result = await dotnet_client.registrar_paciente(onboard_payload)

    if result:
        person_id = result.get("personId", "")
        patient_id = result.get("patientId", "")
        user_id = result.get("userId", "")

        # Guardar contexto de usuario autenticado
        set_user_context(phone, {
            "personId": str(person_id),
            "patientId": str(patient_id),
            "userId": str(user_id),
            "firstName": data.get("firstName", ""),
            "lastName": data.get("lastName", ""),
            "fullName": f"{data.get('firstName', '')} {data.get('lastName', '')}".strip(),
            "documentNumber": data.get("documentNumber", ""),
            "phone": contacto_phone,
        })

        await evolution_client.enviar_mensaje(
            phone,
            f"🎉 *¡Registro exitoso!*\n\n"
            f"¡Bienvenido/a, *{data.get('firstName', '')}*! Tu perfil ha sido creado correctamente en Nexus Odonto.\n\n"
            f"📄 Tu número de documento *{data.get('documentNumber', '')}* es tu identificador para iniciar sesión.\n\n"
            "Ahora puedo ayudarte con:\n"
            "• 📅 *Agendar citas*\n"
            "• 📋 *Ver mis citas programadas*\n"
            "• 🦷 *Consultar servicios y precios*\n"
            "• 👨‍⚕️ *Conocer nuestros especialistas*\n"
            "• 💡 *Resolver dudas sobre tratamientos*\n"
            "• 🚪 *Cerrar sesión* (escribe *cerrar sesión* cuando desees salir)\n\n"
            "¿En qué puedo ayudarte hoy? 😊"
        )

        # Limpiar el estado de registro
        _active_registrations.pop(normalized, None)
        logger.info(f"[Registration] Registro completado exitosamente para {normalized} (PersonId: {person_id})")
    else:
        await evolution_client.enviar_mensaje(
            phone,
            "❌ *Hubo un problema al registrarte.*\n\n"
            "Es posible que ya exista una cuenta con ese número de documento.\n"
            "Si ya tienes una cuenta, intenta *iniciar sesión*.\n\n"
            "Si el problema persiste, contáctanos al +57 324 6030217."
        )
        # Reiniciar flujo
        state.step = RegistrationStep.WELCOME
        state.mode = FlowMode.NONE
        state.data = {}
        logger.warning(f"[Registration] Error en onboarding para {normalized}")


# ─────────────────────────────────────────────────────────
# Handlers para el flujo de LOGIN
# ─────────────────────────────────────────────────────────

async def _handle_login_doc_number(phone: str, state: RegistrationState, texto: str) -> None:
    """Procesa el número de documento para login."""
    doc_number = texto.strip()
    if len(doc_number) < 3:
        await evolution_client.enviar_mensaje(
            phone,
            "❌ El número de documento debe tener al menos 3 caracteres. Inténtalo de nuevo:"
        )
        return

    state.data["loginDocNumber"] = doc_number
    state.step = RegistrationStep.LOGIN_PASSWORD
    await evolution_client.enviar_mensaje(
        phone,
        f"✅ Documento: *{doc_number}*\n\n"
        "Ahora ingresa tu *contraseña*:"
    )


async def _handle_login_password(phone: str, state: RegistrationState, texto: str) -> None:
    """Procesa la contraseña de login y autentica."""
    normalized = _normalize_phone(phone)
    password = texto.strip()

    await evolution_client.enviar_mensaje(
        phone,
        "⏳ Verificando tus credenciales..."
    )

    result = await dotnet_client.login_paciente(
        document_number=state.data.get("loginDocNumber", ""),
        password=password,
    )

    if result:
        # Login exitoso
        state.step = RegistrationStep.LOGIN_COMPLETED

        # Intentar obtener datos del paciente para el contexto
        token = result.get("token", "")
        doc_number = state.data.get("loginDocNumber", "")
        user_context = {
            "token": token,
            "documentNumber": doc_number,
            "phone": normalized,
        }

        # Buscar persona por número de documento para soportar login desde cualquier dispositivo
        persona = await dotnet_client.buscar_persona_por_documento(doc_number)
        if not persona:
            persona = await dotnet_client.buscar_persona_por_telefono(phone)

        if persona:
            person_id = str(persona.get("id", ""))
            user_context.update({
                "personId": person_id,
                "firstName": persona.get("firstName", ""),
                "lastName": persona.get("lastName", ""),
                "fullName": f"{persona.get('firstName', '')} {persona.get('lastName', '')}".strip(),
                "phone": persona.get("phone") or normalized,
            })
            patient = await dotnet_client.buscar_paciente_por_person_id(person_id)
            if patient:
                user_context["patientId"] = str(patient.get("id", ""))

        set_user_context(phone, user_context)

        nombre = user_context.get("firstName", "")
        saludo = f", *{nombre}*" if nombre else ""

        await evolution_client.enviar_mensaje(
            phone,
            f"🎉 *¡Sesión iniciada exitosamente{saludo}!* 🦷✨\n\n"
            "¡Qué gusto tenerte de vuelta en Nexus Odonto!\n\n"
            "¿En qué puedo ayudarte hoy? 😊\n\n"
            "• 📅 *Agendar citas*\n"
            "• 📋 *Ver mis citas programadas*\n"
            "• 🦷 *Consultar servicios y precios*\n"
            "• 👨‍⚕️ *Conocer nuestros especialistas*\n"
            "• 💡 *Resolver dudas sobre tratamientos*\n"
            "• 🚪 *Cerrar sesión* (escribe *cerrar sesión* cuando desees salir)"
        )

        _active_registrations.pop(normalized, None)
        logger.info(f"[Registration] Login exitoso para {normalized}")
    else:
        # Login fallido
        await evolution_client.enviar_mensaje(
            phone,
            "❌ *Credenciales incorrectas.*\n\n"
            "El número de documento o la contraseña no son válidos.\n\n"
            "¿Qué deseas hacer?\n"
            "1️⃣ *Intentar de nuevo* — escribe tu número de documento\n"
            "2️⃣ *Registrarme* — escribe \"registrarme\"\n"
        )
        state.step = RegistrationStep.LOGIN_DOC_NUMBER
        logger.warning(f"[Registration] Login fallido para {normalized}")


def cleanup_expired_registrations() -> int:
    """Limpia estados de registro expirados. Retorna el número de estados limpiados."""
    expired_phones = [
        phone for phone, state in _active_registrations.items()
        if state.is_expired
    ]
    for phone in expired_phones:
        _active_registrations.pop(phone, None)
    return len(expired_phones)
