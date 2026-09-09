import datetime
import logging
import re
import unicodedata
import uuid
from typing import Optional, Dict, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.agents.tools.qdrant_tool import get_embeddings, get_qdrant_client
from app.core.config import settings

logger = logging.getLogger(__name__)

# Caché L1 en memoria (máximo 500 pares query -> respuesta) para respuesta en 0.1ms sin llamadas a red
_L1_MEMORY_CACHE: Dict[str, str] = {}
_MAX_L1_ENTRIES = 500

# Respuestas pre-armadas deterministas de máxima velocidad (0 tokens, 0 red, 0ms)
_FAST_RESPONSES: Dict[str, str] = {
    "saludo": (
        "¡Hola! Qué gusto saludarte 👋✨ Bienvenido a *Nexus Odonto* 🦷.\n\n"
        "Soy tu asistente virtual. ¿En qué te puedo ayudar hoy?\n\n"
        "• 📅 *Agendar una cita:* Consulta disponibilidad y reserva tu turno.\n"
        "• 📋 *Ver mis citas:* Consulta tus citas programadas con tu número de cédula.\n"
        "• ✏️ *Modificar una cita:* Reprograma una cita existente.\n"
        "• ❌ *Cancelar una cita:* Cancela una cita que no puedas atender.\n"
        "• 🦷 *Servicios y tratamientos:* Conoce nuestros procedimientos y tarifas.\n"
        "• 👨‍⚕️ *Nuestros especialistas:* Conoce a nuestro equipo de odontólogos.\n"
        "• 💡 *Dudas odontológicas:* Cuidados bucales, recomendaciones o preparaciones.\n\n"
        "¿Cómo podemos ayudarte hoy? 😊"
    ),
    "horario": (
        "⏰ *Horarios de Atención - Nexus Odonto:*\n\n"
        "• *Lunes a Sábado:* 8:00 AM a 6:00 PM (Jornada Continua)\n"
        "• *Domingos y Festivos:* Cerrado\n\n"
        "¿Te gustaría consultar disponibilidad o agendar una cita para algún día en específico? 📅🦷"
    ),
    "ubicacion": (
        "📍 *Ubicación de Nexus Odonto:*\n"
        "Cr 24 #35-12, Santander 🏥\n\n"
        "🚗 Contamos con excelente ubicación y fácil acceso.\n\n"
        "¿Deseas conocer más sobre nuestros servicios o agendar una cita con nuestros especialistas? ✨"
    ),
    "contacto": (
        "📞 *Canales de Contacto - Nexus Odonto:*\n\n"
        "• 📱 *WhatsApp / Teléfono:* +57 324 6030217\n"
        "• 📍 *Dirección:* Cr 24 #35-12, Santander\n"
        "• ⏰ *Horario:* Lunes a Sábado de 8:00 AM a 6:00 PM\n\n"
        "¡Estamos listos para cuidar de tu sonrisa! ¿En qué más te puedo orientar? 😊"
    ),
    "agradecimiento": (
        "¡Con el mayor de los gustos! 😊 En *Nexus Odonto* siempre estamos listos para cuidar de tu salud bucal 🦷✨.\n\n"
        "Si necesitas algo más, solo escríbeme. ¡Que tengas un excelente día! 👋"
    ),
}

# Mapeo de frases exactas / patrones hacia respuestas rápidas
_FAST_MATCH_PATTERNS = [
    # Saludos
    (r"^(hola|buenas|buen dia|buenos dias|buenas tardes|buenas noches|hola buenas|hola buen dia|hey|hola que tal|saludos|inicio|menu|opciones|ayuda|que puedes hacer|empezar)$", "saludo"),
    # Horarios
    (r"^(horario|horarios|que horario tienen|a que hora abren|a que hora cierran|atienden sabados|atienden los sabados|que dias atienden|horario de atencion)$", "horario"),
    # Ubicación
    (r"^(ubicacion|donde estan|donde quedan|direccion|donde estan ubicados|donde estan localizados|cual es la direccion|donde queda la clinica|como llegar)$", "ubicacion"),
    # Contacto
    (r"^(contacto|telefono|numero de telefono|whatsapp|linea telefonica|numero de contacto|canales de atencion)$", "contacto"),
    # Despedidas / Agradecimientos
    (r"^(gracias|muchas gracias|muchisimas gracias|mil gracias|chao|adios|hasta luego|vale gracias|ok gracias)$", "agradecimiento"),
]


def _normalize_query_key(text: str) -> str:
    """Normaliza un texto eliminando tildes, signos de puntuación y espacios extras."""
    t = text.lower().strip()
    t = "".join(
        c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn"
    )
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _get_fast_response(text: str) -> Optional[str]:
    """Retorna una respuesta instantánea si el mensaje coincide con un patrón determinista común."""
    norm = _normalize_query_key(text)
    if not norm:
        return None

    for pattern, cat in _FAST_MATCH_PATTERNS:
        if re.match(pattern, norm):
            return _FAST_RESPONSES.get(cat)
    return None


def ensure_cache_collection(client: Optional[QdrantClient] = None) -> None:
    """Crea la colección de caché semántico en Qdrant si no existe o si cambió la dimensión."""
    if client is None:
        client = get_qdrant_client()

    col_name = settings.semantic_cache_collection
    if client.collection_exists(col_name):
        try:
            info = client.get_collection(col_name)
            vectors_config = info.config.params.vectors
            existing_size = getattr(vectors_config, "size", None)
            if existing_size is None and isinstance(vectors_config, dict):
                first_val = next(iter(vectors_config.values()), None)
                existing_size = getattr(first_val, "size", None)

            if existing_size is not None and existing_size != settings.embedding_dimension:
                logger.warning(
                    f"[Semantic Cache] Dimensión incompatible detectada ({existing_size} != {settings.embedding_dimension}). "
                    f"Recreando colección '{col_name}'..."
                )
                client.delete_collection(col_name)
                client.create_collection(
                    collection_name=col_name,
                    vectors_config=models.VectorParams(
                        size=settings.embedding_dimension,
                        distance=models.Distance.COSINE,
                    ),
                )
        except Exception as exc:
            logger.warning(f"[Semantic Cache] Error validando dimensiones de colección: {exc}")
    else:
        client.create_collection(
            collection_name=col_name,
            vectors_config=models.VectorParams(
                size=settings.embedding_dimension,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info(f"[Semantic Cache] Colección '{col_name}' creada exitosamente en Qdrant.")


def _es_contenido_cacheable(pregunta: str, respuesta: str) -> bool:
    """Valida si un par pregunta-respuesta es apto para almacenarse en el caché semántico.
    
    Excluye respuestas transaccionales (citas, disponibilidad en tiempo real, agendamientos,
    emergencias médicas o mensajes de error/escalamiento).
    """
    p_lower = pregunta.lower().strip()
    r_lower = respuesta.lower().strip()

    # Descartar preguntas vacías
    if len(p_lower) < 3 or len(r_lower) < 10:
        return False

    # Excluir comandos directos, confirmaciones o elecciones
    if p_lower.startswith("/") or p_lower in ["si", "no", "1", "2", "3", "4", "cancelar", "ok", "vale"]:
        return False

    # Excluir si la pregunta contiene intenciones de agendamiento o reserva de citas dinámicas
    keywords_pregunta_no_cacheables = [
        "agendar", "agenda", "turno", "turnos", "reserva", "reservar",
        "apartar", "para mañana", "para hoy", "mañana a las", "hoy a las",
        "el lunes a las", "el martes a las", "el miercoles a las", "el jueves a las", "el viernes a las",
        "consultar", "detalles", "mis citas", "ver mis citas", "mi cita", "cédula", "cedula",
    ]
    for kw in keywords_pregunta_no_cacheables:
        if kw in p_lower:
            return False

    # Exclusiones por palabras clave transaccionales o dinámicas en la respuesta
    palabras_no_cacheables = [
        "cita agendada",
        "confirmada para el",
        "tu cita ha sido",
        "asesor de la clínica revisará",
        "escalada",
        "código de cita",
        "urgencia médica",
        "acude de inmediato",
        "servicio de urgencias",
        "lo siento, ocurrió un error",
        "no logré escuchar",
        "inconveniente técnico",
        "horarios disponibles",
        "deseas agendar",
        "te gustaría agendar",
        "por favor confirma",
        "detalles de tu cita",
        "[consultando",
        "consultando información",
    ]

    for palabra in palabras_no_cacheables:
        if palabra in r_lower:
            return False

    return True


async def buscar_en_cache(pregunta: str) -> Optional[str]:
    """Busca una respuesta en el sistema de caché en capas (L1 Memoria -> L2 Qdrant).
    
    1. Interceptor de Respuestas Rápidas (0 tokens, < 1ms).
    2. Caché L1 en Memoria RAM (0 tokens, < 1ms).
    3. Caché L2 Semántico Vectorial en Qdrant (< 50ms).
    """
    if not settings.semantic_cache_enabled:
        return None

    query = pregunta.strip()
    if not query:
        return None

    # 1. Interceptor de respuestas deterministas comunes (saludos, menú, horarios, contacto)
    fast_resp = _get_fast_response(query)
    if fast_resp:
        logger.info(f"[Semantic Cache] ⚡ Fast-Path HIT (0 tokens) para query: '{query[:40]}'")
        return fast_resp

    norm_key = _normalize_query_key(query)

    # 2. Caché L1 en memoria RAM
    if norm_key in _L1_MEMORY_CACHE:
        logger.info(f"[Semantic Cache] ⚡ L1 Memory Cache HIT para query: '{query[:40]}'")
        return _L1_MEMORY_CACHE[norm_key]

    if len(query) < 4:
        return None

    # 3. Caché L2 Semántico en Qdrant
    try:
        client = get_qdrant_client()
        ensure_cache_collection(client)

        embeddings = get_embeddings()
        vector_pregunta = await embeddings.aembed_query(query)

        col_name = settings.semantic_cache_collection
        search_results = client.search(
            collection_name=col_name,
            query_vector=vector_pregunta,
            limit=1,
            with_payload=True,
        )

        if search_results:
            top_match = search_results[0]
            score = float(top_match.score)

            if score >= settings.semantic_cache_threshold:
                payload = top_match.payload or {}
                respuesta = payload.get("respuesta")
                if respuesta:
                    logger.info(
                        f"[Semantic Cache] ⚡ L2 Vector Cache HIT (score: {score:.4f} >= {settings.semantic_cache_threshold}) "
                        f"para query: '{query[:50]}...' | Guardada: '{payload.get('pregunta', '')[:40]}...'"
                    )
                    # Guardar en L1 para futuros hits instantáneos
                    if len(_L1_MEMORY_CACHE) < _MAX_L1_ENTRIES:
                        _L1_MEMORY_CACHE[norm_key] = str(respuesta)
                    return str(respuesta)

            logger.debug(f"[Semantic Cache] Cache MISS (score: {score:.4f} < {settings.semantic_cache_threshold}) para query: '{query[:50]}...'")

        return None
    except Exception as exc:
        logger.warning(f"[Semantic Cache] Error al consultar caché semántico: {exc}")
        return None


async def guardar_en_cache(pregunta: str, respuesta: str, categoria: str = "general") -> bool:
    """Almacena el par (pregunta, respuesta) en el Caché L1 y L2 (Qdrant)."""
    if not settings.semantic_cache_enabled:
        return False

    if not _es_contenido_cacheable(pregunta, respuesta):
        logger.debug(f"[Semantic Cache] Contenido no cacheable omitido: '{pregunta[:40]}...'")
        return False

    norm_key = _normalize_query_key(pregunta)
    if len(_L1_MEMORY_CACHE) < _MAX_L1_ENTRIES:
        _L1_MEMORY_CACHE[norm_key] = respuesta.strip()

    try:
        client = get_qdrant_client()
        ensure_cache_collection(client)

        embeddings = get_embeddings()
        vector_pregunta = await embeddings.aembed_query(pregunta.strip())

        point_id = str(uuid.uuid4())
        col_name = settings.semantic_cache_collection

        client.upsert(
            collection_name=col_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector_pregunta,
                    payload={
                        "pregunta": pregunta.strip(),
                        "respuesta": respuesta.strip(),
                        "categoria": categoria,
                        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    },
                )
            ],
        )

        logger.info(f"[Semantic Cache] 💾 Guardada nueva entrada en caché L2 para: '{pregunta[:50]}...'")
        return True
    except Exception as exc:
        logger.warning(f"[Semantic Cache] Error al guardar en caché semántico L2: {exc}")
        return False
