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
from app.core.llm_concurrency import with_llm_slot

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
        "Calle 100 # 15-20, Centro Médico Odontológico 🏥\n\n"
        "🚗 Contamos con excelente ubicación y fácil acceso.\n\n"
        "¿Deseas conocer más sobre nuestros servicios o agendar una cita con nuestros especialistas? ✨"
    ),
    "contacto": (
        "📞 *Canales de Contacto - Nexus Odonto:*\n\n"
        "• 📱 *WhatsApp / Teléfono:* +57 324 6030217\n"
        "• 📍 *Dirección:* Calle 100 # 15-20, Centro Médico Odontológico\n"
        "• ⏰ *Horario:* Lunes a Sábado de 8:00 AM a 6:00 PM\n"
        "• 📧 *Correo:* soporte@nexusodonto.com\n\n"
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


def _should_skip_l2_embed(text: str) -> bool:
    """Skip L2 Gemini embed for transactional text (no reply invented — just miss faster)."""
    norm = _normalize_query_key(text)
    if not norm:
        return False
    digits = re.sub(r"\D", "", text)
    if 6 <= len(digits) <= 12 and re.fullmatch(r"[\d\s.\-]+", text.strip()):
        return True
    agenda_hints = (
        "agendar",
        "cita",
        "reservar",
        "disponibilidad",
        "cancelar cita",
        "modificar cita",
        "reprogramar",
    )
    return any(h in norm for h in agenda_hints)


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


def purgar_cache_semantico(client: Optional[QdrantClient] = None) -> bool:
    """Elimina todas las entradas del caché en RAM y purga la colección en Qdrant.
    Garantiza que no queden rastros de respuestas transaccionales o datos de pacientes.
    """
    global _L1_MEMORY_CACHE
    _L1_MEMORY_CACHE.clear()
    logger.info("[Semantic Cache] Caché L1 en memoria limpiado.")

    try:
        if client is None:
            client = get_qdrant_client()
        col_name = settings.semantic_cache_collection
        if client.collection_exists(col_name):
            client.delete_collection(col_name)
            logger.info(f"[Semantic Cache] Colección '{col_name}' eliminada de Qdrant.")

        client.create_collection(
            collection_name=col_name,
            vectors_config=models.VectorParams(
                size=settings.embedding_dimension,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info(f"[Semantic Cache] Colección '{col_name}' recreada vacía en Qdrant exitosamente.")
        return True
    except Exception as exc:
        logger.error(f"[Semantic Cache] Error purgando colección en Qdrant: {exc}", exc_info=True)
        return False


def _es_contenido_cacheable(pregunta: str, respuesta: str, categoria: str = "general") -> bool:
    """Valida si un par pregunta-respuesta es apto para almacenarse en el caché semántico.
    
    POLÍTICA ESTRICTA DE SEGURIDAD Y PRIVACIDAD DE DATOS (HABEAS DATA / LEY 1581):
    1. Las respuestas del chat con pacientes NUNCA se almacenan en el caché semántico global.
       Solo se permite si la categoría es explícitamente 'faq_estatica'.
    2. Se descarta absolutamente cualquier texto que contenga cédulas, teléfonos, UUIDs,
       nombres propios, estados de citas o términos dinámicos de agenda.
    """
    # Solo respuestas institucionales / FAQ curadas explícitamente pueden cachearse
    if categoria != "faq_estatica":
        return False

    p_lower = pregunta.lower().strip()
    r_lower = respuesta.lower().strip()

    # Descartar preguntas vacías o demasiado cortas
    if len(p_lower) < 4 or len(r_lower) < 15:
        return False

    # 1. Protección contra filtración de cédulas o números de identificación (7 a 12 dígitos)
    if re.search(r"\b\d{7,12}\b", respuesta) or re.search(r"\b\d{7,12}\b", pregunta):
        logger.warning("[Semantic Cache Guard] Rechazado: contiene números tipo documento/cédula.")
        return False

    # 2. Protección contra filtración de IDs de citas (UUIDs)
    if re.search(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}", respuesta):
        logger.warning("[Semantic Cache Guard] Rechazado: contiene UUID de cita.")
        return False

    # 3. Palabras transaccionales o dinámicas que jamás deben ser compartidas entre usuarios
    PALABRAS_PROHIBIDAS_RESPUESTA = [
        "cita", "citas", "agendad", "cancelad", "programad", "reprogramad",
        "turno", "turnos", "cédula", "cedula", "identificación", "paciente",
        "doctor", "doctora", "dr.", "dra.", "hoy", "mañana", "ayer",
        "septiembre", "octubre", "noviembre", "diciembre", "enero", "febrero",
        "marzo", "abril", "mayo", "junio", "julio", "agosto",
        "asesor de la clínica", "escalada", "urgencia médica", "consultando información",
    ]
    for palabra in PALABRAS_PROHIBIDAS_RESPUESTA:
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

    # Miss-only: skip Gemini embed for pure digits / agenda keywords (no invented reply)
    if _should_skip_l2_embed(query):
        logger.info(f"[Semantic Cache] Skip L2 embed (agenda/cedula) query='{query[:40]}'")
        return None

    # 3. Caché L2 Semántico en Qdrant
    try:
        client = get_qdrant_client()
        ensure_cache_collection(client)

        embeddings = get_embeddings()
        vector_pregunta = await with_llm_slot(
            embeddings.aembed_query(query),
            label="embed_cache_lookup",
        )

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

    if not _es_contenido_cacheable(pregunta, respuesta, categoria=categoria):
        logger.debug(f"[Semantic Cache] Contenido no cacheable omitido: '{pregunta[:40]}...'")
        return False

    norm_key = _normalize_query_key(pregunta)
    if len(_L1_MEMORY_CACHE) < _MAX_L1_ENTRIES:
        _L1_MEMORY_CACHE[norm_key] = respuesta.strip()

    try:
        client = get_qdrant_client()
        ensure_cache_collection(client)

        embeddings = get_embeddings()
        vector_pregunta = await with_llm_slot(
            embeddings.aembed_query(pregunta.strip()),
            label="embed_cache_store",
        )

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
