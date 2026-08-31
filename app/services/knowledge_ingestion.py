"""Módulo de ingestión y carga de conocimiento clínico en Qdrant para Nexus Odonto.

Indexa documentos estructurados sobre preparación de procedimientos, cuidados
postoperatorios, preguntas frecuentes clínicas y políticas de atención.
"""

import logging
from typing import List
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from app.agents.tools.qdrant_tool import get_qdrant_client, get_vector_store, _ensure_collection
from app.core.config import settings

logger = logging.getLogger(__name__)

# Base de conocimiento clínico oficial de Nexus Odonto
CONOCIMIENTO_CLINICO_DOCUMENTOS: List[dict] = [
    # 1. Preparaciones previas a procedimientos
    {
        "titulo": "Preparación previa para Limpieza Dental y Profilaxis",
        "categoria": "preparacion",
        "especialidad": "Odontología General",
        "contenido": (
            "Preparación para Limpieza Dental (Profilaxis):\n"
            "- No ingerir alimentos pesados 30 minutos antes de la cita.\n"
            "- Cepillarse los dientes antes de asistir a la consulta.\n"
            "- Informar al odontólogo sobre encías sangrantes, presencia de marcapasos o sensibilidad dental previa.\n"
            "- Duración estimada: 30 a 45 minutos."
        ),
    },
    {
        "titulo": "Preparación previa para Consulta y Tratamiento de Ortodoncia",
        "categoria": "preparacion",
        "especialidad": "Ortodoncia",
        "contenido": (
            "Preparación para Ortodoncia (Brackets y Alineadores Invisibles):\n"
            "- Para la primera cita de valoración no se requiere preparación especial.\n"
            "- Para la colocación de brackets: se requiere profilaxis dental previa y boca libre de caries.\n"
            "- Traer radiografía panorámica y cefalométrica si el especialista las solicitó previamente.\n"
            "- Comer ligero antes de la cita, ya que la colocación inicial toma entre 60 y 90 minutos."
        ),
    },
    {
        "titulo": "Preparación previa para Endodoncia (Tratamiento de Conductos)",
        "categoria": "preparacion",
        "especialidad": "Endodoncia",
        "contenido": (
            "Preparación para Tratamiento de Endodoncia (Conductos):\n"
            "- Comer antes de la consulta; el procedimiento se realiza bajo anestesia local.\n"
            "- No suspender medicamentos habituales (presión arterial, tiroides, etc.).\n"
            "- Evitar tomar analgésicos fuertes en las 4 horas previas si se va a realizar la prueba de diagnóstico del dolor.\n"
            "- Informar si padece alergias a anestésicos locales (lidocaína, mepivacaína) o antibióticos."
        ),
    },
    {
        "titulo": "Preparación previa para Cirugía Oral y Extracción de Cordales",
        "categoria": "preparacion",
        "especialidad": "Cirugía Oral",
        "contenido": (
            "Preparación para Cirugía Oral y Extracción de Cordales (Muelas del Juicio):\n"
            "- Asistir con un acompañante adulto responsable.\n"
            "- Comer ligero 2 horas antes del procedimiento (no asistir en ayuno si es con anestesia local).\n"
            "- No fumar ni consumir bebidas alcohólicas durante las 24 horas previas.\n"
            "- Usar ropa cómoda de manga corta o holgada.\n"
            "- Informar si toma anticoagulantes, aspirina o si tiene enfermedades sistémicas como diabetes o hipertensión."
        ),
    },
    {
        "titulo": "Preparación previa para Blanqueamiento Dental",
        "categoria": "preparacion",
        "especialidad": "Estética Dental",
        "contenido": (
            "Preparación para Blanqueamiento Dental en Consultorio:\n"
            "- Es requisito indispensable tener una profilaxis dental realizada en los últimos 30 días.\n"
            "- No tener caries activas ni enfermedad periodontal no tratada.\n"
            "- Evitar el consumo de café, té, vino tinto, chocolate o alimentos con colorantes oscuros las 24 horas antes."
        ),
    },
    {
        "titulo": "Preparación previa para Diseño de Sonrisa y Carillas",
        "categoria": "preparacion",
        "especialidad": "Estética Dental",
        "contenido": (
            "Preparación para Diseño de Sonrisa y Carillas en Resina o Cerámica:\n"
            "- Requiere valoración previa con escaneo digital y registro fotográfico.\n"
            "- La boca debe estar sana, sin caries ni inflamación gingival.\n"
            "- Acudir con tiempo disponible (las sesiones de diseño pueden durar de 2 a 3 horas)."
        ),
    },

    # 2. Cuidados posteriores a procedimientos
    {
        "titulo": "Cuidados posteriores a una Extracción Dental o Cirugía Oral",
        "categoria": "postoperatorio",
        "especialidad": "Cirugía Oral",
        "contenido": (
            "Cuidados postoperatorios tras Extracción o Cirugía Dental:\n"
            "1. Mantener la gasa mordida firmemente durante 30 a 45 minutos.\n"
            "2. Aplicar compresas frías o hielo envuelto en un paño sobre la mejilla en intervalos de 15 minutos durante las primeras 24 horas.\n"
            "3. NO escupir, NO enjuagarse con fuerza, NO usar pitillo/pajilla para no desalojar el coágulo.\n"
            "4. Dieta blanda y fría durante las primeras 48 horas (helados, gelatinas, sopas tibias, purés).\n"
            "5. Reposo relativo: no realizar esfuerzo físico, ejercicio ni exponerse al sol las primeras 72 horas.\n"
            "6. Tomar los medicamentos (analgésicos/antibióticos) exactamente según la fórmula médica."
        ),
    },
    {
        "titulo": "Cuidados y Mantenimiento con Brackets de Ortodoncia",
        "categoria": "postoperatorio",
        "especialidad": "Ortodoncia",
        "contenido": (
            "Cuidados de Ortodoncia con Brackets:\n"
            "- Cepillarse después de cada comida usando cepillo ortodóntico y cepillos interdentales.\n"
            "- Evitar alimentos duros (hielo, frutos secos, turrones, chicharrones, morder manzanas o zanahorias enteras).\n"
            "- Evitar alimentos pegajosos (chicles, caramelos masticables).\n"
            "- Si un bracket se despega o un alambre causa molestia, aplicar cera de ortodoncia y solicitar cita de urgencia ortodóntica."
        ),
    },
    {
        "titulo": "Cuidados posteriores a un Blanqueamiento Dental",
        "categoria": "postoperatorio",
        "especialidad": "Estética Dental",
        "contenido": (
            "Cuidados 'Dieta Blanca' tras Blanqueamiento Dental:\n"
            "- Durante las primeras 72 horas seguir una dieta blanca: evitar café, té, gaseosas oscuras, vino tinto, salsa de tomate, salsa de soya, curry y remolacha.\n"
            "- No fumar durante al menos 7 días post-tratamiento.\n"
            "- Es normal sentir una leve sensibilidad dental transitoria las primeras 24-48 horas. Se puede usar crema dental desensibilizante."
        ),
    },
    {
        "titulo": "Cuidados posteriores a Tratamiento de Conducto (Endodoncia)",
        "categoria": "postoperatorio",
        "especialidad": "Endodoncia",
        "contenido": (
            "Cuidados tras Tratamiento de Endodoncia:\n"
            "- Es normal sentir molestia o dolor leve al masticar durante 2 a 4 días; se controla con los analgésicos recetados.\n"
            "- Evitar masticar alimentos duros del lado tratado hasta que se coloque la restauración definitiva (corona o incrustación).\n"
            "- Si el empaste temporal se cae o presenta hinchazón visible, comunicarse de inmediato con la clínica."
        ),
    },

    # 3. Preguntas Frecuentes Clínicas (FAQ Clínica)
    {
        "titulo": "Frecuencia recomendada de Limpieza Dental y Chequeo",
        "categoria": "faq_clinica",
        "especialidad": "Odontología General",
        "contenido": (
            "¿Cada cuánto tiempo se debe realizar una limpieza dental profesional?\n"
            "- Se recomienda realizarse una limpieza dental (profilaxis y destartraje) cada 6 meses.\n"
            "- En pacientes con enfermedad periodontal, antecedentes de cálculo severo o brackets de ortodoncia, se recomienda cada 3 a 4 meses.\n"
            "- La limpieza previene caries interdentales, gingivitis y pérdida ósea."
        ),
    },
    {
        "titulo": "Manejo de la Sensibilidad Dental al frío o caliente",
        "categoria": "faq_clinica",
        "especialidad": "Odontología General",
        "contenido": (
            "¿Por qué se produce la sensibilidad dental y cómo tratarla?\n"
            "- La sensibilidad ocurre por desgaste del esmalte, retracción de encías (exposición de dentina) o caries incipientes.\n"
            "- Recomendaciones: usar cepillo de cerdas suaves, crema dental para dientes sensibles y no cepillarse con fuerza excesiva.\n"
            "- Si el dolor persiste por más de 30 segundos tras retirar el estímulo frío/caliente, se requiere valoración clínica para descartar pulpitis."
        ),
    },
    {
        "titulo": "Síntomas y Tratamiento de la Periodoncia (Enfermedad de las Encías)",
        "categoria": "faq_clinica",
        "especialidad": "Periodoncia",
        "contenido": (
            "Periodoncia y salud de las encías:\n"
            "- Síntomas de alerta: sangrado de encías al cepillarse, encías rojas o inflamadas, mal aliento persistente, movilidad dental.\n"
            "- Las encías sanas NUNCA sangran. El sangrado es signo de gingivitis o periodontitis.\n"
            "- El tratamiento periodóntico incluye raspado y alisado radicular profundo (curetajes) para eliminar el sarro subgingival."
        ),
    },
    {
        "titulo": "Odontopediatría y primera visita al dentista",
        "categoria": "faq_clinica",
        "especialidad": "Odontopediatría",
        "contenido": (
            "Atención odontológica para niños (Odontopediatría):\n"
            "- La primera visita al odontopediatra se recomienda desde la erupción del primer diente de leche (alrededor de los 6 a 12 meses) o al cumplir 1 año.\n"
            "- Se realizan tratamientos preventivos como aplicación de flúor en barniz, sellantes de fosas y fisuras y educación en cepillado."
        ),
    },

    # 4. Información Institucional y Políticas de Nexus Odonto
    {
        "titulo": "Horarios de Atención, Ubicación y Canales de Contacto Nexus Odonto",
        "categoria": "institucional",
        "especialidad": "Información General",
        "contenido": (
            "Información General de Nexus Odonto:\n"
            "- Nombre: Consultorio Odontológico Nexus Odonto.\n"
            "- Horario de atención: Lunes a Viernes de 8:00 AM a 6:00 PM. Sábados de 8:00 AM a 1:00 PM. Domingos y festivos cerrado.\n"
            "- Canales de atención: WhatsApp y llamadas al +57 324 6030217.\n"
            "- Formas de pago: Efectivo, tarjetas de débito/crédito y transferencias bancarias."
        ),
    },
    {
        "titulo": "Políticas de Cancelación y Reprogramación de Citas Nexus Odonto",
        "categoria": "politicas",
        "especialidad": "Información General",
        "contenido": (
            "Políticas de Citas Nexus Odonto:\n"
            "- Cancelaciones o reprogramaciones: Solicitar con un mínimo de 24 horas de anticipación a través del chat de WhatsApp o línea telefónica.\n"
            "- Puntualidad: Se solicita llegar 10 minutos antes de la hora programada para diligenciar la ficha clínica.\n"
            "- Recordatorios: El sistema envía recordatorios automáticos vía WhatsApp el día anterior a su cita."
        ),
    },
]


def poblar_conocimiento_clinico(forzar_reindexacion: bool = False) -> int:
    """
    Ingesta los documentos de conocimiento clínico en la colección Qdrant.
    Si forzar_reindexacion es False, solo indexa si la colección está vacía.
    Retorna la cantidad de documentos indexados.
    """
    client = get_qdrant_client()
    _ensure_collection(client)

    try:
        count_info = client.count(collection_name=settings.qdrant_collection_name)
        puntos_actuales = count_info.count
        logger.info(f"[Qdrant Ingest] Puntos actuales en '{settings.qdrant_collection_name}': {puntos_actuales}")

        if puntos_actuales > 0 and not forzar_reindexacion:
            logger.info(f"[Qdrant Ingest] La colección ya contiene {puntos_actuales} documentos. No se requiere reindexar.")
            return puntos_actuales

        logger.info(f"[Qdrant Ingest] Indexando {len(CONOCIMIENTO_CLINICO_DOCUMENTOS)} documentos en Qdrant...")
        docs: List[Document] = []
        for item in CONOCIMIENTO_CLINICO_DOCUMENTOS:
            doc = Document(
                page_content=item["contenido"],
                metadata={
                    "titulo": item["titulo"],
                    "categoria": item["categoria"],
                    "especialidad": item["especialidad"],
                },
            )
            docs.append(doc)

        vector_store = get_vector_store()
        vector_store.add_documents(docs)

        count_after = client.count(collection_name=settings.qdrant_collection_name).count
        logger.info(f"[Qdrant Ingest] Ingestión completada exitosamente. Total de puntos: {count_after}")
        return count_after
    except Exception as exc:
        logger.error(f"[Qdrant Ingest] Error durante la ingestión de conocimiento clínico: {exc}", exc_info=True)
        return 0
