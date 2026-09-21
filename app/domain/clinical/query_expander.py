"""Expansor, normalizador y clasificador semántico de consultas clínicas para WhatsApp.
Convierte expresiones coloquiales y síntomas descritos por pacientes en términos clínicos
y extrae metadatos para optimizar la recuperación híbrida en Qdrant y BM25.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Any

# Diccionario enriquecido de equivalencias semánticas para odontología
EXPANSIONES_CLINICAS: Dict[str, List[str]] = {
    "muela": ["diente", "molar", "premolar", "cordal", "odontologia"],
    "calza": ["resina", "obturacion", "empaste", "composite", "restauracion"],
    "chucha": ["sarro", "calculo dental", "placa bacteriana", "profilaxis"],
    "sarro": ["calculo dental", "tartaro", "profilaxis", "destartraje"],
    "frenillos": ["brackets", "ortodoncia", "alineadores", "aparatos"],
    "alambre": ["arco de ortodoncia", "bracket", "cera ortodontica"],
    "postizo": ["protesis dental", "dentadura", "implante", "corona"],
    "corona": ["rehabilitacion oral", "protesis fija", "ceramica", "perno"],
    "perno": ["espigo", "poste", "endodoncia", "reconstruccion dental"],
    "sangra": ["sangrado gingival", "gingivitis", "periodoncia", "encias"],
    "sangrado": ["gingivitis", "periodontitis", "hemorragia", "encias"],
    "hinchada": ["inflamacion", "edema", "flemon", "absceso"],
    "flemon": ["absceso dental", "infeccion", "celulitis facial", "edema"],
    "pelota": ["absceso", "flemon", "fistula", "inflamacion gingival"],
    "bolita": ["absceso", "fistula dental", "infeccion", "gingiva"],
    "frio": ["sensibilidad dental", "pulpitis", "esmalte", "dentina"],
    "caliente": ["sensibilidad pulpar", "endodoncia", "conductos"],
    "destemplado": ["sensibilidad dental", "hiperestesia", "esmalte"],
    "destemple": ["sensibilidad dental", "recesion gingival", "cuello dental"],
    "apretar": ["bruxismo", "desgaste dental", "placa miorrelajante", "mandibula"],
    "crujir": ["bruxismo", "articulacion temporomandibular", "atm"],
    "mal aliento": ["halitosis", "higiene bucal", "calculo dental", "periodoncia"],
    "pitillo": ["pajilla", "coagulo", "alveolitis", "cuidados postoperatorios"],
    "dormida": ["anestesia dental", "adormecimiento", "parestesia"],
    "blanquear": ["blanqueamiento dental", "aclaramiento", "dieta blanca", "estetica"],
    "amarillos": ["manchas dentales", "blanqueamiento", "aclaramiento dental"],
    "nino": ["odontopediatria", "dientes de leche", "fluor", "sellantes"],
    "bebe": ["odontopediatria", "erupcion dental", "primera visita"],
    "punzada": ["dolor pulpar", "pulpitis aguda", "dolor dental agudo"],
    "latido": ["dolor pulsatil", "pulpitis aguda", "absceso periapical"],
    "aguantar": ["dolor agudo", "sobrecupo", "urgencia odontologica"],
    "cordal": ["tercer molar", "muela del juicio", "cirugia oral", "extraccion"],
    "muela del juicio": ["cordal", "tercer molar", "extraccion", "cirugia"],
}

# Palabras clave asociadas a categorías clínicas
PATRONES_CATEGORIAS: Dict[str, List[str]] = {
    "urgencias_y_dolor": [
        "urgencia", "emergencia", "dolor", "duele", "aguantar", "insoportable",
        "punzada", "latido", "hinchad", "flemon", "absceso", "fistula",
        "golpe", "trauma", "fractur", "sobrecupo", "no hay cita", "alivio",
    ],
    "postoperatorio": [
        "despues de", "sacaron", "extraccion", "postoperatorio", "fumar",
        "pitillo", "pajilla", "coagulo", "hielo", "salitre", "enjuague",
        "hinchazon despues", "sangrado despues", "gasa", "puntos de sutura",
    ],
    "preparacion": [
        "antes de", "preparacion", "ayunas", "como debo ir", "recomendaciones antes",
        "previa", "medicamentos antes", "sedacion",
    ],
    "politicas": [
        "cancelar", "reprogramar", "horario", "direccion", "ubicacion",
        "donde quedan", "sede", "llegar", "telefono", "contacto", "receso",
    ],
    "estetica": [
        "blanqueamiento", "aclaramiento", "carillas", "diseno de sonrisa",
        "estetica dental", "manchas", "dientes blancos",
    ],
}

# Palabras clave asociadas a especialidades odontológicas
PATRONES_ESPECIALIDADES: Dict[str, str] = {
    "ortodoncia": "Ortodoncia",
    "brackets": "Ortodoncia",
    "frenillos": "Ortodoncia",
    "alineadores": "Ortodoncia",
    "endodoncia": "Endodoncia",
    "conductos": "Endodoncia",
    "nervio": "Endodoncia",
    "periodoncia": "Periodoncia",
    "encias": "Periodoncia",
    "sangrado gingival": "Periodoncia",
    "sarro": "Periodoncia",
    "calculo dental": "Periodoncia",
    "odontopediatria": "Odontopediatría",
    "nino": "Odontopediatría",
    "ninos": "Odontopediatría",
    "bebe": "Odontopediatría",
    "dientes de leche": "Odontopediatría",
    "cirugia": "Cirugía Oral y Maxilofacial",
    "cordal": "Cirugía Oral y Maxilofacial",
    "extraccion": "Cirugía Oral y Maxilofacial",
    "rehabilitacion": "Rehabilitación Oral",
    "protesis": "Rehabilitación Oral",
    "implante": "Implantología",
    "blanqueamiento": "Estética Dental",
}


def normalizar_texto(texto: str) -> str:
    """Elimina tildes y diacríticos, convirtiendo a minúsculas."""
    if not texto:
        return ""
    texto = texto.lower().strip()
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def expandir_consulta_clinica(query: str) -> str:
    """Enriquece la consulta del paciente agregando términos clínicos afines si detecta palabras clave.
    Ejemplo:
        'tengo la encia hinchada y me sangra' ->
        'tengo la encia hinchada y me sangra gingivitis periodoncia inflamacion'
    """
    if not query:
        return ""

    query_norm = normalizar_texto(query)
    palabras = set(re.findall(r"\b[a-z0-9]+\b", query_norm))
    terminos_agregados: List[str] = []

    for clave, sinonimos in EXPANSIONES_CLINICAS.items():
        if clave in query_norm or any(p in palabras for p in clave.split()):
            agregados_para_clave = 0
            for s in sinonimos:
                if s not in query_norm and s not in terminos_agregados:
                    terminos_agregados.append(s)
                    agregados_para_clave += 1
                    if agregados_para_clave >= 2:
                        break

    if terminos_agregados:
        enriquecimiento = " ".join(terminos_agregados[:8])
        return f"{query} {enriquecimiento}".strip()

    return query


def extraer_filtros_metadatos(query: str) -> Dict[str, Optional[str]]:
    """Infiere la categoría y especialidad odontológica a partir de la consulta del paciente.
    Permite aplicar Self-Querying / filtrado inteligente en Qdrant y BM25.
    
    Retorna:
        {"categoria": Optional[str], "especialidad": Optional[str]}
    """
    if not query:
        return {"categoria": None, "especialidad": None}

    query_norm = normalizar_texto(query)
    cat_detectada: Optional[str] = None
    esp_detectada: Optional[str] = None

    # 1. Detección de categoría
    for cat, pistas in PATRONES_CATEGORIAS.items():
        if any(pista in query_norm for pista in pistas):
            cat_detectada = cat
            break

    # 2. Detección de especialidad
    for pista, esp in PATRONES_ESPECIALIDADES.items():
        if pista in query_norm:
            esp_detectada = esp
            break

    return {"categoria": cat_detectada, "especialidad": esp_detectada}
