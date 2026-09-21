"""Módulo de Dominio para Sanitización y Formateo Nativo de WhatsApp.

Convierte texto generado en formato Markdown estándar o con imperfecciones
a sintaxis nativa de WhatsApp:
- Convierte negrita estándar Markdown `**texto**` a negrita nativa `*texto*`.
- Corrige asteriscos dobles huérfanos o pegados a palabras (`palabra**`, `** palabra`).
- Convierte encabezados Markdown (`# `, `## `, `### `) a `*Título*`.
- Convierte enlaces Markdown `[Texto](url)` a `Texto (url)`.
- Convierte listas Markdown (`- ` o `* ` al inicio de línea) a viñetas nativas `• `.
- Limpia saltos de línea excesivos y espacios huérfanos.

Módulo puro de dominio: 0 dependencias externas, determinista y testeable.
"""

import re


def sanitizar_negritas_whatsapp(texto: str) -> str:
    """Convierte negritas de Markdown (**texto**) a negrita simple de WhatsApp (*texto*)
    y limpia asteriscos dobles residuales.
    """
    if not texto:
        return ""

    # 1. Corregir asteriscos triples accidentales: ***texto*** -> *texto*
    texto = re.sub(r"\*{3,}([^\*\n]+?)\*{3,}", r"*\1*", texto)

    # 2. Convertir negritas estándar o con espacios internos: **contenido** -> *contenido*
    def _reemplazar_negrita(match):
        contenido = match.group(1).strip()
        if not contenido:
            return ""
        return f"*{contenido}*"

    texto = re.sub(r"\*\*([^\*\n]+?)\*\*", _reemplazar_negrita, texto)

    # 3. Limpiar cualquier doble asterisco residual (ej. "palabra**dentro" o "**opción")
    texto = re.sub(r"\*\*", "*", texto)

    return texto


def sanitizar_listas_y_encabezados(texto: str) -> str:
    """Convierte encabezados y viñetas de Markdown a formato WhatsApp limpio."""
    if not texto:
        return ""

    lineas = texto.split("\n")
    lineas_procesadas = []

    for linea in lineas:
        l_strip = linea.strip()

        # Encabezados Markdown: ### Titulo -> *Titulo*
        if re.match(r"^#{1,6}\s+", l_strip):
            titulo = re.sub(r"^#{1,6}\s+", "", l_strip).strip()
            # Si el título ya tiene asteriscos, preservarlos limpios
            titulo_limpio = titulo.strip("*")
            lineas_procesadas.append(f"*{titulo_limpio}*")
            continue

        # Viñetas con guión o asterisco al inicio: "- item" o "* item" -> "• item"
        # Previene colisión entre "* item" y negrita "*palabra*"
        if re.match(r"^[ \t]*[-*]\s+", linea):
            contenido = re.sub(r"^[ \t]*[-*]\s+", "", linea).strip()
            lineas_procesadas.append(f"• {contenido}")
            continue

        lineas_procesadas.append(linea)

    return "\n".join(lineas_procesadas)


def sanitizar_enlaces_whatsapp(texto: str) -> str:
    """Convierte enlaces Markdown [Texto](url) a formato legible en WhatsApp: Texto (url)."""
    if not texto:
        return ""
    return re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r"\1 (\2)", texto)


def formatear_para_whatsapp(texto: str) -> str:
    """Función principal de formateo nativo para mensajes salientes a WhatsApp.

    Garantiza una presentación visual impecable, libre de caracteres markdown
    incompatibles y asteriscos dobles dentro de palabras.
    """
    if not texto or not isinstance(texto, str):
        return ""

    # Paso 1: Sanitizar enlaces
    res = sanitizar_enlaces_whatsapp(texto)

    # Paso 2: Sanitizar listas y encabezados
    res = sanitizar_listas_y_encabezados(res)

    # Paso 3: Sanitizar negritas y eliminar **
    res = sanitizar_negritas_whatsapp(res)

    # Paso 4: Normalizar saltos de línea excesivos (máximo 2 saltos consecutivos)
    res = re.sub(r"\n{3,}", "\n\n", res)

    return res.strip()
