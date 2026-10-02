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


def sanitizar_caracteres_escapados(texto: str) -> str:
    """Limpia comillas escapadas accidentales, saltos de línea literales y comillas envolventes."""
    if not texto:
        return ""
    res = texto.strip()
    # Eliminar comillas envolventes si todo el texto viene entre comillas
    if (res.startswith('"') and res.endswith('"') and len(res) > 2) or (
        res.startswith("'") and res.endswith("'") and len(res) > 2
    ):
        res = res[1:-1].strip()

    # Reemplazar comillas y saltos de línea escapados
    res = res.replace('\\"', '"').replace("\\'", "'").replace("\\n", "\n")
    # Limpiar comillas al inicio de párrafos resultantes de serialización de JSON
    res = re.sub(r'^"(.*?)"$', r"\1", res, flags=re.MULTILINE)
    # Eliminar bloques de código markdown ``` o ```markdown
    res = re.sub(r"```[a-zA-Z]*\n?", "", res)
    return res


def sanitizar_negritas_whatsapp(texto: str) -> str:
    """Convierte negritas de Markdown (**texto**) a negrita simple de WhatsApp (*texto*),
    elimina espacios internos en los asteriscos y limpia asteriscos residuales.
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

    # 4. WhatsApp exige que el texto en negrita NO tenga espacios pegados al asterisco (* palabra * no funciona)
    # Corregir '* palabra *' -> ' *palabra* '
    def _ajustar_espacios_negrita(match):
        prefix_space = match.group(1) or ""
        inner = match.group(2).strip()
        suffix_space = match.group(3) or ""
        if not inner:
            return ""
        return f"{prefix_space}*{inner}*{suffix_space}"

    texto = re.sub(r"(\s|^)\*\s*([^\*\n]+?)\s*\*(\s|$|[.,;:!?])", _ajustar_espacios_negrita, texto)

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
            titulo_limpio = titulo.strip("*").strip()
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
    incompatibles, comillas o saltos escapados, y asteriscos mal formados.
    """
    if not texto or not isinstance(texto, str):
        return ""

    # Paso 0: Limpieza de caracteres escapados y comillas accidentales
    res = sanitizar_caracteres_escapados(texto)

    # Paso 1: Sanitizar enlaces
    res = sanitizar_enlaces_whatsapp(res)

    # Paso 2: Sanitizar listas y encabezados
    res = sanitizar_listas_y_encabezados(res)

    # Paso 3: Sanitizar negritas y eliminar espacios internos en *
    res = sanitizar_negritas_whatsapp(res)

    # Paso 4: Normalizar saltos de línea excesivos (máximo 2 saltos consecutivos)
    res = re.sub(r"\n{3,}", "\n\n", res)

    return res.strip()

