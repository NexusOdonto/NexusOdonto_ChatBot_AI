"""Variantes naturales para las respuestas fijas de recepción (sin LLM).

Cada respuesta canned tiene varias redacciones con el mismo significado; `pick`
elige una al azar evitando repetir las usadas recientemente con el mismo teléfono.

Reglas de redacción: español de Colombia, tono de recepción humana, 0-1 emoji,
nunca revelar que es un bot, sin formas "o/a".
Los pools de agenda deben conservar "número de cédula" y "nombre completo":
message_processor / booking_flow detectan el paso de identidad por esos textos.
"""

from __future__ import annotations

import random
import threading
from collections import OrderedDict, deque
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

_MAX_PHONES = 5000
_HISTORY: "OrderedDict[str, dict[str, deque[int]]]" = OrderedDict()
_LOCK = threading.Lock()
_GLOBAL_KEY = "_global"


def _now_bogota() -> datetime:
    try:
        from app.core.config import settings

        tz = ZoneInfo(settings.reminder_timezone or "America/Bogota")
    except Exception:
        try:
            tz = ZoneInfo("America/Bogota")
        except Exception:
            tz = timezone(timedelta(hours=-5))
    return datetime.now(tz)


def momento_del_dia(now: Optional[datetime] = None) -> str:
    hour = (now or _now_bogota()).hour
    if 5 <= hour < 12:
        return "manana"
    if 12 <= hour < 19:
        return "tarde"
    return "noche"


def saludo_del_momento(now: Optional[datetime] = None) -> str:
    return {"manana": "Buenos días", "tarde": "Buenas tardes", "noche": "Buenas noches"}[
        momento_del_dia(now)
    ]


def despedida_del_momento(now: Optional[datetime] = None) -> str:
    return {"manana": "un buen día", "tarde": "una buena tarde", "noche": "una buena noche"}[
        momento_del_dia(now)
    ]


def primer_nombre(nombre: Optional[str]) -> str:
    parts = (nombre or "").strip().split()
    return parts[0].title() if parts else ""


def _recent_indexes(phone: str, key: str, window: int) -> deque:
    with _LOCK:
        per_phone = _HISTORY.get(phone)
        if per_phone is None:
            per_phone = {}
            _HISTORY[phone] = per_phone
            while len(_HISTORY) > _MAX_PHONES:
                _HISTORY.popitem(last=False)
        else:
            _HISTORY.move_to_end(phone)
        recent = per_phone.get(key)
        if recent is None or recent.maxlen != window:
            recent = deque(recent or (), maxlen=window)
            per_phone[key] = recent
        return recent


def pick(
    key: str,
    variants: Sequence[str],
    *,
    phone: Optional[str] = None,
    nombre: Optional[str] = None,
    avoid: Optional[str] = None,
    **fmt: str,
) -> str:
    """Elige una variante de `variants` sin repetir las últimas usadas para `phone`.

    Placeholders disponibles en cada variante: {nombre} (", Ana" o ""),
    {saludo} ("Buenos días"...), {saludo_min}, {despedida} ("un buen día"...) y
    cualquier kwarg extra de `fmt`. `avoid` excluye una variante literal (p. ej. el último
    mensaje del hilo tras un reinicio del proceso).
    """
    if not variants:
        return ""
    window = max(0, min(len(variants) - 1, len(variants) // 2 + 1))
    phone_key = (phone or "").strip() or _GLOBAL_KEY
    recent = _recent_indexes(phone_key, key, window) if window else deque()
    all_idx = [i for i in range(len(variants)) if variants[i] != avoid] or list(range(len(variants)))
    candidates = [i for i in all_idx if i not in recent] or all_idx
    idx = random.choice(candidates)
    if window:
        with _LOCK:
            recent.append(idx)

    first = primer_nombre(nombre)
    now = _now_bogota()
    saludo = saludo_del_momento(now)
    values = {
        "nombre": f", {first}" if first else "",
        "saludo": saludo,
        "saludo_min": saludo.lower(),
        "despedida": despedida_del_momento(now),
        **fmt,
    }
    return variants[idx].format(**values)


def reset_history(phone: Optional[str] = None) -> None:
    with _LOCK:
        if phone is None:
            _HISTORY.clear()
        else:
            _HISTORY.pop(phone, None)


# ─────────────────────────────────────────────────────────────────────────────
# Pools
# ─────────────────────────────────────────────────────────────────────────────

SALUDO = (
    "{saludo}{nombre}, gracias por escribir a *Nexus Odonto*. ¿En qué te puedo ayudar?",
    "¡Hola{nombre}! Qué gusto saludarte. Cuéntame qué necesitas: citas, precios o cualquier duda de tus dientes.",
    "Hola{nombre}, {saludo_min}. Con gusto te atiendo. ¿Quieres agendar, revisar una cita que ya tienes o tienes alguna pregunta?",
    "¡{saludo}{nombre}! Aquí en *Nexus Odonto* estamos para ayudarte 😊 ¿Qué se te ofrece?",
    "Hola{nombre}, ¿cómo estás? Dime en qué te colaboro: agendar o cambiar una cita, precios, especialistas o alguna duda dental.",
    "{saludo}{nombre}. Estás hablando con *Nexus Odonto*, cuéntame qué necesitas y lo revisamos.",
    "¡Hola{nombre}! ¿Cómo te puedo ayudar hoy? Si quieres una cita, cambiar la que tienes o saber precios, me cuentas.",
    "Hola{nombre}, qué bueno que nos escribes. ¿Buscas una cita o tienes alguna consulta?",
)

AGRADECIMIENTO = (
    "Con mucho gusto{nombre}. Cualquier cosa, aquí estamos. ¡Que te vaya muy bien!",
    "Fue un placer{nombre}. Si necesitas algo más, me escribes cuando quieras.",
    "Con gusto. Que tengas {despedida}; aquí seguimos para lo que necesites.",
    "¡Listo{nombre}! Un gusto ayudarte. Cuídate mucho.",
    "De nada{nombre}. Cuando quieras, por aquí te atendemos 😊",
)

HORARIO = (
    "Atendemos de *lunes a sábado* de 8:00 AM a 6:00 PM. Domingos y festivos estamos cerrados.\n\n"
    "¿Quieres que miremos disponibilidad para algún día?",
    "Nuestro horario es de *lunes a sábado*, 8:00 AM a 6:00 PM; domingos y festivos no abrimos.\n\n"
    "¿Te busco un espacio para algún día en especial?",
    "Estamos abiertos de *lunes a sábado* entre 8:00 AM y 6:00 PM. Los domingos y festivos descansamos.\n\n"
    "Si quieres, te reviso qué horarios hay libres.",
    "De *lunes a sábado*, de 8:00 AM a 6:00 PM (domingos y festivos cerrado).\n\n"
    "¿Para qué día te gustaría venir?",
)

UBICACION = (
    "Estamos en *Calle 100 # 15-20*, Centro Médico Odontológico.\n\n"
    "¿Te ayudo a agendar o quieres saber algo más de la clínica?",
    "Nos encuentras en la *Calle 100 # 15-20*, en el Centro Médico Odontológico.\n\n"
    "¿Quieres que te separe una cita?",
    "La clínica queda en *Calle 100 # 15-20* (Centro Médico Odontológico).\n\n"
    "¿Algo más en lo que te pueda ayudar?",
    "Quedamos en la *Calle 100 # 15-20*, dentro del Centro Médico Odontológico.\n\n"
    "Si quieres venir, te miro disponibilidad.",
)

CONTACTO = (
    "Puedes escribirnos por aquí o llamar al *+57 324 6030217*.\n"
    "También estamos en Calle 100 # 15-20 y el correo es soporte@nexusodonto.com "
    "(lun-sáb 8:00 AM–6:00 PM).\n\n"
    "¿En qué te puedo orientar?",
    "Por este mismo chat te atendemos, y si prefieres llamar el número es *+57 324 6030217*.\n"
    "Correo: soporte@nexusodonto.com · Dirección: Calle 100 # 15-20 (lun-sáb 8:00 AM–6:00 PM).\n\n"
    "¿Te ayudo con algo más?",
    "Te dejo nuestros datos: teléfono *+57 324 6030217*, correo soporte@nexusodonto.com "
    "y estamos en la Calle 100 # 15-20, de lunes a sábado de 8:00 AM a 6:00 PM.\n\n"
    "Igual por aquí te puedo ayudar, ¿qué necesitas?",
)

AGENDAR_INICIO = (
    "Claro, te ayudo a agendar.\n\n"
    "¿Me regalas tu *número de cédula* y tu *nombre completo* (nombre y apellido)? "
    "Con eso miramos el tratamiento y los horarios.",
    "Con gusto te separo la cita. Para empezar, ¿me pasas tu *número de cédula* y tu *nombre completo*?",
    "Perfecto, miremos tu cita. Primero necesito tu *número de cédula* y tu *nombre completo* "
    "(nombre y apellido).",
    "Dale, te ayudo con la cita. Escríbeme tu *número de cédula* y tu *nombre completo*, "
    "y seguimos con el tratamiento y el horario.",
    "Listo, vamos a agendar. ¿Me compartes tu *número de cédula* y tu *nombre completo*?",
)

PEDIR_SERVICIO = (
    "Listo{nombre}, gracias.\n\n"
    "¿Qué tratamiento te gustaría agendar? Puede ser *Profilaxis*, *Resina* o *Blanqueamiento*, "
    "o si prefieres te paso la lista de *servicios y precios*.",
    "Perfecto{nombre}, ya tengo tus datos.\n\n"
    "¿Para qué tratamiento sería la cita? Si tienes dudas, te comparto los *servicios y precios*.",
    "Gracias{nombre}.\n\n"
    "Cuéntame, ¿qué tratamiento necesitas? Por ejemplo *Profilaxis* (limpieza), *Resina* o *Blanqueamiento*.",
    "Muy bien{nombre}.\n\n"
    "¿Qué tratamiento te quieres hacer? Dime cuál, o si quieres primero te muestro *servicios y precios*.",
    "Anotado{nombre}.\n\n"
    "¿Qué tratamiento buscas? Si quieres te cuento los *servicios y precios* que manejamos.",
)

# Tras recibir cédula + nombre cuando ya eligió servicio: no volver a preguntar el tratamiento.
PEDIR_FECHA = (
    "Listo{nombre}, ya tengo tus datos para *{servicio}*.\n\n"
    "¿Qué día y a qué hora te queda bien? Atendemos de lunes a viernes de 8:00 AM a 12:00 PM "
    "y de 2:00 PM a 5:00 PM, y los sábados de 8:00 AM a 12:00 PM.",
    "Perfecto{nombre}, anotado: *{servicio}*.\n\n¿Qué fecha y hora prefieres para la cita?",
    "Gracias{nombre}. ¿Para qué día agendamos *{servicio}*? Dime también si te sirve más en la mañana o en la tarde.",
    "Muy bien{nombre}. Para *{servicio}*, ¿qué día te queda cómodo y a qué hora más o menos?",
)

# ── Agenda sin LLM: fecha → horarios → confirmación ─────────────────────────
# Identity re-asks keep "número de cédula" / "nombre completo" + "cita" (booking_flow detects them).
PEDIR_NOMBRE_CITA = (
    "Gracias. ¿Y tu *nombre completo* para la cita?",
    "Listo, ya tengo la cédula. ¿Me regalas tu *nombre completo* para la cita?",
    "Perfecto. Para dejar la cita a tu nombre, ¿cuál es tu *nombre completo*?",
)

PEDIR_CEDULA_CITA = (
    "Gracias{nombre}. ¿Me pasas tu *número de cédula* para la cita?",
    "Listo{nombre}. Me falta tu *número de cédula* para apartar la cita.",
    "Perfecto{nombre}, ¿y tu *número de cédula*? Con eso te separo la cita.",
)

REPEDIR_IDENTIDAD = (
    "Para seguir con la cita de *{servicio}* necesito tu *número de cédula* y tu *nombre completo*.",
    "¿Me compartes tu *número de cédula* y tu *nombre completo*? Así te aparto la cita de *{servicio}*.",
    "Me faltan tu *número de cédula* y tu *nombre completo* para la cita de *{servicio}*.",
)

REPEDIR_FECHA = (
    "Perdona{nombre}, no te entendí bien el día. ¿Para cuándo te agendo *{servicio}*? "
    "Por ejemplo «mañana en la tarde» o «el jueves a las 9».",
    "¿Qué día te sirve para *{servicio}*{nombre}? Me puedes decir algo como «el viernes en la mañana».",
    "Cuéntame qué día y más o menos a qué hora te queda bien para *{servicio}*, y te busco espacio.",
    "Para buscarte espacio en *{servicio}*, dime un día (hoy, mañana, el sábado...) y si prefieres mañana o tarde.",
)

REPEDIR_IDENTIDAD_GENERAL = (
    "Para apartarte la cita me faltan tu *número de cédula* y tu *nombre completo*.",
    "¿Me regalas tu *número de cédula* y tu *nombre completo*? Con eso seguimos con la cita.",
)

REPEDIR_FECHA_GENERAL = (
    "¿Qué día y a qué hora te queda bien para la cita? Por ejemplo «mañana en la tarde» o «el jueves a las 9».",
    "Cuéntame qué día te sirve y si prefieres mañana o tarde, y te busco espacio.",
)

MOTIVO_DOMINGO = (
    "Los domingos no abrimos.",
    "El domingo la clínica está cerrada.",
    "Ese día no atendemos, los domingos cerramos.",
)

MOTIVO_FECHA_PASADA = (
    "Esa fecha ya pasó.",
    "Ese día ya quedó atrás.",
)

MOTIVO_HORA_PASADA = (
    "Las {hora} de hoy ya pasaron.",
    "Para hoy a las {hora} ya no alcanzamos.",
)

MOTIVO_ALMUERZO = (
    "A las {hora} estamos en el horario de almuerzo (de 12 a 2).",
    "De 12:00 a 2:00 PM no hay citas porque es la hora de almuerzo.",
)

MOTIVO_FUERA_JORNADA = (
    "A las {hora} no estamos atendiendo; la jornada es de 8:00 AM a 12:00 PM y de 2:00 PM a 5:00 PM.",
    "Las {hora} quedan por fuera del horario (8 a 12 y 2 a 5).",
)

MOTIVO_SABADO_TARDE = (
    "Los sábados solo atendemos en la mañana, hasta las 12.",
    "El sábado la jornada es solo de 8:00 AM a 12:00 PM.",
)

MOTIVO_NO_CABE = (
    "A las {hora} no alcanza a quedar completo el tratamiento antes del cierre.",
    "Empezando a las {hora} el tratamiento no alcanza a terminar dentro de la jornada.",
)

MOTIVO_OCUPADO = (
    "{dia_cap} a las {hora} ya está ocupado.",
    "Ese espacio de las {hora} ya lo tomaron.",
    "A las {hora} ya no me queda libre.",
)

MOTIVO_SIN_CUPO_DIA = (
    "Para {dia} ya no me quedan espacios.",
    "{dia_cap} ya está lleno.",
    "{dia_cap} no tengo cupo.",
)

MOTIVO_SIN_CUPO_PERIODO = (
    "{dia_cap} en la {periodo} ya está lleno.",
    "En la {periodo} de {dia} no me queda espacio.",
)

OPCIONES_CERCANAS = (
    "Lo más cercano que tengo para *{servicio}*:",
    "Te puedo ofrecer estos espacios:",
    "Mira, estas opciones te pueden servir:",
    "Tengo disponible:",
)

OPCIONES_DIA = (
    "Para {dia} tengo estos espacios{nombre}:",
    "{dia_cap} me quedan libres:",
    "Listo, {dia} tengo:",
    "Te cuento lo que hay {dia}:",
)

ELEGIR_CIERRE = (
    "¿Cuál te queda mejor?",
    "¿Te sirve alguno?",
    "Dime cuál prefieres y te la aparto.",
    "¿Con cuál nos quedamos?",
)

ELEGIR_SLOT_REPROMPT = (
    "¿Cuál de estos te sirve{nombre}?\n{opciones}",
    "Dale, ¿con cuál te quedas?\n{opciones}",
    "Solo dime cuál de estos prefieres:\n{opciones}",
)

CONFIRMAR_CITA = (
    "Te cuento cómo quedaría{nombre}:\n\n{resumen}\n\n¿Te la dejo agendada?",
    "Perfecto. Así quedaría la cita:\n\n{resumen}\n\n¿La confirmo?",
    "Listo{nombre}, tengo este espacio:\n\n{resumen}\n\n¿Te la aparto?",
    "Me queda así:\n\n{resumen}\n\n¿Confirmamos?",
)

REPEDIR_CONFIRMACION = (
    "¿Te la dejo agendada entonces?\n\n{resumen}\n\nDime sí, o si prefieres otro día u hora.",
    "Solo me falta tu confirmación{nombre}:\n\n{resumen}\n\n¿La aparto?",
)

NO_CONFIRMA = (
    "Sin problema{nombre}. ¿Qué otro día u hora te sirve?",
    "Listo, no la agendo todavía. ¿Prefieres otro día u otra hora?",
    "Dale, la dejamos quieta. ¿Qué día te acomoda mejor?",
)

CITA_AGENDADA = (
    "¡Listo{nombre}! Tu cita quedó agendada:",
    "Hecho{nombre}, ya quedó tu cita:",
    "Perfecto{nombre}, te quedó agendada así:",
    "¡Quedó{nombre}! Estos son los datos de tu cita:",
)

CITA_DESPEDIDA = (
    "Te esperamos en la Calle 100 # 15-20. Si necesitas algo, me escribes por aquí.",
    "Nos vemos en la clínica (Calle 100 # 15-20). Cualquier cosa me cuentas.",
    "Te esperamos. Si te surge algo, por aquí mismo lo cambiamos.",
)

MISMA_PERSONA = (
    "Veo que la cédula {cedula} está registrada a nombre de *{registrado}*. ¿Eres tú? "
    "Si es así, dime «sí» y te la agendo.",
    "Esa cédula ({cedula}) la tengo a nombre de *{registrado}*. ¿Confirmas que eres tú?",
)

MISMA_PERSONA_NO = (
    "Entendido. Revisemos los datos: ¿me pasas de nuevo tu *número de cédula* y tu *nombre completo* para la cita?",
    "Listo, corrijamos. ¿Cuál es tu *número de cédula* y tu *nombre completo* para la cita?",
)

SIN_CUPOS = (
    "Por ahora no veo espacios para *{servicio}* en los próximos días. Si quieres, llámanos al "
    "*+57 324 6030217* y te buscamos un hueco.",
    "En los próximos días está todo lleno para *{servicio}*. ¿Te sirve otra fecha más adelante? "
    "También puedes llamarnos al *+57 324 6030217*.",
)

PREGUNTA_EN_CITA = (
    "Eso te lo explican con calma en la consulta.",
    "Buena pregunta; eso te lo confirman en la cita según lo que vean.",
    "Eso depende de la valoración, allá te lo explican bien.",
)

SEGUIMOS_CITA = (
    "Cuando quieras seguimos con la cita de *{servicio}*: ¿qué día te sirve?",
    "Y para la cita de *{servicio}*, ¿qué día te acomoda?",
    "Si quieres seguimos con *{servicio}*: dime el día y la hora que prefieras.",
)

# Cola de una frase ("Para buscarte el turno, {pedir}") → minúscula inicial.
PEDIR_TRATAMIENTO_CORTO = (
    "¿qué tratamiento necesitas?",
    "¿para qué tratamiento sería?",
    "cuéntame qué tratamiento te quieres hacer.",
)

PEDIR_DATOS_Y_TRATAMIENTO = (
    "¿me pasas tu *número de cédula*, tu *nombre completo* y el tratamiento que necesitas?",
    "¿me regalas tu *número de cédula*, tu *nombre completo* y qué tratamiento quieres?",
    "necesito tu *número de cédula*, tu *nombre completo* y el tratamiento que buscas.",
)

HORA_PASADA_CON_JORNADA = (
    "Las {hora} de hoy ya pasaron, pero todavía atendemos {restantes}.\n\n"
    "Para buscarte el turno más cercano, {pedir}",
    "Uy, las {hora} de hoy ya se nos pasaron. Aún tenemos atención {restantes}.\n\n"
    "Para mirarte el espacio más próximo, {pedir}",
    "Esa hora (las {hora}) ya pasó por hoy, pero seguimos atendiendo {restantes}.\n\n"
    "Si te sirve, te busco algo cercano; {pedir}",
    "Para las {hora} de hoy ya no alcanzamos, pero aún hay jornada {restantes}.\n\n"
    "Para ubicarte en el turno más cercano, {pedir}",
)

HORA_PASADA_SIN_JORNADA = (
    "Las {hora} de hoy ya pasaron y por hoy ya cerramos la jornada. "
    "Con gusto te busco turno para el próximo día hábil, {pedir}",
    "Uy, las {hora} ya pasaron y hoy ya terminamos de atender. "
    "Te puedo buscar espacio para el siguiente día hábil; {pedir}",
    "Para hoy ya no alcanzamos: las {hora} ya pasaron y la jornada terminó. "
    "Miremos el próximo día hábil, {pedir}",
)

HORA_HOY_VALIDA = (
    "Listo, miramos para hoy a las {hora}.\n\nPara apartarla, {pedir}",
    "Dale, revisemos hoy a las {hora}.\n\nPara separarla, {pedir}",
    "Perfecto, hoy a las {hora}.\n\nPara dejarla apartada, {pedir}",
)

ESCALAMIENTO = (
    "Entiendo. Un asesor de la clínica revisará tu solicitud y te contactará pronto.",
    "Listo, ya le paso tu caso a un asesor de la clínica; te escribe en un momento.",
    "Claro que sí. Un asesor de la clínica va a revisar tu solicitud y te responde pronto.",
    "Entendido, dejo tu caso con un asesor de la clínica para que te contacte en breve.",
)

MEDIOS_NO_SOPORTADOS = (
    "Por aquí no alcanzo a ver fotos, videos ni documentos.\n\n"
    "¿Me cuentas por texto o nota de voz qué necesitas (síntoma, tratamiento u orden médica)? "
    "Así te ayudo mejor.",
    "Qué pena, por este chat no puedo abrir archivos ni fotos.\n\n"
    "¿Me escribes o me mandas una nota de voz contándome qué necesitas?",
    "No me carga el archivo por aquí.\n\n"
    "Si me cuentas por texto o audio qué necesitas, te ayudo de una.",
)

NO_ENTENDI = (
    "No te entendí bien. ¿Me lo dices otra vez con palabras? Por ejemplo: agendar, precios o servicios.",
    "Perdón, no te entendí. ¿Me cuentas de nuevo qué necesitas? Puede ser una cita, precios o una duda.",
    "Uy, no logré entender el mensaje. ¿Me lo escribes otra vez? Te ayudo con citas, precios o servicios.",
    "Creo que se te fue un mensaje incompleto. ¿Qué necesitas: agendar, saber precios o algo más?",
)

REINICIO = (
    "Listo, empezamos de nuevo. Hola, estás hablando con *Nexus Odonto*. ¿En qué te ayudo?",
    "Perfecto, arrancamos desde cero. ¿Qué necesitas?",
    "Listo, conversación nueva. Cuéntame en qué te puedo ayudar.",
    "Dale, empezamos otra vez. ¿Qué se te ofrece?",
)

AUDIO_NO_ENTENDIDO = (
    "No alcancé a escuchar bien tu nota de voz.\n\n"
    "¿La grabas otra vez en un lugar más silencioso, o me lo escribes por texto?",
    "Qué pena, el audio no se escuchó bien.\n\n¿Me lo repites o me escribes?",
    "No logré entender la nota de voz.\n\n¿Me la mandas de nuevo o me cuentas por texto?",
)

AUDIO_ERROR = (
    "Se nos complicó procesar tu nota de voz.\n\n"
    "¿La intentas de nuevo o me cuentas por texto lo que necesitas?",
    "No pude abrir tu audio.\n\n¿Me lo envías otra vez o me escribes?",
    "Tu nota de voz no me cargó bien.\n\n¿La reenvías o me lo escribes por aquí?",
)

INACTIVIDAD_CITA_INCOMPLETA = (
    "Como no alcanzamos a completar los datos, dejamos el agendamiento por ahora.\n\n"
    "Cuando quieras retomarlo o consultar algo de la clínica, escríbeme y con gusto te ayudo.",
    "Dejamos la cita en pausa porque no alcanzamos a terminar.\n\n"
    "Cuando quieras la retomamos, solo escríbeme.",
    "Veo que quedó pendiente el agendamiento; lo dejamos por ahora.\n\n"
    "Si quieres terminarlo más tarde, me escribes y seguimos.",
)

INACTIVIDAD_GENERAL = (
    "Te dejo por ahora por si estás ocupado.\n\n"
    "Cuando quieras agendar, consultar un servicio o resolver una duda, aquí estoy.",
    "Parece que andas con muchas cosas, te dejo por ahora.\n\n"
    "Cuando necesites algo de la clínica, me escribes.",
    "Por ahora cierro la conversación.\n\n"
    "Si luego necesitas una cita o tienes alguna pregunta, aquí te atendemos.",
)

SPAM = (
    "Estás enviando muchos mensajes seguidos. "
    "Espera un momento y escríbeme tu consulta en un solo mensaje para atenderte bien.",
    "Me están llegando muchos mensajes muy rápido. "
    "Dame un momentico y cuéntame todo en un solo mensaje, porfa.",
    "Uy, van muchos mensajes seguidos. Mejor escríbeme lo que necesitas en uno solo y te ayudo.",
)

FUERA_DE_ALCANCE = (
    "En *Nexus Odonto* solo atendemos *salud oral* (dientes, encías y boca). "
    "Por un dolor en otra parte del cuerpo te conviene consultar un médico general "
    "o el especialista correspondiente. "
    "Si tienes una molestia dental o quieres una cita odontológica, aquí te ayudamos.",
    "Eso ya se sale de lo que manejamos: en *Nexus Odonto* solo vemos *salud oral*. "
    "Para ese dolor lo mejor es un médico general o el especialista. "
    "Si es algo de dientes, encías o boca, con gusto te ayudo.",
    "Qué pena, aquí solo atendemos *salud oral*: dientes, encías y boca. "
    "Para esa molestia te recomiendo ir al médico. "
    "Si necesitas algo odontológico, me cuentas.",
)

SOLO_ODONTOLOGIA = (
    "Solo puedo ayudarte con temas odontológicos de Nexus Odonto.",
    "Por aquí te ayudo solo con temas de la clínica: citas, tratamientos y dudas dentales.",
    "Eso no lo manejo; te puedo ayudar con citas, precios o dudas odontológicas.",
)

PORTAL_RECORDATORIO = (
    "También puedes consultar tu cita en la *plataforma virtual*:\n{url}",
    "Si quieres revisarla, está también en la *plataforma virtual*:\n{url}",
    "Recuerda que tus citas las ves en la *plataforma virtual*:\n{url}",
)

PORTAL_PRIMERA_VEZ = (
    "También puedes ver tu cita en la *plataforma virtual*:\n{url}\n"
    "Tu *usuario* es tu número de cédula y la *contraseña* también es tu número de cédula "
    "(acceso temporal). Al entrar, cámbiala por tu seguridad; desde aquí no podemos modificar contraseñas.",
    "Tu cita también queda en la *plataforma virtual*:\n{url}\n"
    "Para entrar, el *usuario* y la *contraseña* son tu número de cédula (acceso temporal). "
    "Apenas ingreses, cámbiala; desde este chat no podemos cambiar contraseñas.",
    "Puedes revisarla cuando quieras en la *plataforma virtual*:\n{url}\n"
    "Entras con tu número de cédula como *usuario* y también como *contraseña* temporal. "
    "Te recomiendo cambiarla al ingresar; por aquí no manejamos contraseñas.",
)

PORTAL_SOLO_CREDENCIALES = (
    "Tu *usuario* es tu número de cédula y la *contraseña* también es tu número de cédula "
    "(acceso temporal). Al entrar, cámbiala por tu seguridad; desde aquí no podemos modificar contraseñas.",
    "Para entrar, el *usuario* y la *contraseña* son tu número de cédula (acceso temporal). "
    "Apenas ingreses, cámbiala; desde este chat no podemos cambiar contraseñas.",
    "Entras con tu número de cédula como *usuario* y también como *contraseña* temporal. "
    "Te recomiendo cambiarla al ingresar; por aquí no manejamos contraseñas.",
)
