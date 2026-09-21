"""Reglas clínicas oficiales de agenda y horarios para Nexus Odonto.
Funciones puras deterministas, sin efectos secundarios ni dependencias externas.
"""

from datetime import time, datetime, date
from typing import Optional, Tuple

SLOT_DURATION_MINUTES = 30

# Jornadas laborales oficiales
HORA_APERTURA_MANANA = time(8, 0)
HORA_CIERRE_MANANA = time(12, 0)

HORA_INICIO_ALMUERZO = time(12, 0)
HORA_FIN_ALMUERZO = time(14, 0)

HORA_APERTURA_TARDE = time(14, 0)
HORA_CIERRE_TARDE = time(17, 0)


def es_horario_almuerzo(t: time) -> bool:
    """Retorna True si la hora dada se encuentra dentro del receso de almuerzo médico (12:00 PM a 2:00 PM)."""
    return HORA_INICIO_ALMUERZO <= t < HORA_FIN_ALMUERZO


def es_dia_laboral(d: date) -> Tuple[bool, str]:
    """Valida si el día consultado es laboral en Nexus Odonto.
    Lunes a Viernes: Completo (8-12, 2-5)
    Sábados: Media jornada (8-12)
    Domingos: Cerrado
    """
    weekday = d.weekday()  # 0=Lunes, 5=Sábado, 6=Domingo
    if weekday == 6:
        return False, "Los domingos el consultorio se encuentra cerrado."
    return True, "Abierto"


def es_hora_laboral_valida(t: time, es_sabado: bool = False) -> Tuple[bool, str]:
    """Determina si una hora específica está dentro de los turnos autorizados de la clínica."""
    # 1. Verificar receso de almuerzo
    if es_horario_almuerzo(t):
        return False, (
            "El horario seleccionado (12:00 PM a 2:00 PM) corresponde al receso de almuerzo de nuestros especialistas. "
            "La atención de la tarde inicia a las 2:00 PM. 😊"
        )

    # 2. Verificar si es sábado (solo mañanas)
    if es_sabado:
        if HORA_APERTURA_MANANA <= t < HORA_CIERRE_MANANA:
            return True, "Horario válido (Sábado)"
        return False, "Los sábados la clínica atiende únicamente en jornada de la mañana (8:00 AM a 12:00 PM)."

    # 3. Lunes a Viernes
    en_manana = HORA_APERTURA_MANANA <= t < HORA_CIERRE_MANANA
    en_tarde = HORA_APERTURA_TARDE <= t < HORA_CIERRE_TARDE

    if en_manana or en_tarde:
        return True, "Horario laboral válido"

    if t < HORA_APERTURA_MANANA:
        return False, "Nuestro horario de atención inicia a las 8:00 AM."
    if t >= HORA_CIERRE_TARDE:
        return False, "Nuestra última cita de la tarde finaliza a las 5:00 PM."

    return False, "Horario fuera de turno laboral."


def redondear_a_bloque_30_minutos(t: time) -> time:
    """Si el paciente solicita una hora intermedia (ej. 1:42 PM o 2:15 PM),
    calcula el turno estándar posterior o más cercano en bloques de 30 min.
    """
    hora = t.hour
    minuto = t.minute

    if minuto == 0 or minuto == 30:
        return t

    if minuto < 30:
        return time(hora, 30)
    else:
        # Pasa a la siguiente hora en punto
        nueva_hora = min(23, hora + 1)
        return time(nueva_hora, 0)


def obtener_primer_turno_tarde() -> time:
    """Devuelve la hora del primer turno de la tarde tras el almuerzo (2:00 PM)."""
    return HORA_APERTURA_TARDE
