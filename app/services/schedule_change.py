"""Aviso al paciente cuando su cita queda por fuera del nuevo horario del odontólogo.

La API .NET llama POST /api/v1/agent/schedule-change por cada cita afectada. Aquí se calculan
las alternativas (mismo odontólogo con su horario nuevo y otros odontólogos del servicio), el
LLM redacta el mensaje (nunca hay plantilla fija), se envía por WhatsApp, se siembra en el hilo
de LangGraph y queda un registro en `pending_schedule_changes` hasta que la cita se reprograme o
cancele. Si el LLM falla el aviso queda en cola y el scheduler lo reintenta.
"""

import asyncio
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.clients.dotnet_client import dotnet_client
from app.clients.evolution_client import evolution_client
from app.core.config import settings

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 6
RETRY_INTERVAL_MINUTES = 10

_ALTERNATIVES_TIMEOUT_S = 5.0
_LLM_BUDGET_S = 7.0
_ENDPOINT_BUDGET_S = 14.0
_STALE_SENDING = timedelta(minutes=10)

STATUS_QUEUED = "QUEUED"
STATUS_SENDING = "SENDING"
STATUS_SENT = "SENT"
STATUS_FAILED = "FAILED"

_DDL = (
    """
    CREATE TABLE IF NOT EXISTS pending_schedule_changes (
        appointment_id TEXT PRIMARY KEY,
        phone TEXT NOT NULL,
        phone_digits TEXT NOT NULL DEFAULT '',
        lid TEXT NOT NULL DEFAULT '',
        patient_id TEXT NOT NULL,
        starts_at TIMESTAMP NOT NULL,
        data JSONB NOT NULL,
        alternatives JSONB NOT NULL DEFAULT '[]'::jsonb,
        message TEXT,
        status TEXT NOT NULL DEFAULT 'QUEUED',
        attempts INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        sent_at TIMESTAMPTZ,
        resolved_at TIMESTAMPTZ,
        resolution TEXT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS ix_pending_schedule_changes_open
        ON pending_schedule_changes (phone) WHERE resolved_at IS NULL
    """,
)


class ScheduleChangeError(Exception):
    """Bad notice input (maps to HTTP 4xx)."""


# ─── Utilidades ──────────────────────────────────────────────────────────────


def _tz() -> ZoneInfo:
    try:
        return ZoneInfo(settings.reminder_timezone)
    except Exception:
        return ZoneInfo("America/Bogota")


def _now_local() -> datetime:
    return datetime.now(_tz()).replace(tzinfo=None)


def parse_wallclock(value: Any) -> Optional[datetime]:
    """Colombian wall-clock datetime; a trailing 'Z' or offset is ignored (the API stores local time)."""
    raw = str(value or "").strip().replace(" ", "T")
    if not raw:
        return None
    raw = re.sub(r"(Z|[+-]\d{2}:?\d{2})$", "", raw).split(".")[0]
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _pool():
    from app.session.postgres_checkpointer import get_checkpointer_instance

    cp = get_checkpointer_instance()
    return cp.pool if cp else None


async def _fetch(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    from psycopg.rows import dict_row

    pool = _pool()
    if pool is None:
        return []
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            if cur.description is None:
                return []
            return list(await cur.fetchall())


async def _exec(sql: str, params: tuple = ()) -> None:
    pool = _pool()
    if pool is None:
        raise RuntimeError("Postgres del bot no disponible")
    async with pool.connection() as conn:
        await conn.execute(sql, params)


def _jsonb(value: Any):
    from psycopg.types.json import Jsonb

    return Jsonb(value)


async def ensure_schema() -> None:
    pool = _pool()
    if pool is None:
        return
    async with pool.connection() as conn:
        for stmt in _DDL:
            await conn.execute(stmt)


# ─── Identidad WhatsApp ──────────────────────────────────────────────────────


async def _resolver_identidad(raw_phone: str) -> tuple[str, str, str]:
    """(canonical thread id, last 10 phone digits or '', LID or '') for an E.164 phone or LID chatIdentifier."""
    from app.services.whatsapp_identity import (
        es_identificador_lid,
        jid_desde_telefono,
        limpiar_digitos,
        normalizar_e164,
        obtener_destino_envio,
        obtener_telefono_canonico,
        registrar_asociacion_lid,
    )

    raw = str(raw_phone or "").strip()
    if not raw:
        return "", "", ""
    lid = ""
    if es_identificador_lid(raw):
        lid = raw if "@" in raw else f"{limpiar_digitos(raw)}@lid"
        canon = obtener_telefono_canonico(lid)
        if es_identificador_lid(canon):
            try:
                phone = await asyncio.wait_for(evolution_client.resolver_telefono_desde_lid(lid), timeout=3.0)
            except Exception:
                phone = None
            if phone:
                registrar_asociacion_lid(phone, lid)
                canon = jid_desde_telefono(phone)
    else:
        e164 = normalizar_e164(raw)
        if not e164:
            return "", "", ""
        canon = jid_desde_telefono(e164)
        dest = obtener_destino_envio(canon)
        if es_identificador_lid(dest):
            lid = dest
    e164 = normalizar_e164(canon)
    digits = limpiar_digitos(e164)[-10:] if e164 else ""
    return canon, digits, lid


def _claves_chat(thread_id: str) -> tuple[list[str], list[str], str]:
    """Identifiers a chat can be stored under: (phones, lids, last 10 digits)."""
    from app.services.whatsapp_identity import (
        es_identificador_lid,
        limpiar_digitos,
        normalizar_e164,
        obtener_destino_envio,
        obtener_telefono_canonico,
    )

    raw = str(thread_id or "").strip()
    canon = obtener_telefono_canonico(raw)
    idents = {i for i in (raw, canon, obtener_destino_envio(canon)) if i}
    lids = sorted(
        (i if "@" in i else f"{limpiar_digitos(i)}@lid") for i in idents if es_identificador_lid(i)
    )
    e164 = normalizar_e164(canon)
    digits = limpiar_digitos(e164)[-10:] if e164 else ""
    return sorted(idents), lids, digits


# ─── Registro persistente ────────────────────────────────────────────────────


async def _obtener(appointment_id: str) -> Optional[Dict[str, Any]]:
    rows = await _fetch("SELECT * FROM pending_schedule_changes WHERE appointment_id = %s", (appointment_id,))
    return rows[0] if rows else None


async def avisos_pendientes_para(thread_id: str) -> List[Dict[str, Any]]:
    """Open notices already sent to this chat whose appointment is still ahead."""
    if not thread_id or _pool() is None:
        return []
    phones, lids, digits = _claves_chat(thread_id)
    return await _fetch(
        """
        SELECT * FROM pending_schedule_changes
        WHERE resolved_at IS NULL AND status = %s AND starts_at > %s
          AND (phone = ANY(%s) OR lid = ANY(%s) OR (phone_digits <> '' AND phone_digits = %s))
        ORDER BY starts_at
        """,
        (STATUS_SENT, _now_local(), phones, lids or ["-"], digits or "-"),
    )


async def aviso_para_chat(thread_id: str, cita_id: Optional[str], permitir_unico: bool) -> Optional[Dict[str, Any]]:
    """Notice of this chat for `cita_id`; with `permitir_unico`, the only open notice when no id matches."""
    avisos = await avisos_pendientes_para(thread_id)
    sel = str(cita_id or "").strip().lower()
    for a in avisos:
        if sel and a["appointment_id"].lower() == sel:
            return a
    if permitir_unico and len(avisos) == 1:
        return avisos[0]
    return None


async def resolver_aviso(appointment_id: str, resolution: str) -> None:
    if not appointment_id or _pool() is None:
        return
    try:
        await _exec(
            """
            UPDATE pending_schedule_changes
            SET resolved_at = NOW(), resolution = %s, updated_at = NOW()
            WHERE lower(appointment_id) = lower(%s) AND resolved_at IS NULL
            """,
            (resolution, str(appointment_id)),
        )
    except Exception as exc:
        logger.warning(f"[ScheduleChange] No se pudo resolver el aviso {appointment_id}: {exc}")


async def _marcar_intento_fallido(appointment_id: str, error: str) -> int:
    rows = await _fetch(
        """
        UPDATE pending_schedule_changes
        SET status = %s, attempts = attempts + 1, last_error = %s, updated_at = NOW()
        WHERE appointment_id = %s
        RETURNING attempts
        """,
        (STATUS_QUEUED, error[:500], appointment_id),
    )
    return int(rows[0]["attempts"]) if rows else 0


# ─── Alternativas (calculadas en código) ─────────────────────────────────────


def _slot_mas_cercano(slots: List[str], objetivo_min: int, excluir: Optional[str] = None) -> Optional[str]:
    candidatos = [s for s in slots if s != excluir]
    if not candidatos:
        return None
    return min(candidatos, key=lambda s: (abs(int(s[:2]) * 60 + int(s[3:5]) - objetivo_min), s))


def _opcion(dia: datetime, hhmm: str, prof_id: Any, prof_nombre: str, mismo: bool) -> Dict[str, Any]:
    from app.agents.tools.agenda_helpers import fecha_legible, hora_corta

    return {
        "fecha_hora": f"{dia.strftime('%Y-%m-%d')} {hhmm}",
        "texto": f"{fecha_legible(dia, con_anio=False)} a las {hora_corta(hhmm)} con {prof_nombre}",
        "hora": hora_corta(hhmm),
        "profesional_id": str(prof_id or ""),
        "profesional": prof_nombre,
        "mismo_odontologo": mismo,
    }


async def calcular_alternativas(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """2-3 options: the same dentist under the new schedule near the original date, and other
    dentists of the same service that day or the next business day. Never the original slot."""
    from app.agents.tools.agenda_helpers import (
        _buscar_servicio_por_texto,
        _normalizar_texto,
        _obtener_valor,
        _resolver_profesional,
        motivo_dia_cerrado,
    )
    from app.agents.tools.catalog_tools import (
        _dias_con_turnos,
        _turnos_por_profesional,
        profesionales_para_servicio,
    )

    original = parse_wallclock(data.get("starts_at"))
    if not original:
        return []
    orig_hhmm = original.strftime("%H:%M")
    orig_min = original.hour * 60 + original.minute
    orig_day = original.replace(hour=0, minute=0, second=0, microsecond=0)

    servicios = await dotnet_client.obtener_servicios() or []
    servicio = next(
        (s for s in servicios if str(_obtener_valor(s, "id", "serviceId") or "").lower() == str(data.get("service_id") or "").lower()),
        None,
    )
    if not servicio and data.get("service"):
        servicio = _buscar_servicio_por_texto(_normalizar_texto(str(data["service"])), servicios)
    if not servicio:
        logger.warning(f"[ScheduleChange] Servicio no encontrado para la cita {data.get('appointment_id')}")
        return []
    servicio_id = _obtener_valor(servicio, "id", "servicioId", "serviceId")
    duracion = int(_obtener_valor(servicio, "durationMinutes", "duracionMinutos") or 60)

    todos = await dotnet_client.obtener_profesionales() or []
    doctor = next((p for p in todos if str(p.get("id") or "").lower() == str(data.get("professional_id") or "").lower()), None)
    if not doctor and data.get("professional"):
        doctor, _ = _resolver_profesional(data["professional"], todos)
    doctor_id = str((doctor or {}).get("id") or data.get("professional_id") or "").lower()

    mismo: List[Dict[str, Any]] = []
    if doctor:
        dias = await _dias_con_turnos([doctor], servicio_id, duracion, original, max_dias=2)
        for dia, libres in dias:
            for nombre, slots in libres:
                excluir = orig_hhmm if dia == orig_day else None
                hhmm = _slot_mas_cercano(slots, orig_min, excluir)
                if hhmm:
                    mismo.append(_opcion(dia, hhmm, doctor.get("id"), nombre, True))

    otros_prof = [p for p in await profesionales_para_servicio(servicio) if str(p.get("id") or "").lower() != doctor_id]
    otros: List[Dict[str, Any]] = []
    if otros_prof:
        hoy = _now_local().replace(hour=0, minute=0, second=0, microsecond=0)
        dia = max(orig_day, hoy)
        dias_objetivo: List[datetime] = []
        if dia == orig_day and not motivo_dia_cerrado(dia):
            dias_objetivo.append(dia)
        siguiente = dia + timedelta(days=1)
        while motivo_dia_cerrado(siguiente):
            siguiente += timedelta(days=1)
        dias_objetivo.append(siguiente)
        for d in dias_objetivo:
            turnos = await _turnos_por_profesional(otros_prof, d.strftime("%Y-%m-%d"), servicio_id, duracion)
            candidatos = []
            for pid, nombre, slots in turnos:
                hhmm = _slot_mas_cercano(slots or [], orig_min)
                if hhmm:
                    candidatos.append((abs(int(hhmm[:2]) * 60 + int(hhmm[3:]) - orig_min), hhmm, pid, nombre))
            for _dist, hhmm, pid, nombre in sorted(candidatos)[:2]:
                otros.append(_opcion(d, hhmm, pid, nombre, False))
            if otros:
                break

    opciones = mismo[:1] + otros[:2]
    if len(opciones) < 3:
        opciones += mismo[1:2]
    return opciones[:3]


# ─── Redacción por el LLM ────────────────────────────────────────────────────

_REDACCION_SYSTEM = (
    "Eres una persona de recepción de Nexus Odonto escribiendo por WhatsApp (español de Colombia, "
    "cálido, natural y breve). Redacta UN solo mensaje para avisarle a un paciente que su cita quedó "
    "por fuera del nuevo horario de su odontólogo y ofrecerle cómo reprogramarla.\n"
    "REGLAS:\n"
    "- Saluda con su primer nombre y di que le escribes de Nexus Odonto.\n"
    "- Explica en una frase que el odontólogo ajustó su horario y que por eso su cita (servicio, día, "
    "fecha y hora tal como vienen en los datos) quedó por fuera. Sin tecnicismos ni disculpas largas.\n"
    "- Si hay opciones, ofrécelas copiando EXACTAMENTE el día, la fecha, la hora y el odontólogo de cada "
    "una (no recalcules ni cambies nada); si son 2 o más, en viñetas con •.\n"
    "- Si no hay opciones, pregúntale qué día y hora le quedan bien, con el mismo odontólogo u otro.\n"
    "- Cierra preguntando cuál prefiere o si le sirve mejor otro día u hora.\n"
    "- Entre 4 y 8 líneas. Sin emojis. *negrita* con un solo asterisco, sin títulos "
    "ni firma.\n"
    "- PROHIBIDO decir que eres asistente, bot, IA, sistema o mensaje automático; PROHIBIDO inventar "
    "datos, precios, direcciones o teléfonos; no pidas la cédula; no digas que la cita ya se movió "
    "o se canceló."
)

_AUTO_REVEAL_RE = re.compile(
    r"\b(bot|chatbot|asistente virtual|inteligencia artificial|mensaje autom[aá]tico)\b|\bIA\b",
    re.IGNORECASE,
)


def _norm_hora(texto: str) -> str:
    return re.sub(r"[\s.\u00a0\u202f]", "", texto).lower()


def _datos_para_llm(data: Dict[str, Any], alternativas: List[Dict[str, Any]]) -> str:
    from app.agents.tools.agenda_helpers import fecha_legible, hora_corta

    inicio = parse_wallclock(data.get("starts_at"))
    nombre = str(data.get("patient_name") or "").strip()
    primer_nombre = nombre.split()[0].capitalize() if nombre else ""
    lineas = [
        f"Primer nombre del paciente: {primer_nombre or '(desconocido: saluda sin nombre)'}",
        f"Servicio de la cita: {data.get('service') or 'consulta odontológica'}",
        f"Odontólogo que cambió su horario: {data.get('professional') or 'su odontólogo'}",
        f"Cita actual: {fecha_legible(inicio, con_anio=False)} a las {hora_corta(inicio.strftime('%H:%M'))}",
    ]
    if alternativas:
        lineas.append("Opciones disponibles para ofrecer:")
        lineas.extend(f"• {a['texto']}" for a in alternativas)
    else:
        lineas.append("Opciones disponibles para ofrecer: ninguna calculada.")
    return "\n".join(lineas)


async def redactar_aviso(data: Dict[str, Any], alternativas: List[Dict[str, Any]]) -> str:
    """Patient-facing notice written by gpt-4o-mini; raises if the model fails or breaks the rules."""
    from app.core.llm_concurrency import with_llm_slot
    from app.core.llm_factory import extract_text_content, get_chat_llm
    from app.core.llm_runtime import end_turn_budget, start_turn_budget
    from app.services.chat.message_processor import _to_whatsapp_format

    llm = get_chat_llm(model="gpt-4o-mini", provider="openai", temperature=0.6, max_tokens=350)
    messages = [SystemMessage(content=_REDACCION_SYSTEM), HumanMessage(content=_datos_para_llm(data, alternativas))]
    horas = {_norm_hora(a["hora"]) for a in alternativas}

    token = start_turn_budget(_LLM_BUDGET_S)
    try:
        ultimo_error = "respuesta vacía"
        for intento in range(2):
            response = await asyncio.wait_for(
                with_llm_slot(llm.ainvoke(messages), label="schedule_change_notice"),
                timeout=_LLM_BUDGET_S,
            )
            texto = re.sub(r"[ \t]+\n", "\n", _to_whatsapp_format(extract_text_content(response.content).strip()))
            if not texto:
                ultimo_error = "respuesta vacía"
            elif _AUTO_REVEAL_RE.search(texto):
                ultimo_error = "se identificó como bot"
            elif any(h not in _norm_hora(texto) for h in horas):
                ultimo_error = "cambió o omitió las horas de las opciones"
            else:
                return texto
            logger.warning(f"[ScheduleChange] Redacción descartada (intento {intento + 1}): {ultimo_error}")
        raise RuntimeError(f"LLM: {ultimo_error}")
    finally:
        end_turn_budget(token)


# ─── Envío, siembra en el hilo y registro ────────────────────────────────────


async def _sembrar_hilo(canon: str, mensaje: str) -> None:
    """The notice becomes the last AI turn of the chat so the patient's reply has context."""
    from app.services.chat.message_processor import _USER_LAST_ACTIVE, _update_thread_state
    from app.session.memory_store import get_thread_config
    from app.session.postgres_checkpointer import get_checkpointer_instance

    cp = get_checkpointer_instance()
    if cp:
        inactivo = await cp.obtener_segundos_inactividad(canon)
        if inactivo is None or inactivo > settings.session_ttl_seconds:
            # Stale history would otherwise survive because the activity is refreshed below.
            await cp.clear_thread(canon)
    await _update_thread_state(get_thread_config(canon), {"messages": [AIMessage(content=mensaje)]})
    if cp:
        await cp.actualizar_actividad(canon)
    _USER_LAST_ACTIVE[canon] = time.time()


async def _entregar(row: Dict[str, Any], mensaje: str) -> bool:
    from app.services.chat.message_processor import get_chat_lock

    appointment_id = row["appointment_id"]
    canon = row["phone"]
    lock = await get_chat_lock(canon)
    try:
        async with lock:
            enviado = await evolution_client.enviar_mensaje(canon, mensaje)
            if not enviado:
                attempts = await _marcar_intento_fallido(appointment_id, "Evolution no confirmó el envío")
                await _notificar_si_agotado(row, attempts)
                return False
            await _exec(
                """
                UPDATE pending_schedule_changes
                SET status = %s, message = %s, sent_at = NOW(), last_error = NULL, updated_at = NOW()
                WHERE appointment_id = %s
                """,
                (STATUS_SENT, mensaje, appointment_id),
            )
            try:
                await _sembrar_hilo(canon, mensaje)
            except Exception as seed_err:
                logger.warning(f"[ScheduleChange] No se pudo sembrar el hilo {canon}: {seed_err}")
    except Exception as exc:
        logger.error(f"[ScheduleChange] Error entregando aviso {appointment_id}: {exc}", exc_info=True)
        try:
            await _marcar_intento_fallido(appointment_id, f"envío: {exc}")
        except Exception:
            pass
        return False

    asyncio.create_task(
        dotnet_client.registrar_mensaje(
            chat_identifier=canon,
            rol="CHATBOT",
            contenido=mensaje,
            patient_id=row.get("patient_id"),
        )
    )
    logger.info(f"[ScheduleChange] Aviso enviado a {canon} por la cita {appointment_id}")
    return True


async def _notificar_si_agotado(row: Dict[str, Any], attempts: int) -> None:
    """After MAX_ATTEMPTS failed tries, reception gets a notification to call the patient."""
    if attempts < MAX_ATTEMPTS:
        return
    from app.agents.tools.agenda_helpers import fecha_legible, hora_corta

    await _exec(
        "UPDATE pending_schedule_changes SET status = %s, updated_at = NOW() WHERE appointment_id = %s",
        (STATUS_FAILED, row["appointment_id"]),
    )
    data = row.get("data") or {}
    inicio = parse_wallclock(data.get("starts_at"))
    cuando = f"{fecha_legible(inicio, con_anio=False)} a las {hora_corta(inicio.strftime('%H:%M'))}" if inicio else ""
    telefono = data.get("phone") or row["phone"]
    try:
        conv_id = await dotnet_client.obtener_o_crear_conversacion(row["phone"], patient_id=row.get("patient_id"))
        await dotnet_client.crear_notificacion(
            titulo="Aviso de cambio de horario sin enviar",
            mensaje=(
                f"No se pudo avisar por WhatsApp a {data.get('patient_name') or 'el paciente'} ({telefono}) "
                f"que su cita de {data.get('service') or 'odontología'} con {data.get('professional') or 'su odontólogo'} "
                f"del {cuando} quedó por fuera del nuevo horario. Contáctalo para reprogramarla."
            ),
            prioridad="ALTA",
            conversation_id=conv_id,
            telefono=telefono,
        )
    except Exception as exc:
        logger.warning(f"[ScheduleChange] No se pudo notificar a recepción el aviso {row['appointment_id']}: {exc}")


async def _preparar_y_enviar(row: Dict[str, Any], deadline: Optional[float]) -> str:
    """One delivery attempt: alternatives + LLM + send. Returns 'sent' or 'queued'.

    `deadline` (monotonic) bounds how long the caller waits for the WhatsApp send.
    """
    appointment_id = row["appointment_id"]
    data = row["data"]
    try:
        try:
            alternativas = await asyncio.wait_for(calcular_alternativas(data), timeout=_ALTERNATIVES_TIMEOUT_S)
        except asyncio.TimeoutError:
            raise RuntimeError("agenda: timeout calculando alternativas")
        mensaje = await redactar_aviso(data, alternativas)
    except Exception as exc:
        logger.warning(f"[ScheduleChange] Aviso {appointment_id} queda en cola: {exc}")
        attempts = await _marcar_intento_fallido(appointment_id, str(exc) or type(exc).__name__)
        await _notificar_si_agotado(row, attempts)
        return "queued"

    await _exec(
        "UPDATE pending_schedule_changes SET status = %s, alternatives = %s, updated_at = NOW() WHERE appointment_id = %s",
        (STATUS_SENDING, _jsonb(alternativas), appointment_id),
    )
    tarea = asyncio.create_task(_entregar(row, mensaje))
    if deadline is None:
        return "sent" if await tarea else "queued"
    try:
        # Shielded: if the send outlives the HTTP budget it still finishes and marks the row.
        ok = await asyncio.wait_for(asyncio.shield(tarea), timeout=max(2.0, deadline - time.monotonic()))
    except asyncio.TimeoutError:
        logger.warning(f"[ScheduleChange] Envío del aviso {appointment_id} sigue en curso tras el plazo")
        return "queued"
    return "sent" if ok else "queued"


# ─── Entradas públicas ───────────────────────────────────────────────────────


async def notificar_cambio_horario(data: Dict[str, Any]) -> Dict[str, Any]:
    """Handles one affected appointment from the API within ~14 s."""
    t0 = time.monotonic()
    if _pool() is None:
        raise RuntimeError("Postgres del bot no disponible")

    appointment_id = str(data.get("appointment_id") or "").strip()
    patient_id = str(data.get("patient_id") or "").strip()
    if not appointment_id or not patient_id:
        raise ScheduleChangeError("appointment_id y patient_id son obligatorios")
    inicio = parse_wallclock(data.get("starts_at"))
    if not inicio:
        raise ScheduleChangeError("starts_at inválido")
    fin = parse_wallclock(data.get("ends_at")) if data.get("ends_at") else None
    if data.get("ends_at") and (not fin or fin <= inicio):
        raise ScheduleChangeError("ends_at inválido")
    if inicio <= _now_local():
        raise ScheduleChangeError("La cita ya pasó")
    canon, digits, lid = await _resolver_identidad(str(data.get("phone") or ""))
    if not canon:
        raise ScheduleChangeError("phone no es un teléfono WhatsApp ni un chatIdentifier válido")

    previo = await _obtener(appointment_id)
    if (
        previo
        and previo["resolved_at"] is None
        and previo["status"] in (STATUS_SENT, STATUS_SENDING)
        and previo["starts_at"] == inicio
        and str((previo["data"] or {}).get("professional_id") or "").lower() == str(data.get("professional_id") or "").lower()
    ):
        return {"status": "sent" if previo["status"] == STATUS_SENT else "queued", "duplicate": True}

    await _exec(
        """
        INSERT INTO pending_schedule_changes
            (appointment_id, phone, phone_digits, lid, patient_id, starts_at, data, status, attempts)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 0)
        ON CONFLICT (appointment_id) DO UPDATE SET
            phone = EXCLUDED.phone, phone_digits = EXCLUDED.phone_digits, lid = EXCLUDED.lid,
            patient_id = EXCLUDED.patient_id, starts_at = EXCLUDED.starts_at, data = EXCLUDED.data,
            alternatives = '[]'::jsonb, message = NULL, status = EXCLUDED.status, attempts = 0,
            last_error = NULL, created_at = NOW(), updated_at = NOW(), sent_at = NULL,
            resolved_at = NULL, resolution = NULL
        """,
        (appointment_id, canon, digits, lid, patient_id, inicio, _jsonb(data), STATUS_QUEUED),
    )
    row = await _obtener(appointment_id)
    status = await _preparar_y_enviar(row, deadline=t0 + _ENDPOINT_BUDGET_S)
    return {"status": status}


async def _cita_sigue_afectada(row: Dict[str, Any]) -> bool:
    """False when the appointment was cancelled or moved outside the chat since the notice was queued."""
    try:
        resp = await dotnet_client.transport.request("GET", f"Appointments/{row['appointment_id']}")
    except Exception:
        return True
    if resp is None:
        return True
    if resp.status_code == 404:
        return False
    if resp.status_code != 200:
        return True
    cita = resp.json() or {}
    estado = str(cita.get("statusName") or cita.get("status") or "").lower()
    if cita.get("cancelledAt") or str(cita.get("appointmentStatusId") or "").lower() == "10000000-0000-0000-0000-000000000005" or "cancel" in estado:
        return False
    data = row.get("data") or {}
    if parse_wallclock(cita.get("startsAt")) != row["starts_at"]:
        return False
    prof = str(data.get("professional_id") or "").lower()
    return not prof or str(cita.get("professionalId") or "").lower() == prof


async def reintentar_avisos_pendientes() -> None:
    """Scheduler job (every 10 min): retry notices whose LLM or send failed."""
    if _pool() is None:
        return
    try:
        rows = await _fetch(
            """
            SELECT * FROM pending_schedule_changes
            WHERE resolved_at IS NULL AND attempts < %s AND (
                (status = %s AND updated_at < NOW() - INTERVAL '5 minutes')
                OR (status = %s AND updated_at < NOW() - %s)
            )
            ORDER BY starts_at
            LIMIT 20
            """,
            (MAX_ATTEMPTS, STATUS_QUEUED, STATUS_SENDING, _STALE_SENDING),
        )
    except Exception as exc:
        logger.warning(f"[ScheduleChange] No se pudo leer la cola de avisos: {exc}")
        return
    for row in rows:
        try:
            if row["starts_at"] <= _now_local():
                await resolver_aviso(row["appointment_id"], "vencida_sin_envio")
                continue
            if not await _cita_sigue_afectada(row):
                await resolver_aviso(row["appointment_id"], "cambiada_fuera_del_chat")
                continue
            status = await _preparar_y_enviar(row, deadline=None)
            logger.info(f"[ScheduleChange] Reintento del aviso {row['appointment_id']}: {status}")
        except Exception as exc:
            logger.error(f"[ScheduleChange] Error reintentando aviso {row['appointment_id']}: {exc}", exc_info=True)


def bloque_cita_afectada(avisos: List[Dict[str, Any]]) -> str:
    """Per-turn prompt block while a notice is open for this chat."""
    from app.agents.tools.agenda_helpers import fecha_legible, hora_corta

    partes = []
    for a in avisos[:3]:
        data = a.get("data") or {}
        inicio = a["starts_at"]
        opciones = a.get("alternatives") or []
        texto_opc = (
            "; ".join(f"{o['texto']} (nueva_fecha_hora='{o['fecha_hora']}', odontólogo='{o['profesional']}')" for o in opciones)
            or "ninguna calculada"
        )
        partes.append(
            f"cita_id='{a['appointment_id']}': {data.get('service') or 'cita'} con {data.get('professional') or 'su odontólogo'}, "
            f"{fecha_legible(inicio, con_anio=False)} a las {hora_corta(inicio.strftime('%H:%M'))}. "
            f"Opciones que ya le ofrecimos: {texto_opc}."
        )
    return (
        "\n[CITA AFECTADA POR CAMBIO DE HORARIO] Le escribimos a este paciente porque su cita quedó por fuera "
        "del nuevo horario de su odontólogo: " + " | ".join(partes) + " "
        "Ayúdale a moverla a otra hora con el mismo odontólogo o con otro disponible (consultar_disponibilidad_tool "
        "si pide otro día u hora). Para esta cita NO pidas cédula: si ya eligió con claridad una opción o un día, "
        "hora y odontólogo concretos, llama modificar_cita_tool(cita_id='<cita_id>', nueva_fecha_hora='YYYY-MM-DD HH:MM', "
        "nuevo_profesional_id='<nombre del odontólogo>' si cambia de odontólogo); si no es claro, pregunta cuál prefiere. "
        "Si quiere cancelarla: cancelar_cita_tool(cita_id='<cita_id>'). Confirma el cambio solo con el resultado "
        "exitoso de la herramienta."
    )
