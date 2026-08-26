# Contrato de integracion del bot con la API .NET

Estado: provisional, pendiente de validacion contra Swagger ejecutable.

## 1. Datos base

- Base URL esperada: `http://localhost:5000/api/v1` en desarrollo.
- Formato: JSON con propiedades `camelCase`.
- Fechas: `YYYY-MM-DD`.
- Marcas de tiempo: ISO 8601 con zona horaria.
- Las respuestas de listados usan paginacion mediante `items`, `page`, `pageSize`, `totalItems` y `totalPages`.
- El bot no accede directamente a Oracle.

## 1.1 Seguridad del webhook de Evolution

El endpoint `POST /webhook/whatsapp` exige el encabezado configurado en `WEBHOOK_SIGNATURE_HEADER` y una firma HMAC-SHA256 calculada sobre el body JSON original usando `WEBHOOK_SECRET`. Se acepta la firma hexadecimal directa o con el prefijo `sha256=`.

Las solicitudes sin firma, con firma incorrecta o con un payload alterado reciben `403` y se rechazan antes de llegar al webhook o a LangGraph. El secreto debe ser el mismo en Evolution API y en el contenedor del agente, y nunca debe registrarse en logs.

## 2. Autenticacion tecnica

Todas las rutas operativas requieren autenticacion. El bot debe usar una identidad tecnica con los permisos minimos necesarios.

Permisos iniciales esperados:

- `AGENTE.VER`
- `AGENTE.CREAR`
- `AGENTE.VINCULAR_PACIENTE`
- `AGENDA.VER`
- `CITAS.VER`
- `CITAS.CREAR`
- `CITAS.EDITAR`
- `CITAS.REPROGRAMAR`
- `CITAS.CANCELAR`

Pendiente de confirmacion con .NET:

- Si la credencial tecnica sera un JWT de servicio o una API key.
- Nombre del encabezado para la API key.
- Duracion y renovacion del token si se usa JWT.
- Respuesta y endpoint para obtener o renovar la credencial.

## 3. Disponibilidad

```http
GET /api/v1/profesionales/{profesionalId}/horarios-disponibles
```

Query parameters:

- `fecha`: fecha a consultar.
- `servicioId`: servicio que determina la duracion de la cita.

Pendiente de confirmacion:

- Forma exacta de cada horario disponible.
- Si la respuesta es una lista directa o un objeto paginado.
- Nombre de los campos de inicio y fin.

## 4. Citas

### Listar citas

```http
GET /api/v1/citas
```

Filtros utilizados por el bot:

- `fecha`
- `pacienteId`
- `profesionalId`
- `estado`
- `page`
- `pageSize`

El scheduler usa `fecha` para consultar las citas del dia siguiente y debe recorrer todas las paginas.

### Crear cita

```http
POST /api/v1/citas
Content-Type: application/json
```

Payload documentado:

```json
{
  "pacienteId": 31,
  "profesionalId": 7,
  "servicioId": 3,
  "fechaHoraInicio": "2026-09-01T10:00:00-05:00",
  "motivoConsulta": "Dolor en molar inferior derecho",
  "origen": "MANUAL"
}
```

El bot debe enviar `origen` con el valor que el equipo .NET defina para el canal conversacional.

### Reprogramar cita

```http
POST /api/v1/citas/{citaId}/reprogramar
```

Pendiente de confirmacion:

- Payload exacto para la nueva fecha y hora.
- Si requiere nuevamente `profesionalId` y `servicioId`.
- Forma de la respuesta actualizada.

### Cancelar cita

```http
POST /api/v1/citas/{citaId}/cancelar
```

Pendiente de confirmacion:

- Campo obligatorio para el motivo de cancelacion.
- Estados que permiten cancelar.
- Forma de la respuesta.

## 5. Pacientes y conversaciones

Para identificar al usuario del chat, el bot necesita consultar o crear la conversacion y vincular el paciente cuando corresponda.

Endpoints documentados:

```http
POST /api/v1/conversaciones-chatbot
PATCH /api/v1/conversaciones-chatbot/{conversacionChatbotId}/paciente
```

Pendiente de confirmacion:

- Campo que representa el identificador del chat de WhatsApp.
- Flujo de busqueda de paciente por telefono o documento.
- Campos necesarios para registrar un paciente nuevo.
- Respuestas de ambas operaciones.

## 6. Recordatorios

El requerimiento inicial indica que el agente debe consultar las citas del dia siguiente y enviar el mensaje por Evolution API.

El contrato tambien documenta una alternativa centralizada:

```http
POST /api/v1/notificaciones-whatsapp
```

Por ahora el bot mantiene el envio directo por Evolution porque el payload de notificaciones .NET aun no esta definido. Antes de cambiar a ese flujo se debe confirmar:

- Payload de programacion.
- Tipo de notificacion para recordatorio.
- Fecha programada.
- Relacion con `citaId` y `pacienteId`.
- Regla de idempotencia para no enviar dos recordatorios.

## 7. Errores que el bot debe manejar

- `400`: parametros o JSON invalidos.
- `401`: credencial ausente, invalida o vencida.
- `403`: identidad tecnica sin permiso.
- `404`: recurso no encontrado.
- `409`: conflicto de horario o transicion incompatible.
- `422`: regla de negocio incumplida.
- `429`: limite de solicitudes.
- `500`: error interno sin detalles sensibles.

El bot debe mostrar un mensaje amigable al usuario y registrar el detalle tecnico solo en logs.

## 8. Escalamiento a recepcion

El bot escala la conversacion cuando el paciente solicita una persona o cuando la confianza del RAG queda por debajo de `RAG_MIN_CONFIDENCE` (por defecto `0.65`). En ese caso:

1. Crea un ticket mediante `POST /api/v1/tickets-soporte`.
2. Cambia el estado del `thread_id` a `ESCALADA`.
3. Informa al paciente que un asesor lo contactara pronto.
4. Ignora mensajes posteriores de ese `thread_id` mientras recepcion atiende el ticket.

Pendiente de confirmacion contra Swagger:

- Payload exacto de `tickets-soporte`.
- Nombres validos para `motivo` y `prioridad`.
- Identificador de la conversacion esperado por el backend.

## 9. Criterio de validacion con el equipo .NET

Este documento queda listo para revisarse contra Swagger. Una vez disponible, se deben reemplazar las secciones pendientes y probar, en este orden:

1. Autenticacion tecnica.
2. Consulta de disponibilidad.
3. Consulta paginada de citas.
4. Creacion de una cita de prueba.
5. Reprogramacion y cancelacion de esa cita.
6. Vinculacion de una conversacion con un paciente.
7. Programacion de una notificacion, si se adopta el flujo centralizado.
