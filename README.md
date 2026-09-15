#  NexusOdonto ChatBot AI — Asistente Clínico Inteligente

[![FastAPI](https://img.shields.io/badge/FastAPI-0.112.1-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.14-blue?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.14-1C3C3C?style=flat-square)](https://www.langchain.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-dc2626?style=flat-square&logo=qdrant)](https://qdrant.tech/)
[![Evolution API](https://img.shields.io/badge/Evolution_API-WhatsApp_Baileys-25D366?style=flat-square&logo=whatsapp)](https://evolution-api.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15_Checkpointer-336791?style=flat-square&logo=postgresql)](https://www.postgresql.org/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python)](https://www.python.org/)

**NexusOdonto ChatBot AI** es un agente conversacional autónomo de grado clínico diseñado para operar en **WhatsApp** mediante **Evolution API**. Está construido con **FastAPI**, **LangGraph** y un motor **RAG híbrido (Qdrant + Re-ranking)**, integrándose bidireccionalmente con el backend hospitalario en **.NET / Oracle Database**.

El sistema permite a los pacientes consultar servicios y tarifas, verificar preparación previa y cuidados postoperatorios, agendar, reprogramar o cancelar citas en tiempo real, interactuar mediante texto o **notas de voz (audio)**, y recibir recordatorios automatizados.

---

##  Tabla de Contenidos

1. [¿Qué es y Para Qué Sirve?](#-qué-es-y-para-qué-sirve)
2. [Beneficios y Utilidad del Chatbot](#-beneficios-y-utilidad-del-chatbot)
3. [Arquitectura y Cómo Funciona](#-arquitectura-y-cómo-funciona)
   - [Diagrama de Flujo y Componentes](#diagrama-de-flujo-y-componentes)
   - [El Grafo de LangGraph](#el-grafo-de-langgraph)
   - [Ciclo de Vida del Mensaje](#ciclo-de-vida-del-mensaje)
4. [Estructura del Proyecto](#-estructura-del-proyecto)
5. [Herramientas del Agente (Tools)](#-herramientas-del-agente-tools)
6. [Triage de Emergencias y Seguridad](#-triage-de-emergencias-y-seguridad)
7. [Requisitos Previos y Variables de Entorno](#-requisitos-previos-y-variables-de-entorno)
8. [Guía de Instalación y Uso](#-guía-de-instalación-y-uso)
   - [Opción 1: Docker Compose (Recomendada)](#opción-1-despliegue-con-docker-compose-recomendada)
   - [Opción 2: Ejecución Local en Desarrollo](#opción-2-ejecución-local-en-desarrollo)
   - [Carga de Conocimiento Clínico (RAG)](#carga-de-conocimiento-clínico-rag)
   - [Vinculación de WhatsApp (Panel QR)](#vinculación-de-whatsapp-panel-qr)
9. [Comandos y Atajos del Usuario](#-comandos-y-atajos-del-usuario)
10. [Endpoints Principales de la API](#-endpoints-principales-de-la-api)

---

##   **NexusOdonto ChatBot AI**?

- Actúa como un **recepcionista virtual 24/7** con lenguaje natural humano, empático y profesional.
- **Acceso directo a la agenda médica:** interactúa con el backend en .NET para consultar huecos libres reales, crear citas en firme y cancelarlas sin solapamientos.
- **Base de Conocimiento Clínico Verificada (RAG):** responde preguntas sobre tratamientos, indicaciones prequirúrgicas y cuidados postoperatorios anclado estrictamente a guías clínicas aprobadas (anti-alucinación).
- **Procesamiento de Voz:** transcribe notas de voz de WhatsApp automáticamente mediante modelos de transcripción (Whisper) para pacientes que prefieren hablar en lugar de escribir.
- **Triage de Emergencias:** detecta situaciones de riesgo (hemorragias abundantes, asfixia, flemón con fiebre, traumatismo mandibular) y deriva de inmediato a la línea telefónica y centro de urgencias.

---

##  Beneficios y Utilidad del Chatbot

| Beneficio | Impacto en la Clínica y el Paciente |
| :--- | :--- |
| **Disponibilidad 24/7** | Los pacientes pueden agendar citas a las 11:00 PM o domingos sin esperar a horario hábil. |
| **Cero Ausentismo (No-Show)** | Cron automático con recordatorios el día anterior (8:00 AM) y alerta de confirmación 30 minutos antes. |
| **Reducción de Espera** | Respuestas inmediatas (< 2s) apoyadas por **Caché Semántico** en Qdrant. |
| **Soporte de Notas de Voz** | Ideal para personas mayores o usuarios en movimiento que envían audios por WhatsApp. |
| **Integración Segura** | No expone la base de datos Oracle directamente; consume la API REST de .NET con tokens de servicio. |
| **Escalamiento a Humanos** | Si el paciente solicita un humano o el bot no tiene suficiente certeza clínica, genera un ticket en recepción y pausa la intervención del bot. |

---

##  Arquitectura y Cómo Funciona

### Diagrama de Flujo y Componentes

```mermaid
flowchart TD
    subgraph WhatsApp["Canal Paciente"]
        User[" Paciente (WhatsApp)"]
        Voice[" Nota de Voz / Texto"]
        User --> Voice
    end

    subgraph Gateway["Integración WhatsApp"]
        Evo["Evolution API (Baileys)"]
        Voice --> Evo
    end

    subgraph FastAPI_Server["NexusOdonto Bot (FastAPI)"]
        Webhook["POST /webhook/whatsapp\n(Deduplicación + Locks + Auth)"]
        Evo -->|Webhook JSON| Webhook

        AudioSvc["Audio Service\n(Descarga + Whisper)"]
        SemCache["Caché Semántico\n(Qdrant Cosine >= 0.90)"]
        
        Webhook -->|Es Audio| AudioSvc --> SemCache
        Webhook -->|Es Texto| SemCache
    end

    subgraph LangGraph_Flow["Orquestador LangGraph"]
        CheckEmerg["1. emergency_check_node\n(Regex + LLM Triage)"]
        CheckSec["2. security_check_node\n(Prompt Injection & Estado)"]
        Compress["3. summarize_conversation\n(Compresión de Historial)"]
        Chatbot["4. chatbot_node\n(OpenAI / Gemini con Tools)"]
        ToolsNode["5. tools (Ejecutor de Herramientas)"]

        SemCache -->|Miss| CheckEmerg
        CheckEmerg -->|Normal| CheckSec
        CheckSec -->|Historial Largo| Compress --> Chatbot
        CheckSec -->|Normal| Chatbot
        Chatbot -->|Requiere Acción| ToolsNode --> Chatbot
    end

    subgraph External_Services["Sistemas Externos y Almacenamiento"]
        Qdrant[("Qdrant Vector DB\n(Conocimiento RAG)")]
        Postgres[("PostgreSQL\n(Memoria LangGraph)")]
        DotNetAPI["Backend .NET REST API\n(Agenda, Citas, Pacientes)"]
        OracleDB[("Oracle Database\n(Datos Maestros)")]

        ToolsNode -->|Búsqueda Semántica| Qdrant
        ToolsNode -->|Agendar / Consultar / Cancelar| DotNetAPI
        DotNetAPI --> OracleDB
        FastAPI_Server <-->|Checkpoints & Threads| Postgres
    end

    CheckEmerg -->|🚨 Emergencia Severa| UrgenciasMsg["🚨 Alerta Inmediata + Escalar a Recepción"]
    SemCache -->|Hit| DirectResponse["Respuesta Instantánea"]
    Chatbot -->|Respuesta Final| Evo --> User
    UrgenciasMsg --> Evo
    DirectResponse --> Evo
```

### El Grafo de LangGraph

El flujo de procesamiento conversacional está compuesto por un **`StateGraph`** con memoria persistente en PostgreSQL:

1. **`emergency_check`**: Evalúa si el paciente describe síntomas de riesgo vital o emergencia odontológica crítica. Si se detecta, se interrumpe el flujo normal, se envía la advertencia médica con el teléfono de urgencias y se escala.
2. **`security_check`**: Analiza intentos de inyección de prompts, verifica si la conversación está bloqueada o si ya fue escalada a un asesor humano.
3. **`summarize_conversation`**: Si la conversación supera un umbral de mensajes (`SUMMARY_THRESHOLD`), comprime el historial mediante un resumen estructurado para optimizar tokens y evitar exceder la ventana de contexto.
4. **`chatbot`**: Nodo central con el modelo LLM configurado (Google Gemini o OpenAI) con *Function Calling*. Analiza el mensaje, decide qué herramientas invocar o formula la respuesta final.
5. **`tools`**: Ejecuta de forma determinista las herramientas invocadas y devuelve los resultados estructurados al chatbot.

### Ciclo de Vida del Mensaje

1. **Recepción:** Evolution API recibe el mensaje de WhatsApp y dispara un evento `MESSAGES_UPSERT` hacia `/webhook/whatsapp`.
2. **Deduplicación y Bloqueo:** El webhook filtra mensajes duplicados (TTL de 90s) y adquiere un candado asíncrono (`asyncio.Lock`) por cada chat para evitar respuestas desordenadas si el usuario envía mensajes seguidos.
3. **Audio a Texto:** Si el mensaje es una nota de voz (`audioMessage`), se descarga el binario base64 y se transcribe a texto en español.
4. **Caché Semántico:** Se consulta Qdrant para preguntas frecuentes idénticas o semánticamente similares (umbral `>= 0.90`). Si hay acierto, se responde de inmediato sin consumir tokens del LLM.
5. **Ejecución del Grafo:** Si no está en caché, LangGraph procesa el hilo (`thread_id = número_whatsapp`), guarda el punto de control en PostgreSQL y ejecuta las llamadas a la API de .NET necesarias.
6. **Entrega:** La respuesta final se envía de vuelta al chat del paciente vía Evolution API con simulación de presencia (*escribiendo...*).

---

##  Estructura del Proyecto

```text
NexusOdonto_ChatBot_AI/
│
├── app/
│   ├── agents/
│   │   └── tools/
│   │       ├── agenda_tools.py        # Tools de interacción con el backend .NET (citas, doctores, servicios)
│   │       └── qdrant_tool.py         # Tool de RAG clínico y conexión con Qdrant
│   │
│   ├── api/
│   │   └── routes/
│   │       ├── agent_handoff.py       # Endpoints para pausar bot y transferir a asesores humanos
│   │       ├── reminders.py           # Disparadores manuales y consulta de recordatorios de citas
│   │       └── webhook.py             # Webhook principal receptor de eventos de Evolution API (WhatsApp)
│   │
│   ├── clients/
│   │   ├── dotnet_client.py           # Cliente HTTP asíncrono con reintentos para la API de .NET
│   │   └── evolution_client.py        # Cliente HTTP para envío de mensajes, presencia y estado en Evolution API
│   │
│   ├── core/
│   │   ├── config.py                  # Configuración central tipada mediante Pydantic BaseSettings (.env)
│   │   └── llm_factory.py             # Fábrica desacoplada para instanciar OpenAI o Google Gemini
│   │
│   ├── graph/
│   │   ├── builder.py                 # Definición y compilación del StateGraph de LangGraph
│   │   ├── nodes.py                   # Lógica de cada nodo (chatbot, emergency, security, summarize)
│   │   └── state.py                   # Esquema de estado AgentState (mensajes, banderas, paciente)
│   │
│   ├── rag/
│   │   └── data_loader.py             # Script de carga y chunking de documentos a Qdrant (.json, .txt, .md)
│   │
│   ├── schemas/
│   │   └── chat.py                    # Modelos Pydantic para payloads de WhatsApp, Evolution y estado
│   │
│   ├── security/
│   │   ├── emergency_detector.py      # Detección determinista (Regex) y heurística de emergencias
│   │   └── webhook_signature.py       # Validación de firmas HMAC-SHA256 del webhook
│   │
│   ├── services/
│   │   ├── appointment_reminders.py   # Tareas programadas de recordatorios (día anterior y -30 minutos)
│   │   ├── audio_service.py           # Descarga de notas de voz de WhatsApp y transcripción con Whisper
│   │   ├── knowledge_ingestion.py     # Base de conocimiento clínico estructurada e indexación
│   │   └── semantic_cache.py          # Caché semántico vectorial de preguntas frecuentes en Qdrant
│   │
│   ├── session/
│   │   ├── memory_store.py            # Utilidades de configuración de sesión y threads
│   │   └── postgres_checkpointer.py   # Checkpointer asíncrono en PostgreSQL para persistencia de LangGraph
│   │
│   └── main.py                        # Punto de entrada FastAPI, ciclo de vida, middleware y servidor QR
│
├── docs/
│   └── bot-api-contract.md            # Especificación técnica y contrato de endpoints con el backend .NET
│
├── Dockerfile                         # Imagen Docker optimizada para producción
├── docker-compose.yml                 # Orquestación completa (Agente + Evolution + PostgreSQL + Qdrant)
├── requirements.txt                   # Dependencias de Python del proyecto
├── .env.example                       # Plantilla documentada de variables de entorno
└── README.md                          # Documentación oficial del proyecto
```

---

##  Herramientas del Agente (Tools)

El agente dispone de 9 herramientas principales con validación estricta de argumentos:

| Herramienta | Función Principal | Origen de Datos |
| :--- | :--- | :--- |
| `clinical_knowledge_tool` | Responde dudas sobre cuidados pre y postoperatorios, tratamientos, duración y preguntas frecuentes. | **Qdrant (RAG)** + Re-ranker |
| `consultar_disponibilidad_tool` | Consulta turnos libres de un doctor para una fecha y servicio específicos. | `GET /profesionales/{id}/horarios-disponibles` (.NET) |
| `agendar_cita_tool` | Reserva una nueva cita médica en el calendario oficial de la clínica. | `POST /citas` (.NET) |
| `consultar_cita_por_cedula_tool` | Localiza citas activas o históricas del paciente usando su documento de identidad. | `GET /citas` + filtro paciente (.NET) |
| `confirmar_cita_tool` | Valida y confirma la asistencia del paciente a una cita agendada. | `PUT/POST /citas/{id}/confirmar` (.NET) |
| `modificar_cita_tool` | Reprograma la fecha y hora de una cita ya existente. | `POST /citas/{id}/reprogramar` (.NET) |
| `cancelar_cita_tool` | Cancela una cita médica registrando el motivo de la cancelación. | `POST /citas/{id}/cancelar` (.NET) |
| `consultar_doctores_tool` | Lista los odontólogos del centro, sus especialidades y registro médico. | `GET /profesionales` (.NET) |
| `consultar_servicios_y_precios_tool` | Retorna el catálogo oficial de servicios odontológicos, precios y duraciones. | `GET /servicios` (.NET) |

---

##  Triager de Emergencias y Seguridad

El chatbot no reemplaza la atención de urgencia presencial. Implementa un sistema de protección en 2 capas:

1. **Capa Determinista (Regex Médico):** Monitorea patrones críticos como:
   - Hemorragias incontrolables / *"sangrado que no para"*
   - Dolor intolerable / *"no aguanto el dolor"*
   - Infección grave, flemón con fiebre o dificultad respiratoria / asfixia
   - Traumatismos maxilofaciales o fracturas dentales por impacto
2. **Acción Inmediata:** Si se detecta cualquiera de estos casos:
   - Se suspende de inmediato la atención del bot.
   - Se instruye al usuario a acudir de urgencia a la dirección del consultorio (**Cr 24 #35-12, Santander**) o llamar al **+57 324 6030217**.
   - Se marca el caso con prioridad `CRITICA` en el sistema.

---

##  Requisitos Previos y Variables de Entorno

### Requisitos del Sistema
- **Docker y Docker Compose** (versión recomendada: Docker 24+, Compose v2).
- O alternativamente: **Python 3.11+**, instancia de **PostgreSQL 15+** y **Qdrant 1.11+**.
- Acceso a la API de **OpenAI** (clave `sk-...`) o **Google Gemini** (gratuita en [Google AI Studio](https://aistudio.google.com/)).

### Configuración del archivo `.env`

Copia el archivo de plantilla y configura tus credenciales:

```bash
cp .env.example .env
```

Principales variables configurables:

```ini
# --- Proveedor de IA (openai o gemini) ---
LLM_PROVIDER="gemini"
GEMINI_API_KEY="AIzaSy..."
GEMINI_MODEL="gemini-2.0-flash"

# --- Embeddings ---
EMBEDDING_PROVIDER="gemini"
GEMINI_EMBEDDING_MODEL="models/text-embedding-004"

# --- Base Vectorial Qdrant ---
QDRANT_URL="http://localhost:6333"

# --- Integración WhatsApp (Evolution API) ---
EVOLUTION_API_URL="http://localhost:8080"
EVOLUTION_INSTANCE_NAME="clinica_odonto"
EVOLUTION_API_KEY="CLAVE_SECRETA_ODONTO_2026"
WEBHOOK_SECRET="SECRETO_COMPARTIDO_CON_EVOLUTION"

# --- Backend .NET y Memoria ---
DOTNET_API_URL="http://localhost:5170/api/v1"
DOTNET_AUTH_LOGIN="usuario_bot@nexusodonto.com"
DOTNET_AUTH_PASSWORD="tu_password_aqui"
POSTGRES_CHECKPOINT_URL="postgresql://bot_user:bot_password@localhost:5433/bot_memory"

# --- Programación de Recordatorios ---
REMINDER_SCHEDULE_HOUR=8
REMINDER_SCHEDULE_MINUTE=0
REMINDER_TIMEZONE="America/Bogota"
```

---

##  Guía de Instalación y Uso

### Opción 1: Despliegue con Docker Compose 

El repositorio incluye un archivo [`docker-compose.yml`](docker-compose.yml) listo para producción que orquesta todos los microservicios:
- **`evolution-whatsapp`**: Servidor WhatsApp con motor Baileys.
- **`evolution-postgres`**: Base de datos de sesiones para Evolution API.
- **`qdrant-db`**: Base de datos vectorial para RAG y Caché Semántico.
- **`bot-memory-db`**: PostgreSQL para checkpoints de LangGraph.
- **`agente-python`**: Contenedor de NexusOdonto ChatBot AI (FastAPI).

Para iniciar toda la infraestructura:

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/NexusOdonto_ChatBot_AI.git
cd NexusOdonto_ChatBot_AI

# 2. Configurar variables
cp .env.example .env
# Edita las claves API en .env

# 3. Levantar los contenedores
docker compose up -d --build

# 4. Verificar logs
docker compose logs -f agente-python
```

### Opción 2: Ejecución Local en Desarrollo

Si prefieres correr el agente en tu máquina local:

```bash
# 1. Crear entorno virtual
python -m venv .venv

# En Windows:
.venv\Scripts\activate
# En Linux/macOS:
# source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Iniciar el servidor FastAPI
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

###  Carga de Conocimiento Clínico (RAG)

Para que el agente responda preguntas clínicas con precisión, debes indexar el catálogo de procedimientos y cuidados en Qdrant:

```bash
# Carga estructurada oficial
python -m app.services.knowledge_ingestion

# O cargar un archivo de texto/markdown/JSON personalizado:
python -m app.rag.data_loader --source data/catalogo_clinico.json
```

---

###  Vinculación de WhatsApp (Panel QR)

NexusOdonto incluye una interfaz web interactiva en tiempo real con recarga reactiva para vincular la línea de WhatsApp:

1. Abre en tu navegador: **`http://localhost:8000/qr`**
2. En tu celular, abre **WhatsApp** > **Dispositivos vinculados** > **Vincular un dispositivo**.
3. Escanea el código QR que aparece en pantalla.
4. El panel detectará la sincronización automáticamente y mostrará el estado **¡WhatsApp Conectado!** con la instancia en línea.

*(También puedes gestionar la instancia desde el panel oficial de Evolution Manager en `http://localhost:8085/manager/`).*

---

##  Comandos y Atajos del Usuario

Los usuarios pueden utilizar comandos rápidos en el chat de WhatsApp:

- **`/reset` o `/limpiar`:** Reinicia la memoria de la conversación actual y comienza un nuevo hilo limpio.
- **`/start` o `/inicio`:** Mensaje de bienvenida inicial y presentación de servicios principales.
- **`"Hablar con un asesor"`:** Escala la conversación a recepción y pausa las respuestas automáticas del bot.
- **`"Volver al bot"` o `"/bot"`:** Reactiva el bot automático si la conversación estaba pausada tras una atención humana.

---

##  Endpoints Principales de la API

Una vez iniciado el servidor, accede a la documentación interactiva en Swagger:
 **`http://localhost:8000/docs`**

| Método | Ruta | Descripción |
| :--- | :--- | :--- |
| `GET` | `/health` | Chequeo de salud del servicio. |
| `POST` | `/webhook/whatsapp` | Webhook receptor de mensajes de Evolution API. |
| `GET` | `/qr` | Interfaz visual interactiva para escaneo y vinculación de WhatsApp. |
| `GET` | `/qr/data` | Estado de conexión JSON y string base64 del código QR activo. |
| `POST` | `/qr/restart` | Fuerza la regeneración de una nueva sesión y código QR limpio. |
| `POST` | `/api/v1/handoff/pause` | Pausa la atención del bot para permitir atención humana manual. |
| `POST` | `/api/v1/handoff/resume` | Reanuda el bot tras la intervención del asesor de recepción. |
| `POST` | `/api/v1/reminders/trigger-daily` | Ejecución manual de la ronda de recordatorios para citas del día siguiente. |
| `POST` | `/api/v1/reminders/trigger-30m` | Ejecución manual del barrido de recordatorios para citas en los próximos 30 minutos. |

---

##  Equipo y Créditos

Proyecto desarrollado para el ecosistema **NexusOdonto** integrando:
- **Backend .NET:** Lógica de negocio y persistencia en Oracle Database.
- **Frontend React:** Panel de gestión administrativa para la clínica.
- **Agente Inteligente:** Orquestación conversacional con LangGraph, RAG y WhatsApp.

---

##  Licencia

Este proyecto está bajo la licencia [MIT](LICENSE).
