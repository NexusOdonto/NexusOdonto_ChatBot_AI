# 🦷 NexusOdonto ChatBot AI — Asistente Clínico Inteligente

[![FastAPI](https://img.shields.io/badge/FastAPI-0.112.1-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.14-blue?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![Clean Architecture](https://img.shields.io/badge/Architecture-Clean%20%2F%20Layered-orange?style=flat-square)](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
[![Tests](https://img.shields.io/badge/Tests-41%20Passing%20(100%25)-brightgreen?style=flat-square)](tests/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB%20%2B%20Reranker-dc2626?style=flat-square&logo=qdrant)](https://qdrant.tech/)
[![Evolution API](https://img.shields.io/badge/Evolution_API-WhatsApp_Native-25D366?style=flat-square&logo=whatsapp)](https://evolution-api.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15_Checkpointer-336791?style=flat-square&logo=postgresql)](https://www.postgresql.org/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python)](https://www.python.org/)

**NexusOdonto ChatBot AI** es un agente conversacional autónomo de grado clínico diseñado para operar en **WhatsApp** mediante **Evolution API**. Está estructurado bajo **Arquitectura Limpia (Clean Architecture)** con desacoplamiento estricto en capas de Dominio, Aplicación e Infraestructura, operando sobre **FastAPI**, **LangGraph** y un motor **RAG híbrido (Qdrant + Cross-Encoder Re-ranking)**, con integración bidireccional al backend hospitalario en **.NET / Oracle Database**.

El sistema permite a los pacientes consultar servicios y tarifas, verificar preparación previa y cuidados postoperatorios, agendar, reprogramar o cancelar citas en tiempo real, interactuar mediante texto o **notas de voz (audio)**, y recibir recordatorios automatizados con simulación de presencia (*"Escribiendo..."*) y formato nativo optimizado para WhatsApp.

---

## 📋 Tabla de Contenidos

1. [¿Qué es y Para Qué Sirve?](#-qué-es-y-para-qué-sirve)
2. [Beneficios y Utilidad del Chatbot](#-beneficios-y-utilidad-del-chatbot)
3. [Arquitectura y Cómo Funciona](#-arquitectura-y-cómo-funciona)
   - [Diagrama de Flujo por Capas](#diagrama-de-flujo-por-capas)
   - [El Grafo de LangGraph](#el-grafo-de-langgraph)
   - [Ciclo de Vida del Mensaje](#ciclo-de-vida-del-mensaje)
4. [Estructura del Proyecto (Clean Architecture)](#-estructura-del-proyecto-clean-architecture)
5. [Herramientas del Agente (Tools)](#-herramientas-del-agente-tools)
6. [Triage de Emergencias y Seguridad](#-triage-de-emergencias-y-seguridad)
7. [Experiencia Nativa de WhatsApp](#-experiencia-nativa-de-whatsapp)
8. [Requisitos Previos y Variables de Entorno](#-requisitos-previos-y-variables-de-entorno)
9. [Guía de Instalación y Uso](#-guía-de-instalación-y-uso)
   - [Opción 1: Docker Compose (Recomendada)](#opción-1-despliegue-con-docker-compose-recomendada)
   - [Opción 2: Ejecución Local en Desarrollo](#opción-2-ejecución-local-en-desarrollo)
   - [Carga de Conocimiento Clínico (RAG)](#carga-de-conocimiento-clínico-rag)
   - [Vinculación de WhatsApp (Panel QR)](#vinculación-de-whatsapp-panel-qr)
10. [Ejecución de Pruebas Automatizadas](#-ejecución-de-pruebas-automatizadas)
11. [Comandos y Atajos del Usuario](#-comandos-y-atajos-del-usuario)
12. [Endpoints Principales de la API](#-endpoints-principales-de-la-api)

---

## 💡 ¿Qué es y Para Qué Sirve?

- **Recepcionista virtual 24/7:** Atención cálida, humana y profesional sin menús robóticos fríos.
- **Acceso directo a la agenda médica:** Interacción en tiempo real con .NET para consultar turnos libres, bloquear franjas de almuerzo (12:00 PM a 2:00 PM) y registrar o cancelar citas sin solapamientos.
- **Base de Conocimiento Clínico Verificada (RAG Robusto):** 24 guías odontológicas oficiales, expansor de modismos coloquiales y re-ranking semántico para responder dudas clínicas con precisión anti-alucinación.
- **Procesamiento de Voz:** Transcribe notas de voz automáticamente mediante Whisper para pacientes que prefieren hablar.
- **Triage de Emergencias Odontológicas:** Detección inmediata de hemorragias, dolor agudo, flemón con fiebre y traumas maxilofaciales con derivación prioritaria.
- **Privacidad y Habeas Data (Ley 1581):** Cero asunción de identidad; validación estricta de cédula antes de revelar o procesar citas.
- **Presencia en Vivo:** Simulación de estado *"Escribiendo..."* (`composing`) y *"Grabando audio..."* (`recording`) en tiempo real mientras el bot procesa.

---

## 🏆 Beneficios y Utilidad del Chatbot

| Beneficio | Impacto en la Clínica y el Paciente |
| :--- | :--- |
| **Disponibilidad 24/7** | Los pacientes pueden agendar citas a las 11:00 PM o domingos sin esperar a horario hábil. |
| **Cero Ausentismo (No-Show)** | Cron automático con recordatorios el día anterior (8:00 AM) y alerta de confirmación 30 minutos antes. |
| **Reducción de Espera** | Respuestas inmediatas (< 2s) apoyadas por **Caché Semántico** en Qdrant (0 tokens consumidos). |
| **Soporte de Notas de Voz** | Ideal para personas mayores o usuarios en movimiento que envían audios por WhatsApp. |
| **Reglas Clínicas Estrictas** | Cero citas en horario de almuerzo de doctores (12-2 PM) y redondeo a bloques exactos de 30 minutos. |
| **Escalamiento a Humanos** | Si el paciente solicita un humano o el bot no tiene suficiente certeza clínica, genera un ticket en recepción y pausa la intervención del bot. |
| **Purga Inteligente de Inactividad** | Tras 15 minutos sin respuesta, cierra la sesión ordenadamente, detecta si quedó una cita a medias y notifica al usuario sin dejar sesiones huérfanas. |

---

## 🏛️ Arquitectura y Cómo Funciona

### Diagrama de Flujo por Capas

```mermaid
flowchart TD
    subgraph WhatsApp["Canal Paciente"]
        User["👤 Paciente (WhatsApp)"]
        Voice["🎙️ Nota de Voz / 💬 Texto"]
        User --> Voice
    end

    subgraph Gateway["Integración WhatsApp"]
        Evo["Evolution API (Baileys)"]
        Voice --> Evo
    end

    subgraph Presentation["Capa de Presentación / API"]
        Webhook["POST /webhook/whatsapp\n(Deduplicación 90s + Auth)"]
        QRRoute["GET /qr (Panel Interactivo)"]
        HealthRoute["GET /health/deep (Diagnóstico)"]
        Evo -->|Webhook JSON| Webhook
    end

    subgraph Application["Capa de Aplicación / Servicios"]
        Orchestrator["ChatOrchestrator\n(Debounce 2s + Anti-Spam + Presence)"]
        MsgProc["MessageProcessor\n(Orquestación Grafo + Control TTL 15m)"]
        AudioSvc["AudioService\n(Descarga + Whisper)"]
        SemCache["Caché Semántico\n(Qdrant Cosine >= 0.90)"]
        Inactivity["InactivityService\n(Sweeper 60s + Detección Cita Incompleta)"]
        
        Webhook --> Orchestrator --> MsgProc
        MsgProc --> AudioSvc
        MsgProc --> SemCache
    end

    subgraph LangGraph_Flow["Orquestador Conversacional (LangGraph)"]
        CheckEmerg["1. emergency_check\n(30+ Patrones Regex + Triage)"]
        CheckSec["2. security_check\n(Prompt Injection & Estado)"]
        Compress["3. summarize_conversation\n(Compresión de Historial)"]
        Chatbot["4. chatbot_node\n(LLM con Formato WhatsApp Nativo)"]
        ToolsNode["5. tools (Ejecutor de Herramientas Atómicas)"]

        SemCache -->|Miss| CheckEmerg
        CheckEmerg -->|Normal| CheckSec
        CheckSec -->|Historial Largo| Compress --> Chatbot
        CheckSec -->|Normal| Chatbot
        Chatbot -->|Requiere Acción| ToolsNode --> Chatbot
    end

    subgraph Domain["Capa de Dominio Puro (0 Dependencias)"]
        Rules["Reglas Clínicas (Schedule Rules)\nBloqueo Almuerzo 12-2 PM"]
        Habeas["Habeas Data (Ley 1581)\nValidación Cédula >= 7 dígitos"]
        Formatter["WhatsApp Formatter\nLimpieza ** a *, Viñetas y Enlaces"]
        KB["Base de Conocimiento (24 Guías)\nExpansor de Consultas Coloquiales"]
    end

    subgraph Infrastructure["Capa de Infraestructura Externa"]
        Qdrant[("Qdrant Vector DB\nRAG + Cross-Encoder Re-ranker")]
        Postgres[("PostgreSQL\nMemoria LangGraph & Checkpoints")]
        DotNetAPI["Backend .NET REST API\n(Appointments, Patients, Catalog, Tickets)"]
        OracleDB[("Oracle Database\nDatos Maestros Clínicos")]

        ToolsNode --> DotNetAPI --> OracleDB
        ToolsNode --> Qdrant
        MsgProc <--> Postgres
    end

    Chatbot -->|Texto en Crudo| Formatter
    Formatter -->|Texto Nativo WhatsApp| Evo --> User
```

### El Grafo de LangGraph

1. **`emergency_check`**: Evalúa si el paciente describe síntomas de riesgo vital o urgencia médica severa. Si se detecta, interrumpe el flujo, envía la advertencia con el teléfono de urgencias y escala.
2. **`security_check`**: Analiza intentos de inyección de prompts, verifica si la conversación está bloqueada o si ya fue escalada a un asesor humano.
3. **`summarize_conversation`**: Si la conversación supera el umbral de mensajes (`SUMMARY_THRESHOLD`), comprime el historial mediante un resumen estructurado para optimizar tokens.
4. **`chatbot`**: Nodo central con el modelo LLM configurado (Google Gemini u OpenAI) con *Function Calling*. Analiza el mensaje, decide qué herramientas invocar y formula la respuesta con estilo 100% humano y empático.
5. **`tools`**: Ejecuta de forma determinista las herramientas atómicas invocadas y devuelve los resultados estructurados al chatbot.

### Ciclo de Vida del Mensaje

1. **Recepción:** Evolution API recibe el mensaje de WhatsApp y dispara un evento `MESSAGES_UPSERT` hacia `/webhook/whatsapp`.
2. **Deduplicación:** El webhook descarta entregas duplicadas (TTL 90 segundos) y detecta si el mensaje fue enviado manualmente por un asesor humano (`fromMe`).
3. **Presencia Inmediata:** Se envía `sendPresence("composing")` a WhatsApp para mostrar *"Escribiendo..."* desde el primer milisegundo.
4. **Debounce:** `ChatOrchestrator` acumula ráfagas de mensajes durante 2 segundos para responder con un solo mensaje consolidado.
5. **Transcripción:** Si el mensaje es una nota de voz, se transcribe a texto mediante Whisper.
6. **Caché Semántico:** Se consulta Qdrant para preguntas frecuentes (similitud `>= 0.90`). Si hay acierto, responde en milisegundos a costo cero de tokens.
7. **Ejecución del Grafo:** LangGraph procesa el hilo con memoria en PostgreSQL, aplicando las reglas de agenda y catálogo en .NET.
8. **Sanitización y Entrega:** `whatsapp_formatter` limpia el texto final (eliminando asteriscos dobles y adaptando viñetas) antes de entregarlo por WhatsApp.

---

## 📁 Estructura del Proyecto (Clean Architecture)

```text
NexusOdonto_ChatBot_AI/
│
├── app/
│   ├── agents/
│   │   └── tools/
│   │       ├── agenda_tools.py            # Facade retrocompatible para el agente
│   │       ├── appointment_tools.py       # Herramientas atómicas de agendamiento y citas
│   │       ├── catalog_tools.py           # Herramientas atómicas de catálogo y disponibilidad
│   │       ├── clinical_rag_tool.py       # Herramienta RAG de conocimiento clínico
│   │       ├── agenda_helpers.py          # Validaciones de horarios y diccionarios
│   │       └── qdrant_tool.py             # Conector con la base vectorial Qdrant
│   │
│   ├── api/
│   │   └── routes/
│   │       ├── webhook.py                 # Controlador HTTP de eventos WhatsApp (Evolution API)
│   │       ├── qr.py                      # Panel web interactivo para vincular WhatsApp
│   │       ├── health.py                  # Endpoints /health y /health/deep (diagnóstico completo)
│   │       ├── agent_handoff.py           # Pausa y reactivación entre bot y asesores humanos
│   │       └── reminders.py               # Disparadores manuales de recordatorios de citas
│   │
│   ├── clients/
│   │   ├── dotnet_client.py               # Facade retrocompatible para la API .NET
│   │   └── evolution_client.py            # Cliente Evolution API con presencia, botones y texto
│   │
│   ├── domain/                            # CAPA DE DOMINIO PURA (0 dependencias externas)
│   │   ├── clinical/
│   │   │   ├── schedule_rules.py          # Reglas de horario laboral y bloqueo de almuerzo
│   │   │   ├── emergency_triage.py        # Triage clínico determinista (30+ patrones)
│   │   │   ├── knowledge_base.py          # 24 guías clínicas estructuradas oficiales
│   │   │   └── query_expander.py          # Expansor semántico de jerga colombiana a términos médicos
│   │   ├── formatters/
│   │   │   └── whatsapp_formatter.py      # Sanitizador nativo de WhatsApp (elimina **, viñetas •)
│   │   ├── models/
│   │   │   ├── appointment.py             # Entidades tipadas de Citas
│   │   │   ├── patient.py                 # Entidades tipadas de Pacientes y Contexto Seguro
│   │   │   ├── catalog.py                 # Entidades tipadas de Servicios y Disponibilidad
│   │   │   └── emergency.py               # Entidades tipadas de Triage de Emergencias
│   │   └── security/
│   │       └── habeas_data.py             # Reglas Ley 1581 y validación de cédulas (>= 7 dígitos)
│   │
│   ├── graph/                             # ORQUESTADOR LANGGRAPH
│   │   ├── nodes/
│   │   │   ├── chatbot_node.py            # LLM principal con prompt humano y formato WhatsApp
│   │   │   ├── emergency_node.py          # Nodo de detección de emergencias
│   │   │   ├── security_node.py           # Nodo de defensa contra prompt injection
│   │   │   └── summarizer_node.py         # Nodo de compresión de historial
│   │   ├── builder.py                     # Compilador del StateGraph
│   │   ├── nodes.py                       # Facade retrocompatible de nodos
│   │   └── state.py                       # Esquema de estado AgentState
│   │
│   ├── infra/                             # CAPA DE INFRAESTRUCTURA EXTERNA
│   │   ├── external/
│   │   │   └── dotnet/
│   │   │       ├── http_transport.py      # Transporte HTTP con auto-refresh JWT (401)
│   │   │       ├── appointments_api.py    # Endpoints de Citas y Agendamiento en .NET
│   │   │       ├── patients_api.py        # Endpoints de Pacientes y Onboarding en .NET
│   │   │       ├── catalog_api.py         # Endpoints de Catálogo, Servicios y Doctores en .NET
│   │   │       └── tickets_api.py         # Endpoints de Tickets de Soporte y Mensajes en .NET
│   │   └── persistence/
│   │       └── clinical_retriever.py      # Búsqueda semántica + Cross-Encoder Re-ranking
│   │
│   ├── services/                          # CAPA DE APLICACIÓN
│   │   ├── chat/
│   │   │   ├── chat_orchestrator.py       # Debounce 2s, control de spam y presencia
│   │   │   └── message_processor.py       # Procesamiento de hilos, TTL de 15 min y memoria
│   │   ├── appointment/
│   │   │   └── appointment_service.py     # Enriquecimiento de citas y validación de pacientes
│   │   ├── appointment_reminders.py       # Cron APScheduler (diario 8:00 AM y cada 2 min)
│   │   ├── inactivity_service.py          # Sweeper de inactividad 15 min y detección de citas a medias
│   │   ├── audio_service.py               # Transcripción Whisper de notas de voz
│   │   └── semantic_cache.py              # Caché semántico vectorial en Qdrant
│   │
│   ├── core/
│   │   ├── config.py                      # Configuración central tipada con Pydantic (.env)
│   │   └── llm_factory.py                 # Fábrica agnóstica para OpenAI y Google Gemini
│   │
│   ├── session/
│   │   ├── postgres_checkpointer.py       # Checkpointer asíncrono en PostgreSQL
│   │   └── memory_store.py                # Configuración de threads de LangGraph
│   │
│   └── main.py                            # Punto de entrada FastAPI, ciclo de vida y schedulers
│
├── tests/                                 # SUITE DE PRUEBAS AUTOMATIZADAS (41 TESTS)
│   ├── test_domain_and_rag.py             # Pruebas de reglas clínicas, Habeas Data y RAG
│   ├── test_phase2_tools_and_nodes.py     # Pruebas de herramientas atómicas y nodos del grafo
│   ├── test_phase3_reminders_and_inactivity.py # Pruebas de recordatorios e inactividad
│   ├── test_phase4_e2e_concurrency.py     # Pruebas E2E de concurrencia y aislamiento
│   └── test_whatsapp_formatter.py         # Pruebas de formato WhatsApp y presencia en tiempo real
│
├── scripts/
│   └── limpiar_cache_y_sesiones.py        # Script para purgar caché semántico y sesiones
├── Dockerfile                             # Imagen Docker de producción
├── docker-compose.yml                     # Orquestación (Agente + Evolution + Postgres + Qdrant)
├── requirements.txt                       # Dependencias del proyecto
├── .env.example                           # Plantilla de variables de entorno
└── README.md                              # Documentación oficial
```

---

## 🛠️ Herramientas del Agente (Tools)

El agente dispone de herramientas atómicas especializadas agrupadas bajo el principio de Responsabilidad Única (SRP):

| Herramienta | Módulo Fuente | Función Principal |
| :--- | :--- | :--- |
| `consultar_servicios_y_precios_tool` | `catalog_tools.py` | Consulta el catálogo oficial de procedimientos activos y tarifas. |
| `consultar_doctores_tool` | `catalog_tools.py` | Retorna los especialistas de la clínica y sus especialidades. |
| `consultar_disponibilidad_tool` | `catalog_tools.py` | Consulta turnos libres aplicando reglas de almuerzo (12-2 PM). |
| `agendar_cita_tool` | `appointment_tools.py` | Valida cédula, asegura al paciente y reserva la cita en el sistema. |
| `consultar_cita_por_cedula_tool` | `appointment_tools.py` | Busca citas activas o históricas asociadas al número de documento. |
| `modificar_cita_tool` | `appointment_tools.py` | Reprograma fecha u hora de una cita previa verificando disponibilidad. |
| `cancelar_cita_tool` | `appointment_tools.py` | Cancela una cita registrando el motivo y liberando el horario. |
| `confirmar_cita_tool` | `appointment_tools.py` | Confirma la asistencia a una cita programada o recordatorio. |
| `clinical_knowledge_tool` | `clinical_rag_tool.py` | Responde dudas odontológicas anclado a las 24 guías clínicas oficiales. |

---

## 🚨 Triage de Emergencias y Seguridad

El chatbot implementa una estrategia de protección médica en dos capas:

1. **Capa Determinista (30+ Patrones Regex):**
   - Hemorragias incontrolables / *"sangrado que no para"*
   - Dolor intolerable / *"no aguanto el dolor"* (dolor severo > 8/10)
   - Flemón con fiebre o hinchazón en el cuello (compromiso de vía aérea)
   - Traumatismos severos o fracturas dentales por golpe
2. **Acción Inmediata:**
   - Se suspende de inmediato la atención del bot para no demorar la atención médica.
   - Se instruye al paciente a acudir de urgencia a **Calle 100 # 15-20, Centro Médico Odontológico** o comunicarse a la línea **+57 324 6030217**.
   - Se genera un ticket en el sistema hospitalario marcado con prioridad `CRITICA`.

---

## 💬 Experiencia Nativa de WhatsApp

El bot incorpora optimizaciones específicas para el canal de WhatsApp:

- **Sanitizador Automático (`whatsapp_formatter.py`):** Convierte negritas Markdown (`**texto**`) a sintaxis nativa de WhatsApp (`*texto*`), repara asteriscos dentro de palabras y transforma enlaces en texto legible.
- **Listas Conversacionales:** Presenta tratamientos y opciones en viñetas limpias (`•`) eliminando los menús numéricos robóticos (*"escribe el número 1 o 2"*).
- **Presencia en Tiempo Real:** Dispara estados de `composing` (*"Escribiendo..."*) y `recording` (*"Grabando audio..."*) inmediatamente al recibir mensajes o notas de voz, dando retroalimentación visual al paciente mientras se genera la respuesta.

---

## ⚙️ Requisitos Previos y Variables de Entorno

### Requisitos del Sistema
- **Docker y Docker Compose** (versión recomendada: Docker 24+, Compose v2).
- O alternativamente: **Python 3.11+**, instancia de **PostgreSQL 15+** y **Qdrant 1.11+**.
- Clave de API de **OpenAI** o **Google Gemini** (gratuita en [Google AI Studio](https://aistudio.google.com/)).

### Configuración del archivo `.env`

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
SESSION_TTL_SECONDS=900
```

---

## 🚀 Guía de Instalación y Uso

### Opción 1: Despliegue con Docker Compose (Recomendada)

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/NexusOdonto_ChatBot_AI.git
cd NexusOdonto_ChatBot_AI

# 2. Configurar variables de entorno
cp .env.example .env

# 3. Iniciar todos los servicios
docker compose up -d --build

# 4. Monitorear logs del agente
docker compose logs -f agente-python
```

### Opción 2: Ejecución Local en Desarrollo

```bash
# 1. Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Iniciar el servidor FastAPI
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

### 📚 Carga de Conocimiento Clínico (RAG)

Para inicializar o reindexar las 24 guías clínicas oficiales en Qdrant:

```bash
python -m app.services.knowledge_ingestion
```

---

### 📱 Vinculación de WhatsApp (Panel QR)

1. Abre en tu navegador: **`http://localhost:8000/qr`**
2. En tu celular, abre **WhatsApp** > **Dispositivos vinculados** > **Vincular un dispositivo**.
3. Escanea el código QR que se actualiza automáticamente en pantalla.
4. El panel detectará la conexión y mostrará el estado **¡WhatsApp Conectado!**.

---

## 🧪 Ejecución de Pruebas Automatizadas

El proyecto cuenta con una suite completa de **41 pruebas unitarias e integradas** que se ejecutan sin depender de servicios externos levantados:

```bash
python -m unittest discover tests
```

**Resultado esperado:**
```text
Ran 41 tests in 0.128s
OK
```

---

## ⌨️ Comandos y Atajos del Usuario

- **`/reset` o `/limpiar`:** Reinicia la memoria de la conversación actual e inicia un nuevo hilo.
- **`/start` o `/inicio`:** Envía el saludo de bienvenida y la presentación de opciones.
- **`"Hablar con un asesor"`:** Escala la conversación a recepción y pausa las respuestas automáticas del bot.
- **`"Volver al bot"` o `"/bot"`:** Reactiva las respuestas automáticas del bot tras haber finalizado la atención con un humano.

---

## 🔌 Endpoints Principales de la API

Acceso a la documentación interactiva en Swagger: **`http://localhost:8000/docs`**

| Método | Ruta | Descripción |
| :--- | :--- | :--- |
| `GET` | `/health` | Chequeo rápido de disponibilidad (liveness probe). |
| `GET` | `/health/deep` | Diagnóstico profundo de conectividad con .NET API, Qdrant y PostgreSQL. |
| `POST` | `/webhook/whatsapp` | Webhook receptor de eventos y mensajes de Evolution API. |
| `GET` | `/qr` | Interfaz visual interactiva para vinculación de WhatsApp con recarga reactiva. |
| `GET` | `/qr/data` | Estado de conexión JSON y string base64 del código QR activo. |
| `POST` | `/qr/restart` | Fuerza la regeneración de una nueva sesión y código QR limpio. |
| `POST` | `/api/v1/handoff/pause` | Pausa la atención del bot para permitir atención humana manual. |
| `POST` | `/api/v1/handoff/resume` | Reanuda el bot tras la intervención del asesor de recepción. |
| `POST` | `/api/v1/reminders/trigger-daily` | Disparo manual de la ronda de recordatorios diarios (8:00 AM). |
| `POST` | `/api/v1/reminders/trigger-30m` | Disparo manual de recordatorios de proximidad (30 minutos antes). |
| `POST` | `/webhook/cache/purge` | Purga manual de la colección de caché semántico en Qdrant. |

---

## 👥 Equipo y Créditos

Proyecto desarrollado para el ecosistema **NexusOdonto**:
- **Backend .NET:** Lógica hospitalaria central y persistencia en Oracle Database.
- **Frontend React:** Panel de gestión clínica y recepción administrativa.
- **Agente IA (Este repositorio):** Orquestación conversacional con LangGraph, RAG y WhatsApp.

---

## 📄 Licencia

Este proyecto está bajo la licencia [MIT](LICENSE).
