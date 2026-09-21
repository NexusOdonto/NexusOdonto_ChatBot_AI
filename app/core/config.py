from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Configuración general de la aplicación.
    app_env: str = "development"  # Ambiente actual de ejecución
    app_name: str = "Nexus Odonto ChatBot AI"
    debug: bool = True
    log_level: str = "INFO"

    # Host y puerto donde se ejecuta FastAPI.
    host: str = "0.0.0.0" # Host donde correrá FastAPI
    port: int = 8000 # Puerto de la aplicación

    # Configuración del modelo de lenguaje y del modelo de embeddings.
    llm_provider: str = Field(
        default="gemini",
        validation_alias=AliasChoices("LLM_PROVIDER", "AI_PROVIDER"),
    )
    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"
    gemini_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )
    gemini_model: str = "gemini-1.5-flash"
    embedding_provider: str = Field(
        default="gemini",
        validation_alias=AliasChoices("EMBEDDING_PROVIDER"),
    )
    embedding_model: str = "text-embedding-3-small"
    gemini_embedding_model: str = "models/gemini-embedding-001"

    # Datos de conexión y configuración de la colección vectorial de Qdrant.
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "conocimiento_clinico"
    embedding_dimension: int = 3072

    # Configuración de Caché Semántico en Qdrant
    semantic_cache_enabled: bool = True
    semantic_cache_collection: str = "semantic_cache"
    semantic_cache_threshold: float = 0.90

    # Configuración de las integraciones que se habilitarán más adelante.
    evolution_api_url: str = "http://localhost:8080"
    evolution_api_key: str = ""
    instance_name: str = "Nexus_Odonto"
    evolution_instance_name: str = "Nexus_Odonto"

    # Secreto y encabezado usados para autenticar webhooks de Evolution API.
    webhook_secret: str = ""
    webhook_signature_header: str = "x-webhook-signature"

    # Configuración de la API externa y reintentos de comunicación.
    agent_internal_secret: str = "TOKEN_SECRETO_INTERNO_NET_2026"
    dotnet_auth_login: str = Field(
        default="",
        validation_alias=AliasChoices("DOTNET_AUTH_LOGIN", "DOTNET_API_USER"),
    )
    dotnet_auth_password: str = Field(
        default="",
        validation_alias=AliasChoices("DOTNET_AUTH_PASSWORD", "DOTNET_API_PASSWORD"),
    )
    dotnet_api_timeout: int = 30
    dotnet_max_retries: int = 3
    dotnet_retry_backoff_seconds: float = 0.5

    # Hora local para enviar recordatorios de citas del día siguiente.
    reminder_schedule_hour: int = 8
    reminder_schedule_minute: int = 0
    reminder_timezone: str = "America/Bogota"

    # Configuración de memoria, tiempo de sesión y flujo conversacional.
    session_store: str = "memory"
    session_ttl_seconds: int = 900  # 15 minutos de inactividad antes de expirar la sesión
    session_cleanup_interval_seconds: int = 300
    graph_timeout_seconds: int = 60
    postgres_checkpoint_url: str = "postgresql://bot_user:bot_password@localhost:5433/bot_memory"
    rag_min_confidence: float = 0.50
    rag_candidate_count: int = 4
    rag_hybrid_enabled: bool = True
    rag_bm25_weight: float = 0.4
    rag_dense_weight: float = 0.6
    rag_rrf_k: int = 60
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    allowed_origins: str = "*"

    # Carga las variables desde .env y permite ignorar variables adicionales.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,
    )

    # Acepta el nombre actual DOTNET_API_URL y el nombre alternativo anterior.
    dotnet_api_url: str = Field(
        default="http://localhost:5000/api/v1",
        validation_alias=AliasChoices("DOTNET_API_URL", "DOTNET_API_BASE_URL"),
    )

    @property
    def allowed_origins_list(self) -> list[str]:
        # Convierte los orígenes separados por coma en una lista para CORS.
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    @model_validator(mode="after")
    def _validate_settings(self) -> "Settings":
        prov = (self.llm_provider or "openai").lower().strip()
        emb_prov = (self.embedding_provider or "openai").lower().strip()

        # Sincronizar dimensiones según el proveedor de embeddings
        if emb_prov == "gemini":
            if self.embedding_dimension in (1536, 768):
                self.embedding_dimension = 3072
        elif emb_prov == "openai":
            if self.embedding_dimension in (3072, 768):
                self.embedding_dimension = 1536

        # En producción se valida la clave del proveedor que esté configurado
        if self.app_env.lower() == "production":
            if prov == "gemini" and not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY es obligatoria en producción cuando LLM_PROVIDER='gemini'")
            elif prov == "openai" and not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY es obligatoria en producción cuando LLM_PROVIDER='openai'")
        return self


# Instancia única reutilizada por el resto de la aplicación.
settings = Settings()