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
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    # Datos de conexión y configuración de la colección vectorial de Qdrant.
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "conocimiento_clinico"
    embedding_dimension: int = 1536

    # Configuración de las integraciones que se habilitarán más adelante.
    evolution_api_url: str = "http://localhost:8080"
    evolution_api_key: str = ""
    instance_name: str = "Nexus_Odonto"

    # Configuración de la API externa y reintentos de comunicación.
    agent_internal_secret: str = ""
    dotnet_api_timeout: int = 30
    dotnet_max_retries: int = 3
    dotnet_retry_backoff_seconds: float = 0.5

    # Hora local para enviar recordatorios de citas del día siguiente.
    reminder_schedule_hour: int = 8
    reminder_schedule_minute: int = 0
    reminder_timezone: str = "America/Bogota"

    # Configuración de memoria, tiempo de sesión y flujo conversacional.
    session_store: str = "memory"
    session_ttl_seconds: int = 3600
    session_cleanup_interval_seconds: int = 300
    graph_timeout_seconds: int = 60
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
    def _validate_production_requirements(self) -> "Settings":
        # En producción no se permite iniciar sin la clave de OpenAI.
        if self.app_env.lower() == "production" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY es obligatoria en producción")
        return self


# Instancia única reutilizada por el resto de la aplicación.
settings = Settings()