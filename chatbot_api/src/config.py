from typing import Any, Literal, Optional
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM provider configuration ('openai' | 'ollama')
    LLM_PROVIDER: Literal["openai", "ollama"] = "openai"

    # Required Neo4j connection parameters (fail fast at startup if missing)
    NEO4J_URI: str
    NEO4J_USERNAME: str
    NEO4J_PASSWORD: str

    # OpenAI configuration (required when LLM_PROVIDER is 'openai')
    OPENAI_API_KEY: Optional[str]

    # Ollama configuration
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    OLLAMA_MODEL: str = "llama3.1"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"

    # Redis Cypher cache configuration
    REDIS_URL: str = "redis://redis:6379/0"
    REDIS_CYPHER_CACHE_TTL: int = 86400  # 24 hours

    # Model parameters with defaults
    HOSPITAL_AGENT_MODEL: str = "gpt-4o-mini"
    HOSPITAL_CYPHER_MODEL: str = "gpt-4o-mini"
    HOSPITAL_QA_MODEL: str = "gpt-4o-mini"

    # Cypher few-shot example index settings
    NEO4J_CYPHER_EXAMPLES_INDEX_NAME: str = "questions"
    NEO4J_CYPHER_EXAMPLES_NODE_NAME: str = "Question"
    NEO4J_CYPHER_EXAMPLES_TEXT_NODE_PROPERTY: str = "question"
    NEO4J_CYPHER_EXAMPLES_METADATA_NAME: str = "cypher"

    @model_validator(mode="before")
    @classmethod
    def handle_provider_credentials(cls, values: Any) -> Any:
        if isinstance(values, dict):
            provider = values.get("LLM_PROVIDER", "openai")
            # If running offline with Ollama, OPENAI_API_KEY is not required
            if provider == "ollama" and "OPENAI_API_KEY" not in values:
                values["OPENAI_API_KEY"] = None
        return values


settings = Settings()
