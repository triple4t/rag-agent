"""Production Configuration Management."""

from functools import lru_cache
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    APP_NAME: str = "RAG System API"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json or text

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # API Keys (Required)
    OPENAI_API_KEY: str = Field(default="")
    AZURE_API_KEY: str = Field(default="")
    AZURE_MISTRAL_OCR_ENDPOINT: str = Field(default="")
    AZURE_MISTRAL_MODEL: str = "mistral-document-ai-2505"

    # Optional API Keys
    COHERE_API_KEY: Optional[str] = None
    LANGCHAIN_API_KEY: Optional[str] = None

    # LangSmith Configuration
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "rag-system"
    LANGCHAIN_ENDPOINT: Optional[str] = None

    # Model Configuration
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536  # text-embedding-3-small dimension
    LLM_MODEL: str = "gpt-4o"
    LLM_TEMPERATURE: float = 0.0
    RERANK_MODEL: str = "rerank-english-v3.0"

    # Search Configuration
    VECTOR_WEIGHT: float = 0.5
    KEYWORD_WEIGHT: float = 0.5
    TOP_K_SEARCH: int = 15  # Increased for better coverage, especially for comparison queries
    TOP_K_RERANK: int = 8   # Increased to provide more context for complex queries
    RRF_K: int = 60  # RRF constant

    # Chunking Configuration
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Qdrant Configuration
    QDRANT_URL: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "rag_docs"
    QDRANT_TIMEOUT: int = 30

    # OCR Configuration
    OCR_ENABLED: bool = True
    OCR_LANGUAGE: str = "eng"
    OCR_DPI: int = 300

    # Performance Configuration
    MAX_CONTEXT_TOKENS: int = 2000
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    REQUEST_TIMEOUT: int = 60

    # Caching Configuration
    CACHE_ENABLED: bool = True
    REDIS_URL: Optional[str] = None
    CACHE_TTL: int = 3600  # 1 hour

    # Success Metrics Targets (from roadmap lines 96-100)
    TARGET_MRR: float = 0.7
    TARGET_QUALITY: float = 0.75
    TARGET_LATENCY_P95: float = 2.0  # seconds
    TARGET_COST_PER_QUERY: float = 0.01  # dollars

    # Security
    SECRET_KEY: str = Field(default="change-me-in-production")
    CORS_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ]
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience alias
settings = get_settings()

