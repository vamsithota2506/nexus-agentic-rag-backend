import os
from typing import List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Nexus Agentic RAG Enterprise Backend"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]

    # Retrieval Configuration
    DENSE_WEIGHT: float = 0.6
    SPARSE_WEIGHT: float = 0.4
    DEFAULT_TOP_K: int = 5
    RRF_K_CONSTANT: int = 60
    SIMILARITY_THRESHOLD: float = 0.35

    # Document Chunking Configuration
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100

    # LLM & Embedding Providers ("mock", "gemini", "openai")
    LLM_PROVIDER: str = "mock"
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # Self-Correction Agent Parameters
    ENABLE_SELF_CORRECTION: bool = True
    MAX_REWRITE_ATTEMPTS: int = 2
    CONFIDENCE_HALLUCINATION_THRESHOLD: float = 0.70

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore"
    }


settings = Settings()
