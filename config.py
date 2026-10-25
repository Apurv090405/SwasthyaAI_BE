"""
Configuration module for SwasthyaAI application.

Centralizes all configuration settings, environment variables, and constants.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load `.env` from the project root (this file's directory), not from the shell's
# current working directory. That way `uvicorn` works even when started from ~ or another folder.
# `override=True` ensures values in `.env` win over stale exports in the shell (e.g. placeholder keys).
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)


class Settings:
    """Application settings and configuration."""
    
    # API Keys
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    GEMINI_API_KEY: Optional[str] = os.environ.get("GEMINI_API_KEY")
    SERPAPI_KEY: Optional[str] = os.environ.get("SERPAPI_KEY")
    
    # Paths
    LANCEDB_PATH: str = os.environ.get("LANCEDB_PATH", "./lancedb_store")
    
    # Model Configuration
    OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4.1")
    GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    DEFAULT_LLM_PROVIDER: str = os.environ.get("DEFAULT_LLM_PROVIDER", "openai")
    EMBEDDING_MODEL_NAME: str = os.environ.get(
        "EMBEDDING_MODEL", 
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # PDF Processing Defaults
    DEFAULT_CHUNK_SIZE: int = 1000
    DEFAULT_CHUNK_OVERLAP: int = 200
    MIN_PDF_TEXT_LENGTH: int = 50
    
    # RAG Configuration
    DEFAULT_TOP_K: int = 5
    EMBEDDING_BATCH_SIZE: int = 32
    LANCEDB_TABLE_NAME: str = "clinical_guidelines"
    
    # Search Configuration
    SEARCH_RESULTS_LIMIT: int = 5
    SEARCH_LANGUAGE: str = "hi"  # Hindi
    SEARCH_REGION: str = "in"  # India
    
    # Report Generation
    DEFAULT_LANGUAGE: str = "hi"
    DEFAULT_CONSENT_STATUS: str = "verified"
    TOKEN_CHAR_RATIO: int = 4  # Approximate: 1 token ≈ 4 characters
    
    # Voice Agent Configuration (Pipecat/Exotel)
    HOST: str = os.environ.get("HOST", "localhost")
    PORT: int = int(os.environ.get("PORT", 8000))
    EXOTEL_ACCOUNT_SID: Optional[str] = os.environ.get("EXOTEL_SID")
    EXOTEL_AUTH_TOKEN: Optional[str] = os.environ.get("EXOTEL_API_TOKEN")
    EXOTEL_API_KEY: Optional[str] = os.environ.get("EXOTEL_API_KEY")
    EXOTEL_API_BASE: str = os.environ.get("EXOTEL_API_BASE", "https://api.exotel.com/v1")
    DEEPGRAM_API_KEY: Optional[str] = os.environ.get("DEEPGRAM_API_KEY")
    SARVAM_API_KEY: Optional[str] = os.environ.get("SARVAM_API_KEY")
    SARVAM_VOICE_ID: str = os.environ.get("SARVAM_VOICE_ID", "default_voice")
    WEBHOOK_URL: Optional[str] = os.environ.get("WEBHOOK_URL")

    # Supabase (optional — service role for server-side voice_calls / analytics rows)
    SUPABASE_URL: Optional[str] = os.environ.get("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

    # Vobiz Configuration
    VOBIZ_AUTH_ID: Optional[str] = os.environ.get("VOBIZ_AUTH_ID")
    VOBIZ_AUTH_TOKEN: Optional[str] = os.environ.get("VOBIZ_AUTH_TOKEN")
    VOBIZ_PHONE_NUMBER: Optional[str] = os.environ.get("VOBIZ_PHONE_NUMBER")

    # Telnyx Configuration
    TELNYX_API_KEY: Optional[str] = os.environ.get("TELNYX_API_KEY")
    TELNYX_TEXML_APP_ID: Optional[str] = os.environ.get("TELNYX_TEXML_APP_ID")
    # Managed (sub-) account id — only used when TELNYX_MANAGER_SCOPING=true (see below).
    TELNYX_SUB_ACCOUNT_ID: Optional[str] = os.environ.get("TELNYX_SUB_ACCOUNT_ID")
    TELNYX_MANAGED_ACCOUNT_ID: Optional[str] = os.environ.get("TELNYX_MANAGED_ACCOUNT_ID")
    # Set true ONLY if TELNYX_API_KEY belongs to the parent *manager* account. Sends
    # x-managed-account-id. Sub-account API keys must leave this false or Telnyx returns 10009.
    TELNYX_MANAGER_SCOPING: bool = os.environ.get("TELNYX_MANAGER_SCOPING", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    TELNYX_BUSINESS_NAME: Optional[str] = os.environ.get("TELNYX_BUSINESS_NAME")
    TELNYX_PHONE_NUMBER: Optional[str] = os.environ.get("TELNYX_PHONE_NUMBER")
    # DNS hostname only (e.g. abc123.ngrok-free.app). Telnyx must reach /telnyx/answer and wss://.../ws/telnyx/
    # — set this when HOST is localhost. Optional path may be stripped if pasted by mistake.
    TELNYX_PUBLIC_HOST: Optional[str] = os.environ.get("TELNYX_PUBLIC_HOST")
    
    def __init__(self):
        """Validate required settings on initialization."""
        if not self.OPENAI_API_KEY and not self.GEMINI_API_KEY:
            raise RuntimeError("At least one of OPENAI_API_KEY or GEMINI_API_KEY must be set")
        if self.DEFAULT_LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY must be set when DEFAULT_LLM_PROVIDER is 'openai'")
        if self.DEFAULT_LLM_PROVIDER == "gemini" and not self.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY must be set when DEFAULT_LLM_PROVIDER is 'gemini'")
    
    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load embedding model."""
        if not hasattr(self, "_embedding_model"):
            print(f"Loading embedding model: {self.EMBEDDING_MODEL_NAME}")
            self._embedding_model = SentenceTransformer(self.EMBEDDING_MODEL_NAME)
            print("Embedding model loaded successfully")
        return self._embedding_model


# Global settings instance
settings = Settings()

