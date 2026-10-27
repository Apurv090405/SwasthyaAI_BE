"""
Configuration module for SwasthyaAI application.

Centralizes all configuration settings, environment variables, and constants.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

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
    # Post-call report: "fast" = fixed order with parallel steps (services/clinical_pipeline.py);
    # "supervisor" = the LangGraph supervisor routing all ten agents (slower, more LLM calls).
    CLINICAL_MODE: str = os.environ.get("CLINICAL_MODE", "fast").strip().lower()
    # Live calls use smaller, faster models than the post-call report agents
    # (time to first token dominates voice-to-voice latency).
    VOICE_OPENAI_MODEL: str = os.environ.get("VOICE_OPENAI_MODEL", "gpt-4.1-mini")
    VOICE_GEMINI_MODEL: str = os.environ.get("VOICE_GEMINI_MODEL", "gemini-2.5-flash-lite")
    VOICE_MAX_TOKENS: int = int(os.environ.get("VOICE_MAX_TOKENS", 150))
    VOICE_TEMPERATURE: float = float(os.environ.get("VOICE_TEMPERATURE", 0.3))
    # API embedding model; empty = provider default (see llm_factory.get_embeddings)
    EMBEDDING_MODEL_NAME: str = os.environ.get("EMBEDDING_MODEL", "")
    
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
    # Browser origins allowed to call the API (comma-separated); add your tunnel / deployed FE host
    CORS_ORIGINS: list = [
        o.strip().rstrip("/")
        for o in os.environ.get(
            "CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if o.strip()
    ]
    PORT: int = int(os.environ.get("PORT", 8000))
    EXOTEL_ACCOUNT_SID: Optional[str] = os.environ.get("EXOTEL_SID")
    EXOTEL_AUTH_TOKEN: Optional[str] = os.environ.get("EXOTEL_API_TOKEN")
    EXOTEL_API_KEY: Optional[str] = os.environ.get("EXOTEL_API_KEY")
    EXOTEL_API_BASE: str = os.environ.get("EXOTEL_API_BASE", "https://api.exotel.com/v1")
    DEEPGRAM_API_KEY: Optional[str] = os.environ.get("DEEPGRAM_API_KEY")
    SARVAM_API_KEY: Optional[str] = os.environ.get("SARVAM_API_KEY")
    # bulbul:v3 speaker (e.g. priya, ritu, neha, kavya, shubh, aditya); v2 names like anushka no longer work
    SARVAM_VOICE_ID: str = os.environ.get("SARVAM_VOICE_ID", "priya")
    # bulbul:v3 (default, 11 languages) or bulbul:v4-flash (lower latency, no Malayalam/Odia;
    # needs a v4-flash SARVAM_VOICE_ID from Sarvam's docs)
    SARVAM_TTS_MODEL: str = os.environ.get("SARVAM_TTS_MODEL", "bulbul:v3")
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

    # VoiceLink (India DIDs) — platform reseller account used for number purchase
    VOICELINK_USERNAME: Optional[str] = os.environ.get("VOICELINK_USERNAME")
    VOICELINK_PASSWORD: Optional[str] = os.environ.get("VOICELINK_PASSWORD")

    # Bolna (second call engine — Bolna runs the live call, our clinical agents run after it).
    # Create the agent once in the Bolna dashboard and set its webhook to <public host>/bolna/webhook.
    BOLNA_API_KEY: Optional[str] = os.environ.get("BOLNA_API_KEY")
    BOLNA_AGENT_ID: Optional[str] = os.environ.get("BOLNA_AGENT_ID")
    BOLNA_FROM_NUMBER: Optional[str] = os.environ.get("BOLNA_FROM_NUMBER")
    BOLNA_API_BASE: str = os.environ.get("BOLNA_API_BASE", "https://api.bolna.ai")

    def __init__(self):
        """Validate required settings on initialization."""
        if not self.OPENAI_API_KEY and not self.GEMINI_API_KEY:
            raise RuntimeError("At least one of OPENAI_API_KEY or GEMINI_API_KEY must be set")
        if self.DEFAULT_LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY must be set when DEFAULT_LLM_PROVIDER is 'openai'")
        if self.DEFAULT_LLM_PROVIDER == "gemini" and not self.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY must be set when DEFAULT_LLM_PROVIDER is 'gemini'")
    


# Global settings instance
settings = Settings()

