"""
Pydantic models for request/response schemas.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class AgentChatTurn(BaseModel):
    """One prior message in an agent chat thread."""

    role: Literal["user", "assistant"]
    content: str


class AgentChatRequest(BaseModel):
    """Chat with a single clinical agent (not the full supervisor)."""

    agent_id: str = Field(..., description="Agent id, e.g. diagnosis_agent, rag_agent")
    message: str = Field(..., description="Latest user message")
    history: list[AgentChatTurn] = Field(
        default_factory=list,
        description="Prior turns in order: user/assistant pairs",
    )
    llm_provider: Literal["openai", "gemini"] = Field(
        default="openai",
        description="LLM backend for this request",
    )


class AgentChatResponse(BaseModel):
    reply: str
    agent_id: str


class TranscriptRequest(BaseModel):
    """Request model for transcript analysis."""
    userid: str
    transcript: str
    location: Optional[str] = None  # Optional, useful for local outbreak search/filters
    llm_provider: Literal["openai", "gemini"] = Field(
        default="openai",
        description="LLM provider to use for this request. 'openai' (default) or 'gemini'.",
    )


class PDFUploadResponse(BaseModel):
    """Response model for PDF upload endpoint."""
    success: bool
    message: str
    document_id: Optional[str] = None
    chunks_added: Optional[int] = None


class TextIngestRequest(BaseModel):
    """Plain-text / markdown ingest for LanceDB RAG (same pipeline as PDF after extraction)."""

    text: str = Field(..., description="Raw document text")
    source: str = Field(default="pasted_doc", description="Knowledge-base / corpus label stored on chunks")
    chunk_size: Optional[int] = Field(default=None, description="Words per chunk; default from settings")
    overlap: Optional[int] = Field(default=None, description="Word overlap; default from settings")


# ── Dashboard Models ──────────────────────────────────────────────────

class ApproveReportRequest(BaseModel):
    """Approve or revoke a clinical report."""
    call_id: str
    user_id: str
    notes: Optional[str] = None


class AgentConfigRequest(BaseModel):
    """Save calling agent configuration."""
    user_id: str
    name: str = Field(default="Default")
    config: dict = Field(..., description="Full agent config object (telco, llm, voice, etc.)")


class UserPreferencesRequest(BaseModel):
    """Save user preferences (language, voice, theme)."""
    user_id: str
    preferences: dict = Field(..., description="Preferences object (language, voice_uri, theme, etc.)")


class PhoneRegistryRequest(BaseModel):
    """Add a phone number to the manual registry."""
    user_id: str
    phone_number: str
    provider: str = Field(default="manual")
    label: Optional[str] = None
    is_default: bool = False


class PhoneRegistryDeleteRequest(BaseModel):
    """Delete a phone registry entry."""
    entry_id: str
    user_id: str

