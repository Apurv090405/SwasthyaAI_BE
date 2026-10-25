"""
LLM Factory module for SwasthyaAI.

Provides a unified interface to create LLM instances for either
OpenAI (default) or Google Gemini, selected per-request via a
contextvars.ContextVar so concurrent async calls remain isolated.

Usage:
    # In an endpoint handler, before calling any agent/tool:
    set_llm_provider("gemini")          # or "openai"

    # Inside tools / agents (reads from async context automatically):
    llm = get_llm()                     # uses current context provider
    llm = get_llm("openai")             # override explicitly
"""

import contextvars
from typing import Optional

# Per-async-task context variable — thread-safe and coroutine-safe
_llm_provider_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "llm_provider", default="openai"
)


def set_llm_provider(provider: str) -> None:
    """Set the LLM provider for the current async context/task."""
    _llm_provider_var.set(provider)


def get_current_provider() -> str:
    """Return the LLM provider set for the current async context."""
    return _llm_provider_var.get()


def get_llm(provider: Optional[str] = None):
    """
    Return a LangChain chat model for the requested provider.

    Args:
        provider: 'openai' or 'gemini'.
                  Falls back to the context-var value (set by set_llm_provider),
                  which itself defaults to settings.DEFAULT_LLM_PROVIDER.

    Returns:
        A LangChain BaseChatModel instance (ChatOpenAI or ChatGoogleGenerativeAI).

    Raises:
        RuntimeError: If the required API key is not configured.
        ValueError: If an unknown provider name is given.
    """
    from config import settings

    resolved = provider or get_current_provider()

    if resolved == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file to use Gemini."
            )
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
        )

    if resolved == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file to use OpenAI."
            )
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
        )

    raise ValueError(
        f"Unknown LLM provider '{resolved}'. Supported values: 'openai', 'gemini'."
    )


def get_openai_compat_params(provider: Optional[str] = None) -> dict:
    """
    Return kwargs needed to initialise Pipecat's OpenAILLMService for the
    given provider.  Gemini exposes an OpenAI-compatible REST endpoint so
    Pipecat can use it without a dedicated service class.

    Returns:
        dict with keys: api_key, model, and optionally base_url.
    """
    from config import settings

    resolved = provider or get_current_provider()

    if resolved == "gemini":
        if not settings.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file to use Gemini."
            )
        return {
            "api_key": settings.GEMINI_API_KEY,
            "model": settings.GEMINI_MODEL,
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        }

    # Default: OpenAI
    if not settings.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file to use OpenAI."
        )
    return {
        "api_key": settings.OPENAI_API_KEY,
        "model": settings.OPENAI_MODEL,
    }
