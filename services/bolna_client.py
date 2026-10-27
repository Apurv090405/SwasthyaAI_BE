"""
Minimal async client for the Bolna API (https://www.bolna.ai/docs/quickstarts/api).

Bolna runs the live call (its own telephony + STT/LLM/TTS); we only place calls and read back
the finished execution, whose transcript feeds our post-call clinical agents.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from loguru import logger

from config import settings

# Statuses after which an execution will not change any more (per Bolna's polling guide).
# `call-disconnected` is not final: duration/recording arrive with `completed` a few seconds later.
FINAL_STATUSES = {
    "completed",
    "no-answer",
    "busy",
    "failed",
    "canceled",
    "stopped",
    "error",
    "balance-low",
}


class BolnaError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"Bolna API {status}: {message}")
        self.status = status


def is_configured() -> bool:
    return bool(settings.BOLNA_API_KEY and settings.BOLNA_AGENT_ID)


def _headers() -> Dict[str, str]:
    if not settings.BOLNA_API_KEY:
        raise BolnaError(500, "BOLNA_API_KEY is not set")
    return {"Authorization": f"Bearer {settings.BOLNA_API_KEY}", "Content-Type": "application/json"}


async def _request(method: str, path: str, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from api.voice_routes import get_global_session

    session = await get_global_session()
    url = f"{settings.BOLNA_API_BASE.rstrip('/')}{path}"
    async with session.request(method, url, json=json, headers=_headers()) as response:
        try:
            data = await response.json(content_type=None)
        except Exception:
            data = {"message": (await response.text())[:500]}
        if response.status >= 400:
            message = data.get("message") if isinstance(data, dict) else str(data)
            logger.warning(f"[Bolna] {method} {path} → {response.status}: {message}")
            raise BolnaError(response.status, str(message))
        return data if isinstance(data, dict) else {"data": data}


async def place_call(
    to: str,
    *,
    from_number: Optional[str] = None,
    user_data: Optional[Dict[str, Any]] = None,
    agent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """POST /call → {"message", "status": "queued", "execution_id"}."""
    body: Dict[str, Any] = {
        "agent_id": agent_id or settings.BOLNA_AGENT_ID,
        "recipient_phone_number": to,
    }
    if from_number or settings.BOLNA_FROM_NUMBER:
        body["from_phone_number"] = from_number or settings.BOLNA_FROM_NUMBER
    if user_data:
        body["user_data"] = user_data
    if not body["agent_id"]:
        raise BolnaError(500, "BOLNA_AGENT_ID is not set")
    return await _request("POST", "/call", body)


async def get_execution(execution_id: str) -> Dict[str, Any]:
    """GET /executions/{id} → full execution (status, transcript, telephony_data, latency_data…)."""
    return await _request("GET", f"/executions/{execution_id}")
