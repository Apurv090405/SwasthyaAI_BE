"""Bolna as a second call engine.

Bolna runs the live call; when its execution reaches ``completed`` we convert it into the same
payload our own pipelines send (see ``api/voice_routes.send_call_webhook``) and run the usual
post-call path: saved webhook event → Supabase ``voice_calls`` row → 10-agent clinical report.

  POST /bolna/outbound                  place a call through the configured Bolna agent
  POST /bolna/webhook                   Bolna's status webhook (set it on the agent)
  POST /bolna/executions/{id}/sync      pull a finished call by id (no public URL needed)
  GET  /bolna/executions/{id}           raw execution, for debugging
  GET  /bolna/status                    whether Bolna is configured
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger
from pydantic import BaseModel

from auth import call_owner, optional_user
from config import settings
from services import bolna_client
from services.bolna_client import FINAL_STATUSES, BolnaError

bolna_router = APIRouter(prefix="/bolna", tags=["bolna"])

# execution_id → what we knew when placing the call (user, provider choice)
BOLNA_CALLS: Dict[str, Dict[str, Any]] = {}
# executions already handed to the post-call pipeline (Bolna sends several webhooks per call)
_PROCESSED: set[str] = set()

_SPEAKER_RE = re.compile(r"^\s*(assistant|agent|bot|user|customer|human)\s*:\s*(.*)$", re.IGNORECASE)
_ROLE = {"assistant": "assistant", "agent": "assistant", "bot": "assistant"}


class BolnaOutboundParams(BaseModel):
    to: str
    from_number: Optional[str] = None
    user_id: Optional[str] = None
    llm_provider: str = "openai"  # used for our post-call clinical agents


def _parse_transcript(raw: Any) -> List[Dict[str, str]]:
    """Bolna's transcript is usually 'assistant: …\\nuser: …' text; accept a message list too."""
    if isinstance(raw, list):
        messages = []
        for m in raw:
            if isinstance(m, dict) and m.get("content"):
                role = _ROLE.get(str(m.get("role", "")).lower(), "user")
                messages.append({"role": role, "content": str(m["content"]).strip()})
        return messages

    messages: List[Dict[str, str]] = []
    for line in str(raw or "").splitlines():
        match = _SPEAKER_RE.match(line)
        if match:
            role = _ROLE.get(match.group(1).lower(), "user")
            messages.append({"role": role, "content": match.group(2).strip()})
        elif line.strip() and messages:
            messages[-1]["content"] = f"{messages[-1]['content']} {line.strip()}"
    return [m for m in messages if m["content"]]


def _user_data(execution: Dict[str, Any]) -> Dict[str, Any]:
    ctx = execution.get("context_details") or {}
    for candidate in (execution.get("user_data"), ctx.get("recipient_data"), ctx.get("user_data")):
        if isinstance(candidate, dict):
            return candidate
    return {}


def execution_to_payload(execution: Dict[str, Any]) -> Dict[str, Any]:
    """Map a Bolna execution onto our call-webhook payload shape."""
    execution_id = str(execution.get("id") or execution.get("execution_id") or "unknown")
    known = BOLNA_CALLS.get(execution_id, {})
    user_data = _user_data(execution)
    telephony = execution.get("telephony_data") or {}

    messages = _parse_transcript(execution.get("transcript"))
    full_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in messages
    )

    latency = execution.get("latency_data") or {}
    duration = telephony.get("duration") or execution.get("conversation_duration")
    call_type = (telephony.get("call_type") or known.get("direction") or "inbound").lower()

    return {
        "stream_id": execution_id,
        "call_id": execution_id,
        "call_info": {
            "from": execution.get("user_number") or telephony.get("from_number") or "unknown",
            "to": execution.get("agent_number") or telephony.get("to_number") or "unknown",
            "direction": "outbound" if "out" in call_type else "inbound",
            "status": execution.get("status", "completed"),
            "supabase_user_id": known.get("supabase_user_id") or user_data.get("supabase_user_id"),
            "llm_provider": known.get("llm_provider") or user_data.get("llm_provider") or "openai",
            "provider": "bolna",
            "engine": "bolna",
            "telco": telephony.get("provider"),
            "recording_url": telephony.get("recording_url"),
        },
        "transcript": {"full_text": full_text, "message_count": len(messages), "messages": messages},
        "metrics": {
            "duration_secs": duration,
            "total_cost": execution.get("total_cost"),
            "cost_breakdown": execution.get("cost_breakdown"),
            "termination_reason": execution.get("termination_reason"),
            "hangup_by": telephony.get("hangup_by"),
        },
        "usage_metrics": {},
        # Bolna reports its own timings (e.g. time_to_first_audio); keep them under the engine name
        "latency_metrics": {"engine": "bolna", "bolna": latency} if latency else {"engine": "bolna"},
        "timestamp": execution.get("updated_at") or datetime.utcnow().isoformat() + "Z",
        "bolna_summary": execution.get("summary"),
        "bolna_extracted_data": execution.get("extracted_data"),
    }


async def _ingest(execution: Dict[str, Any]) -> Dict[str, Any]:
    """Save + run the post-call pipeline once per completed execution."""
    from webhook import ingest_call_payload

    execution_id = str(execution.get("id") or execution.get("execution_id") or "")
    status = execution.get("status")
    if status != "completed":
        return {"status": "ignored", "execution_status": status}
    if not execution_id or execution_id in _PROCESSED:
        return {"status": "duplicate", "execution_id": execution_id}
    _PROCESSED.add(execution_id)

    payload = execution_to_payload(execution)
    saved_file = ingest_call_payload(payload)
    BOLNA_CALLS.pop(execution_id, None)
    logger.info(
        f"[Bolna] execution {execution_id} completed: {payload['transcript']['message_count']} "
        f"messages → post-call pipeline (saved {saved_file})"
    )
    return {"status": "processing", "execution_id": execution_id, "saved_file": saved_file}


def _http_error(e: BolnaError) -> HTTPException:
    return HTTPException(status_code=e.status if 400 <= e.status < 600 else 502, detail=str(e))


@bolna_router.get("/status")
async def bolna_status():
    return {
        "configured": bolna_client.is_configured(),
        "agent_id": settings.BOLNA_AGENT_ID if bolna_client.is_configured() else None,
        "from_number": settings.BOLNA_FROM_NUMBER,
        "active_calls": len(BOLNA_CALLS),
    }


@bolna_router.post("/outbound")
async def bolna_outbound(params: BolnaOutboundParams, token_user: Optional[str] = Depends(optional_user)):
    """Place an outbound health call through the Bolna agent."""
    if not bolna_client.is_configured():
        raise HTTPException(status_code=400, detail="Bolna is not configured: set BOLNA_API_KEY and BOLNA_AGENT_ID")
    user_id = call_owner(token_user, params.user_id)
    try:
        result = await bolna_client.place_call(
            params.to.strip(),
            from_number=(params.from_number or "").strip() or None,
            # available to the Bolna prompt as {supabase_user_id}; also lets the webhook map back
            user_data={"supabase_user_id": user_id or "", "llm_provider": params.llm_provider},
        )
    except BolnaError as e:
        raise _http_error(e)

    execution_id = result.get("execution_id")
    if execution_id:
        BOLNA_CALLS[execution_id] = {
            "supabase_user_id": user_id,
            "llm_provider": params.llm_provider,
            "direction": "outbound",
            "to": params.to,
        }
    return {
        "status": result.get("status", "queued"),
        "message": f"Call queued via Bolna ({execution_id})",
        "execution_id": execution_id,
        "engine": "bolna",
    }


@bolna_router.post("/webhook")
async def bolna_webhook(request: Request):
    """Bolna posts the execution on every status change; only `completed` is processed."""
    try:
        execution = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    if not isinstance(execution, dict):
        raise HTTPException(status_code=400, detail="Expected an execution object")
    logger.info(f"[Bolna] webhook: execution={execution.get('id')} status={execution.get('status')}")
    return await _ingest(execution)


@bolna_router.get("/executions/{execution_id}")
async def bolna_get_execution(execution_id: str):
    try:
        return await bolna_client.get_execution(execution_id)
    except BolnaError as e:
        raise _http_error(e)


@bolna_router.post("/executions/{execution_id}/sync")
async def bolna_sync_execution(execution_id: str, wait_secs: int = 0):
    """Fetch an execution and process it if completed; optionally poll up to `wait_secs`."""
    deadline = asyncio.get_event_loop().time() + max(0, min(wait_secs, 600))
    while True:
        try:
            execution = await bolna_client.get_execution(execution_id)
        except BolnaError as e:
            raise _http_error(e)
        status = execution.get("status")
        if status in FINAL_STATUSES or asyncio.get_event_loop().time() >= deadline:
            break
        await asyncio.sleep(5)

    result = await _ingest(execution)
    return {**result, "execution_status": status}
