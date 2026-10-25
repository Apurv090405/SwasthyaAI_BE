"""Supabase REST operations for dashboard: analytics, approvals, agent configs, preferences, phone registry."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger

from config import settings
from services.supabase_voice import _credentials_present, _headers, verify_supabase_connection


def _base_url() -> str:
    return settings.SUPABASE_URL.rstrip("/")


async def _get(path: str, *, params: str = "") -> Optional[Any]:
    if not _credentials_present():
        return None
    ok, _ = await verify_supabase_connection()
    if not ok:
        return None
    url = f"{_base_url()}/rest/v1/{path}"
    if params:
        url = f"{url}?{params}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers={**_headers(prefer="return=representation"), "Accept": "application/json"},
            ) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    logger.warning(f"Supabase GET {path} failed {resp.status}: {text[:300]}")
                    return None
                return json.loads(text) if text.strip() else []
    except Exception as e:
        logger.warning(f"Supabase GET {path} error: {e}")
        return None


async def _post(path: str, body: Dict[str, Any]) -> Optional[Any]:
    if not _credentials_present():
        return None
    ok, _ = await verify_supabase_connection()
    if not ok:
        return None
    url = f"{_base_url()}/rest/v1/{path}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=_headers(), json=body) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    logger.warning(f"Supabase POST {path} failed {resp.status}: {text[:400]}")
                    return None
                return json.loads(text) if text.strip() else None
    except Exception as e:
        logger.warning(f"Supabase POST {path} error: {e}")
        return None


async def _patch(path: str, body: Dict[str, Any], *, filter_str: str) -> bool:
    if not _credentials_present():
        return False
    ok, _ = await verify_supabase_connection()
    if not ok:
        return False
    url = f"{_base_url()}/rest/v1/{path}?{filter_str}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.patch(
                url, headers=_headers(prefer="return=minimal"), json=body
            ) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    logger.warning(f"Supabase PATCH {path} failed {resp.status}: {text[:300]}")
                    return False
                return True
    except Exception as e:
        logger.warning(f"Supabase PATCH {path} error: {e}")
        return False


async def _delete(path: str, *, filter_str: str) -> bool:
    if not _credentials_present():
        return False
    ok, _ = await verify_supabase_connection()
    if not ok:
        return False
    url = f"{_base_url()}/rest/v1/{path}?{filter_str}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.delete(url, headers=_headers(prefer="return=minimal")) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    logger.warning(f"Supabase DELETE {path} failed {resp.status}: {text[:300]}")
                    return False
                return True
    except Exception as e:
        logger.warning(f"Supabase DELETE {path} error: {e}")
        return False


# ── Analytics (from voice_calls table) ──────────────────────────────

async def get_dashboard_stats() -> Dict[str, Any]:
    """Aggregate KPI stats from voice_calls."""
    rows = await _get("voice_calls", params="select=id,created_at,direction,orchestrator_status,ended_at,from_number,to_number,provider,approved&order=created_at.desc&limit=500")
    if not rows:
        rows = []

    now = datetime.now(timezone.utc)
    today_str = now.strftime("%Y-%m-%d")

    total = len(rows)
    today_calls = sum(1 for r in rows if (r.get("created_at") or "")[:10] == today_str)
    completed = sum(1 for r in rows if r.get("orchestrator_status") == "completed")
    failed = sum(1 for r in rows if r.get("orchestrator_status") == "failed")
    approved = sum(1 for r in rows if r.get("approved"))
    inbound = sum(1 for r in rows if r.get("direction") == "inbound")
    outbound = sum(1 for r in rows if r.get("direction") == "outbound")

    # Yesterday comparison
    yesterday_str = (now.replace(hour=0, minute=0, second=0) - __import__("datetime").timedelta(days=1)).strftime("%Y-%m-%d")
    yesterday_calls = sum(1 for r in rows if (r.get("created_at") or "")[:10] == yesterday_str)

    return {
        "total_calls": total,
        "today_calls": today_calls,
        "yesterday_calls": yesterday_calls,
        "completed": completed,
        "failed": failed,
        "pending": total - completed - failed,
        "approved": approved,
        "inbound": inbound,
        "outbound": outbound,
        "success_rate": round(completed / max(total, 1) * 100, 1),
        "today_trend": today_calls - yesterday_calls,
    }


async def get_call_volume(days: int = 7) -> List[Dict[str, Any]]:
    """Daily call volume breakdown for chart."""
    rows = await _get(
        "voice_calls",
        params=f"select=created_at,direction&order=created_at.desc&limit={days * 100}"
    )
    if not rows:
        return []

    from collections import defaultdict
    daily: Dict[str, Dict[str, int]] = defaultdict(lambda: {"inbound": 0, "outbound": 0, "web": 0})

    for r in rows:
        date = (r.get("created_at") or "")[:10]
        if not date:
            continue
        direction = (r.get("direction") or "").lower()
        if direction == "inbound":
            daily[date]["inbound"] += 1
        elif direction == "outbound":
            daily[date]["outbound"] += 1
        else:
            daily[date]["web"] += 1

    result = []
    for date in sorted(daily.keys())[-days:]:
        result.append({"period": date, **daily[date]})
    return result


async def get_triage_outcomes() -> List[Dict[str, Any]]:
    """Triage outcome distribution from clinical reports."""
    rows = await _get(
        "voice_calls",
        params="select=orchestrator_status,clinical_report&orchestrator_status=eq.completed&limit=200"
    )
    if not rows:
        return [
            {"name": "Resolved same session", "value": 0},
            {"name": "Follow-up scheduled", "value": 0},
            {"name": "Escalated clinical", "value": 0},
            {"name": "Patient left / dropped", "value": 0},
        ]

    resolved = 0
    followup = 0
    escalated = 0
    dropped = 0

    for r in rows:
        report = r.get("clinical_report")
        if not report or not isinstance(report, dict):
            resolved += 1
            continue
        triage = report.get("triage", {})
        level = (triage.get("level") or "").lower() if isinstance(triage, dict) else ""
        if "emergent" in level or "urgent" in level:
            escalated += 1
        elif "routine" in level:
            followup += 1
        else:
            resolved += 1

    # Count failed as dropped
    failed_rows = await _get("voice_calls", params="select=id&orchestrator_status=eq.failed&limit=200")
    dropped = len(failed_rows) if failed_rows else 0

    return [
        {"name": "Resolved same session", "value": resolved},
        {"name": "Follow-up scheduled", "value": followup},
        {"name": "Escalated clinical", "value": escalated},
        {"name": "Patient left / dropped", "value": dropped},
    ]


async def get_pipeline_stats() -> Dict[str, Any]:
    """Orchestrator pipeline completion stats."""
    rows = await _get("voice_calls", params="select=orchestrator_status&limit=500")
    if not rows:
        return {"current": 0, "target": 85, "total_runs": 0}
    total = len(rows)
    completed = sum(1 for r in rows if r.get("orchestrator_status") == "completed")
    pct = round(completed / max(total, 1) * 100) if total else 0
    return {"current": pct, "target": 85, "total_runs": total}


async def get_calls_history(
    *,
    page: int = 1,
    limit: int = 20,
    direction: Optional[str] = None,
    provider: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """Paginated call history from voice_calls."""
    offset = (page - 1) * limit
    params = f"select=*&order=created_at.desc&limit={limit}&offset={offset}"
    if direction:
        params += f"&direction=eq.{direction}"
    if provider:
        params += f"&provider=eq.{provider}"
    if status:
        params += f"&orchestrator_status=eq.{status}"

    rows = await _get("voice_calls", params=params)
    # Get total count
    count_rows = await _get("voice_calls", params="select=id")
    total = len(count_rows) if count_rows else 0

    return {
        "calls": rows or [],
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": max(1, -(-total // limit)),
    }


# ── Report Approvals ────────────────────────────────────────────────

async def get_approvals(user_id: str) -> List[Dict[str, Any]]:
    """Get all approved call_ids for a user."""
    rows = await _get("report_approvals", params=f"select=*&user_id=eq.{user_id}&approved=eq.true&order=approved_at.desc")
    return rows or []


async def approve_report(call_id: str, user_id: str, notes: Optional[str] = None) -> bool:
    """Approve a report. Also updates voice_calls row if it exists."""
    body = {
        "call_id": call_id,
        "user_id": user_id,
        "approved": True,
        "notes": notes,
        "approved_at": datetime.now(timezone.utc).isoformat(),
    }
    # Upsert into report_approvals
    if not _credentials_present():
        return False
    ok, _ = await verify_supabase_connection()
    if not ok:
        return False

    url = f"{_base_url()}/rest/v1/report_approvals"
    headers = {
        **_headers(),
        "Prefer": "resolution=merge-duplicates,return=representation",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=body) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    logger.warning(f"Supabase upsert report_approvals failed {resp.status}: {text[:300]}")
                    return False
    except Exception as e:
        logger.warning(f"Supabase upsert report_approvals error: {e}")
        return False

    # Also update voice_calls if call_id matches
    await _patch(
        "voice_calls",
        {"approved": True, "approved_by": user_id, "approved_at": datetime.now(timezone.utc).isoformat()},
        filter_str=f"call_id=eq.{call_id}",
    )
    return True


async def revoke_approval(call_id: str, user_id: str) -> bool:
    """Revoke approval for a report."""
    ok = await _delete("report_approvals", filter_str=f"call_id=eq.{call_id}&user_id=eq.{user_id}")
    if ok:
        await _patch(
            "voice_calls",
            {"approved": False, "approved_by": None, "approved_at": None},
            filter_str=f"call_id=eq.{call_id}",
        )
    return ok


# ── Agent Configs ───────────────────────────────────────────────────

async def get_agent_config(user_id: str) -> Optional[Dict[str, Any]]:
    """Get active agent config for user."""
    rows = await _get(
        "agent_configs",
        params=f"select=*&user_id=eq.{user_id}&is_active=eq.true&order=updated_at.desc&limit=1"
    )
    if rows and len(rows) > 0:
        return rows[0]
    return None


async def save_agent_config(user_id: str, config: Dict[str, Any], name: str = "Default") -> Optional[str]:
    """Save or update agent config. Returns config id."""
    existing = await get_agent_config(user_id)
    now = datetime.now(timezone.utc).isoformat()

    if existing:
        ok = await _patch(
            "agent_configs",
            {"config": config, "name": name, "updated_at": now},
            filter_str=f"id=eq.{existing['id']}",
        )
        return existing["id"] if ok else None
    else:
        body = {
            "user_id": user_id,
            "name": name,
            "config": config,
            "is_active": True,
            "updated_at": now,
        }
        result = await _post("agent_configs", body)
        if result and isinstance(result, list) and len(result) > 0:
            return str(result[0].get("id", ""))
        return None


# ── User Preferences ───────────────────────────────────────────────

async def get_user_preferences(user_id: str) -> Dict[str, Any]:
    """Get user preferences."""
    rows = await _get("user_preferences", params=f"select=*&user_id=eq.{user_id}&limit=1")
    if rows and len(rows) > 0:
        return rows[0].get("preferences", {})
    return {}


async def save_user_preferences(user_id: str, preferences: Dict[str, Any]) -> bool:
    """Save or update user preferences (upsert)."""
    if not _credentials_present():
        return False
    ok, _ = await verify_supabase_connection()
    if not ok:
        return False

    body = {
        "user_id": user_id,
        "preferences": preferences,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    url = f"{_base_url()}/rest/v1/user_preferences"
    headers = {
        **_headers(),
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=body) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    logger.warning(f"Supabase upsert user_preferences failed {resp.status}: {text[:300]}")
                    return False
                return True
    except Exception as e:
        logger.warning(f"Supabase upsert user_preferences error: {e}")
        return False


# ── Phone Registry ─────────────────────────────────────────────────

async def get_phone_registry(user_id: str) -> List[Dict[str, Any]]:
    """Get all manually registered phone numbers for user."""
    rows = await _get(
        "phone_registry",
        params=f"select=*&user_id=eq.{user_id}&order=created_at.desc"
    )
    return rows or []


async def add_phone_entry(
    user_id: str,
    phone_number: str,
    provider: str = "manual",
    label: Optional[str] = None,
    is_default: bool = False,
) -> Optional[str]:
    """Add a phone number to registry."""
    body = {
        "user_id": user_id,
        "phone_number": phone_number,
        "provider": provider,
        "label": label,
        "is_default": is_default,
    }
    result = await _post("phone_registry", body)
    if result and isinstance(result, list) and len(result) > 0:
        return str(result[0].get("id", ""))
    return None


async def delete_phone_entry(entry_id: str, user_id: str) -> bool:
    """Delete a phone registry entry."""
    return await _delete("phone_registry", filter_str=f"id=eq.{entry_id}&user_id=eq.{user_id}")
