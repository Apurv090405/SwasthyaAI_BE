"""Supabase REST operations for dashboard: analytics, approvals, agent configs, preferences, phone registry."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import aiohttp
from fastapi import HTTPException
from loguru import logger

from config import settings
from services.supabase_voice import _credentials_present, _headers, verify_supabase_connection


def _base_url() -> str:
    return settings.SUPABASE_URL.rstrip("/")


async def _require_supabase() -> None:
    """Analytics must fail loudly: empty charts hide a missing config or migration."""
    if not _credentials_present():
        raise HTTPException(503, "Supabase is not configured on the server (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY)")
    ok, _ = await verify_supabase_connection()
    if not ok:
        raise HTTPException(503, "Supabase is unreachable")


async def _get_rows(path: str, *, params: str) -> List[Dict[str, Any]]:
    """Like _get, but a failed query is an HTTP 502 instead of looking like 'no data'."""
    await _require_supabase()
    rows = await _get(path, params=params)
    if rows is None:
        raise HTTPException(
            502,
            f"Supabase query on {path} failed; check that all migrations are applied (see server log)",
        )
    return rows


async def _count(path: str, *, params: str) -> int:
    """Exact row count for a filter via PostgREST Content-Range."""
    url = f"{_base_url()}/rest/v1/{path}?select=id&limit=1"
    if params:
        url = f"{url}&{params}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={**_headers(prefer="count=exact")}) as resp:
                if resp.status >= 400:
                    logger.warning(f"Supabase count {path} failed {resp.status}: {(await resp.text())[:300]}")
                    return 0
                total = resp.headers.get("Content-Range", "*/0").rsplit("/", 1)[-1]
                return int(total) if total.isdigit() else 0
    except Exception as e:
        logger.warning(f"Supabase count {path} error: {e}")
        return 0


def _own(user_id: str) -> str:
    return f"user_id=eq.{quote(user_id)}"


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

async def get_dashboard_stats(user_id: str) -> Dict[str, Any]:
    """Aggregate KPI stats from the caller's voice_calls."""
    rows = await _get_rows(
        "voice_calls",
        params=f"select=id,created_at,direction,orchestrator_status,ended_at,from_number,to_number,provider,approved&{_own(user_id)}&order=created_at.desc&limit=500",
    )

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
    yesterday_str = (now - timedelta(days=1)).strftime("%Y-%m-%d")
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


async def get_call_volume(user_id: str, days: int = 7) -> List[Dict[str, Any]]:
    """Daily call volume breakdown for chart (last `days` days, including empty days)."""
    start = (datetime.now(timezone.utc) - timedelta(days=days - 1)).strftime("%Y-%m-%d")
    rows = await _get_rows(
        "voice_calls",
        params=f"select=created_at,direction&{_own(user_id)}&created_at=gte.{start}&order=created_at.desc&limit=5000",
    )

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

    today = datetime.now(timezone.utc).date()
    dates = [(today - timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]
    return [{"period": date, **daily[date]} for date in dates]


async def get_triage_outcomes(user_id: str) -> List[Dict[str, Any]]:
    """Triage outcome distribution from the caller's clinical reports."""
    rows = await _get_rows(
        "voice_calls",
        params=f"select=orchestrator_status,clinical_report&{_own(user_id)}&orchestrator_status=eq.completed&limit=200",
    )

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
    dropped = await _count("voice_calls", params=f"{_own(user_id)}&orchestrator_status=eq.failed")

    return [
        {"name": "Resolved same session", "value": resolved},
        {"name": "Follow-up scheduled", "value": followup},
        {"name": "Escalated clinical", "value": escalated},
        {"name": "Patient left / dropped", "value": dropped},
    ]


async def get_pipeline_stats(user_id: str) -> Dict[str, Any]:
    """Orchestrator pipeline completion stats."""
    rows = await _get_rows("voice_calls", params=f"select=orchestrator_status&{_own(user_id)}&limit=500")
    if not rows:
        return {"current": 0, "target": 85, "total_runs": 0}
    total = len(rows)
    completed = sum(1 for r in rows if r.get("orchestrator_status") == "completed")
    pct = round(completed / max(total, 1) * 100) if total else 0
    return {"current": pct, "target": 85, "total_runs": total}


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    # nearest-rank percentile (round() would send 2.5 → 2 and under-report the median)
    index = max(0, min(len(ordered) - 1, math.ceil(pct * len(ordered)) - 1))
    return round(ordered[index], 3)


async def get_latency_stats(user_id: str, limit: int = 200) -> Dict[str, Any]:
    """Voice-to-voice latency across the caller's recent calls (from voice_calls.latency_metrics).

    Our pipelines store per-turn latencies (services/voice_pipeline.py); Bolna calls store Bolna's
    own timings under latency_metrics.bolna and are reported separately.
    """
    rows = await _get_rows(
        "voice_calls",
        params=f"select=created_at,provider,latency_metrics&{_own(user_id)}&order=created_at.desc&limit={limit}",
    )
    turns: List[float] = []
    ttfb: Dict[str, List[float]] = defaultdict(list)
    by_provider: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"calls": 0, "turns": []})
    recent: List[Dict[str, Any]] = []
    bolna_first_audio: List[float] = []

    for r in rows:
        lm = r.get("latency_metrics") or {}
        if not isinstance(lm, dict):
            continue
        provider = r.get("provider") or "unknown"
        call_turns = [t for t in (lm.get("turns") or []) if isinstance(t, (int, float))]
        bolna = lm.get("bolna") if isinstance(lm.get("bolna"), dict) else {}
        first_audio = bolna.get("time_to_first_audio")
        if isinstance(first_audio, (int, float)):
            bolna_first_audio.append(float(first_audio))
        if not call_turns and not isinstance(first_audio, (int, float)):
            continue
        turns.extend(call_turns)
        bucket = by_provider[provider]
        bucket["calls"] += 1
        bucket["turns"].extend(call_turns or [float(first_audio)])
        for kind, summary in (lm.get("ttfb") or {}).items():
            if isinstance(summary, dict) and isinstance(summary.get("p50"), (int, float)):
                ttfb[kind].append(summary["p50"])
        if len(recent) < 20:
            recent.append(
                {
                    "created_at": r.get("created_at"),
                    "provider": provider,
                    "p50": _percentile(call_turns, 0.5) if call_turns else first_audio,
                }
            )

    return {
        "calls": sum(b["calls"] for b in by_provider.values()),
        "turns": len(turns),
        "p50": _percentile(turns, 0.5),
        "p95": _percentile(turns, 0.95),
        "ttfb_p50": {k: _percentile(v, 0.5) for k, v in ttfb.items()},
        "by_provider": sorted(
            (
                {
                    "provider": p,
                    "calls": b["calls"],
                    "p50": _percentile(b["turns"], 0.5),
                    "p95": _percentile(b["turns"], 0.95),
                }
                for p, b in by_provider.items()
            ),
            key=lambda x: -x["calls"],
        ),
        "bolna_first_audio_p50": _percentile(bolna_first_audio, 0.5),
        "recent": list(reversed(recent)),
    }


async def get_calls_history(
    user_id: str,
    *,
    page: int = 1,
    limit: int = 20,
    direction: Optional[str] = None,
    provider: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """Paginated call history from voice_calls."""
    offset = (page - 1) * limit
    filters = _own(user_id)
    if direction:
        filters += f"&direction=eq.{quote(direction)}"
    if provider:
        filters += f"&provider=eq.{quote(provider)}"
    if status:
        filters += f"&orchestrator_status=eq.{quote(status)}"

    rows = await _get_rows(
        "voice_calls", params=f"select=*&{filters}&order=created_at.desc&limit={limit}&offset={offset}"
    )
    total = await _count("voice_calls", params=filters)

    return {
        "calls": rows,
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

    # on_conflict: the unique key is (call_id, user_id), not the id primary key
    url = f"{_base_url()}/rest/v1/report_approvals?on_conflict=call_id,user_id"
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
        filter_str=f"call_id=eq.{quote(call_id)}&{_own(user_id)}",
    )
    return True


async def revoke_approval(call_id: str, user_id: str) -> bool:
    """Revoke approval for a report."""
    ok = await _delete("report_approvals", filter_str=f"call_id=eq.{call_id}&user_id=eq.{user_id}")
    if ok:
        await _patch(
            "voice_calls",
            {"approved": False, "approved_by": None, "approved_at": None},
            filter_str=f"call_id=eq.{quote(call_id)}&{_own(user_id)}",
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
    url = f"{_base_url()}/rest/v1/user_preferences?on_conflict=user_id"
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
