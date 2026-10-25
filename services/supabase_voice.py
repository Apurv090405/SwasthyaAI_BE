"""Insert/update voice call rows via Supabase REST (service role)."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import aiohttp
from loguru import logger

from config import settings

_CACHE_TTL_SEC = 45.0
_cache_monotonic: float = 0.0
_cache_ok: Optional[bool] = None
_cache_detail: str = ""


def _credentials_present() -> bool:
    url = (settings.SUPABASE_URL or "").strip()
    key = (settings.SUPABASE_SERVICE_ROLE_KEY or "").strip()
    if url and key:
        return True
    if not url:
        logger.debug(
            "Supabase voice_calls skipped: SUPABASE_URL is missing or empty in environment"
        )
    if not key:
        logger.debug(
            "Supabase voice_calls skipped: SUPABASE_SERVICE_ROLE_KEY is missing or empty "
            "(use API Keys → Secret, or Legacy → service_role — not the publishable key)"
        )
    return False


def _require_service_role_jwt() -> Tuple[bool, str]:
    """
    Accept an elevated Supabase key so PostgREST uses the ``service_role`` DB role (bypasses RLS):

    - New platform keys: ``sb_secret_...`` (Secret API key in dashboard).
    - Legacy: JWT whose payload ``role`` is ``service_role`` (starts with ``eyJ``).
    ``sb_publishable_...`` is rejected (same privilege as anon).
    """

    key = (settings.SUPABASE_SERVICE_ROLE_KEY or "").strip()
    if key.startswith("sb_publishable_"):
        return (
            False,
            "publishable_key_in_env_use_secret_sb_secret_or_legacy_service_role_jwt",
        )
    if key.startswith("sb_secret_"):
        return True, "ok"
    parts = key.split(".")
    if len(parts) != 3:
        return (
            False,
            "key_must_be_sb_secret_or_legacy_service_role_jwt_eyj_three_dot_parts",
        )
    try:
        payload_b64 = parts[1]
        pad_len = (4 - len(payload_b64) % 4) % 4
        raw = base64.urlsafe_b64decode(payload_b64 + ("=" * pad_len))
        payload = json.loads(raw.decode("utf-8"))
        role = payload.get("role")
        if role != "service_role":
            return (
                False,
                f"jwt_role_is_{role}_need_service_role_secret_not_anon_key",
            )
    except Exception:
        return False, "invalid_jwt_could_not_decode_use_dashboard_service_role_secret"

    return True, "ok"


def _headers(prefer: str = "return=representation") -> Dict[str, str]:
    key = (settings.SUPABASE_SERVICE_ROLE_KEY or "").strip()
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": prefer,
    }


async def verify_supabase_connection(*, force: bool = False) -> Tuple[bool, str]:
    """
    Probe PostgREST for ``voice_calls`` (auth + table exists).
    Result is cached for ``_CACHE_TTL_SEC`` unless ``force=True``.
    """
    global _cache_monotonic, _cache_ok, _cache_detail

    now = time.monotonic()
    if (
        not force
        and _cache_ok is not None
        and (now - _cache_monotonic) < _CACHE_TTL_SEC
    ):
        return _cache_ok, _cache_detail

    if not _credentials_present():
        _cache_ok = False
        _cache_detail = "missing_env"
        _cache_monotonic = now
        return False, "missing_env"

    key_ok, key_detail = _require_service_role_jwt()
    if not key_ok:
        _cache_ok = False
        _cache_detail = key_detail
        _cache_monotonic = now
        logger.warning(
            "Supabase: SUPABASE_SERVICE_ROLE_KEY is not elevated (need Secret sb_secret_ "
            "or Legacy service_role JWT) — {}",
            key_detail,
        )
        return False, key_detail

    base = settings.SUPABASE_URL.rstrip("/")
    url = f"{base}/rest/v1/voice_calls?select=id&limit=1"
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                url,
                headers={
                    **_headers(prefer="return=minimal"),
                    "Accept": "application/json",
                },
            ) as resp:
                text = await resp.text()
                if resp.status == 200:
                    _cache_ok = True
                    _cache_detail = "ok"
                    _cache_monotonic = now
                    logger.info("Supabase voice_calls: connection check OK")
                    return True, "ok"
                if resp.status in (401, 403):
                    _cache_ok = False
                    _cache_detail = "unauthorized_invalid_service_role_key"
                    _cache_monotonic = now
                    logger.warning(
                        "Supabase connection check failed: HTTP %s (check service_role key)",
                        resp.status,
                    )
                    return False, _cache_detail
                if resp.status == 404:
                    _cache_ok = False
                    _cache_detail = "table_not_found_run_migration_002"
                    _cache_monotonic = now
                    logger.warning(
                        "Supabase voice_calls returned 404 — apply migration 002_voice_calls.sql"
                    )
                    return False, _cache_detail
                _cache_ok = False
                _cache_detail = f"http_{resp.status}"
                _cache_monotonic = now
                logger.warning(
                    "Supabase connection check failed: HTTP %s — %s",
                    resp.status,
                    text[:300],
                )
                return False, _cache_detail
    except asyncio.TimeoutError:
        _cache_ok = False
        _cache_detail = "timeout"
        _cache_monotonic = now
        logger.warning("Supabase connection check: timeout")
        return False, "timeout"
    except Exception as e:
        _cache_ok = False
        _cache_detail = f"error:{type(e).__name__}"
        _cache_monotonic = now
        logger.warning(f"Supabase connection check error: {e}")
        return False, _cache_detail


async def supabase_health_dict(*, force: bool = False) -> Dict[str, Any]:
    """For GET / and dashboards: env + live connectivity snapshot."""
    if not _credentials_present():
        return {
            "configured": False,
            "connected": None,
            "detail": "not_configured",
        }
    ok, detail = await verify_supabase_connection(force=force)
    return {
        "configured": True,
        "connected": ok,
        "detail": detail,
    }


async def insert_voice_call_row(row: Dict[str, Any]) -> Optional[str]:
    """
    Insert one completed voice call. Returns new row ``id`` (uuid) or None if skipped/failed.
    """
    if not _credentials_present():
        return None
    ok, reason = await verify_supabase_connection()
    if not ok:
        logger.debug(f"Supabase insert skipped: not connected ({reason})")
        return None

    base = settings.SUPABASE_URL.rstrip("/")
    url = f"{base}/rest/v1/voice_calls"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=_headers(), json=row) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    low = text.lower()
                    if "row-level security" in low or "42501" in text:
                        logger.warning(
                            "Supabase insert blocked by RLS — use a Secret API key "
                            "(sb_secret_...) or Legacy service_role JWT in "
                            "SUPABASE_SERVICE_ROLE_KEY, not the publishable key. "
                            "Response: {}",
                            text[:400],
                        )
                    else:
                        logger.warning(
                            f"Supabase insert voice_calls failed {resp.status}: {text[:500]}"
                        )
                    return None
                try:
                    data = json.loads(text) if text.strip() else None
                except Exception:
                    logger.warning(f"Supabase insert: unexpected body {text[:200]}")
                    return None
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    return str(data[0].get("id", "")) or None
                if isinstance(data, dict) and data.get("id"):
                    return str(data["id"])
    except Exception as e:
        logger.warning(f"Supabase insert voice_calls error: {e}")
    return None


async def update_voice_call_clinical(
    *,
    row_id: str,
    clinical_report: Dict[str, Any],
    clinical_saved_path: Optional[str],
    orchestrator_status: str,
    orchestrator_error: Optional[str] = None,
) -> None:
    if not _credentials_present() or not row_id:
        return
    ok, reason = await verify_supabase_connection()
    if not ok:
        logger.debug(f"Supabase patch skipped: not connected ({reason})")
        return

    base = settings.SUPABASE_URL.rstrip("/")
    url = f"{base}/rest/v1/voice_calls?id=eq.{row_id}"
    patch: Dict[str, Any] = {
        "clinical_report": clinical_report,
        "orchestrator_status": orchestrator_status,
        "clinical_saved_path": clinical_saved_path,
        "orchestrator_error": orchestrator_error,
        "orchestrator_completed_at": datetime.utcnow().isoformat() + "Z",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.patch(
                url,
                headers=_headers(prefer="return=minimal"),
                json=patch,
            ) as resp:
                if resp.status >= 400:
                    t = await resp.text()
                    logger.warning(
                        f"Supabase patch voice_calls failed {resp.status}: {t[:400]}"
                    )
    except Exception as e:
        logger.warning(f"Supabase patch voice_calls error: {e}")


async def mark_voice_call_orchestrator_failed(
    *,
    row_id: str,
    error_message: str,
) -> None:
    if not _credentials_present() or not row_id:
        return
    ok, reason = await verify_supabase_connection()
    if not ok:
        logger.debug(f"Supabase patch (failed) skipped: not connected ({reason})")
        return

    base = settings.SUPABASE_URL.rstrip("/")
    url = f"{base}/rest/v1/voice_calls?id=eq.{row_id}"
    patch = {
        "orchestrator_status": "failed",
        "orchestrator_error": error_message[:8000],
        "orchestrator_completed_at": datetime.utcnow().isoformat() + "Z",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.patch(
                url, headers=_headers(prefer="return=minimal"), json=patch
            ) as resp:
                if resp.status >= 400:
                    t = await resp.text()
                    logger.warning(
                        f"Supabase patch (failed) voice_calls: {t[:300]}"
                    )
    except Exception as e:
        logger.warning(f"Supabase patch voice_calls error: {e}")

