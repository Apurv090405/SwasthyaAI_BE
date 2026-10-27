"""After voice webhook: optional Supabase row + clinical supervisor orchestration."""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any, Dict, Optional

from loguru import logger

from services.clinical_service import run_clinical_supervisor_report
from services.supabase_voice import (
    insert_voice_call_row,
    mark_voice_call_orchestrator_failed,
    update_voice_call_clinical,
)
from services.webhook_event_file import save_analyze_transcript_event


def _extract_llm_provider(payload: Dict[str, Any]) -> str:
    ci = payload.get("call_info") or {}
    p = (ci.get("llm_provider") or "openai").strip().lower() or "openai"
    return p if p in ("openai", "gemini") else "openai"


def _build_row_from_webhook(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    call_info = payload.get("call_info") or {}
    transcript = payload.get("transcript") or {}
    user_id = (call_info.get("supabase_user_id") or "").strip()
    if not user_id:
        return None

    msgs = transcript.get("messages")
    return {
        "user_id": user_id,
        "provider": call_info.get("provider") or "telnyx",
        "call_id": payload.get("call_id") or "unknown",
        "stream_id": payload.get("stream_id"),
        "direction": call_info.get("direction"),
        "from_number": call_info.get("from"),
        "to_number": call_info.get("to"),
        "transcript_full": transcript.get("full_text") or "",
        "transcript_messages": msgs if isinstance(msgs, list) else [],
        "metrics": payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {},
        "usage_metrics": payload.get("usage_metrics")
        if isinstance(payload.get("usage_metrics"), dict)
        else {},
        "latency_metrics": payload.get("latency_metrics")
        if isinstance(payload.get("latency_metrics"), dict)
        else {},
        "orchestrator_status": "pending",
        "ended_at": payload.get("timestamp"),
    }


async def process_voice_call_webhook(payload: Dict[str, Any]) -> None:
    """
    Fire-and-forget from webhook receiver: Supabase insert, then full supervisor report.
    """
    import time
    t0 = time.time()

    payload = dict(payload)
    call_id = payload.get("call_id") or "unknown"
    call_info = payload.get("call_info") or {}
    direction = call_info.get("direction", "unknown")
    provider_telco = call_info.get("provider", "unknown")

    logger.info(
        f"[PostCall] === Starting post-call pipeline === | "
        f"call_id={call_id} direction={direction} telco={provider_telco}"
    )

    # Step 1: Insert Supabase row
    row = _build_row_from_webhook(payload)
    row_id: Optional[str] = None
    if row:
        logger.info(f"[PostCall] Step 1/4: Inserting voice_calls Supabase row...")
        row_id = await insert_voice_call_row(row)
        if row_id:
            logger.info(f"[PostCall] Supabase row inserted: id={row_id}")
        else:
            logger.warning(f"[PostCall] Supabase row insert returned None (no user_id?)")
    else:
        logger.info(f"[PostCall] Skipping Supabase insert (no supabase_user_id in payload)")

    # Step 2: Extract transcript
    transcript = (payload.get("transcript") or {}).get("full_text") or ""
    transcript = transcript.strip()
    msg_count = (payload.get("transcript") or {}).get("message_count", 0)
    llm_provider = _extract_llm_provider(payload)

    logger.info(
        f"[PostCall] Step 2/4: Transcript extracted | "
        f"length={len(transcript)} chars, messages={msg_count}, llm={llm_provider}"
    )

    if not transcript:
        logger.warning(f"[PostCall] SKIPPED — empty transcript for call_id={call_id}")
        if row_id:
            await mark_voice_call_orchestrator_failed(
                row_id=row_id,
                error_message="empty_transcript",
            )
        return

    logger.debug(f"[PostCall] Transcript preview: {transcript[:200]}...")

    # Step 3: Run clinical supervisor orchestration
    supabase_uid = call_info.get("supabase_user_id")
    userid_for_report = str(supabase_uid).strip() if supabase_uid else str(call_id)

    logger.info(
        f"[PostCall] Step 3/4: Running clinical supervisor orchestration | "
        f"userid={userid_for_report} provider={llm_provider}"
    )

    runner = partial(
        run_clinical_supervisor_report,
        transcript=transcript,
        userid=userid_for_report,
        location="",
        provider=llm_provider,
    )
    try:
        report, _ = await asyncio.to_thread(runner)
    except RuntimeError as e:
        elapsed = time.time() - t0
        logger.error(
            f"[PostCall] Orchestrator FAILED after {elapsed:.1f}s | "
            f"call_id={call_id} error={e}"
        )
        if row_id:
            await mark_voice_call_orchestrator_failed(row_id=row_id, error_message=str(e))
        return
    except Exception as e:
        elapsed = time.time() - t0
        logger.exception(
            f"[PostCall] Orchestrator EXCEPTION after {elapsed:.1f}s | "
            f"call_id={call_id} error={e}"
        )
        if row_id:
            await mark_voice_call_orchestrator_failed(row_id=row_id, error_message=str(e))
        return

    # Emergencies the live pipeline flagged (services/red_flags.py): always shown to the doctor
    live_flags = call_info.get("red_flags") or []
    if live_flags:
        report["live_red_flags"] = live_flags
        meta = report.setdefault("report_meta", {})
        if isinstance(meta, dict):
            meta["human_review_required"] = True

    # Step 4: Save report
    logger.info(f"[PostCall] Step 4/4: Saving clinical report to disk and Supabase...")

    saved_rel = ""
    try:
        saved_rel = save_analyze_transcript_event(
            userid=str(call_id),
            provider=llm_provider,
            location="",
            transcript=transcript,
            report=dict(report),
        )
        logger.info(f"[PostCall] Report saved to disk: {saved_rel}")
    except OSError as e:
        logger.warning(f"[PostCall] Could not save analyze event to disk: {e}")

    if row_id:
        await update_voice_call_clinical(
            row_id=row_id,
            clinical_report=dict(report),
            clinical_saved_path=saved_rel or None,
            orchestrator_status="completed",
            orchestrator_error=None,
        )
        logger.info(f"[PostCall] Supabase voice_calls row updated with clinical report")

    elapsed = time.time() - t0
    triage = report.get("triage", {})
    triage_level = triage.get("level", "?") if isinstance(triage, dict) else "?"

    logger.info(
        f"[PostCall] === Pipeline complete === | "
        f"call_id={call_id} triage={triage_level} total_time={elapsed:.1f}s "
        f"saved={saved_rel or 'disk_failed'}"
    )
