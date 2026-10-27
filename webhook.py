"""
FastAPI webhook receiver for call end events.

This endpoint receives webhook POST requests when calls end, containing:
- Transcript data
- Call metrics
- Usage metrics
- Latency metrics
- Call information
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
import json

# Initialize FastAPI app
app = FastAPI(
    title="Call Webhook Receiver",
    description="Receives webhook POST requests when calls end",
    version="1.0.0"
)

# Local storage directory for full webhook payloads (including full transcript)
WEBHOOK_STORE_DIR = Path(__file__).resolve().parent / "webhook_events"
WEBHOOK_STORE_DIR.mkdir(exist_ok=True)


def _safe_id_segment(s: str, maxlen: int = 64) -> str:
    """Sanitize userid for clinical event filenames (must match api/routes.py)."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:maxlen]


def _save_webhook_payload(call_id: str, payload: Dict[str, Any]) -> str:
    """Persist full webhook payload to disk and return file path."""
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_call_id = call_id.replace("/", "_")
    file_path = WEBHOOK_STORE_DIR / f"{ts}_{safe_call_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(file_path)


def _transcript_preview(transcript: Dict[str, Any], max_len: int = 160) -> str:
    """Short plain-text preview for sheet views (dashboard analytics)."""
    ft = (transcript.get("full_text") or "").strip()
    if ft:
        return (ft[:max_len] + "…") if len(ft) > max_len else ft
    msgs = transcript.get("messages") or []
    chunks: List[str] = []
    for m in msgs[:10]:
        role = str(m.get("role", "")).strip()
        c = str(m.get("content") or "").replace("\n", " ").strip()
        if role == "system":
            c = (c[:100] + "…") if len(c) > 100 else c
        if c:
            chunks.append(f"{role}: {c[:120]}")
    s = " · ".join(chunks)
    return (s[:max_len] + "…") if len(s) > max_len else s


def _duration_minutes_from_metrics(metrics: Dict[str, Any]) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None
    for key in ("call_duration_seconds", "duration_seconds", "duration", "total_seconds"):
        v = metrics.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return round(float(v) / 60.0, 2)
    return None


def _critical_from_report(report: Dict[str, Any]) -> bool:
    if not isinstance(report, dict):
        return False
    triage = report.get("triage") or {}
    level = str(triage.get("level") or "").lower()
    if not level:
        return False
    urgent_tokens = ("emergent", "immediate", "urgent", "critical", "esi 1", "esi 2", "emergency")
    return any(t in level for t in urgent_tokens)


def _load_saved_events(limit: int = 50) -> List[Dict[str, Any]]:
    """Load saved webhook event metadata in reverse-chronological order."""
    files = sorted(WEBHOOK_STORE_DIR.glob("*.json"), reverse=True)[:limit]
    events: List[Dict[str, Any]] = []
    for p in files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
            etype = payload.get("event_type", "voice_call")
            if etype == "analyze_transcript":
                req = payload.get("request", {})
                report = payload.get("report", {})
                tr = str(req.get("transcript") or "")
                preview = (tr[:160] + "…") if len(tr) > 160 else tr
                triage = report.get("triage") or {}
                events.append(
                    {
                        "file": p.name,
                        "event_type": "analyze_transcript",
                        "call_id": req.get("userid", payload.get("call_id", "unknown")),
                        "stream_id": report.get("pseudo_id", "unknown"),
                        "timestamp": payload.get("timestamp"),
                        "message_count": None,
                        "from_number": None,
                        "to_number": None,
                        "direction": "supervisor",
                        "status": "completed",
                        "transcript_preview": preview,
                        "telco_provider": None,
                        "llm_provider": req.get("llm_provider"),
                        "duration_minutes": None,
                        "critical_case": _critical_from_report(report),
                        "triage_level": triage.get("level"),
                    }
                )
            else:
                transcript = payload.get("transcript", {})
                call_info = payload.get("call_info", {}) or {}
                metrics = payload.get("metrics") or {}
                events.append(
                    {
                        "file": p.name,
                        "event_type": etype,
                        "call_id": payload.get("call_id", "unknown"),
                        "stream_id": payload.get("stream_id", "unknown"),
                        "timestamp": payload.get("timestamp"),
                        "message_count": transcript.get("message_count", 0),
                        "from_number": call_info.get("from"),
                        "to_number": call_info.get("to"),
                        "direction": call_info.get("direction"),
                        "status": call_info.get("status", "completed"),
                        "transcript_preview": _transcript_preview(transcript),
                        "telco_provider": call_info.get("provider"),
                        "llm_provider": call_info.get("llm_provider"),
                        "duration_minutes": _duration_minutes_from_metrics(metrics),
                        "critical_case": None,
                        "triage_level": None,
                    }
                )
        except Exception:
            continue
    return events


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Call Webhook Receiver",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


def ingest_call_payload(payload: Dict[str, Any]) -> str:
    """Persist a call-end payload and start the post-call pipeline; returns the saved file path.

    Shared by the HTTP webhook, Bolna executions and in-process delivery from our own pipelines.
    """
    call_id = payload.get("call_id", "unknown")
    saved_file = _save_webhook_payload(call_id, payload)
    logger.info(f"Saved full webhook payload to: {saved_file}")
    try:
        from services.post_call_pipeline import process_voice_call_webhook

        asyncio.create_task(process_voice_call_webhook(dict(payload)))
    except Exception as sched_err:
        logger.warning(f"Post-call pipeline not scheduled: {sched_err}")
    return saved_file


@app.post("/webhook")
async def receive_webhook(request: Request):
    """
    Receive webhook POST request with call data.
    
    Expected payload structure:
    {
        "stream_id": str,
        "call_id": str,
        "call_info": {
            "from": str,
            "to": str,
            "direction": str,
            "status": str,
            "websocket_url": str
        },
        "transcript": {
            "full_text": str,
            "message_count": int,
            "messages": list
        },
        "metrics": dict,
        "usage_metrics": dict,
        "latency_metrics": dict,
        "timestamp": str
    }
    """
    try:
        # Parse JSON payload
        payload = await request.json()
        
        # Extract key information
        stream_id = payload.get("stream_id", "unknown")
        call_id = payload.get("call_id", "unknown")
        call_info = payload.get("call_info", {})
        transcript = payload.get("transcript", {})
        metrics = payload.get("metrics", {})
        usage_metrics = payload.get("usage_metrics", {})
        latency_metrics = payload.get("latency_metrics", {})
        timestamp = payload.get("timestamp", datetime.utcnow().isoformat() + "Z")
        
        # Log received webhook
        logger.info(f"Received webhook for call_id={call_id}, stream_id={stream_id}")
        logger.info(f"Call direction: {call_info.get('direction', 'unknown')}")
        logger.info(f"From: {call_info.get('from', 'unknown')} To: {call_info.get('to', 'unknown')}")
        logger.info(f"Transcript message count: {transcript.get('message_count', 0)}")
        
        # Log transcript (truncated for readability)
        full_text = transcript.get("full_text", "")
        if full_text:
            transcript_preview = full_text[:200] + "..." if len(full_text) > 200 else full_text
            logger.info(f"Transcript preview: {transcript_preview}")
            logger.info(f"Transcript full_text:\n{full_text}")
        
        # Log metrics if available
        if metrics:
            logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
        if usage_metrics:
            logger.info(f"Usage metrics: {json.dumps(usage_metrics, indent=2)}")
        if latency_metrics:
            logger.info(f"Latency metrics: {json.dumps(latency_metrics, indent=2)}")
        
        # Persist full payload for later retrieval/debugging, then run the post-call pipeline.
        saved_file = ingest_call_payload(payload)

        # Return success response
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Webhook received successfully",
                "call_id": call_id,
                "stream_id": stream_id,
                "saved_file": saved_file,
                "received_at": datetime.utcnow().isoformat() + "Z"
            }
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in webhook payload: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {str(e)}")
    except Exception as e:
        logger.exception(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/webhook/raw")
async def receive_webhook_raw(request: Request):
    """
    Receive webhook POST request and return raw payload (for debugging).
    """
    try:
        payload = await request.json()
        logger.info(f"Received raw webhook: {json.dumps(payload, indent=2)}")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "received_payload": payload,
                "received_at": datetime.utcnow().isoformat() + "Z"
            }
        )
    except Exception as e:
        logger.exception(f"Error processing raw webhook: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/webhook/events")
async def list_saved_events(limit: int = 50):
    """List recent saved webhook events (metadata only)."""
    if limit < 1:
        limit = 1
    if limit > 200:
        limit = 200
    return {"count": len(_load_saved_events(limit)), "events": _load_saved_events(limit)}


@app.get("/webhook/events/{call_id}")
async def get_event_by_call_id(call_id: str):
    """Fetch the most recent full saved payload by call_id or analyze_transcript userid."""
    voice_matches = list(WEBHOOK_STORE_DIR.glob(f"*_{call_id}.json"))
    safe = _safe_id_segment(call_id)
    clinical_matches = list(WEBHOOK_STORE_DIR.glob(f"*_analyze_{safe}_*.json"))
    files = sorted(
        set(voice_matches) | set(clinical_matches),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise HTTPException(status_code=404, detail=f"No saved event found for call_id={call_id}")
    latest = files[0]
    with open(latest, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload


@app.get("/webhook/events/by-file/{filename}")
async def get_event_by_filename(filename: str):
    """Load one saved event by exact filename under webhook_events/ (for dashboard when call_id lookup fails)."""
    raw = (filename or "").strip()
    if ".." in raw or "/" in raw or "\\" in raw:
        raise HTTPException(status_code=400, detail="Invalid filename")
    name = Path(raw).name
    if not name or name != raw:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not name.endswith(".json"):
        raise HTTPException(status_code=400, detail="Expected a .json file")
    path = (WEBHOOK_STORE_DIR / name).resolve()
    try:
        path.relative_to(WEBHOOK_STORE_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No file {name}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

