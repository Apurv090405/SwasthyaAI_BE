"""Persist analyze_transcript / supervisor events next to voice webhooks on disk."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

_WEBHOOK_EVENTS_DIR = Path(__file__).resolve().parent.parent / "webhook_events"


def save_analyze_transcript_event(
    *,
    userid: str,
    provider: str,
    location: str,
    transcript: str,
    report: Dict[str, Any],
) -> str:
    """
    Persist full supervisor clinical report under webhook_events/.

    Returns path relative to project root (e.g. webhook_events/2026..._analyze_....json).
    """
    _WEBHOOK_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_user = "".join(c if c.isalnum() or c in "-_" else "_" for c in userid)[:64]
    suffix = uuid.uuid4().hex[:8]
    filename = f"{ts}_analyze_{safe_user}_{suffix}.json"
    path = _WEBHOOK_EVENTS_DIR / filename

    pseudo_id = report.get("pseudo_id", "unknown")
    rel = str(path.relative_to(path.parent.parent))
    payload = {
        "event_type": "analyze_transcript",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "saved_event_file": rel,
        "call_id": userid,
        "stream_id": pseudo_id,
        "request": {
            "userid": userid,
            "llm_provider": provider,
            "location": location or None,
            "transcript": transcript,
        },
        "report": report,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return rel


def load_event_by_call_id(call_id: str) -> Dict[str, Any] | None:
    """Load the most recent saved event for a call_id from webhook_events/."""
    if not _WEBHOOK_EVENTS_DIR.exists():
        return None

    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in call_id)[:64]

    voice_matches = list(_WEBHOOK_EVENTS_DIR.glob(f"*_{call_id}.json"))
    clinical_matches = list(_WEBHOOK_EVENTS_DIR.glob(f"*_analyze_{safe}_*.json"))

    files = sorted(
        set(voice_matches) | set(clinical_matches),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        return None

    with open(files[0], "r", encoding="utf-8") as f:
        return json.load(f)
