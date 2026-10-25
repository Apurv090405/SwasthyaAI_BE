#!/usr/bin/env python3
"""One-off probe: insert a minimal voice_calls row using app Supabase settings.

Requires:
  - SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY in project .env:
    Secret API key ``sb_secret_...`` or Legacy ``service_role`` JWT ``eyJ...`` — not ``sb_publishable_...``.
  - A profile id that exists in public.profiles (pass as argv or SUPABASE_TEST_USER_ID)

Usage:
  uv run python test_supabase_voice_insert.py <user_uuid>
  SUPABASE_TEST_USER_ID=<uuid> uv run python test_supabase_voice_insert.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

# Ensure config loads project-root .env before importing supabase_voice
from config import settings  # noqa: E402
from services.supabase_voice import insert_voice_call_row, verify_supabase_connection  # noqa: E402


async def main() -> int:
    uid = (
        (sys.argv[1] if len(sys.argv) > 1 else "").strip()
        or (os.environ.get("SUPABASE_TEST_USER_ID") or "").strip()
    )
    if not uid:
        print(
            "Missing user id: pass as first argument or set SUPABASE_TEST_USER_ID "
            "(must exist in public.profiles).",
            file=sys.stderr,
        )
        return 2

    ok, detail = await verify_supabase_connection(force=True)
    print(f"verify_supabase_connection: ok={ok} detail={detail}")
    if not ok:
        return 1

    call_id = f"test-insert-{uuid.uuid4()}"
    row = {
        "user_id": uid,
        "provider": "telnyx",
        "call_id": call_id,
        "transcript_messages": [],
        "metrics": {},
        "usage_metrics": {},
        "latency_metrics": {},
        "transcript_full": "",
        "orchestrator_status": "pending",
    }
    row_id = await insert_voice_call_row(row)
    if row_id:
        print(f"OK inserted voice_calls id={row_id} call_id={call_id}")
        print(f"Check Supabase Table Editor → voice_calls for call_id={call_id}")
        return 0

    print("Insert returned None — see log lines above (RLS, FK user_id, duplicate, etc.)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
