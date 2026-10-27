"""
Browser-based web call routes for SwasthyaAI.

Provides a WebSocket endpoint that runs the Pipecat voice pipeline
directly from the browser — no telephony provider needed.
The browser captures mic audio, sends PCM frames over WebSocket,
and receives TTS audio back.

The pipeline itself is shared with the phone routes (services/voice_pipeline.py).
"""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, WebSocket
from loguru import logger
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from api.browser_serializer import BrowserPCMSerializer
from auth import call_owner, optional_user
from api.voice_routes import load_prompt_from_file
from services.voice_pipeline import (
    WEB_SAMPLE_RATE,
    build_voice_session,
    greeting_for,
    run_voice_session,
)


web_call_router = APIRouter(prefix="/web", tags=["web-call"])

# Track active web call sessions
WEB_CALLS: dict[str, dict] = {}


@web_call_router.post("/start")
async def start_web_call(
    llm_provider: str = "openai",
    user_id: str = None,
    token_user: Optional[str] = Depends(optional_user),
):
    """Create a web call session and return the WebSocket URL to connect to."""
    stream_id = str(uuid.uuid4())

    WEB_CALLS[stream_id] = {
        "direction": "web",
        "llm_provider": llm_provider,
        "status": "waiting",
        "supabase_user_id": call_owner(token_user, user_id),
        "provider": "web",
    }

    return {
        "status": "ready",
        "stream_id": stream_id,
        "ws_url": f"/ws/web/{stream_id}",
        "llm_provider": llm_provider,
    }


@web_call_router.get("/calls")
async def list_web_calls():
    """List active/recent web call sessions."""
    return {"calls": list(WEB_CALLS.values()), "count": len(WEB_CALLS)}


async def run_web_pipeline(websocket: WebSocket, stream_id: str, call_data: dict):
    """Run the Pipecat pipeline for browser-based web calls.

    Audio format: PCM 16-bit signed LE, 16kHz mono.
    The browser sends raw PCM frames, and receives raw PCM frames back.
    """
    llm_provider = call_data.get("llm_provider", "openai")
    logger.info(f"Starting WEB pipeline: stream={stream_id}, llm={llm_provider}")

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=BrowserPCMSerializer(sample_rate=WEB_SAMPLE_RATE, num_channels=1),
        ),
    )
    session = build_voice_session(
        transport=transport,
        sample_rate=WEB_SAMPLE_RATE,
        system_prompt=load_prompt_from_file("inbound_prompt.md"),
        llm_provider=llm_provider,
        greeting=greeting_for("inbound"),
        label="web",
    )

    @transport.event_handler("on_client_connected")
    async def on_connect(trans, client):
        logger.info(f"Web call client connected: {stream_id}")
        await session.speak_greeting()

    @transport.event_handler("on_client_disconnected")
    async def on_disconnect(trans, client):
        logger.info(f"Web call client disconnected: {stream_id}")
        await session.task.cancel()

    if stream_id in WEB_CALLS:
        WEB_CALLS[stream_id]["status"] = "active"

    await run_voice_session(
        session,
        websocket,
        stream_id=stream_id,
        call_id=stream_id,
        pop_call_info=lambda: {**call_data, **(WEB_CALLS.pop(stream_id, None) or {})},
    )
