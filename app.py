"""
FastAPI application entry point for SwasthyaAI.

This module creates a LangGraph Supervisor orchestrating multiple clinical agents:
- diagnosis_agent
- rag_agent (LanceDB + embeddings)
- google_search_agent (SerpAPI / Google Custom Search)
- eval_agent
- clinical_rules_agent
- risk_scoring_agent
- differential_diagnosis_agent
- triage_decision_agent
- recommendation_agent
- report_formatting_agent

Endpoints (see /docs):
- GET / — API health JSON
- GET /clinical_agents — ids and summaries for single-agent chat
- POST /agent_chat — one turn with a chosen clinical agent
- POST /analyze_transcript — clinical supervisor report
- POST /upload_pdf — LanceDB RAG ingest
- POST /webhook — call-end transcript webhook (alias of /webhook-receiver/webhook)
- Mounted webhook app: /webhook-receiver/* (e.g. GET /webhook-receiver/webhook/events)
- WebSocket /ws/exotel/{stream_id}, /ws/vobiz/{stream_id}, /ws/telnyx/{stream_id}
  (optional query on Telnyx: direction, llm_provider / llm — for local ``test_telnyx_websocket.py``)
- POST /voice/exotel/inbound, /voice/exotel/outbound — Exotel
- /vobiz/* — Vobiz integration

CORS allows local Vite dev (e.g. http://localhost:3000).
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api import router
from api.voice_routes import voice_router, ws_exotel_endpoint, close_global_session
from api.vobiz_routes import vobiz_router, run_vobiz_pipeline, VOBIZ_CALLS
from api.telnyx_routes import telnyx_router, run_telnyx_pipeline, TELNYX_CALLS
from api.web_call_routes import web_call_router, run_web_pipeline, WEB_CALLS
from api.dashboard_routes import dashboard_router
from api.voicelink_routes import voicelink_router
from services.supabase_voice import supabase_health_dict
from webhook import app as webhook_receiver_app, receive_webhook


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan: startup and shutdown events."""
    # Startup: Global session will be created on first use
    yield
    # Shutdown: Close global aiohttp session
    await close_global_session()


# Initialize FastAPI app with lifespan management
app = FastAPI(
    title="SwasthyaAI - Supervisor Orchestrator",
    description="Clinical AI system for patient transcript analysis and diagnosis with voice agent capabilities",
    lifespan=lifespan,
)

# Browser dev server (Vite on :3000) and local variants
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Lightweight health route for dashboards and probes."""
    supabase = await supabase_health_dict()
    return {
        "status": "ok",
        "service": "SwasthyaAI API",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "supabase": supabase,
    }


@app.get("/health/supabase")
async def health_supabase(force: bool = False):
    """Re-check Supabase PostgREST (``voice_calls``); use ``?force=true`` to bypass cache."""
    return await supabase_health_dict(force=force)


# Mounted: same webhook app as `python webhook.py` but under /webhook-receiver
app.mount("/webhook-receiver", webhook_receiver_app)


@app.post("/webhook")
async def call_end_webhook_root(request: Request):
    """Receive call-end payloads; same handler as POST /webhook-receiver/webhook.

    Use WEBHOOK_URL=http://<host>:<port>/webhook when the voice API and receiver
    share one uvicorn process (e.g. port 8000 or 8001).
    """
    return await receive_webhook(request)

# Include API routes
app.include_router(router)
app.include_router(voice_router)
app.include_router(vobiz_router)
app.include_router(telnyx_router)
app.include_router(web_call_router)
app.include_router(dashboard_router)
app.include_router(voicelink_router)

# WebSocket endpoint at root level (Exotel calls this directly)
@app.websocket("/ws/exotel/{stream_id}")
async def ws_exotel_root(stream_id: str, websocket: WebSocket):
    """WebSocket endpoint for Exotel - root level for external access."""
    await ws_exotel_endpoint(stream_id, websocket)

# WebSocket endpoint for Vobiz
@app.websocket("/ws/vobiz/{stream_id}")
async def ws_vobiz_root(stream_id: str, websocket: WebSocket):
    """WebSocket endpoint for Vobiz."""
    await websocket.accept()
    
    # Get call data if available
    call_data = VOBIZ_CALLS.get(stream_id, {})
    
    try:
        await run_vobiz_pipeline(websocket, stream_id, call_data)
    except Exception as e:
        # logger.error(f"Vobiz WebSocket error: {e}")
        pass


@app.websocket("/ws/telnyx/{stream_id}")
async def ws_telnyx_root(stream_id: str, websocket: WebSocket):
    """WebSocket endpoint for Telnyx TeXML media streams."""
    await websocket.accept()

    call_data = dict(TELNYX_CALLS.get(stream_id, {}))
    qp = websocket.query_params
    if qp.get("direction"):
        call_data["direction"] = qp["direction"]
    if qp.get("llm_provider"):
        call_data["llm_provider"] = qp["llm_provider"]
    elif qp.get("llm"):
        call_data["llm_provider"] = qp["llm"]

    try:
        await run_telnyx_pipeline(websocket, stream_id, call_data)
    except Exception as e:
        pass


@app.websocket("/ws/web/{stream_id}")
async def ws_web_root(stream_id: str, websocket: WebSocket):
    """WebSocket endpoint for browser-based web calls (no telephony)."""
    await websocket.accept()

    call_data = dict(WEB_CALLS.get(stream_id, {}))
    qp = websocket.query_params
    if qp.get("llm_provider") or qp.get("llm"):
        call_data["llm_provider"] = qp.get("llm_provider") or qp.get("llm")
    if not call_data.get("llm_provider"):
        call_data["llm_provider"] = "openai"

    try:
        await run_web_pipeline(websocket, stream_id, call_data)
    except Exception as e:
        logger.error(f"Web call WebSocket error: {e}")
        pass

