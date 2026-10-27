"""Voice agent routes for Pipecat/Exotel integration."""

import os
import uuid
import base64
from pathlib import Path
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

import aiohttp
from aiohttp import client_exceptions as aiohttp_client_exc
import httpx
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse
from loguru import logger
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.serializers.exotel import ExotelFrameSerializer
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

# Project root (parent of `api/`) — prompt files are always loaded from here
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

from auth import call_owner, optional_user
from config import settings
from services.voice_pipeline import (
    TELEPHONY_SAMPLE_RATE,
    LatencyCollector,
    build_voice_session,
    greeting_for,
    run_voice_session,
)

# Try to import RAG components (optional, for endpoints only)
# Environment is loaded once from project-root `.env` in config.py

# Initialize router
voice_router = APIRouter(prefix="/voice", tags=["voice"])

# In-memory store for calls (demo only - consider using Redis in production)
CALLS: Dict[str, Dict[str, Any]] = {}

# Global aiohttp session with connection pooling for better performance
# This reduces connection overhead and improves latency
_global_session: Optional[aiohttp.ClientSession] = None


async def get_global_session() -> aiohttp.ClientSession:
    """Get or create global aiohttp session with optimized connection pooling."""
    global _global_session
    if _global_session is None or _global_session.closed:
        # Configure connector with connection pooling limits
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=30,  # Max connections per host
            ttl_dns_cache=300,  # DNS cache TTL
            force_close=False,  # Reuse connections
            enable_cleanup_closed=True,  # Clean up closed connections
        )
        timeout = aiohttp.ClientTimeout(
            total=30,  # Total timeout
            connect=10,  # Connection timeout
            sock_read=10,  # Socket read timeout
        )
        _global_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": "SwasthyaAI-VoiceAgent/1.0"},
        )
        logger.info("Created global aiohttp session with connection pooling")
    return _global_session


async def close_global_session():
    """Close global aiohttp session (called on app shutdown)."""
    global _global_session
    if _global_session and not _global_session.closed:
        await _global_session.close()
        logger.info("Closed global aiohttp session")


def load_prompt_from_file(filepath: str) -> str:
    """Load prompt content from a markdown file under the project root."""
    path = Path(filepath)
    if not path.is_absolute():
        path = _PROJECT_ROOT / filepath

    default_voice = (
        "You are SwasthyaAI, a Hindi health-assistant on a phone call. "
        "Reply only in simple spoken Hindi (Devanagari). Do not use emoji. "
        "Keep answers short. You give general health guidance, not a diagnosis."
    )

    def _read(p: Path) -> str:
        with open(p, "r", encoding="utf-8") as f:
            return f.read().strip()

    try:
        content = _read(path)
    except FileNotFoundError:
        logger.warning(f"Prompt file {path} not found, using default prompt")
        return default_voice
    except Exception as e:
        logger.error(f"Error loading prompt from {path}: {e}")
        return default_voice

    # Detect corrupted / placeholder files (e.g. only the word "string")
    if len(content) < 80 or content.lower() in ("string", "none", "null"):
        logger.error(
            f"Prompt file {path} is invalid or too short ({len(content)} chars); "
            "trying outbound_prompt.md fallback"
        )
        fallback = _PROJECT_ROOT / "outbound_prompt.md"
        try:
            fb = _read(fallback)
            if len(fb) >= 80:
                return fb
        except OSError:
            pass
        return default_voice

    return content


def make_ws_url(stream_id: str, call_id: Optional[str] = None) -> str:
    """Construct the WebSocket URL that Exotel (or dev) should connect to."""
    query = f"?call_id={call_id}" if call_id else ""
    return f"wss://{settings.HOST}/ws/exotel/{stream_id}{query}"


def _webhook_is_local() -> bool:
    """True when WEBHOOK_URL is unset or points back at this server (localhost:PORT or its ngrok host)."""
    from urllib.parse import urlparse

    if not settings.WEBHOOK_URL:
        return True
    url = urlparse(settings.WEBHOOK_URL)
    host = (url.hostname or "").lower()
    public_host = (settings.TELNYX_PUBLIC_HOST or "").split("/")[0].lower()
    if public_host and host == public_host:
        return True
    return host in {"localhost", "127.0.0.1", "0.0.0.0"} and url.port == settings.PORT


async def send_call_webhook(
    stream_id: str,
    call_id: str,
    context: Optional[LLMContext] = None,
    task: Optional[PipelineTask] = None,
    call_info: Optional[Dict[str, Any]] = None,
    latency_observer: Optional[LatencyCollector] = None
):
    """Deliver call transcript, metrics and call info to the post-call pipeline.

    With no WEBHOOK_URL, or one pointing back at this server, the payload is ingested
    in-process (no HTTP hop through ngrok, and calls are never silently dropped). An external
    WEBHOOK_URL still receives an HTTP POST.
    """
    try:
        # Collect transcript from context
        transcript_messages = []
        full_transcript = ""
        if context and hasattr(context, 'messages'):
            transcript_messages = context.messages
            # Format transcript as readable text
            transcript_parts = []
            for msg in transcript_messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role == "user":
                    transcript_parts.append(f"User: {content}")
                elif role == "assistant":
                    transcript_parts.append(f"Assistant: {content}")
                elif role == "system":
                    # Skip system messages in transcript
                    continue
            full_transcript = "\n".join(transcript_parts)
        
        # Collect metrics from task
        metrics = {}
        usage_metrics = {}
        if task:
            # Try to get metrics from task
            if hasattr(task, 'metrics'):
                metrics = task.metrics or {}
            if hasattr(task, 'usage_metrics'):
                usage_metrics = task.usage_metrics or {}
            # Also check if metrics are stored elsewhere
            if hasattr(task, '_metrics'):
                metrics = task._metrics or {}
            # Try to get metrics from runner if available
            if hasattr(task, 'runner') and task.runner:
                if hasattr(task.runner, 'metrics'):
                    metrics = {**metrics, **(task.runner.metrics or {})}
        
        # Collect latency metrics from observer if available
        latency_metrics = {}
        if latency_observer:
            try:
                # Try to get latency data from observer
                if hasattr(latency_observer, 'latencies'):
                    latency_metrics = {"latencies": latency_observer.latencies}
                if hasattr(latency_observer, 'get_metrics'):
                    latency_metrics = {**latency_metrics, **latency_observer.get_metrics()}
            except Exception as e:
                logger.debug(f"Could not extract latency metrics: {e}")
        
        # Get call information
        call_data = call_info or {}
        
        # Prepare webhook payload
        payload = {
            "stream_id": stream_id,
            "call_id": call_id,
            "call_info": {
                "from": call_data.get("from", "unknown"),
                "to": call_data.get("to", "unknown"),
                "direction": call_data.get("direction", "unknown"),
                "status": call_data.get("status", "ended"),
                "websocket_url": call_data.get("websocket_url"),
                "supabase_user_id": call_data.get("supabase_user_id"),
                "llm_provider": call_data.get("llm_provider", "openai"),
                "provider": call_data.get("provider", "telnyx"),
                # emergencies spotted live during the call (services/red_flags.py)
                "red_flags": call_data.get("red_flags") or [],
            },
            "transcript": {
                "full_text": full_transcript,
                "message_count": len(transcript_messages),
                "messages": transcript_messages,
            },
            "metrics": metrics,
            "usage_metrics": usage_metrics,
            "latency_metrics": latency_metrics,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        
        if _webhook_is_local():
            from webhook import ingest_call_payload

            saved = ingest_call_payload(payload)
            logger.info(f"Call {call_id} ingested in-process → post-call pipeline ({saved})")
            return

        # Send to webhook
        session = await get_global_session()
        async with session.post(
            settings.WEBHOOK_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            if response.status >= 200 and response.status < 300:
                logger.info(f"Successfully sent webhook for call {call_id}")
            else:
                error_text = await response.text()
                logger.warning(
                    f"Webhook returned status {response.status} for call {call_id}: {error_text}"
                )
    except aiohttp_client_exc.ClientConnectorError as e:
        logger.warning(
            f"Webhook unreachable for call {call_id} ({settings.WEBHOOK_URL}): {e}"
        )
    except Exception as e:
        # Don't fail the call cleanup if webhook fails
        logger.exception(f"Error sending webhook for call {call_id}: {e}")


async def _run_exotel_pipeline(
    websocket: WebSocket,
    stream_id: str,
    call_id: str,
    direction: str,
    llm_provider: str = "openai",
):
    """Run the shared Pipecat voice pipeline over an Exotel media stream."""
    logger.info(
        f"Starting {direction.upper()} pipeline for stream_id={stream_id}, "
        f"call_id={call_id}, llm_provider={llm_provider}"
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=ExotelFrameSerializer(stream_sid=stream_id, call_sid=call_id),
        ),
    )
    session = build_voice_session(
        transport=transport,
        sample_rate=TELEPHONY_SAMPLE_RATE,
        system_prompt=load_prompt_from_file(f"{direction}_prompt.md"),
        llm_provider=llm_provider,
        greeting=greeting_for(direction),
        label=f"exotel-{direction}",
    )

    @transport.event_handler("on_client_connected")
    async def _on_connect(trans, client):
        logger.info(f"{direction} client connected on stream {stream_id}")
        await session.speak_greeting()

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnect(trans, client):
        logger.info(f"{direction} client disconnected stream {stream_id}")
        await session.task.cancel()

    await run_voice_session(
        session,
        websocket,
        stream_id=stream_id,
        call_id=call_id,
        pop_call_info=lambda: {
            "provider": "exotel",
            "direction": direction,
            "llm_provider": llm_provider,
            **(CALLS.pop(stream_id, None) or {}),
        },
    )


async def run_inbound_pipeline(
    websocket: WebSocket,
    stream_id: str,
    call_id: str,
    llm_provider: str = "openai",
):
    """Run the Pipecat pipeline for inbound Exotel calls."""
    await _run_exotel_pipeline(websocket, stream_id, call_id, "inbound", llm_provider)


async def run_outbound_pipeline(
    websocket: WebSocket,
    stream_id: str,
    call_id: str,
    llm_provider: str = "openai",
):
    """Run the Pipecat pipeline for outbound Exotel calls."""
    await _run_exotel_pipeline(websocket, stream_id, call_id, "outbound", llm_provider)


async def ws_exotel_endpoint(stream_id: str, websocket: WebSocket):
    """WebSocket endpoint handler for Exotel/dev testing."""
    await websocket.accept()
    query_params = websocket.query_params
    call_id = query_params.get("call_id", "")
    logger.info(f"WebSocket connection received stream_id={stream_id}, call_id={call_id}")

    # Determine call direction and LLM provider from stored call metadata
    call_info = CALLS.get(stream_id, {})
    direction = call_info.get("direction", "inbound")
    llm_provider = call_info.get("llm_provider", "openai")

    # Store call meta
    CALLS[stream_id] = {**call_info, "call_id": call_id, "status": "active"}

    try:
        # Route to appropriate pipeline based on direction
        if direction == "outbound":
            await run_outbound_pipeline(
                websocket, stream_id=stream_id, call_id=call_id, llm_provider=llm_provider
            )
        else:
            await run_inbound_pipeline(
                websocket, stream_id=stream_id, call_id=call_id, llm_provider=llm_provider
            )
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for stream {stream_id}")
    except Exception as e:
        logger.exception(f"WebSocket handling error for stream {stream_id}: {e}")
    finally:
        CALLS.pop(stream_id, None)


@voice_router.post("/exotel/inbound")
async def exotel_inbound(request: Request):
    """Handle inbound call webhook from Exotel.

    Accepts an optional `llm_provider` field ('openai' or 'gemini') in the
    JSON payload to select the LLM backend for this call. Defaults to 'openai'.
    """
    payload = await request.json()
    logger.info(f"Incoming inbound webhook: {payload}")

    from_number = payload.get("From") or payload.get("from") or "unknown"
    to_number = payload.get("To") or payload.get("to") or "unknown"
    call_id = payload.get("CallSid") or payload.get("call_id") or str(uuid.uuid4())
    llm_provider = payload.get("llm_provider", "openai")

    stream_id = str(uuid.uuid4())
    ws_url = make_ws_url(stream_id, call_id)

    # Store meta including chosen LLM provider
    CALLS[stream_id] = {
        "call_id": call_id,
        "from": from_number,
        "to": to_number,
        "direction": "inbound",
        "websocket_url": ws_url,
        "llm_provider": llm_provider,
        "status": "pending",
    }

    logger.info(
        f"Inbound call created: stream_id={stream_id}, "
        f"llm_provider={llm_provider}, websocket_url={ws_url}"
    )
    return JSONResponse({
        "stream_id": stream_id,
        "call_id": call_id,
        "websocket_url": ws_url,
        "llm_provider": llm_provider,
    })


@voice_router.post("/exotel/outbound")
async def exotel_outbound(
    payload: Dict[str, Any],
    background_tasks: BackgroundTasks,
    token_user: Optional[str] = Depends(optional_user),
):
    """Handle outbound call request.

    Accepts an optional `llm_provider` field ('openai' or 'gemini') to select
    the LLM backend for this call. Defaults to 'openai'.
    """
    logger.info(f"Outbound request payload: {payload}")
    if not settings.EXOTEL_ACCOUNT_SID or not settings.EXOTEL_AUTH_TOKEN:
        raise HTTPException(status_code=500, detail="Exotel credentials not configured")

    to_number = payload.get("to")
    from_number = payload.get("from")
    llm_provider = payload.get("llm_provider", "openai")
    if not to_number or not from_number:
        raise HTTPException(status_code=400, detail="Both `to` and `from` are required")

    call_id = str(uuid.uuid4())
    stream_id = str(uuid.uuid4())
    ws_url = make_ws_url(stream_id, call_id)

    # Exotel REST call payload
    exotel_payload = {
        "From": from_number,
        "To": to_number,
        "StreamSid": stream_id,
        "WebsocketUrl": ws_url,
    }

    auth = (settings.EXOTEL_ACCOUNT_SID, settings.EXOTEL_AUTH_TOKEN)
    exotel_call_url = f"{settings.EXOTEL_API_BASE}/Calls"
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(exotel_call_url, data=exotel_payload, auth=auth)
        if resp.status_code >= 400:
            logger.error(f"Exotel outbound call failed: {resp.status_code} {resp.text}")
            raise HTTPException(status_code=502, detail=f"Exotel API error: {resp.text}")

    result = resp.json()
    CALLS[stream_id] = {
        "call_id": call_id,
        "from": from_number,
        "to": to_number,
        "direction": "outbound",
        "websocket_url": ws_url,
        "llm_provider": llm_provider,
        "exotel_response": result,
        "status": "initiated",
        "provider": "exotel",
        "supabase_user_id": call_owner(token_user, payload.get("user_id")),
    }

    logger.info(
        f"Outbound call initiated: stream_id={stream_id}, "
        f"llm_provider={llm_provider}, ws_url={ws_url}"
    )
    return JSONResponse({
        "stream_id": stream_id,
        "call_id": call_id,
        "websocket_url": ws_url,
        "llm_provider": llm_provider,
        "exotel_response": result,
    })


@voice_router.get("/calls")
async def get_calls():
    """List active/initiated calls for debugging."""
    return CALLS


@voice_router.get("/exotel/phone-numbers")
async def exotel_list_phone_numbers():
    """List incoming phone numbers on the Exotel account (when credentials are configured)."""
    sid = settings.EXOTEL_ACCOUNT_SID
    token = settings.EXOTEL_AUTH_TOKEN
    if not sid or not token:
        return JSONResponse(
            {
                "count": 0,
                "numbers": [],
                "note": "Set EXOTEL_SID and EXOTEL_API_TOKEN on the server to list numbers.",
            }
        )
    auth_b64 = base64.b64encode(f"{sid}:{token}".encode()).decode()
    base = settings.EXOTEL_API_BASE.rstrip("/")
    url = f"{base}/Accounts/{sid}/IncomingPhoneNumbers.json"
    session = await get_global_session()
    headers = {"Authorization": f"Basic {auth_b64}", "Accept": "application/json"}
    try:
        async with session.get(url, headers=headers) as resp:
            text = await resp.text()
            if resp.status >= 400:
                logger.warning("Exotel IncomingPhoneNumbers %s: %s", resp.status, text[:400])
                return JSONResponse(
                    {
                        "count": 0,
                        "numbers": [],
                        "note": f"Exotel returned HTTP {resp.status}. Verify EXOTEL_API_BASE for your region.",
                    }
                )
            try:
                body = json.loads(text)
            except json.JSONDecodeError:
                return JSONResponse(
                    {"count": 0, "numbers": [], "note": "Could not parse Exotel JSON response."}
                )
            entries = body.get("IncomingPhoneNumbers") or body.get("incoming_phone_numbers") or []
            if isinstance(entries, dict):
                inner = entries.get("IncomingPhoneNumber")
                if isinstance(inner, list):
                    entries = inner
                elif isinstance(inner, dict):
                    entries = [inner]
                else:
                    entries = [entries] if entries else []
            rows = []
            for e in entries:
                if not isinstance(e, dict):
                    continue
                pn = e.get("PhoneNumber") or e.get("phone_number")
                rows.append(
                    {
                        "phone_number": pn,
                        "id": e.get("Sid") or e.get("sid"),
                        "status": e.get("Status") or e.get("status"),
                    }
                )
            return JSONResponse({"count": len(rows), "numbers": rows})
    except Exception as e:
        logger.exception("exotel_list_phone_numbers: %s", e)
        return JSONResponse({"count": 0, "numbers": [], "note": str(e)})


@voice_router.post("/rag/upload-pdf")
async def upload_pdf_rag(
    file: UploadFile = File(...),
    metadata: Optional[str] = None
):
    """Upload a PDF file to the RAG knowledge base."""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG service not available")
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        pdf_bytes = await file.read()
        
        metadata_dict = None
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning(f"Invalid metadata JSON, ignoring: {metadata}")
        
        rag_service = get_rag_service()
        doc_id = rag_service.add_pdf(
            pdf_bytes=pdf_bytes,
            filename=file.filename,
            metadata=metadata_dict
        )
        
        logger.info(f"Successfully uploaded PDF: {file.filename} (doc_id: {doc_id})")
        
        return JSONResponse({
            "status": "success",
            "doc_id": doc_id,
            "filename": file.filename,
            "message": f"PDF '{file.filename}' has been added to the knowledge base"
        })
        
    except Exception as e:
        logger.exception(f"Error uploading PDF {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@voice_router.post("/rag/add-text")
async def add_text_document(
    text: str,
    source: str,
    metadata: Optional[str] = None
):
    """Add a text document to the RAG knowledge base."""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG service not available")
    
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text content cannot be empty")
    
    try:
        metadata_dict = None
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning(f"Invalid metadata JSON, ignoring: {metadata}")
        
        rag_service = get_rag_service()
        doc_id = rag_service.add_document(
            text=text,
            source=source,
            metadata=metadata_dict
        )
        
        logger.info(f"Successfully added text document: {source} (doc_id: {doc_id})")
        
        return JSONResponse({
            "status": "success",
            "doc_id": doc_id,
            "source": source,
            "message": f"Document '{source}' has been added to the knowledge base"
        })
        
    except Exception as e:
        logger.exception(f"Error adding text document {source}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@voice_router.get("/rag/search")
async def search_rag(query: str, top_k: int = 3):
    """Search the RAG knowledge base."""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG service not available")
    
    try:
        rag_service = get_rag_service()
        results = rag_service.search(query, top_k=top_k)
        
        return JSONResponse({
            "query": query,
            "results": results,
            "count": len(results)
        })
        
    except Exception as e:
        logger.exception(f"Error searching RAG: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")

