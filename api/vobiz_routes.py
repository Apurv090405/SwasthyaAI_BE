"""Vobiz integration routes for Pipecat."""

import os
import uuid
import json
import base64
import asyncio
from typing import Dict, Any, Optional

import aiohttp
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, HTTPException, Query, BackgroundTasks, Body
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.serializers.plivo import PlivoFrameSerializer
from services.sarvam_voice import make_stt, make_tts
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.transcriptions.language import Language

from config import settings
from llm_factory import get_openai_compat_params
from api.voice_routes import get_global_session, load_prompt_from_file

vobiz_router = APIRouter(prefix="/vobiz", tags=["vobiz"])

# In-memory store for Vobiz calls
VOBIZ_CALLS: Dict[str, Dict[str, Any]] = {}
# In-memory store for agent config (persist to file in prod)
AGENT_CONFIG: Dict[str, str] = {}
CONFIG_FILE = "agent_config.json"

def load_agent_config():
    """Load agent config from file."""
    global AGENT_CONFIG
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                AGENT_CONFIG = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load agent config: {e}")

def save_agent_config():
    """Save agent config to file."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(AGENT_CONFIG, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save agent config: {e}")

# Load config on startup
load_agent_config()


@vobiz_router.get("/phone-numbers")
async def vobiz_list_phone_numbers():
    """List incoming numbers on the Vobiz account."""
    auth_id = settings.VOBIZ_AUTH_ID
    auth_token = settings.VOBIZ_AUTH_TOKEN
    if not auth_id or not auth_token:
        return JSONResponse(
            {
                "count": 0,
                "numbers": [],
                "note": "Set VOBIZ_AUTH_ID and VOBIZ_AUTH_TOKEN on the server to list numbers.",
            }
        )
    session = await get_global_session()
    headers = {
        "X-Auth-ID": auth_id,
        "X-Auth-Token": auth_token,
    }
    url = f"https://api.vobiz.ai/api/v1/Account/{auth_id}/IncomingPhoneNumber/"
    try:
        async with session.get(url, headers=headers) as resp:
            text = await resp.text()
            if resp.status >= 400:
                logger.warning("Vobiz list numbers %s: %s", resp.status, text[:400])
                return JSONResponse(
                    {
                        "count": 0,
                        "numbers": [],
                        "note": f"Vobiz API HTTP {resp.status}",
                    }
                )
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                return JSONResponse({"count": 0, "numbers": [], "note": "Invalid JSON from Vobiz"})
            objs = data.get("objects") or data.get("results") or data.get("data") or []
            rows = []
            for num in objs:
                if not isinstance(num, dict):
                    continue
                n = num.get("number") or num.get("phone_number")
                rows.append(
                    {
                        "phone_number": n,
                        "id": num.get("number_id") or num.get("id"),
                        "status": num.get("status"),
                    }
                )
            return JSONResponse({"count": len(rows), "numbers": rows})
    except Exception as e:
        logger.exception("vobiz_list_phone_numbers: %s", e)
        return JSONResponse({"count": 0, "numbers": [], "note": str(e)})


def get_websocket_url(request: Request, stream_id: str) -> str:
    """Construct WebSocket URL for Vobiz Stream."""
    # Use configured public URL or derive from request
    host = settings.HOST
    
    # If using ngrok/tunnel, we might need to be careful about http/https vs ws/wss
    # For Vobiz, it usually expects wss://
    
    # If HOST contains protocol, strip it
    if "://" in host:
        host = host.split("://")[1]
        
    return f"wss://{host}/ws/vobiz/{stream_id}"

async def make_vobiz_call_api(to_number: str, from_number: str, answer_url: str):
    """Make an outbound call using Vobiz's REST API."""
    auth_id = settings.VOBIZ_AUTH_ID
    auth_token = settings.VOBIZ_AUTH_TOKEN
    
    if not auth_id or not auth_token:
        raise ValueError("Vobiz credentials not configured")

    headers = {
        "Content-Type": "application/json",
        "X-Auth-ID": auth_id,
        "X-Auth-Token": auth_token,
    }

    data = {
        "to": to_number,
        "from": from_number,
        "answer_url": answer_url,
        "answer_method": "POST",
    }

    url = f"https://api.vobiz.ai/api/v1/Account/{auth_id}/Call/"
    
    session = await get_global_session()
    async with session.post(url, headers=headers, json=data) as response:
        if response.status != 201:
            text = await response.text()
            logger.error(f"Vobiz API error: {response.status} - {text}")
            raise Exception(f"Vobiz API failed: {text}")
        return await response.json()

async def run_vobiz_pipeline(
    websocket: WebSocket,
    stream_id: str,
    call_data: Dict[str, Any],
):
    """Run the Pipecat pipeline for Vobiz calls."""
    llm_provider = call_data.get("llm_provider", "openai")
    logger.info(f"Starting Vobiz pipeline for stream {stream_id}, llm_provider={llm_provider}")

    # Load prompt — use per-number override if configured
    to_number = call_data.get("data", {}).get("To") or call_data.get("data", {}).get("to")

    prompt_content = None
    if to_number and to_number in AGENT_CONFIG:
        logger.info(f"Using specific prompt for number: {to_number}")
        prompt_content = AGENT_CONFIG[to_number]

    if not prompt_content:
        prompt_file = "outbound_prompt.md" if call_data.get("direction") == "outbound" else "inbound_prompt.md"
        prompt_content = load_prompt_from_file(prompt_file)

    system_prompt = prompt_content

    # Initialize LLM service — Gemini uses its OpenAI-compatible endpoint
    llm_params = get_openai_compat_params(llm_provider)
    llm_service = OpenAILLMService(**llm_params)
    
    stt_service = make_stt()  # auto-detects the caller's language
    tts_service = make_tts(sample_rate=8000)

    # Dynamic language switcher — updates TTS language when STT detects a new language
    from services.language_detect import LanguageDetectTTSSwitcher
    language_switcher = LanguageDetectTTSSwitcher(default_language=Language.HI)

    # Context
    messages = [{"role": "system", "content": system_prompt}]
    context = LLMContext(messages=messages)

    # Transport & Serializer
    # Vobiz uses Plivo-compatible XML and WebSocket framing
    serializer = PlivoFrameSerializer(
        stream_id=stream_id,
        call_id=call_data.get("call_uuid", ""),
        auth_id=settings.VOBIZ_AUTH_ID,
        auth_token=settings.VOBIZ_AUTH_TOKEN,
        params=PlivoFrameSerializer.InputParams(
            plivo_sample_rate=8000,
            auto_hang_up=True
        )
    )

    vad_analyzer = SileroVADAnalyzer(params=VADParams(start_secs=0.1, stop_secs=0.2))
    # Pipecat 1.x: VAD lives on the user aggregator; Smart Turn v3 is the default stop strategy.
    context_aggregator = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=vad_analyzer)
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False, # Crucial for telephony
            serializer=serializer,
        ),
    )

    # Pipeline
    processors = [
        transport.input(),
        stt_service,
        language_switcher,
        context_aggregator.user(),
        llm_service,
        tts_service,
        transport.output(),
        context_aggregator.assistant(),
    ]

    pipeline = Pipeline(processors)

    task = PipelineTask(
        pipeline=pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_connect(trans, client):
        logger.info(f"Vobiz client connected: {stream_id}")
        # Initial greeting
        greeting = "नमस्ते! मैं स्वास्थ्य AI से बोल रहा हूँ। मैं आपकी कैसे मदद कर सकता हूँ?"
        await task.queue_frames([TTSSpeakFrame(greeting)])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnect(trans, client):
        logger.info(f"Vobiz client disconnected: {stream_id}")
        await task.cancel()

    runner = PipelineRunner()
    try:
        await runner.run(task)
    except Exception as e:
        logger.error(f"Pipeline error for {stream_id}: {e}")
    finally:
        # Cleanup
        VOBIZ_CALLS.pop(stream_id, None)
        logger.info(f"Pipeline finished for {stream_id}")


@vobiz_router.post("/outbound")
async def vobiz_outbound(payload: Dict[str, Any], request: Request):
    """Initiate an outbound call via Vobiz.

    Accepts an optional `llm_provider` field ('openai' or 'gemini') in the
    JSON payload to select the LLM backend for this call. Defaults to 'openai'.
    """
    to_number = payload.get("to")
    from_number = payload.get("from") or settings.VOBIZ_PHONE_NUMBER
    llm_provider = payload.get("llm_provider", "openai")

    if not to_number:
        raise HTTPException(status_code=400, detail="Missing 'to' number")
    if not from_number:
        raise HTTPException(status_code=400, detail="Missing 'from' number (and VOBIZ_PHONE_NUMBER not set)")

    protocol = "https" if "localhost" not in settings.HOST else "http"
    host = settings.HOST
    answer_url = f"{protocol}://{host}/vobiz/answer"

    try:
        result = await make_vobiz_call_api(to_number, from_number, answer_url)
        call_uuid = result.get("request_uuid") or result.get("call_uuid")

        # Store provider so the WebSocket pipeline picks it up
        VOBIZ_CALLS[call_uuid or to_number] = {
            "direction": "outbound",
            "llm_provider": llm_provider,
        }

        return JSONResponse({
            "status": "initiated",
            "call_uuid": call_uuid,
            "to": to_number,
            "llm_provider": llm_provider,
        })
    except Exception as e:
        logger.exception("Failed to initiate Vobiz call")
        raise HTTPException(status_code=500, detail=str(e))


@vobiz_router.api_route("/answer", methods=["GET", "POST"])
async def vobiz_answer(request: Request):
    """Handle Vobiz answer webhook (inbound or outbound pickup)."""
    # Vobiz might send data as form or JSON depending on configuration
    # Usually POST with form data for webhooks
    try:
        data = await request.form()
    except:
        data = await request.json() if request.headers.get("content-type") == "application/json" else {}

    call_uuid = data.get("CallUUID") or str(uuid.uuid4())
    stream_id = str(uuid.uuid4())

    # Inherit llm_provider if stored from the outbound initiation
    prior_info = VOBIZ_CALLS.pop(call_uuid, {})
    llm_provider = prior_info.get("llm_provider", "openai")

    # Store call info including llm_provider for the WebSocket handler
    VOBIZ_CALLS[stream_id] = {
        "call_uuid": call_uuid,
        "status": "connecting",
        "llm_provider": llm_provider,
        "data": dict(data),
    }
    
    # Construct WebSocket URL
    ws_url = get_websocket_url(request, stream_id)
    
    # Generate XML
    xml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Speak>Connecting to Swasthya AI agent.</Speak>
    <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">
        {ws_url}
    </Stream>
</Response>'''

    return HTMLResponse(content=xml_content, media_type="application/xml")


@vobiz_router.api_route("/recording-finished", methods=["GET", "POST"])
async def vobiz_recording_finished(request: Request):
    """Handle recording finished callback."""
    data = await request.form()
    logger.info(f"Recording finished: {dict(data)}")
    return HTMLResponse("<Response></Response>", media_type="application/xml")


class InboundConfig(BaseModel):
    phone_number: str
    prompt: str


@vobiz_router.post("/config/inbound")
async def config_inbound_agent(config: InboundConfig):
    """Configure the inbound agent for a specific phone number."""
    try:
        # Update config
        AGENT_CONFIG[config.phone_number] = config.prompt
        save_agent_config()
        
        # Also update default if it's the first one or requested (optional logic)
        # For now, just saving specific mapping
        
        return JSONResponse({
            "status": "success", 
            "message": f"Inbound prompt updated for {config.phone_number}",
            "config": config.dict()
        })
    except Exception as e:
        logger.error(f"Failed to update inbound config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")


@vobiz_router.post("/setup")
async def setup_vobiz_application(request: Request):
    """
    Auto-configure Vobiz Application and Phone Number.
    1. Creates/Updates 'SwasthyaAI' Application with current webhook URL.
    2. Links VOBIZ_PHONE_NUMBER to this Application.
    """
    auth_id = settings.VOBIZ_AUTH_ID
    auth_token = settings.VOBIZ_AUTH_TOKEN
    phone_number = settings.VOBIZ_PHONE_NUMBER
    
    if not auth_id or not auth_token:
        raise HTTPException(status_code=500, detail="Vobiz credentials not configured")

    # Determine public URL
    protocol = "https" if "localhost" not in settings.HOST else "http"
    host = settings.HOST
    # If host has protocol, strip it
    if "://" in host:
        protocol = host.split("://")[0]
        host = host.split("://")[1]
        
    base_url = f"{protocol}://{host}"
    answer_url = f"{base_url}/vobiz/answer"
    
    logger.info(f"Configuring Vobiz with Answer URL: {answer_url}")
    
    session = await get_global_session()
    headers = {
        "X-Auth-ID": auth_id,
        "X-Auth-Token": auth_token,
        "Content-Type": "application/json"
    }
    
    try:
        # 1. Check/Create Application
        app_id = None
        apps_url = f"https://api.vobiz.ai/api/v1/Account/{auth_id}/Application/"
        
        async with session.get(apps_url, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                apps = data.get("objects", [])
                for app in apps:
                    if app.get("app_name") == "SwasthyaAI":
                        app_id = app.get("app_id")
                        break
        
        if app_id:
            # Update existing
            logger.info(f"Updating existing application: {app_id}")
            update_url = f"{apps_url}{app_id}/"
            await session.post(update_url, headers=headers, json={
                "answer_url": answer_url,
                "app_name": "SwasthyaAI"
            })
        else:
            # Create new
            logger.info("Creating new application 'SwasthyaAI'")
            async with session.post(apps_url, headers=headers, json={
                "app_name": "SwasthyaAI",
                "answer_url": answer_url
            }) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise Exception(f"Failed to create application: {text}")
                data = await resp.json()
                app_id = data.get("app_id")
        
        if not app_id:
            raise Exception("Failed to get App ID")
            
        # 2. Link Phone Number
        if phone_number:
            # Find number ID
            numbers_url = f"https://api.vobiz.ai/api/v1/Account/{auth_id}/IncomingPhoneNumber/"
            number_id = None
            
            async with session.get(numbers_url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    numbers = data.get("objects", [])
                    for num in numbers:
                        # Check exact match or with/without +
                        n = num.get("number", "")
                        if n == phone_number or n == f"+{phone_number}" or f"+{n}" == phone_number:
                            number_id = num.get("number_id") # Vobiz might use 'id' or 'number_id'
                            if not number_id:
                                number_id = num.get("id")
                            break
            
            if number_id:
                logger.info(f"Linking number {phone_number} (ID: {number_id}) to App {app_id}")
                update_num_url = f"{numbers_url}{number_id}/"
                async with session.post(update_num_url, headers=headers, json={
                    "app_id": app_id
                }) as resp:
                    if resp.status >= 400:
                        text = await resp.text()
                        logger.error(f"Failed to link number: {text}")
                        # Don't fail the whole request, just warn
            else:
                logger.warning(f"Phone number {phone_number} not found in Vobiz account")
        
        return JSONResponse({
            "status": "success",
            "message": "Vobiz configured successfully",
            "app_id": app_id,
            "answer_url": answer_url
        })
        
    except Exception as e:
        logger.exception("Vobiz setup failed")
        raise HTTPException(status_code=500, detail=str(e))

