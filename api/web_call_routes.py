"""
Browser-based web call routes for SwasthyaAI.

Provides a WebSocket endpoint that runs the Pipecat voice pipeline
directly from the browser — no telephony provider needed.
The browser captures mic audio, sends PCM frames over WebSocket,
and receives TTS audio back.

Compatible with Pipecat 0.0.94.
"""

import asyncio
import uuid

from fastapi import APIRouter, WebSocket
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.frames.frames import TTSSpeakFrame, InterruptionFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.openai.llm import OpenAILLMService
from services.sarvam_voice import make_stt, make_tts
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.transcriptions.language import Language

from llm_factory import get_openai_compat_params
from services.language_detect import LanguageDetectTTSSwitcher
from api.browser_serializer import BrowserPCMSerializer

try:
    from api.voice_routes import load_prompt_from_file, send_call_webhook
except ImportError:
    from voice_routes import load_prompt_from_file, send_call_webhook


web_call_router = APIRouter(prefix="/web", tags=["web-call"])

# Track active web call sessions
WEB_CALLS: dict[str, dict] = {}


@web_call_router.post("/start")
async def start_web_call(
    llm_provider: str = "openai",
    user_id: str = None,
):
    """Create a web call session and return the WebSocket URL to connect to."""
    stream_id = str(uuid.uuid4())

    WEB_CALLS[stream_id] = {
        "direction": "web",
        "llm_provider": llm_provider,
        "status": "waiting",
        "supabase_user_id": (user_id or "").strip() or None,
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

    logger.info(
        f"Starting WEB pipeline: stream={stream_id}, llm={llm_provider}"
    )

    system_prompt = load_prompt_from_file("inbound_prompt.md")

    # LLM
    llm_params = get_openai_compat_params(llm_provider)
    llm_service = OpenAILLMService(**llm_params)

    # STT auto-detects language; TTS starts in Hindi and follows the caller via the switcher
    stt_service = make_stt()
    tts_service = make_tts(sample_rate=16000)

    # Dynamic language switcher
    language_switcher = LanguageDetectTTSSwitcher(default_language=Language.HI)

    # Context
    messages = [{"role": "system", "content": system_prompt}]
    context = LLMContext(messages=messages)

    # Pipecat 1.x: VAD lives on the user aggregator; Smart Turn v3 is the default stop strategy.
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(start_secs=0.2, stop_secs=0.3)
    )
    context_aggregator = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=vad_analyzer)
    )

    # Serializer for raw browser PCM audio
    serializer = BrowserPCMSerializer(sample_rate=16000, num_channels=1)

    # Transport — browser WebSocket with PCM serializer
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    pipeline = Pipeline([
        transport.input(),
        stt_service,
        language_switcher,
        context_aggregator.user(),
        llm_service,
        tts_service,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline=pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_connect(trans, client):
        logger.info(f"Web call client connected: {stream_id}")
        greeting = (
            "\u0928\u092e\u0938\u094d\u0924\u0947, \u092e\u0948\u0902 SwasthyaAI "
            "\u0938\u0947 \u092c\u094b\u0932 \u0930\u0939\u0940 \u0939\u0942\u0901\u0964 "
            "\u0906\u092a \u092e\u0941\u091d\u0947 \u0905\u092a\u0928\u0940 "
            "\u0938\u094d\u0935\u093e\u0938\u094d\u0925\u094d\u092f \u0938\u092e\u0938\u094d\u092f\u093e "
            "\u092c\u0924\u093e \u0938\u0915\u0924\u0947 \u0939\u0948\u0902\u0964"
        )
        await task.queue_frames([TTSSpeakFrame(greeting)])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnect(trans, client):
        logger.info(f"Web call client disconnected: {stream_id}")
        await task.cancel()

    WEB_CALLS[stream_id]["status"] = "active"

    runner = PipelineRunner()
    try:
        await runner.run(task)
    except asyncio.CancelledError:
        logger.info(f"Web pipeline cancelled: {stream_id}")
    except Exception as e:
        logger.error(f"Web pipeline error for {stream_id}: {e}")
    finally:
        logger.info(f"Cleaning up web pipeline: {stream_id}")
        try:
            await task.queue_frames([InterruptionFrame()])
        except Exception:
            pass

        call_info = WEB_CALLS.get(stream_id, {}).copy()
        WEB_CALLS.pop(stream_id, None)

        # Send transcript, metrics, and call info to webhook (same as phone calls)
        try:
            await send_call_webhook(
                stream_id=stream_id,
                call_id=stream_id,
                context=context,
                task=task,
                call_info=call_info,
            )
        except Exception as e:
            logger.warning(f"Web call webhook send error (non-fatal): {e}")

        try:
            await websocket.close()
        except Exception:
            pass

        logger.info(f"Web pipeline closed: {stream_id}")
