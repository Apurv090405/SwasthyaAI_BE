"""
Shared Pipecat voice pipeline for every call route (Exotel, Vobiz, Telnyx, web).

    transport.in → Sarvam STT → LanguageDetectTTSSwitcher → RedFlagDetector
      → user aggregator (VAD + Smart Turn) → OpenAI-compatible LLM (fast VOICE_* model)
      → Sarvam TTS → transport.out → assistant aggregator

Each route only builds its own transport/serializer, then calls ``build_voice_session`` and
``run_voice_session``. This module also:
- collects per-turn voice-to-voice latency and STT/LLM/TTS TTFB (``LatencyCollector``), which
  ``send_call_webhook`` stores as ``latency_metrics``;
- plays greetings from pre-synthesized audio (``warm_greetings``) so the caller hears us at once.
"""

from __future__ import annotations

import asyncio
import base64
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from fastapi import WebSocket
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import InterruptionFrame, SpeechOutputAudioRawFrame, TTSSpeakFrame
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transcriptions.language import Language
from pipecat.workers.runner import WorkerRunner

from config import settings
from llm_factory import get_openai_compat_params
from services.language_detect import LanguageDetectTTSSwitcher
from services.red_flags import RedFlagDetector
from services.sarvam_voice import make_stt, make_tts, tts_model, tts_voice

# Greeting per call type. The persona is female (see inbound_prompt.md).
GREETINGS: Dict[str, str] = {
    "inbound": "नमस्ते, मैं SwasthyaAI से बोल रही हूँ। आप मुझे अपनी स्वास्थ्य समस्या बता सकते हैं।",
    "outbound": "नमस्ते, मैं SwasthyaAI से बोल रही हूँ। क्या यह बात करने का सही समय है?",
    "report_delivery": (
        "नमस्ते, मैं SwasthyaAI से बोल रही हूँ। आपने थोड़ी देर पहले हमसे बात की थी। "
        "आपकी रिपोर्ट डॉक्टर ने रिव्यू कर ली है। मैं आपको रिजल्ट बताती हूँ।"
    ),
}

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
TELEPHONY_SAMPLE_RATE = 8000
WEB_SAMPLE_RATE = 16000
_GREETING_CHUNK_MS = 20


def greeting_for(direction: Optional[str]) -> str:
    return GREETINGS.get(direction or "inbound", GREETINGS["inbound"])


# ─── Latency ────────────────────────────────────────────────────────────────


def _summary(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)  # nearest-rank
    return {
        "count": len(ordered),
        "avg": round(statistics.fmean(ordered), 3),
        "p50": round(statistics.median(ordered), 3),
        "p95": round(ordered[p95_index], 3),
        "max": round(ordered[-1], 3),
    }


def _service_kind(processor: str) -> str:
    name = processor.upper()
    for kind in ("STT", "LLM", "TTS"):
        if kind in name:
            return kind.lower()
    return processor


class LatencyCollector:
    """Wraps UserBotLatencyObserver and keeps every measurement for the call webhook."""

    def __init__(self, label: str):
        self._label = label
        self.observer = UserBotLatencyObserver()
        self.turns: List[float] = []
        self.first_bot_speech: Optional[float] = None
        self.user_turn: List[float] = []
        self.ttfb: Dict[str, List[float]] = defaultdict(list)

        @self.observer.event_handler("on_latency_measured")
        async def _on_latency(_observer, latency: float):
            self.turns.append(latency)
            logger.info(f"[Latency] {self._label} turn {len(self.turns)}: voice-to-voice {latency:.3f}s")

        @self.observer.event_handler("on_latency_breakdown")
        async def _on_breakdown(_observer, breakdown):
            for item in breakdown.ttfb:
                self.ttfb[_service_kind(item.processor)].append(item.duration_secs)
            if breakdown.user_turn_secs is not None:
                self.user_turn.append(breakdown.user_turn_secs)

        @self.observer.event_handler("on_first_bot_speech_latency")
        async def _on_first(_observer, latency: float):
            self.first_bot_speech = latency
            logger.info(f"[Latency] {self._label} first bot speech after {latency:.3f}s")

    def get_metrics(self) -> Dict[str, Any]:
        """Called by ``send_call_webhook``; stored as ``voice_calls.latency_metrics``."""
        return {
            "voice_to_voice": _summary(self.turns),
            "turns": [round(t, 3) for t in self.turns],
            "first_bot_speech_secs": round(self.first_bot_speech, 3) if self.first_bot_speech else None,
            "user_turn": _summary(self.user_turn),
            "ttfb": {kind: _summary(values) for kind, values in self.ttfb.items()},
        }


# ─── Greeting cache ─────────────────────────────────────────────────────────

# (text, sample_rate, model, voice) → raw PCM16 mono
_GREETING_AUDIO: Dict[Tuple[str, int, str, str], bytes] = {}
_warming: set = set()


def _cache_key(text: str, sample_rate: int) -> Tuple[str, int, str, str]:
    model = tts_model()
    return (text, sample_rate, model, tts_voice(model))


async def _synthesize(text: str, sample_rate: int) -> Optional[bytes]:
    """Sarvam REST TTS → PCM16 (same request Pipecat's SarvamHttpTTSService sends)."""
    if not settings.SARVAM_API_KEY:
        return None
    from api.voice_routes import get_global_session

    model = tts_model()
    payload = {
        "text": text,
        "target_language_code": "hi-IN",
        "speaker": tts_voice(model),
        "speech_sample_rate": sample_rate,
        "model": model,
        "pace": 1.0,
    }
    headers = {"api-subscription-key": settings.SARVAM_API_KEY, "Content-Type": "application/json"}
    session = await get_global_session()
    async with session.post(SARVAM_TTS_URL, json=payload, headers=headers) as response:
        if response.status != 200:
            logger.warning(f"[Greeting] Sarvam TTS {response.status}: {(await response.text())[:200]}")
            return None
        data = await response.json()
    audios = data.get("audios") or []
    if not audios:
        return None
    audio = base64.b64decode(audios[0])
    if len(audio) > 44 and audio.startswith(b"RIFF"):
        audio = audio[44:]
    return audio


async def _warm_one(text: str, sample_rate: int) -> None:
    key = _cache_key(text, sample_rate)
    if key in _GREETING_AUDIO or key in _warming:
        return
    _warming.add(key)
    try:
        audio = await _synthesize(text, sample_rate)
        if audio:
            _GREETING_AUDIO[key] = audio
            logger.info(f"[Greeting] cached {len(audio)} bytes @ {sample_rate} Hz: {text[:30]}…")
    except Exception as e:
        logger.warning(f"[Greeting] could not pre-synthesize greeting: {e}")
    finally:
        _warming.discard(key)


async def warm_greetings(sample_rates: Iterable[int] = (TELEPHONY_SAMPLE_RATE, WEB_SAMPLE_RATE)) -> None:
    """Pre-synthesize every greeting at every sample rate (called once from app lifespan)."""
    await asyncio.gather(
        *(_warm_one(text, rate) for text in set(GREETINGS.values()) for rate in sample_rates)
    )


def _greeting_frames(audio: bytes, sample_rate: int) -> List[SpeechOutputAudioRawFrame]:
    chunk = sample_rate * 2 * _GREETING_CHUNK_MS // 1000  # PCM16 mono
    return [
        SpeechOutputAudioRawFrame(audio=audio[i : i + chunk], sample_rate=sample_rate, num_channels=1)
        for i in range(0, len(audio), chunk)
    ]


# ─── Pipeline ───────────────────────────────────────────────────────────────


@dataclass
class VoiceSession:
    task: PipelineTask
    context: LLMContext
    latency: LatencyCollector
    sample_rate: int
    greeting: str
    label: str
    extra: Dict[str, Any] = field(default_factory=dict)

    async def speak_greeting(self) -> None:
        """Play the cached greeting audio; fall back to live TTS (and cache it for next time)."""
        audio = _GREETING_AUDIO.get(_cache_key(self.greeting, self.sample_rate))
        if audio:
            await self.task.queue_frames(_greeting_frames(audio, self.sample_rate))
            return
        asyncio.create_task(_warm_one(self.greeting, self.sample_rate))
        await self.task.queue_frames([TTSSpeakFrame(self.greeting)])


def make_voice_llm(llm_provider: str) -> OpenAILLMService:
    params = get_openai_compat_params(llm_provider, purpose="voice")
    return OpenAILLMService(
        api_key=params["api_key"],
        base_url=params.get("base_url"),
        settings=OpenAILLMService.Settings(
            model=params["model"],
            temperature=settings.VOICE_TEMPERATURE,
            max_tokens=settings.VOICE_MAX_TOKENS,
        ),
    )


def build_voice_session(
    *,
    transport,
    sample_rate: int,
    system_prompt: str,
    llm_provider: str = "openai",
    greeting: Optional[str] = None,
    label: str = "call",
    vad_start_secs: float = 0.1,
    vad_stop_secs: float = 0.2,
    extra_processors: Iterable[FrameProcessor] = (),
) -> VoiceSession:
    """Build the STT → LLM → TTS task for one call.

    VAD stop_secs=0.2 is Pipecat's recommendation with Smart Turn v3 (the default user-turn stop
    strategy in Pipecat 1.x), which decides end of turn; VAD lives on the user aggregator.
    ``extra_processors`` run after language detection and live red-flag detection, before the
    user aggregator.
    """
    greeting = greeting or GREETINGS["inbound"]
    red_flags: List[Dict[str, str]] = []  # filled live by RedFlagDetector

    # The greeting is in context so the LLM knows it already introduced itself.
    context = LLMContext(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": greeting},
        ]
    )
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(start_secs=vad_start_secs, stop_secs=vad_stop_secs)
            )
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            make_stt(),  # auto-detects the caller's language
            LanguageDetectTTSSwitcher(default_language=Language.HI),
            RedFlagDetector(red_flags, label=label),
            *extra_processors,
            context_aggregator.user(),
            make_voice_llm(llm_provider),
            make_tts(sample_rate=sample_rate),
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    latency = LatencyCollector(label)
    task = PipelineTask(
        pipeline=pipeline,
        params=PipelineParams(
            audio_in_sample_rate=sample_rate,
            audio_out_sample_rate=sample_rate,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[latency.observer],
    )
    logger.info(
        f"[Voice] {label}: llm={llm_provider}/{get_openai_compat_params(llm_provider)['model']} "
        f"tts={tts_model()} sample_rate={sample_rate}"
    )
    return VoiceSession(
        task=task,
        context=context,
        latency=latency,
        sample_rate=sample_rate,
        greeting=greeting,
        label=label,
        extra={"red_flags": red_flags},
    )


async def run_voice_session(
    session: VoiceSession,
    websocket: WebSocket,
    *,
    stream_id: str,
    pop_call_info: Callable[[], Dict[str, Any]],
    call_id: Optional[str] = None,
    call_id_from_info: Optional[Callable[[Dict[str, Any]], str]] = None,
) -> None:
    """Run the task, then always send the call webhook (transcript + latency) and close the socket."""
    from api.voice_routes import send_call_webhook

    try:
        # handle_sigint=False: we run inside uvicorn, whose Ctrl+C/SIGTERM handling must stay intact
        await WorkerRunner(handle_sigint=False).run(session.task)
    except asyncio.CancelledError:
        logger.info(f"[Voice] {session.label} pipeline cancelled: {stream_id}")
    except Exception as e:
        logger.exception(f"[Voice] {session.label} pipeline error for {stream_id}: {e}")
    finally:
        logger.info(f"[Voice] cleaning up {session.label} pipeline: {stream_id}")
        try:
            await session.task.cancel()
        except Exception:
            pass
        try:
            await session.task.queue_frames([InterruptionFrame()])
        except Exception:
            pass

        call_info = pop_call_info()
        if session.extra:
            call_info = {**call_info, **session.extra}
        resolved_call_id = call_id or (call_id_from_info(call_info) if call_id_from_info else stream_id)

        summary = session.latency.get_metrics()["voice_to_voice"]
        if summary.get("count"):
            logger.info(
                f"[Latency] {session.label} {stream_id}: {summary['count']} turns, "
                f"p50={summary['p50']}s p95={summary['p95']}s"
            )

        try:
            await send_call_webhook(
                stream_id=stream_id,
                call_id=resolved_call_id,
                context=session.context,
                task=session.task,
                call_info=call_info,
                latency_observer=session.latency,
            )
        except Exception as e:
            logger.warning(f"[Voice] webhook send error (non-fatal): {e}")

        try:
            await websocket.close()
        except Exception:
            pass
        logger.info(f"[Voice] {session.label} pipeline closed: {stream_id}")
