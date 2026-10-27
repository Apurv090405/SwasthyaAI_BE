"""
End-to-end check of the voice pipeline, with fake provider APIs (no keys, no network).

Real: the FastAPI app under uvicorn, /web/start + /ws/web and /ws/exotel routes, serializers,
Silero VAD + Smart Turn on real speech, language switcher, live red-flag detector, the
OpenAI-compatible LLM service (talking to a local fake server), context aggregators, latency
observer, cached greeting, call webhook (HTTP, or in-process with E2E_INPROCESS=1).
Fake: Sarvam STT (fixed transcripts), Sarvam TTS (tone), greeting REST call, and the post-call
clinical pipeline (captured instead of calling OpenAI / Supabase).

Speech clips (macOS):
    mkdir -p test_audio && cd test_audio
    say -o u1.aiff "Hello. I have had chest pain and fever for two days."
    say -o u2.aiff "Yes, it is getting worse at night."
    for f in u1 u2; do
      afconvert -f WAVE -d LEI16@16000 -c 1 $f.aiff ${f}_16k.wav
      afconvert -f WAVE -d LEI16@8000 -c 1 $f.aiff ${f}_8k.wav
    done

Run (uses local ports 8898/8899; removes the webhook_events files it creates):
    uv run python test_e2e_voice_pipeline.py              # HTTP webhook path
    E2E_INPROCESS=1 uv run python test_e2e_voice_pipeline.py   # in-process post-call path
"""

import asyncio
import base64
import json
import os
import sys
import threading
import time
import uuid
import wave

BE = os.path.dirname(os.path.abspath(__file__))
AUDIO = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BE, "test_audio")
APP_PORT, LLM_PORT = 8899, 8898

os.environ.update(
    OPENAI_API_KEY="sk-test",
    SARVAM_API_KEY="test",
    OPENAI_BASE_URL=f"http://127.0.0.1:{LLM_PORT}/v1",
)
# E2E_INPROCESS=1: no WEBHOOK_URL, so payloads are ingested in-process instead of via HTTP
if os.environ.get("E2E_INPROCESS") != "1":
    os.environ["WEBHOOK_URL"] = f"http://127.0.0.1:{APP_PORT}/webhook"
else:
    os.environ.pop("WEBHOOK_URL", None)
os.chdir(BE)
sys.path.insert(0, BE)

import numpy as np
import uvicorn
import websockets
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", filter=lambda r: r["name"].startswith(("services", "api", "webhook")))

# ─── Fake OpenAI-compatible LLM server ──────────────────────────────────────
LLM_REQUESTS = []
llm_app = FastAPI()


@llm_app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    LLM_REQUESTS.append(body)
    last_user = next((m for m in reversed(body["messages"]) if m["role"] == "user"), {})
    reply = f"ठीक है, मैं समझ गई। कितने दिन से? (reply {len(LLM_REQUESTS)})"

    async def stream():
        for i, word in enumerate(reply.split(" ")):
            chunk = {
                "id": "c1", "object": "chat.completion.chunk", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ("" if i == 0 else " ") + word}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.02)
        done = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(done)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


# ─── Fake Sarvam STT / TTS ──────────────────────────────────────────────────
from pipecat.frames.frames import TranscriptionFrame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

TRANSCRIPTS = ["मुझे दो दिन से सीने में दर्द और बुखार है", "हाँ, रात को ज़्यादा होता है"]
STT_CALLS = []


class FakeSTT(SegmentedSTTService):
    async def run_stt(self, audio: bytes):
        text = TRANSCRIPTS[min(len(STT_CALLS), len(TRANSCRIPTS) - 1)]
        STT_CALLS.append(len(audio))
        await asyncio.sleep(0.15)  # pretend network
        yield TranscriptionFrame(text, "user", time_now_iso8601(), language=Language.HI_IN)


TTS_TEXTS = []


class FakeTTS(TTSService):
    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str):
        TTS_TEXTS.append(text)
        await self.start_ttfb_metrics()
        await asyncio.sleep(0.1)
        await self.stop_ttfb_metrics()
        sr = self.sample_rate
        t = np.arange(int(sr * 0.4)) / sr
        pcm = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16).tobytes()
        yield TTSAudioRawFrame(audio=pcm, sample_rate=sr, num_channels=1, context_id=context_id)


GREETING_SYNTH = []


async def fake_synthesize(text, sample_rate):
    GREETING_SYNTH.append(sample_rate)
    t = np.arange(int(sample_rate * 0.5)) / sample_rate
    return (np.sin(2 * np.pi * 300 * t) * 6000).astype(np.int16).tobytes()


import services.voice_pipeline as vp

vp.make_stt = lambda: FakeSTT()
vp.make_tts = lambda sample_rate: FakeTTS(sample_rate=sample_rate)
vp._synthesize = fake_synthesize

import services.post_call_pipeline as pcp

POSTCALL = []


async def fake_postcall(payload):
    POSTCALL.append(payload)


pcp.process_voice_call_webhook = fake_postcall

import app as app_module  # noqa: E402

# ─── Servers ────────────────────────────────────────────────────────────────


def serve(application, port):
    server = uvicorn.Server(uvicorn.Config(application, host="127.0.0.1", port=port, log_level="warning", ws="websockets"))
    threading.Thread(target=server.run, daemon=True).start()
    return server


def read_pcm(name):
    with wave.open(os.path.join(AUDIO, name)) as w:
        return w.readframes(w.getnframes()), w.getframerate()


async def stream_audio(send, pcm, rate, frame_ms=20):
    step = rate * 2 * frame_ms // 1000
    for i in range(0, len(pcm), step):
        await send(pcm[i : i + step])
        await asyncio.sleep(frame_ms / 1000)


async def silence(send, rate, secs):
    await stream_audio(send, b"\x00\x00" * int(rate * secs), rate)


FAIL = []


def check(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


async def web_call():
    import aiohttp

    print("\n== Web call (/web/start + /ws/web)")
    async with aiohttp.ClientSession() as http:
        async with http.post(f"http://127.0.0.1:{APP_PORT}/web/start?llm_provider=openai&user_id=u-web") as r:
            session = await r.json()
    stream_id = session["stream_id"]
    got_audio_before_user = 0
    audio_after = {1: 0, 2: 0}
    clears = 0
    phase = 0

    async with websockets.connect(f"ws://127.0.0.1:{APP_PORT}/ws/web/{stream_id}?llm_provider=openai", max_size=None) as ws:

        async def receiver():
            nonlocal got_audio_before_user, clears
            async for msg in ws:
                if isinstance(msg, str):
                    if json.loads(msg).get("type") == "clear":
                        clears += 1
                elif phase == 0:
                    got_audio_before_user += len(msg)
                else:
                    audio_after[phase] += len(msg)

        rx = asyncio.create_task(receiver())

        async def send(b):
            await ws.send(b)

        await silence(send, 16000, 1.5)  # greeting plays
        for idx, name in ((1, "u1_16k.wav"), (2, "u2_16k.wav")):
            phase = idx
            pcm, rate = read_pcm(name)
            await stream_audio(send, pcm, rate)
            await silence(send, 16000, 3.0)  # end of turn + bot reply
        await ws.close()
        rx.cancel()
    await asyncio.sleep(2.5)  # teardown → webhook → post-call

    check(got_audio_before_user >= 15000, f"greeting audio played from cache ({got_audio_before_user} bytes)")
    check(16000 in GREETING_SYNTH, "greeting was pre-synthesized at 16 kHz")
    check(len(STT_CALLS) >= 2, f"VAD + Smart Turn detected both utterances (STT calls={len(STT_CALLS)})")
    check(audio_after[1] > 0 and audio_after[2] > 0, f"bot replied after each turn (bytes {audio_after})")
    return stream_id


async def exotel_call():
    print("\n== Exotel call (/ws/exotel, 8 kHz)")
    before_stt, before_post = len(STT_CALLS), len(POSTCALL)
    stream_id, call_sid = "ex-" + uuid.uuid4().hex[:6], "CA123"
    media_out = 0
    async with websockets.connect(f"ws://127.0.0.1:{APP_PORT}/ws/exotel/{stream_id}?call_id={call_sid}", max_size=None) as ws:
        await ws.send(json.dumps({"event": "connected"}))
        await ws.send(json.dumps({"event": "start", "stream_sid": stream_id, "start": {"stream_sid": stream_id, "call_sid": call_sid,
                                   "media_format": {"encoding": "base64", "sample_rate": "8000"}}}))

        async def receiver():
            nonlocal media_out
            async for msg in ws:
                data = json.loads(msg)
                if data.get("event") == "media":
                    media_out += 1

        rx = asyncio.create_task(receiver())
        seq = 0

        async def send(b):
            nonlocal seq
            seq += 1
            await ws.send(json.dumps({"event": "media", "stream_sid": stream_id, "sequence_number": str(seq),
                                      "media": {"chunk": str(seq), "timestamp": str(seq * 20), "payload": base64.b64encode(b).decode()}}))

        await silence(send, 8000, 1.5)
        pcm, rate = read_pcm("u1_8k.wav")
        await stream_audio(send, pcm, rate)
        await silence(send, 8000, 3.0)
        await ws.send(json.dumps({"event": "stop", "stream_sid": stream_id}))
        await asyncio.sleep(0.5)
        await ws.close()
        rx.cancel()
    await asyncio.sleep(2.5)
    check(media_out > 0, f"Exotel media sent back to caller ({media_out} media events)")
    check(len(STT_CALLS) > before_stt, "Exotel utterance reached STT")
    check(len(POSTCALL) > before_post, "Exotel call reached the post-call pipeline")
    if len(POSTCALL) > before_post:
        p = POSTCALL[-1]
        check(p["call_info"]["provider"] == "exotel" and p["call_id"] == call_sid, f"Exotel payload provider/call_id ({p['call_info']['provider']}, {p['call_id']})")


async def main():
    events_dir = os.path.join(BE, "webhook_events")
    existing_events = set(os.listdir(events_dir)) if os.path.isdir(events_dir) else set()
    llm_server = serve(llm_app, LLM_PORT)
    app_server = serve(app_module.app, APP_PORT)
    for _ in range(100):
        if app_server.started and llm_server.started:
            break
        await asyncio.sleep(0.1)
    await asyncio.sleep(1.0)  # greeting warm-up task

    stream_id = await web_call()

    print("\n== Web call → webhook → post-call payload")
    web = [p for p in POSTCALL if p["stream_id"] == stream_id]
    check(len(web) == 1, f"exactly one post-call payload for the web call (got {len(web)})")
    if web:
        p = web[0]
        msgs = p["transcript"]["messages"]
        roles = [m["role"] for m in msgs]
        check(p["call_info"]["supabase_user_id"] == "u-web", "user id carried to post-call")
        check(p["call_info"]["provider"] == "web", f"provider=web ({p['call_info']['provider']})")
        check(roles[:2] == ["system", "assistant"], f"context starts with system + greeting ({roles[:3]})")
        check(roles.count("user") >= 2 and roles.count("assistant") >= 3, f"transcript has both turns ({roles})")
        check(TRANSCRIPTS[0] in p["transcript"]["full_text"], "user speech in full_text")
        lm = p["latency_metrics"]
        v2v = lm.get("voice_to_voice", {})
        check(v2v.get("count", 0) >= 2, f"voice-to-voice measured per turn ({v2v})")
        check({"llm", "tts"} <= set(lm.get("ttfb", {})), f"TTFB per service ({ {k: v.get('p50') for k, v in lm.get('ttfb', {}).items()} })")
        check(lm.get("first_bot_speech_secs") is not None, f"first bot speech latency ({lm.get('first_bot_speech_secs')})")
        flags = [f["category"] for f in p["call_info"].get("red_flags", [])]
        check("chest_pain" in flags, f"live red flag carried to post-call ({flags})")

    print("\n== Voice LLM request")
    check(len(LLM_REQUESTS) >= 2, f"LLM called per turn ({len(LLM_REQUESTS)})")
    if LLM_REQUESTS:
        r = LLM_REQUESTS[0]
        check(r["model"] == "gpt-4.1-mini", f"voice model ({r['model']})")
        check(r.get("max_tokens") == 150 or r.get("max_completion_tokens") == 150, "max tokens cap sent")
        check("EMERGENCY" in r["messages"][0]["content"], "inbound_prompt.md loaded (has emergency rules)")
        check(abs(r.get("temperature", 0) - 0.3) < 1e-6, f"temperature ({r.get('temperature')})")
        emergency = [m for m in r["messages"][1:] if m["role"] == "system" and m["content"].startswith("EMERGENCY")]
        check(bool(emergency), "EMERGENCY instruction in context for the chest-pain turn's LLM call")
        later = LLM_REQUESTS[-1]["messages"]
        check(sum(m["content"].startswith("EMERGENCY") for m in later if m["role"] == "system") == 1,
              "emergency instruction added once, not repeated every turn")

    await exotel_call()

    for name in set(os.listdir(events_dir)) - existing_events:  # files this run created
        os.remove(os.path.join(events_dir, name))
    app_server.should_exit = True
    llm_server.should_exit = True
    await asyncio.sleep(0.5)
    print(f"\n{'ALL PASSED' if not FAIL else f'{len(FAIL)} FAILED'}")
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(1 if FAIL else 0)


asyncio.run(main())
