#!/usr/bin/env python3
"""
Local mic/speaker test client for the Telnyx Pipecat WebSocket (`/ws/telnyx/{id}`).

Mirrors ``test_voice_websocket.py`` but speaks the **Telnyx** JSON protocol:
``connected`` → ``start`` → ``media`` frames with **PCMU (μ-law)** payloads, matching
what Telnyx sends on the wire and what ``TelnyxFrameSerializer`` expects.

Usage (API must be running, same as production Telnyx pipeline):

    uv run python test_telnyx_websocket.py
    uv run python test_telnyx_websocket.py --url ws://localhost:8000 --direction outbound

Requires: ``sounddevice``, ``websockets``, ``numpy`` (same as Exotel test).
"""

from __future__ import annotations

import argparse
import asyncio
import audioop
import base64
import json
import os
import queue
import threading
import time
import uuid
from typing import Optional

import numpy as np
import sounddevice as sd
import websockets

SAMPLE_RATE = 8000
CHANNELS = 1
CHUNK_DURATION_MS = 200
CHUNK_FRAMES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
BYTES_PER_SAMPLE = 2

mic_queue: queue.Queue = queue.Queue()
play_pcm_buffer = bytearray()
play_buffer_lock = threading.Lock()


def mic_callback(indata, frames, time_info, status):
    if status:
        print(f"Mic status: {status}")
    pcm_data = (indata * 32767).astype(np.int16)
    mic_queue.put(pcm_data.tobytes())


def speaker_callback(outdata, frames, time_info, status):
    global play_pcm_buffer
    required_bytes = frames * CHANNELS * BYTES_PER_SAMPLE
    with play_buffer_lock:
        available = len(play_pcm_buffer)
        if available > 0:
            take = min(available, required_bytes)
            pcm_chunk = play_pcm_buffer[:take]
            del play_pcm_buffer[:take]
            if take < required_bytes:
                pcm_chunk += bytes(required_bytes - take)
        else:
            pcm_chunk = bytes(required_bytes)
    audio_array = np.frombuffer(pcm_chunk, dtype=np.int16)
    outdata[:] = (audio_array.astype(np.float32) / 32767.0).reshape(frames, CHANNELS)


async def send_telnyx_audio(ws, stream_id: str):
    """PCM from mic → μ-law → Telnyx ``media`` JSON (same as PSTN PCMU track)."""
    while True:
        try:
            pcm_bytes = await asyncio.to_thread(mic_queue.get)
            ulaw_chunk = audioop.lin2ulaw(pcm_bytes, 2)
            payload = base64.b64encode(ulaw_chunk).decode("utf-8")
            msg = {
                "event": "media",
                "sequence_number": str(int(time.time() * 1000)),
                "stream_id": stream_id,
                "media": {
                    "track": "inbound",
                    "payload": payload,
                },
            }
            await ws.send(json.dumps(msg))
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket closed during send.")
            break
        except Exception as e:
            print(f"Error in send_telnyx_audio: {e}")
            break


async def receive_telnyx_events(ws):
    """Decode outbound PCMU from server → PCM for speakers."""
    async for message in ws:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            print(f"Non-JSON: {message[:120]!r}")
            continue

        ev = data.get("event", "")
        if ev == "media":
            payload = (data.get("media") or {}).get("payload", "")
            if not payload:
                continue
            try:
                ulaw = base64.b64decode(payload)
                pcm = audioop.ulaw2lin(ulaw, 2)
                with play_buffer_lock:
                    play_pcm_buffer.extend(pcm)
                print(f"Received audio: {len(pcm)} PCM bytes")
            except Exception as e:
                print(f"Decode media error: {e}")

        elif ev == "mark":
            print(f"Mark: {data.get('mark', {})}")

        elif ev == "clear":
            with play_buffer_lock:
                play_pcm_buffer.clear()
            print("Clear — playback buffer flushed")

        elif ev == "stop":
            print("Stop — closing")
            break

        elif ev in ("connected", "start"):
            print(f"Server/shared event echo or ignore: {ev}")

        else:
            print(f"Event: {ev}")


async def main(
    url: str = "ws://localhost:8000",
    stream_id: Optional[str] = None,
    direction: str = "outbound",
    llm_provider: str = "openai",
):
    if not stream_id:
        stream_id = str(uuid.uuid4())

    q = f"direction={direction}&llm_provider={llm_provider}"
    ws_url = f"{url.rstrip('/')}/ws/telnyx/{stream_id}?{q}"

    print("=" * 60)
    print("Telnyx-format voice WebSocket test (local mic / speakers)")
    print("=" * 60)
    print(f"URL: {ws_url}")
    print(f"stream_id (use same in start + media): {stream_id}")
    print("=" * 60)

    try:
        async with websockets.connect(ws_url) as ws:
            print("Connected.")

            await ws.send(json.dumps({"event": "connected", "version": "1.0.0"}))

            await ws.send(
                json.dumps(
                    {
                        "event": "start",
                        "sequence_number": "1",
                        "stream_id": stream_id,
                        "start": {
                            "call_control_id": "local-test",
                            "media_format": {
                                "encoding": "PCMU",
                                "sample_rate": 8000,
                                "channels": 1,
                            },
                        },
                    }
                )
            )
            print("Sent start — speak after the greeting plays.")

            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=CHUNK_FRAMES,
                callback=mic_callback,
            ), sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=CHUNK_FRAMES,
                callback=speaker_callback,
            ):
                await asyncio.gather(
                    send_telnyx_audio(ws, stream_id),
                    receive_telnyx_events(ws),
                )

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"Error: {e}")
        raise



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test /ws/telnyx with mic (Telnyx PCMU protocol)")
    parser.add_argument(
        "--url",
        default=os.getenv("WS_URL", "ws://localhost:8000"),
        help="WebSocket origin (no path), e.g. ws://localhost:8000",
    )
    parser.add_argument("--stream-id", default=None, help="UUID stream id (default: random)")
    parser.add_argument(
        "--direction",
        default="outbound",
        help="Prompt/pipeline flavor: outbound | inbound",
    )
    parser.add_argument("--llm", default="openai", help="llm_provider query param")

    args = parser.parse_args()
    try:
        asyncio.run(
            main(
                url=args.url,
                stream_id=args.stream_id,
                direction=args.direction,
                llm_provider=args.llm,
            )
        )
    except KeyboardInterrupt:
        print("\nExiting.")
