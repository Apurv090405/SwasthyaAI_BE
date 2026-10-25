#!/usr/bin/env python3
"""
Real-time voice-to-voice WebSocket test client for app.py

This script tests the WebSocket endpoint at /ws/exotel/{stream_id} with:
- Real-time microphone input (captures your voice)
- Real-time speaker output (plays bot responses)
- Exotel frame format (JSON events with media, mark, start)
- 8kHz, 16-bit PCM audio format (linear16)

Requirements:
    pip install sounddevice websockets numpy
    OR
    uv pip install sounddevice websockets numpy

Note: This script uses linear16 PCM format (matching app.py TTS config).
      If you encounter audio issues, you may need to use µ-law encoding
      instead (see test_local_call.py for µ-law example).

Usage:
    python test_voice_websocket.py
    python test_voice_websocket.py --url ws://localhost:8000
    python test_voice_websocket.py --stream-id test123 --call-id call456

Before running:
    1. Make sure app.py is running (uvicorn app:app)
    2. Ensure your microphone and speakers are working
    3. Check that all environment variables are set in .env
"""

import asyncio
import json
import time
import base64
import websockets
import sounddevice as sd
import numpy as np
import queue
import threading
import uuid
import argparse
import os
from typing import Optional

# -------------------------------
# Configuration
# -------------------------------
SAMPLE_RATE = 8000        # 8 kHz sample rate (matching app.py)
CHANNELS = 1              # mono
CHUNK_DURATION_MS = 200   # send every 200ms
CHUNK_FRAMES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
BYTES_PER_SAMPLE = 2      # 16-bit PCM

# Thread-safe queues
mic_queue = queue.Queue()
play_pcm_buffer = bytearray()
play_buffer_lock = threading.Lock()

# -------------------------------
# Microphone callback: capture audio and queue it
# -------------------------------
def mic_callback(indata, frames, time_info, status):
    """Capture audio from microphone and convert to PCM bytes."""
    if status:
        print(f"Mic status: {status}")
    # Convert float32 to int16 PCM
    pcm_data = (indata * 32767).astype(np.int16)
    pcm_bytes = pcm_data.tobytes()
    mic_queue.put(pcm_bytes)

# -------------------------------
# Speaker callback: play audio from buffer
# -------------------------------
def speaker_callback(outdata, frames, time_info, status):
    """Play audio from buffer to speaker."""
    global play_pcm_buffer

    required_bytes = frames * CHANNELS * BYTES_PER_SAMPLE
    with play_buffer_lock:
        available = len(play_pcm_buffer)
        if available > 0:
            take = min(available, required_bytes)
            pcm_chunk = play_pcm_buffer[:take]
            del play_pcm_buffer[:take]
            if take < required_bytes:
                # Pad with silence if not enough data
                pcm_chunk += bytes(required_bytes - take)
        else:
            # No data available, output silence
            pcm_chunk = bytes(required_bytes)

    # Convert bytes to numpy array and normalize to float32
    audio_array = np.frombuffer(pcm_chunk, dtype=np.int16)
    outdata[:] = (audio_array.astype(np.float32) / 32767.0).reshape(frames, CHANNELS)

# -------------------------------
# Async: send microphone audio frames to WebSocket
# -------------------------------
async def send_audio(ws, stream_id: str, call_id: str):
    """Send audio chunks from microphone to WebSocket in Exotel format."""
    print("Starting audio send loop...")
    while True:
        try:
            # Get audio chunk from microphone queue
            pcm_bytes = await asyncio.to_thread(mic_queue.get)
            
            # Encode to base64 for Exotel media event
            b64_payload = base64.b64encode(pcm_bytes).decode("utf-8")
            
            # Create Exotel media event
            media_event = {
                "event": "media",
                "streamSid": stream_id,
                "callSid": call_id,
                "media": {
                    "payload": b64_payload,
                    "timestamp": int(time.time() * 1000),
                    "track": "inbound"
                }
            }
            
            # Send as JSON
            await ws.send(json.dumps(media_event))
            
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket closed during send_audio.")
            break
        except Exception as e:
            print(f"Error in send_audio: {e}")
            break

# -------------------------------
# Async: receive events from WebSocket and process audio
# -------------------------------
async def receive_events(ws):
    """Receive events from WebSocket and process media/mark events."""
    print("Starting event receive loop...")
    async for message in ws:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            print(f"Received non-JSON message: {message[:100]}")
            continue

        event_type = data.get("event", "")
        
        if event_type == "media":
            # Extract audio payload
            media = data.get("media", {})
            payload = media.get("payload", "")
            
            if payload:
                try:
                    # Decode base64 to PCM bytes
                    pcm_bytes = base64.b64decode(payload)
                    
                    # Add to playback buffer
                    with play_buffer_lock:
                        play_pcm_buffer.extend(pcm_bytes)
                    
                    print(f"Received audio chunk: {len(pcm_bytes)} bytes")
                    
                except Exception as e:
                    print(f"Error decoding media: {e}")

        elif event_type == "mark":
            # Handle mark events (for synchronization)
            mark = data.get("mark", {})
            mark_name = mark.get("name", "")
            if mark_name:
                print(f"Received mark event: {mark_name}")

        elif event_type == "clear":
            # Clear playback buffer
            with play_buffer_lock:
                play_pcm_buffer.clear()
            print("Received clear event → playback buffer cleared")

        elif event_type == "stop":
            print("Received stop event — closing.")
            break

        else:
            print(f"Unknown event type: {event_type}")

# -------------------------------
# Main function: connect WebSocket, start audio streams, run tasks
# -------------------------------
async def main(
    url: str = "ws://localhost:8000",
    stream_id: Optional[str] = None,
    call_id: Optional[str] = None
):
    """Main function to run the voice-to-voice test."""
    
    # Generate IDs if not provided
    if not stream_id:
        stream_id = str(uuid.uuid4())
    if not call_id:
        call_id = str(uuid.uuid4())
    
    # Construct WebSocket URL
    ws_url = f"{url}/ws/exotel/{stream_id}?call_id={call_id}"
    
    print("=" * 60)
    print("Voice-to-Voice WebSocket Test Client")
    print("=" * 60)
    print(f"WebSocket URL: {ws_url}")
    print(f"Stream ID: {stream_id}")
    print(f"Call ID: {call_id}")
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Chunk Size: {CHUNK_FRAMES} frames ({CHUNK_DURATION_MS}ms)")
    print("=" * 60)
    print("\nStarting connection...")
    print("Press Ctrl+C to stop\n")
    
    try:
        # Connect to WebSocket
        async with websockets.connect(ws_url) as ws:
            print(f"✓ Connected to {ws_url}")
            
            # Send start event (Exotel format)
            start_event = {
                "event": "start",
                "streamSid": stream_id,
                "callSid": call_id,
                "start": {
                    "streamSid": stream_id,
                    "callSid": call_id
                }
            }
            await ws.send(json.dumps(start_event))
            print("✓ Sent start event")
            
            # Start audio input/output streams
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=CHUNK_FRAMES,
                callback=mic_callback
            ), sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=CHUNK_FRAMES,
                callback=speaker_callback
            ):
                print("✓ Audio streams started")
                print("\n🎤 Speak into your microphone...")
                print("🔊 Bot responses will play through your speakers\n")
                
                # Run send and receive tasks concurrently
                await asyncio.gather(
                    send_audio(ws, stream_id, call_id),
                    receive_events(ws)
                )
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

# -------------------------------
# CLI Entry Point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test voice-to-voice WebSocket connection for app.py"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=os.getenv("WS_URL", "ws://localhost:8000"),
        help="WebSocket server URL (default: ws://localhost:8000)"
    )
    parser.add_argument(
        "--stream-id",
        type=str,
        default=None,
        help="Stream ID (default: auto-generated UUID)"
    )
    parser.add_argument(
        "--call-id",
        type=str,
        default=None,
        help="Call ID (default: auto-generated UUID)"
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(main(
            url=args.url,
            stream_id=args.stream_id,
            call_id=args.call_id
        ))
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

