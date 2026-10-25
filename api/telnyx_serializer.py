"""Telnyx WebSocket serializer aligned with current Media Stream protocol.

Pipecat's TelnyxFrameSerializer omits ``stream_id`` on outbound ``media`` frames.
Telnyx assigns the real id in a ``start`` event; without it, bidirectional RTP
often fails quickly with the socket closing.

Also signals an asyncio.Event when ``start`` is received so the voice pipeline
can defer TTS until the stream is ready.
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from loguru import logger
from pipecat.frames.frames import Frame
from pipecat.serializers.telnyx import TelnyxFrameSerializer


class TelnyxMediaStreamSerializer(TelnyxFrameSerializer):
    """Extends Pipecat Telnyx serializer with stream_id from ``start`` events."""

    def __init__(
        self,
        stream_id: str,
        outbound_encoding: str,
        inbound_encoding: str,
        call_control_id: Optional[str] = None,
        api_key: Optional[str] = None,
        params: Optional[TelnyxFrameSerializer.InputParams] = None,
        stream_ready: Optional[asyncio.Event] = None,
    ):
        super().__init__(
            stream_id,
            outbound_encoding,
            inbound_encoding,
            call_control_id,
            api_key,
            params,
        )
        self._stream_ready = stream_ready
        self._telnyx_ws_stream_id: Optional[str] = None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        message = json.loads(data)
        ev = message.get("event")

        if ev == "connected":
            logger.debug("Telnyx WS: connected")
            return None

        if ev == "start":
            sid = message.get("stream_id")
            if sid:
                self._telnyx_ws_stream_id = str(sid)
                logger.info(f"Telnyx WS: start, stream_id={self._telnyx_ws_stream_id}")
            if self._stream_ready and not self._stream_ready.is_set():
                self._stream_ready.set()
            return None

        if ev == "stop":
            logger.info("Telnyx WS: stop")
            return None

        if ev == "error":
            logger.error(f"Telnyx WS error frame: {message}")
            return None

        # Log media frame stats to debug audio flow
        if ev == "media":
            payload = message.get("media", {})
            track = payload.get("track", "?")
            if not hasattr(self, '_media_counts'):
                self._media_counts = {}
            self._media_counts[track] = self._media_counts.get(track, 0) + 1
            count = self._media_counts[track]
            # Log first frame per track, then every 100th
            if count == 1 or count % 100 == 0:
                logger.debug(
                    f"Telnyx WS: media track={track} frame#{count} "
                    f"chunk_size={len(payload.get('payload', ''))}"
                )

        return await super().deserialize(data)

    async def serialize(self, frame: Frame) -> str | bytes | None:
        out = await super().serialize(frame)
        if not out or not isinstance(out, str):
            return out
        if not self._telnyx_ws_stream_id:
            return out
        try:
            d = json.loads(out)
        except json.JSONDecodeError:
            return out
        ev = d.get("event")
        if ev == "media":
            d["stream_id"] = self._telnyx_ws_stream_id
            return json.dumps(d)
        if ev == "clear":
            d["stream_id"] = self._telnyx_ws_stream_id
            return json.dumps(d)
        return out
