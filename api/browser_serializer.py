"""
Raw PCM frame serializer for browser WebSocket audio.

Handles binary PCM 16-bit 16kHz audio frames sent from the browser
and serializes TTS audio frames back as raw binary PCM.
"""

import json

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer


class BrowserPCMSerializer(FrameSerializer):
    """Simple serializer for raw PCM 16-bit audio from browser WebSocket.

    - Deserializes: binary WebSocket messages → InputAudioRawFrame
    - Serializes: OutputAudioRawFrame → raw binary PCM bytes
      (Pipecat 1.x sends bytes as binary frames; the old `type` property is gone)
    - Serializes: InterruptionFrame → text '{"type": "clear"}' so the browser drops queued
      bot audio the moment the caller barges in
    """

    def __init__(self, sample_rate: int = 16000, num_channels: int = 1):
        super().__init__()  # base sets _params, which should_ignore_frame() reads
        self._sample_rate = sample_rate
        self._num_channels = num_channels

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio
        if isinstance(frame, InterruptionFrame):
            return json.dumps({"type": "clear"})
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, bytes) and len(data) > 0:
            return InputAudioRawFrame(
                audio=data,
                sample_rate=self._sample_rate,
                num_channels=self._num_channels,
            )
        return None
