"""
Live emergency (red-flag) detection inside the voice pipeline.

A deterministic keyword check on every caller transcript — no LLM call, so it adds no latency.
On the first mention of an emergency it adds an ``EMERGENCY`` system instruction to the context
*before* the transcript reaches the user aggregator, so the very next reply tells the caller to
call 108 / go to the nearest hospital. Detected flags are kept on the session and travel with the
call payload (``call_info.red_flags``) to the post-call report and the doctor's queue.

The clinical agents still do the full assessment after the call; this only makes sure nobody in
an emergency has to wait for it. Matching favours recall: a false alarm costs one extra sentence.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from loguru import logger
from pipecat.frames.frames import Frame, LLMMessagesAppendFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# category → phrases (Hindi Devanagari + romanised, English, and common Tamil / Marathi /
# Bengali / Gujarati forms). Lower-cased substring match.
RED_FLAGS: Dict[str, Tuple[str, ...]] = {
    "chest_pain": (
        "सीने में दर्द", "छाती में दर्द", "सीने में तेज", "seene mein dard", "chhati mein dard",
        "chest pain", "pain in my chest", "நெஞ்சு வலி", "छातीत दुख", "বুকে ব্যথা", "છાતીમાં દુખ",
    ),
    "unconscious": (
        "बेहोश", "होश नहीं", "behosh", "unconscious", "fainted", "passed out",
        "மயக்கம்", "बेशुद्ध", "অজ্ঞান", "બેભાન",
    ),
    "breathing": (
        "सांस नहीं", "साँस नहीं", "सांस लेने में", "साँस लेने में", "सांस फूल", "साँस फूल",
        "saans nahi", "can't breathe", "cannot breathe", "difficulty breathing", "breathless",
        "மூச்சு விட", "श्वास घेण्यास", "শ্বাস নিতে", "શ્વાસ લેવામાં",
    ),
    "heavy_bleeding": (
        "बहुत खून", "खून बह", "खून रुक नहीं", "khoon beh", "heavy bleeding", "bleeding a lot",
        "won't stop bleeding", "இரத்தம் நிற்க", "रक्तस्त्राव", "অনেক রক্ত", "ખૂબ લોહી",
    ),
    "snake_bite": (
        "सांप ने काट", "साँप ने काट", "सांप काट", "साँप काट", "saanp ne kaat", "snake bite",
        "snake bit", "பாம்பு கடி", "साप चावला", "সাপে কামড়", "સાપ કરડ",
    ),
    "poisoning": ("ज़हर खा", "जहर खा", "ज़हर पी", "जहर पी", "कीटनाशक", "poison", "pesticide"),
    "seizure": ("दौरा पड़", "दौरे पड़", "झटके आ", "seizure", "convulsion", "fits", "வலிப்பு", "फिट आ"),
    "pregnancy_danger": (
        "बच्चा हिल नहीं", "बच्चे की हलचल नहीं", "baby not moving", "baby is not moving",
        "प्रेग्नेंसी में खून", "गर्भ में खून", "bleeding in pregnancy", "pregnant and bleeding",
    ),
    "self_harm": (
        "आत्महत्या", "मरना चाहता", "मरना चाहती", "जान दे दूं", "suicide", "kill myself",
        "end my life", "தற்கொலை", "আত্মহত্যা", "આત્મહત્યા",
    ),
}

_EMERGENCY = (
    "EMERGENCY: the caller just described a possible {what}. In your very next reply, calmly and "
    "clearly tell them to call 108 now or go to the nearest hospital or PHC immediately, in their "
    "language, in one or two short sentences. Then ask if someone is with them who can help."
)
_SELF_HARM = (
    "EMERGENCY: the caller may be thinking of harming themselves. In your very next reply, speak "
    "gently, tell them they are not alone, and ask them to call Tele-MANAS 14416 or 112 right now. "
    "Keep it to one or two short sentences, in their language, and stay with them."
)
_LABELS = {
    "chest_pain": "heart emergency (chest pain)",
    "unconscious": "loss of consciousness",
    "breathing": "breathing emergency",
    "heavy_bleeding": "heavy bleeding",
    "snake_bite": "snake bite",
    "poisoning": "poisoning",
    "seizure": "seizure",
    "pregnancy_danger": "pregnancy danger sign",
}


def detect_red_flags(text: str) -> List[str]:
    """Categories mentioned in `text` (empty if none)."""
    lowered = (text or "").lower()
    return [cat for cat, phrases in RED_FLAGS.items() if any(p in lowered for p in phrases)]


class RedFlagDetector(FrameProcessor):
    """Place after STT / language detection and before the user aggregator."""

    def __init__(self, flags: List[Dict[str, str]], label: str = "call", **kwargs):
        super().__init__(**kwargs)
        self._flags = flags  # shared with the session, ends up in call_info["red_flags"]
        self._label = label
        self._alerted: set[str] = set()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            new = [c for c in detect_red_flags(frame.text) if c not in self._alerted]
            if new:
                self._alerted.update(new)
                self._flags.extend({"category": c, "text": frame.text[:200]} for c in new)
                logger.warning(f"[RedFlag] {self._label}: {new} in {frame.text[:80]!r}")
                instruction = (
                    _SELF_HARM
                    if "self_harm" in new
                    else _EMERGENCY.format(what=", ".join(_LABELS[c] for c in new))
                )
                # Pushed before the transcript, so it is in context when this turn's LLM call runs
                await self.push_frame(
                    LLMMessagesAppendFrame(
                        messages=[{"role": "system", "content": instruction}], run_llm=False
                    ),
                    FrameDirection.DOWNSTREAM,
                )

        await self.push_frame(frame, direction)
