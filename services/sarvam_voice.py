"""
Sarvam STT/TTS for every call pipeline (Exotel, Vobiz, Telnyx, web).

Models per https://docs.sarvam.ai/api-reference-docs/getting-started/models (Oct 2026):
- STT saaras:v4 — Sarvam's recommended ASR; language left unset = auto-detect across 22 Indian
  languages + English. saarika:v2.5 is gone (pipecat >=1.x rejects it).
- TTS bulbul:v3 — the API default; covers all 11 languages LanguageDetectTTSSwitcher switches
  between. bulbul:v2 is rejected by the API. bulbul:v4-flash is lower latency but drops
  Malayalam and Odia, so it is not the default.
"""

from loguru import logger
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService, get_speakers_for_model
from pipecat.transcriptions.language import Language

from config import settings

STT_MODEL = "saaras:v4"
TTS_MODEL = "bulbul:v3"
DEFAULT_VOICE = "priya"  # bulbul:v3 female voice; v2 names (anushka, …) are not valid here


def tts_voice() -> str:
    """SARVAM_VOICE_ID if bulbul:v3 knows it, else DEFAULT_VOICE (old v2 names would fail every call)."""
    voice = (settings.SARVAM_VOICE_ID or "").strip().lower()
    if voice in get_speakers_for_model(TTS_MODEL):
        return voice
    if voice:
        logger.warning(f"[Sarvam] SARVAM_VOICE_ID={voice!r} is not a {TTS_MODEL} speaker; using {DEFAULT_VOICE!r}")
    return DEFAULT_VOICE


def make_stt() -> SarvamSTTService:
    return SarvamSTTService(
        api_key=settings.SARVAM_API_KEY,
        settings=SarvamSTTService.Settings(model=STT_MODEL),
    )


def make_tts(sample_rate: int, language: Language = Language.HI) -> SarvamTTSService:
    """Streaming TTS starting in `language`; LanguageDetectTTSSwitcher retargets it per caller."""
    return SarvamTTSService(
        api_key=settings.SARVAM_API_KEY,
        sample_rate=sample_rate,
        settings=SarvamTTSService.Settings(model=TTS_MODEL, voice=tts_voice(), language=language),
    )
