"""
Sarvam STT/TTS for every call pipeline (Exotel, Vobiz, Telnyx, web).

Models per https://docs.sarvam.ai/api-reference-docs/getting-started/models (Oct 2026):
- STT saaras:v4 — Sarvam's recommended ASR; language left unset = auto-detect across 22 Indian
  languages + English. saarika:v2.5 is gone (pipecat >=1.x rejects it).
- TTS: SARVAM_TTS_MODEL, default bulbul:v3 — covers all 11 languages LanguageDetectTTSSwitcher
  switches between. bulbul:v4-flash is lower latency but drops Malayalam and Odia; when it is
  selected, the switcher falls back to bulbul:v3 for those two (see TTS_FALLBACK_MODEL).
  bulbul:v2 is rejected by the API.
"""

from loguru import logger
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import TTS_MODEL_CONFIGS, SarvamTTSService, get_speakers_for_model
from pipecat.transcriptions.language import Language

from config import settings

STT_MODEL = "saaras:v4"
TTS_FALLBACK_MODEL = "bulbul:v3"
DEFAULT_VOICE = "priya"  # bulbul:v3 female voice; v2 names (anushka, …) are not valid here

# Languages bulbul:v4-flash cannot speak
V4_FLASH_UNSUPPORTED = {Language.ML, Language.ML_IN, Language.OR, Language.OR_IN}


def tts_model() -> str:
    """SARVAM_TTS_MODEL if Pipecat knows it, else bulbul:v3."""
    model = (settings.SARVAM_TTS_MODEL or "").strip().lower()
    if model in TTS_MODEL_CONFIGS and model != "bulbul:v2":
        return model
    if model:
        logger.warning(f"[Sarvam] SARVAM_TTS_MODEL={model!r} is not supported; using {TTS_FALLBACK_MODEL!r}")
    return TTS_FALLBACK_MODEL


def tts_voice(model: str | None = None) -> str:
    """Speaker for `model`; old v2 names would fail every call, so they are replaced."""
    model = model or tts_model()
    voice = (settings.SARVAM_VOICE_ID or "").strip().lower()
    speakers = get_speakers_for_model(model)
    if not speakers:
        # bulbul:v4-flash: Pipecat has no speaker list, the API validates the name
        return voice or TTS_MODEL_CONFIGS[model].default_speaker
    if voice in speakers:
        return voice
    if voice:
        logger.warning(f"[Sarvam] SARVAM_VOICE_ID={voice!r} is not a {model} speaker; using {DEFAULT_VOICE!r}")
    return DEFAULT_VOICE


def make_stt() -> SarvamSTTService:
    return SarvamSTTService(
        api_key=settings.SARVAM_API_KEY,
        settings=SarvamSTTService.Settings(model=STT_MODEL),
    )


def make_tts(sample_rate: int, language: Language = Language.HI) -> SarvamTTSService:
    """Streaming TTS starting in `language`; LanguageDetectTTSSwitcher retargets it per caller."""
    model = tts_model()
    return SarvamTTSService(
        api_key=settings.SARVAM_API_KEY,
        sample_rate=sample_rate,
        settings=SarvamTTSService.Settings(model=model, voice=tts_voice(model), language=language),
    )
