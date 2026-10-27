"""
Dynamic language detection and TTS language switching for voice pipelines.

Listens for TranscriptionFrame language tags from Sarvam STT (auto-detect mode)
and:
1. Pushes TTSUpdateSettingsFrame to switch TTS language on the fly
2. Pushes LLMMessagesAppendFrame to tell the LLM to respond in the detected language

Uses Pipecat 1.x typed settings deltas (the `settings={...}` dict form is deprecated).
"""

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMMessagesAppendFrame,
    TranscriptionFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import TTSSettings
from pipecat.transcriptions.language import Language

from services.sarvam_voice import TTS_FALLBACK_MODEL, V4_FLASH_UNSUPPORTED, tts_model, tts_voice


# Languages bulbul:v3 can speak; bulbul:v4-flash lacks ML/OR (see services/sarvam_voice.py)
SARVAM_TTS_LANGUAGES = {
    Language.HI, Language.HI_IN,
    Language.BN, Language.BN_IN,
    Language.GU, Language.GU_IN,
    Language.KN, Language.KN_IN,
    Language.ML, Language.ML_IN,
    Language.MR, Language.MR_IN,
    Language.TA, Language.TA_IN,
    Language.TE, Language.TE_IN,
    Language.PA, Language.PA_IN,
    Language.OR, Language.OR_IN,
    Language.EN, Language.EN_IN, Language.EN_US,
}

# Map base language codes to their Indian locale variant for Sarvam
_BASE_TO_INDIAN = {
    Language.HI: Language.HI_IN,
    Language.BN: Language.BN_IN,
    Language.GU: Language.GU_IN,
    Language.KN: Language.KN_IN,
    Language.ML: Language.ML_IN,
    Language.MR: Language.MR_IN,
    Language.TA: Language.TA_IN,
    Language.TE: Language.TE_IN,
    Language.PA: Language.PA_IN,
    Language.OR: Language.OR_IN,
    Language.EN: Language.EN_IN,
    Language.EN_US: Language.EN_IN,
}

# Human-readable language names for the LLM instruction
_LANG_NAMES = {
    Language.HI_IN: "Hindi",
    Language.BN_IN: "Bengali",
    Language.GU_IN: "Gujarati",
    Language.KN_IN: "Kannada",
    Language.ML_IN: "Malayalam",
    Language.MR_IN: "Marathi",
    Language.TA_IN: "Tamil",
    Language.TE_IN: "Telugu",
    Language.PA_IN: "Punjabi",
    Language.OR_IN: "Odia",
    Language.EN_IN: "English",
}


def normalize_to_sarvam_lang(lang: Language) -> Language:
    """Normalize a Language enum to a Sarvam-compatible Indian locale."""
    return _BASE_TO_INDIAN.get(lang, lang)


class LanguageDetectTTSSwitcher(FrameProcessor):
    """Pipecat processor that detects language from STT transcription frames
    and dynamically switches both TTS language and LLM response language.

    Place this in the pipeline AFTER stt_service and BEFORE context_aggregator.
    """

    def __init__(self, default_language: Language = Language.HI_IN, **kwargs):
        super().__init__(**kwargs)
        self._current_language = default_language
        self._switch_count = 0
        self._base_model = tts_model()
        self._current_model = self._base_model

    def _tts_delta(self, language: Language) -> TTSSettings:
        model = (
            TTS_FALLBACK_MODEL
            if self._base_model == "bulbul:v4-flash" and language in V4_FLASH_UNSUPPORTED
            else self._base_model
        )
        if model == self._current_model:
            return TTSSettings(language=language)
        logger.info(f"[LanguageDetect] {language} needs TTS model {model} (was {self._current_model})")
        self._current_model = model
        return TTSSettings(language=language, model=model, voice=tts_voice(model))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.language:
            detected = frame.language
            normalized = normalize_to_sarvam_lang(detected)

            if normalized in SARVAM_TTS_LANGUAGES and normalized != self._current_language:
                self._current_language = normalized
                self._switch_count += 1
                lang_name = _LANG_NAMES.get(normalized, str(normalized))

                logger.info(
                    f"[LanguageDetect] Detected: {detected} -> "
                    f"switching to {lang_name} ({normalized}) (switch #{self._switch_count})"
                )

                # 1. Switch TTS language (and model, when bulbul:v4-flash can't speak it)
                await self.push_frame(
                    TTSUpdateSettingsFrame(delta=self._tts_delta(normalized)),
                    FrameDirection.DOWNSTREAM,
                )

                # 2. Tell the LLM to respond in the detected language
                lang_instruction = (
                    f"[LANGUAGE SWITCH] The caller is now speaking in {lang_name}. "
                    f"You MUST respond in {lang_name} from now on. "
                    f"Do NOT respond in Hindi or any other language — use {lang_name} only."
                )
                llm_frame = LLMMessagesAppendFrame(
                    messages=[{"role": "system", "content": lang_instruction}],
                    run_llm=False,  # Don't trigger LLM — just add to context for next turn
                )
                await self.push_frame(llm_frame, FrameDirection.DOWNSTREAM)

        await self.push_frame(frame, direction)
