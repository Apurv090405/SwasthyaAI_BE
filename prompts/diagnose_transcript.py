"""
Prompt for diagnose_transcript tool.
Analyzes Hindi patient transcripts and extracts symptoms, red flags, and probable conditions.
"""

def get_prompt(transcript: str) -> str:
    """
    Generate prompt for diagnosing Hindi patient transcript.
    
    Args:
        transcript: Hindi patient transcript (Devanagari script)
    
    Returns:
        Formatted prompt string
    """
    return f"""
You are a clinical diagnosis assistant fluent in Hindi and English. The patient transcript below is in Hindi (Devanagari script).

Analyze the Hindi transcript and extract detailed medical information. Return a JSON object with keys:
- symptoms: list of symptom objects, each with:
  - name: string (in Hindi, e.g., "बुखार", "खांसी")
  - duration_days: integer or null (extract from transcript)
  - onset_description: string (in Hindi, e.g., "3 दिन पहले", "कल से")
  - severity: string (in Hindi: "हल्की", "मध्यम", "गंभीर")
  - type: string (in Hindi, optional, e.g., "सूखी" for cough, "गीली" for wet cough)
  - source: string (e.g., "utt_3") or null
- red_flags: list of any emergency warnings (in Hindi if present, empty list if none)
- probable_conditions: list of probable condition names in English medical terminology (short keywords)
- summary: short 2-3 sentence clinical summary in English
- confidence: float 0-1
- vitals: object with:
  - temperature_f: float or null (extract if mentioned, convert Celsius to Fahrenheit if needed)
  - spo2_percent: integer or null
  - heart_rate: integer or null
- past_medical_history: object with:
  - diabetes: boolean (extract from transcript)
  - hypertension: boolean (extract from transcript)
  - medications: list of strings (extract if mentioned)
  - allergies: list of strings (extract if mentioned)
  - prior_infections: list of strings (extract if mentioned)

IMPORTANT: 
- The transcript is in Hindi. Understand the Hindi text and extract medical information accurately.
- Preserve Hindi text for symptom names and descriptions.
- Extract duration, severity, and other metadata from the Hindi transcript.
- Extract vitals if mentioned (temperature, SpO2, heart rate).
- Extract medical history if mentioned.

Hindi Transcript:
\"\"\"{transcript}\"\"\"

Respond ONLY with a JSON object. Symptom names should be in Hindi, but condition names should be in English medical terminology.
"""

