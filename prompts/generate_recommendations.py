"""
Prompt for generate_recommendations tool.
Converts clinical assessment into doctor actionables and patient instructions (Hindi + English).
"""

def get_prompt(diagnosis: dict, triage: dict, differential_diagnosis: dict, risk_scores: dict) -> str:
    """
    Generate prompt for generating recommendations.
    
    Args:
        diagnosis: Diagnosis JSON object
        triage: Triage JSON object
        differential_diagnosis: Differential diagnosis JSON object
        risk_scores: Risk scores JSON object
    
    Returns:
        Formatted prompt string
    """
    return f"""
You are a recommendation specialist. Convert clinical assessment into actionable recommendations.

Inputs:
- DIAGNOSIS: {diagnosis}
- TRIAGE: {triage}
- DIFFERENTIAL_DIAGNOSIS: {differential_diagnosis}
- RISK_SCORES: {risk_scores}

Generate:
1. Doctor actionables (for healthcare providers)
2. Patient instructions in Hindi (for patient communication)
3. Patient instructions in English (for documentation)
4. Test suggestions with priority
5. Referral level and type
6. Follow-up instructions

Return a JSON object with keys:
- for_patient_hi: string (in Hindi/Devanagari script, detailed patient instructions)
- tests_suggested: array of strings (test names in English, e.g., ["CBC", "Typhidot", "Platelet Count"])
- referral_suggestion: object with:
  - urgency: string (e.g., "within 48 hours", "immediate", "within 24 hours")
  - recommended_center: string (in Hindi, e.g., "नजदीकी प्राथमिक स्वास्थ्य केंद्र (PHC)")
- doctor_actionables: list of strings (in English, actionable items for doctor, e.g., ["Advise paracetamol + rest", "Refer to PHC if symptoms persist >48h"])

IMPORTANT:
- for_patient_hi must be in Hindi and provide clear, actionable instructions
- tests_suggested should be standard test names in English
- referral_suggestion.urgency should be in English format
- referral_suggestion.recommended_center should be in Hindi

Respond ONLY with a JSON object.
"""

