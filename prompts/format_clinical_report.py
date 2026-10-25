"""
Prompt for format_clinical_report tool.
Compiles all intermediate outputs into a structured diagnosis report (JSON/FHIR format).
"""

def get_prompt(
    diagnosis: dict,
    triage: dict,
    differential_diagnosis: dict,
    risk_scores: dict,
    recommendations: dict,
    clinical_rules: dict = None,
    rag_hits: list = None,
    transcript: str = ""
) -> str:
    """
    Generate prompt for formatting clinical report.
    
    Args:
        diagnosis: Diagnosis JSON object
        triage: Triage JSON object
        differential_diagnosis: Differential diagnosis JSON object
        risk_scores: Risk scores JSON object
        recommendations: Recommendations JSON object
        clinical_rules: Clinical rules JSON object (optional)
        rag_hits: RAG hits from clinical guidelines (optional)
    
    Returns:
        Formatted prompt string
    """
    return f"""
You are a report formatting specialist. Compile all clinical assessments into a structured diagnosis report following the EXACT JSON structure specified below.

Inputs:
- DIAGNOSIS: {diagnosis}
- TRIAGE: {triage}
- DIFFERENTIAL_DIAGNOSIS: {differential_diagnosis}
- RISK_SCORES: {risk_scores}
- RECOMMENDATIONS: {recommendations}
- CLINICAL_RULES: {clinical_rules or {}}
- GUIDELINE_CITATIONS: {rag_hits or []}
- ORIGINAL_TRANSCRIPT: {transcript or "not provided"}

Create a comprehensive structured report in JSON format with EXACTLY these keys and structure:

{{
  "patient_id": string (from userid or generate),
  "pseudo_id": string (anonymized ID, e.g., "anon_XXXXX"),
  "language": "hi",
  "consent_status": "verified",
  "input_summary": {{
    "transcript_excerpt": string (key excerpt from Hindi transcript, preserve Hindi text),
    "call_date": string (ISO 8601 format with timezone, e.g., "2025-11-15T13:12:08+05:30"),
    "transcript_tokens": integer (approximate token count)
  }},
  "symptoms": [
    {{
      "name": string (in Hindi, e.g., "बुखार"),
      "duration_days": integer or null,
      "onset_description": string (in Hindi, e.g., "3 दिन पहले"),
      "severity": string (in Hindi, e.g., "मध्यम", "हल्की", "गंभीर"),
      "source": string (e.g., "utt_3") or null,
      "type": string (in Hindi, optional, e.g., "सूखी" for cough)
    }}
  ],
  "past_medical_history": {{
    "reported_by_patient": {{
      "diabetes": boolean,
      "hypertension": boolean,
      "medications": array of strings,
      "allergies": array of strings,
      "prior_infections": array of strings
    }},
    "retrieved_from_api": {{
      "record_found": boolean,
      "lookup_time": string (ISO 8601 format) or null
    }}
  }},
  "vitals_reported": {{
    "temperature_f": float or null,
    "spo2_percent": integer or null,
    "heart_rate": integer or null
  }},
  "risk_scores": {{
    "sepsis_risk_score": float (0-1),
    "cardiac_flag": boolean,
    "dehydration_risk": string ("low", "moderate", "high") or null
  }},
  "red_flags": array of strings (in Hindi if any red flags detected, empty array if none),
  "triage": {{
    "level": string ("Emergent", "Urgent", "Routine"),
    "explanation": string (in Hindi, detailed explanation),
    "recommended_action_window": string (in Hindi, e.g., "48 घंटे", "तुरंत"),
    "override_reason": string or null,
    "source_guidelines": [
      {{
        "source": string (e.g., "ICMR", "WHO"),
        "doc_id": string,
        "section": string,
        "snippet": string (in Hindi or English)
      }}
    ]
  }},
  "differential_diagnosis": [
    {{
      "condition": string (in Hindi with English in parentheses, e.g., "वायरल ज्वर (Common viral infection)"),
      "confidence": float (0-1),
      "rationale": string (in Hindi, detailed explanation),
      "citations": [
        {{
          "source": string,
          "doc_id": string,
          "snippet": string (in Hindi or English)
        }}
      ]
    }}
  ],
  "recommendations": {{
    "for_patient_hi": string (in Hindi, detailed patient instructions),
    "tests_suggested": array of strings (test names in English),
    "referral_suggestion": {{
      "urgency": string (e.g., "within 48 hours", "immediate"),
      "recommended_center": string (in Hindi, e.g., "नजदीकी प्राथमिक स्वास्थ्य केंद्र (PHC)")
    }}
  }},
  "doctor_summary_note": {{
    "summary": string (in English, clinical summary),
    "actionables": array of strings (in English, actionable items for doctor)
  }},
  "report_meta": {{
    "created_at": string (ISO 8601 format with timezone),
    "generated_by": string (e.g., "supervisor-agent-v1.2"),
    "confidence": float (0-1),
    "human_review_required": boolean,
    "audit_log_id": string (e.g., "audit_XXXXX")
  }}
}}

IMPORTANT RULES:
1. Preserve Hindi text in symptoms, triage explanation, differential diagnosis, and patient recommendations
2. Use ISO 8601 format for all timestamps with timezone (+05:30 for India)
3. Extract symptoms from diagnosis input with detailed metadata (duration, severity in Hindi) - use diagnosis.symptoms array
4. Extract vitals from diagnosis.vitals if present (temperature_f, spo2_percent, heart_rate)
5. Extract past_medical_history from diagnosis.past_medical_history if present
6. Map risk_scores input to the risk_scores structure (sepsis_risk_score, cardiac_flag, dehydration_risk)
7. Map red_flags from clinical_rules.red_flags to red_flags array (in Hindi, empty array if none)
8. Map triage input to triage structure:
   - triage.level → level
   - triage.explanation → explanation (in Hindi)
   - triage.recommended_action_window → recommended_action_window (in Hindi)
   - triage.source_guidelines → source_guidelines
9. Map differential_diagnosis input:
   - If differential_diagnosis has "diagnoses" key, use that array
   - Otherwise, use differential_diagnosis directly as array
   - Each diagnosis should have: condition (Hindi with English), confidence, rationale (Hindi), citations
10. Map recommendations input: for_patient_hi (Hindi), tests_suggested (array), referral_suggestion (object)
11. Create doctor_summary_note from recommendations.doctor_actionables and diagnosis.summary
12. Use ORIGINAL_TRANSCRIPT for input_summary.transcript_excerpt (first 200 chars or key excerpt)
13. Generate pseudo_id as "anon_" followed by 5 alphanumeric characters
14. Generate audit_log_id as "audit_" followed by 5 alphanumeric characters
15. Set human_review_required to true if confidence < 0.7 or red_flags detected
16. Set past_medical_history.retrieved_from_api.record_found to false (placeholder, can be updated by API integration)

Respond ONLY with a valid JSON object matching this exact structure. Do not include any markdown formatting or code blocks.
"""

