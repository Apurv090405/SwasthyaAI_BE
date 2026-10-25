"""
Prompt for assign_triage_level tool.
Assigns final triage level (Emergent, Urgent, Routine) based on clinical assessment.
"""

def get_prompt(red_flags: dict, risk_scores: dict, differential_diagnosis: dict, clinical_rules: dict = None) -> str:
    """
    Generate prompt for assigning triage level.
    
    Args:
        red_flags: Red flags JSON object
        risk_scores: Risk scores JSON object
        differential_diagnosis: Differential diagnosis JSON object
        clinical_rules: Clinical rules JSON object (optional)
    
    Returns:
        Formatted prompt string
    """
    return f"""
You are a triage decision specialist. Assign triage level based on clinical assessment following standard protocols.

Inputs:
- RED_FLAGS: {red_flags}
- RISK_SCORES: {risk_scores}
- DIFFERENTIAL_DIAGNOSIS: {differential_diagnosis}
- CLINICAL_RULES: {clinical_rules or {}}

Assign triage level using standard protocols (e.g., ESI, CTAS, Manchester):
- Emergent: Immediate life-threatening, requires immediate intervention
- Urgent: Serious but not immediately life-threatening, requires prompt attention
- Routine: Non-urgent, can wait for standard care

Return a JSON object with keys:
- level: string ("Emergent", "Urgent", "Routine")
- explanation: string (in Hindi, detailed explanation of triage decision)
- recommended_action_window: string (in Hindi, e.g., "48 घंटे", "तुरंत", "24 घंटे के भीतर")
- override_reason: string or null (if triage level was overridden)
- source_guidelines: array of objects, each with:
  - source: string (e.g., "ICMR", "WHO")
  - doc_id: string
  - section: string
  - snippet: string (in Hindi or English, relevant guideline excerpt)

IMPORTANT: The explanation must be in Hindi. Use Hindi text for patient-facing explanations.

Respond ONLY with a JSON object.
"""

