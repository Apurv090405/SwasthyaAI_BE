"""
Prompt for apply_clinical_rules tool.
Applies deterministic red-flag rules to detect immediate escalation triggers.
"""

def get_prompt(diagnosis: dict, symptoms: dict) -> str:
    """
    Generate prompt for applying clinical red-flag rules.
    
    Args:
        diagnosis: Diagnosis JSON object
        symptoms: Symptoms JSON object
    
    Returns:
        Formatted prompt string
    """
    return f"""
You are a clinical rules engine. Apply deterministic red-flag rules to detect immediate escalation triggers.

Inputs:
- DIAGNOSIS: {diagnosis}
- SYMPTOMS: {symptoms}

Apply standard clinical red-flag rules for:
- Chest pain with radiation
- Severe abdominal pain
- Altered mental status
- Severe respiratory distress
- Signs of shock
- Severe trauma
- Acute neurological deficits
- Severe allergic reactions
- Overdose/poisoning
- Severe burns

Return a JSON object with keys:
- red_flags_detected: boolean
- escalation_required: boolean (true if immediate medical attention needed)
- red_flags: array of strings (in Hindi, descriptions of red flags detected, empty array if none)
- triggered_rules: list of rule names that were triggered (optional)
- severity_level: string ("critical", "high", "medium", "low")
- rule_explanations: list of explanations for triggered rules (optional)

IMPORTANT: 
- red_flags must be an array of strings in Hindi describing the red flags
- If no red flags detected, return empty array: []
- Each red flag description should be clear and in Hindi

Respond ONLY with a JSON object.
"""

