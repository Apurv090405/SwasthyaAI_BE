"""
Prompt for compute_risk_scores tool.
Computes disease-specific risk scores using validated clinical scoring systems.
"""

def get_prompt(diagnosis: dict, symptoms: dict, patient_age: str = "", comorbidities: str = "") -> str:
    """
    Generate prompt for computing risk scores.
    
    Args:
        diagnosis: Diagnosis JSON object
        symptoms: Symptoms JSON object
        patient_age: Patient age (optional)
        comorbidities: Patient comorbidities (optional)
    
    Returns:
        Formatted prompt string
    """
    return f"""
You are a risk scoring specialist. Compute disease-specific risk scores using validated clinical scoring systems.

Inputs:
- DIAGNOSIS: {diagnosis}
- SYMPTOMS: {symptoms}
- PATIENT_AGE: {patient_age or "not provided"}
- COMORBIDITIES: {comorbidities or "not provided"}

Compute risk scores for:
- Sepsis (qSOFA, SIRS criteria)
- Cardiac events (TIMI, GRACE scores if applicable)
- Diabetes complications
- Stroke risk
- Infection severity
- Overall clinical risk

Return a JSON object with keys:
- sepsis_risk_score: float (0-1, normalized score)
- cardiac_flag: boolean (true if cardiac risk is elevated)
- dehydration_risk: string ("low", "moderate", "high") or null
- additional_scores: object (optional, for other risk scores if computed)

IMPORTANT: Map the computed scores to the required format:
- Extract sepsis score as a float between 0-1
- Set cardiac_flag to true if cardiac risk is elevated
- Determine dehydration_risk based on symptoms and vitals

Respond ONLY with a JSON object.
"""

