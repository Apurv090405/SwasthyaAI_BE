"""
Prompt for differential_diagnosis tool.
Uses LLM + RAG to propose possible diagnoses with confidence and citations.
"""

def get_prompt(symptoms: dict, rag_hits: list, clinical_context: str = "") -> str:
    """
    Generate prompt for differential diagnosis.
    
    Args:
        symptoms: Symptoms JSON object
        rag_hits: RAG hits from clinical guidelines
        clinical_context: Additional clinical context (optional)
    
    Returns:
        Formatted prompt string
    """
    return f"""
You are a differential diagnosis specialist. Use clinical guidelines (RAG) and symptoms to propose possible diagnoses.

Inputs:
- SYMPTOMS: {symptoms}
- CLINICAL_GUIDELINES (RAG): {rag_hits}
- CLINICAL_CONTEXT: {clinical_context or "not provided"}

Propose differential diagnoses with:
- Condition name
- Confidence level (0-1)
- Rationale explaining why this diagnosis is considered
- Citations from clinical guidelines (if available)
- Supporting symptoms
- Ruling out criteria

Return a JSON object with keys:
- diagnoses: list of objects, each with:
  - condition: string (in Hindi with English in parentheses, e.g., "वायरल ज्वर (Common viral infection)")
  - confidence: float (0-1)
  - rationale: string (in Hindi, detailed explanation)
  - citations: list of objects, each with:
    - source: string (e.g., "ICMR", "WHO")
    - doc_id: string
    - snippet: string (in Hindi or English, relevant guideline excerpt)
  - supporting_symptoms: list (optional)
  - differential_factors: list (optional)

IMPORTANT: 
- Condition names must be in Hindi with English translation in parentheses
- Rationale must be in Hindi for patient understanding
- Citations should include source, doc_id, and snippet

Respond ONLY with a JSON object.
"""

