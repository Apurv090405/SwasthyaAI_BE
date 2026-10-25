"""
Prompt for evaluate_patient_report tool.
Synthesizes diagnosis, RAG hits, and news hits into a final patient report.
"""

def get_prompt(diagnosis: dict, rag_hits: list, news_hits: list, transcript: str) -> str:
    """
    Generate prompt for evaluating patient report.
    
    Args:
        diagnosis: Diagnosis JSON object
        rag_hits: RAG hits from clinical guidelines
        news_hits: News hits from search
        transcript: Original Hindi transcript
    
    Returns:
        Formatted prompt string
    """
    return f"""
You are an evaluation agent fluent in Hindi and English. The original patient transcript is in Hindi, but you should produce the report in English for medical documentation standards.

Using the following inputs, create a detailed patient report in English:

Inputs:
- DIAGNOSIS (JSON): {diagnosis}
- RAG hits: {rag_hits}
- Recent News hits: {news_hits}
- Original Hindi Transcript: \"\"\"{transcript}\"\"\"

Produce a JSON with keys:
- final_summary: (3-6 sentence clinical summary in English)
- recommended_next_steps: list of clear actions in English (triage, labs, PHC referral)
- guideline_references: list of guideline titles + short excerpts matched
- public_health_alerts: list of any news evidence impacting the patient (outbreak, recall, region-specific advisory)
- confidence_score: float 0-1

IMPORTANT: The transcript is in Hindi, but all output must be in English for medical documentation consistency.
Respond only with JSON.
"""

