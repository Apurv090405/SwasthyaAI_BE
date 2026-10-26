"""
LangChain tools for clinical agents.

This module contains all tool functions that agents can use.
"""

import json
from typing import Dict, Any, Optional

from langchain_core.tools import tool

from config import settings
from llm_factory import get_embeddings, get_llm
from services.lancedb_service import LanceDBService
from services.search_service import SearchService
from prompts import (
    get_diagnose_transcript_prompt,
    get_evaluate_patient_report_prompt,
    get_apply_clinical_rules_prompt,
    get_compute_risk_scores_prompt,
    get_differential_diagnosis_prompt,
    get_assign_triage_level_prompt,
    get_generate_recommendations_prompt,
    get_format_clinical_report_prompt,
)


# Initialize services
_lancedb_service = LanceDBService()
_search_service = None  # Lazy initialization


def _get_search_service() -> SearchService:
    """Lazy initialization of search service."""
    global _search_service
    if _search_service is None:
        try:
            _search_service = SearchService()
        except (ValueError, ImportError):
            _search_service = None
    return _search_service


def _extract_text(content) -> str:
    """
    Normalise an LLM response content value to a plain string.

    OpenAI always returns a str; Gemini may return a list of parts such as
    [{"type": "text", "text": "..."}, ...] (multimodal format).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def _call_llm(prompt: str) -> str:
    """
    Call the LLM selected for the current request context.

    The provider (openai/gemini) is read from the per-request context variable
    set by set_llm_provider() in the API handler before invoking the workflow.

    Args:
        prompt: Prompt text to send to the model.

    Returns:
        Response content string.
    """
    llm = get_llm()
    result = llm.invoke(prompt)
    return _extract_text(result.content)


def _parse_json_input(json_str: str, default: Any = None) -> Any:
    """
    Safely parse JSON input string.
    
    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON object or default value
    """
    if isinstance(json_str, str):
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return default
    return json_str


@tool
def diagnose_transcript(transcript: str) -> str:
    """
    Analyze a patient transcript in Hindi and extract symptoms, red flags, and probable conditions.
    
    The transcript is in Hindi (Devanagari script). Extract medical information and translate 
    condition names to English for consistency.
    
    Returns a JSON string with keys: symptoms, red_flags, probable_conditions, summary, confidence.
    """
    prompt = get_diagnose_transcript_prompt(transcript)
    return _call_llm(prompt)


@tool
def query_clinical_guidelines(keywords: str, top_k: int = 5) -> str:
    """
    Query LanceDB vector store for relevant clinical guidelines using keywords.
    
    Keywords should be a semicolon-separated string in English (medical terminology).
    The keywords may come from Hindi transcript analysis - ensure they are in English medical terms.
    
    Returns a JSON string with list of hits: [{"id":..., "text":..., "source":..., "score":...}, ...]
    """
    try:
        # Create query embedding
        query_embedding = get_embeddings().embed_query(keywords)
        
        # Search LanceDB
        hits = _lancedb_service.search(query_embedding, top_k=top_k)
        return json.dumps(hits)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def search_health_news(query: str, location: str = "") -> str:
    """
    Search for recent health news, outbreaks, or public health advisories using Google Search via SerpAPI.
    
    Search queries can be in English or Hindi. For Hindi queries, include both Hindi and English 
    terms for better results.
    
    Returns a JSON string with list of news hits: [{"title":..., "snippet":..., "link":..., "source":...}, ...]
    """
    search_service = _get_search_service()
    if search_service is None:
        return json.dumps({"error": "SERPAPI_KEY not configured or package not installed"})
    
    try:
        hits = search_service.search_health_news(query, location=location)
        return json.dumps(hits)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def evaluate_patient_report(
    diagnosis_json: str, 
    rag_hits_json: str, 
    news_hits_json: str, 
    transcript: str
) -> str:
    """
    Synthesize diagnosis, RAG hits, and news hits into a final patient report.
    
    The original transcript is in Hindi, but all analysis should be in English for medical consistency.
    All inputs are JSON strings that need to be parsed.
    
    Returns a JSON string with keys: final_summary, recommended_next_steps, guideline_references, 
    public_health_alerts, confidence_score.
    """
    diagnosis = _parse_json_input(diagnosis_json, {})
    rag_hits = _parse_json_input(rag_hits_json, [])
    news_hits = _parse_json_input(news_hits_json, [])
    
    prompt = get_evaluate_patient_report_prompt(diagnosis, rag_hits, news_hits, transcript)
    return _call_llm(prompt)


@tool
def apply_clinical_rules(diagnosis_json: str, symptoms_json: str) -> str:
    """
    Apply deterministic red-flag rules to detect immediate escalation triggers.
    
    Inputs are JSON strings containing diagnosis and symptoms.
    
    Returns a JSON string with keys: red_flags_detected (bool), escalation_required (bool), 
    triggered_rules (list), severity_level (str: "critical", "high", "medium", "low").
    """
    diagnosis = _parse_json_input(diagnosis_json, {})
    symptoms = _parse_json_input(symptoms_json, {})
    
    prompt = get_apply_clinical_rules_prompt(diagnosis, symptoms)
    return _call_llm(prompt)


@tool
def compute_risk_scores(
    diagnosis_json: str, 
    symptoms_json: str, 
    patient_age: str = "", 
    comorbidities: str = ""
) -> str:
    """
    Compute disease-specific risk scores (sepsis, cardiac, diabetes, etc.).
    
    Inputs are JSON strings. Returns a JSON string with risk scores for various conditions.
    Keys: sepsis_score, cardiac_risk_score, diabetes_risk_score, overall_risk_level, elevated_risks (list).
    """
    diagnosis = _parse_json_input(diagnosis_json, {})
    symptoms = _parse_json_input(symptoms_json, {})
    
    prompt = get_compute_risk_scores_prompt(diagnosis, symptoms, patient_age, comorbidities)
    return _call_llm(prompt)


@tool
def differential_diagnosis(
    symptoms_json: str, 
    rag_hits_json: str, 
    clinical_context: str = ""
) -> str:
    """
    Use LLM + RAG to propose possible diagnoses with confidence and citations.
    
    Inputs are JSON strings. Returns a JSON string with differential diagnoses.
    Keys: diagnoses (list of objects with condition, confidence, rationale, citations).
    """
    symptoms = _parse_json_input(symptoms_json, {})
    rag_hits = _parse_json_input(rag_hits_json, [])
    
    prompt = get_differential_diagnosis_prompt(symptoms, rag_hits, clinical_context)
    return _call_llm(prompt)


@tool
def assign_triage_level(
    red_flags_json: str, 
    risk_scores_json: str, 
    differential_diagnosis_json: str,
    clinical_rules_json: str = ""
) -> str:
    """
    Assign final triage level (Emergent, Urgent, Routine) based on clinical assessment.
    
    Inputs are JSON strings. Returns a JSON string with triage decision.
    Keys: triage_level, reasoning, protocol_alignment, estimated_wait_time, care_location.
    """
    red_flags = _parse_json_input(red_flags_json, {})
    risk_scores = _parse_json_input(risk_scores_json, {})
    differential = _parse_json_input(differential_diagnosis_json, {})
    clinical_rules = _parse_json_input(clinical_rules_json, {})
    
    prompt = get_assign_triage_level_prompt(red_flags, risk_scores, differential, clinical_rules)
    return _call_llm(prompt)


@tool
def generate_recommendations(
    diagnosis_json: str,
    triage_json: str,
    differential_json: str,
    risk_scores_json: str
) -> str:
    """
    Convert clinical assessment into doctor actionables and patient instructions (Hindi).
    
    Inputs are JSON strings. Returns a JSON string with recommendations.
    Keys: doctor_actionables, patient_instructions_hindi, patient_instructions_english, 
    test_suggestions, referral_level, follow_up_instructions.
    """
    diagnosis = _parse_json_input(diagnosis_json, {})
    triage = _parse_json_input(triage_json, {})
    differential = _parse_json_input(differential_json, {})
    risk_scores = _parse_json_input(risk_scores_json, {})
    
    prompt = get_generate_recommendations_prompt(diagnosis, triage, differential, risk_scores)
    return _call_llm(prompt)


@tool
def format_clinical_report(
    diagnosis_json: str,
    triage_json: str,
    differential_json: str,
    risk_scores_json: str,
    recommendations_json: str,
    clinical_rules_json: str = "",
    rag_hits_json: str = "",
    transcript: str = ""
) -> str:
    """
    Compile all intermediate outputs into a structured diagnosis report (JSON/FHIR format).
    
    Inputs are JSON strings. Returns a comprehensive structured report.
    Keys: patient_id, timestamp, clinical_summary, diagnosis, triage, risk_assessment, 
    recommendations, citations, metadata.
    """
    diagnosis = _parse_json_input(diagnosis_json, {})
    triage = _parse_json_input(triage_json, {})
    differential = _parse_json_input(differential_json, {})
    risk_scores = _parse_json_input(risk_scores_json, {})
    recommendations = _parse_json_input(recommendations_json, {})
    clinical_rules = _parse_json_input(clinical_rules_json, {})
    rag_hits = _parse_json_input(rag_hits_json, [])
    
    prompt = get_format_clinical_report_prompt(
        diagnosis, triage, differential, risk_scores, recommendations, 
        clinical_rules, rag_hits, transcript
    )
    return _call_llm(prompt)

