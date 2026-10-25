"""
Prompts module for SwasthyaAI clinical agents.

This module contains all prompts organized by function:
- Tool prompts: prompts for individual tool functions
- Agent prompts: prompts for agent behavior
- Supervisor prompt: prompt for supervisor coordination
"""

# Tool prompts
from .diagnose_transcript import get_prompt as get_diagnose_transcript_prompt
from .evaluate_patient_report import get_prompt as get_evaluate_patient_report_prompt
from .apply_clinical_rules import get_prompt as get_apply_clinical_rules_prompt
from .compute_risk_scores import get_prompt as get_compute_risk_scores_prompt
from .differential_diagnosis import get_prompt as get_differential_diagnosis_prompt
from .assign_triage_level import get_prompt as get_assign_triage_level_prompt
from .generate_recommendations import get_prompt as get_generate_recommendations_prompt
from .format_clinical_report import get_prompt as get_format_clinical_report_prompt

# Agent prompts
from .agent_diagnosis import AGENT_PROMPT as AGENT_DIAGNOSIS_PROMPT
from .agent_rag import AGENT_PROMPT as AGENT_RAG_PROMPT
from .agent_search import AGENT_PROMPT as AGENT_SEARCH_PROMPT
from .agent_eval import AGENT_PROMPT as AGENT_EVAL_PROMPT
from .agent_clinical_rules import AGENT_PROMPT as AGENT_CLINICAL_RULES_PROMPT
from .agent_risk_scoring import AGENT_PROMPT as AGENT_RISK_SCORING_PROMPT
from .agent_differential_diagnosis import AGENT_PROMPT as AGENT_DIFFERENTIAL_DIAGNOSIS_PROMPT
from .agent_triage_decision import AGENT_PROMPT as AGENT_TRIAGE_DECISION_PROMPT
from .agent_recommendation import AGENT_PROMPT as AGENT_RECOMMENDATION_PROMPT
from .agent_report_formatting import AGENT_PROMPT as AGENT_REPORT_FORMATTING_PROMPT

# Supervisor prompt
from .supervisor import SUPERVISOR_PROMPT

__all__ = [
    # Tool prompts
    "get_diagnose_transcript_prompt",
    "get_evaluate_patient_report_prompt",
    "get_apply_clinical_rules_prompt",
    "get_compute_risk_scores_prompt",
    "get_differential_diagnosis_prompt",
    "get_assign_triage_level_prompt",
    "get_generate_recommendations_prompt",
    "get_format_clinical_report_prompt",
    # Agent prompts
    "AGENT_DIAGNOSIS_PROMPT",
    "AGENT_RAG_PROMPT",
    "AGENT_SEARCH_PROMPT",
    "AGENT_EVAL_PROMPT",
    "AGENT_CLINICAL_RULES_PROMPT",
    "AGENT_RISK_SCORING_PROMPT",
    "AGENT_DIFFERENTIAL_DIAGNOSIS_PROMPT",
    "AGENT_TRIAGE_DECISION_PROMPT",
    "AGENT_RECOMMENDATION_PROMPT",
    "AGENT_REPORT_FORMATTING_PROMPT",
    # Supervisor prompt
    "SUPERVISOR_PROMPT",
]

