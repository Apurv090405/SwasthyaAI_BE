"""
Agent creation and configuration module.

This module creates all specialized agents for the clinical workflow.
Supports both OpenAI and Google Gemini as LLM backends.
"""

from typing import Optional

from loguru import logger
from langgraph.prebuilt import create_react_agent

from llm_factory import get_llm
from prompts import (
    AGENT_DIAGNOSIS_PROMPT,
    AGENT_RAG_PROMPT,
    AGENT_SEARCH_PROMPT,
    AGENT_EVAL_PROMPT,
    AGENT_CLINICAL_RULES_PROMPT,
    AGENT_RISK_SCORING_PROMPT,
    AGENT_DIFFERENTIAL_DIAGNOSIS_PROMPT,
    AGENT_TRIAGE_DECISION_PROMPT,
    AGENT_RECOMMENDATION_PROMPT,
    AGENT_REPORT_FORMATTING_PROMPT,
)
from tools import (
    diagnose_transcript,
    query_clinical_guidelines,
    search_health_news,
    evaluate_patient_report,
    apply_clinical_rules,
    compute_risk_scores,
    differential_diagnosis,
    assign_triage_level,
    generate_recommendations,
    format_clinical_report,
)


def create_all_agents(llm_provider: Optional[str] = None):
    """
    Create all specialized agents for the clinical workflow.

    Args:
        llm_provider: 'openai' or 'gemini'. Falls back to the context-var
                      value (set per-request) then to DEFAULT_LLM_PROVIDER.

    Returns:
        Dictionary mapping agent names to agent instances.
    """
    model = get_llm(llm_provider)
    logger.info(f"[Agents] Creating 10 clinical agents with provider={llm_provider} model={model.model_name if hasattr(model, 'model_name') else '?'}")

    agents = {
        "diagnosis_agent": create_react_agent(
            model=model,
            tools=[diagnose_transcript],
            name="diagnosis_agent",
            prompt=AGENT_DIAGNOSIS_PROMPT
        ),
        "rag_agent": create_react_agent(
            model=model,
            tools=[query_clinical_guidelines],
            name="rag_agent",
            prompt=AGENT_RAG_PROMPT
        ),
        "search_agent": create_react_agent(
            model=model,
            tools=[search_health_news],
            name="search_agent",
            prompt=AGENT_SEARCH_PROMPT
        ),
        "eval_agent": create_react_agent(
            model=model,
            tools=[evaluate_patient_report],
            name="eval_agent",
            prompt=AGENT_EVAL_PROMPT
        ),
        "clinical_rules_agent": create_react_agent(
            model=model,
            tools=[apply_clinical_rules],
            name="clinical_rules_agent",
            prompt=AGENT_CLINICAL_RULES_PROMPT
        ),
        "risk_scoring_agent": create_react_agent(
            model=model,
            tools=[compute_risk_scores],
            name="risk_scoring_agent",
            prompt=AGENT_RISK_SCORING_PROMPT
        ),
        "differential_diagnosis_agent": create_react_agent(
            model=model,
            tools=[differential_diagnosis, query_clinical_guidelines],
            name="differential_diagnosis_agent",
            prompt=AGENT_DIFFERENTIAL_DIAGNOSIS_PROMPT
        ),
        "triage_decision_agent": create_react_agent(
            model=model,
            tools=[assign_triage_level],
            name="triage_decision_agent",
            prompt=AGENT_TRIAGE_DECISION_PROMPT
        ),
        "recommendation_agent": create_react_agent(
            model=model,
            tools=[generate_recommendations],
            name="recommendation_agent",
            prompt=AGENT_RECOMMENDATION_PROMPT
        ),
        "report_formatting_agent": create_react_agent(
            model=model,
            tools=[format_clinical_report],
            name="report_formatting_agent",
            prompt=AGENT_REPORT_FORMATTING_PROMPT
        ),
    }

    logger.info(f"[Agents] All 10 agents created: {list(agents.keys())}")
    return agents


def create_supervisor_workflow(llm_provider: Optional[str] = None):
    """
    Create and compile the supervisor workflow for the given LLM provider.

    Args:
        llm_provider: 'openai' or 'gemini'. Falls back to context-var then
                      DEFAULT_LLM_PROVIDER.

    Returns:
        Compiled supervisor workflow application.
    """
    from langgraph_supervisor import create_supervisor
    from prompts import SUPERVISOR_PROMPT

    logger.info(f"[Supervisor] Creating supervisor workflow with provider={llm_provider}")

    model = get_llm(llm_provider)
    agents = create_all_agents(llm_provider)
    agent_list = list(agents.values())

    logger.info(f"[Supervisor] Building supervisor graph with {len(agent_list)} agents")

    workflow = create_supervisor(
        agent_list,
        model=model,
        prompt=SUPERVISOR_PROMPT
    )

    compiled = workflow.compile()
    logger.info(f"[Supervisor] Workflow compiled successfully for provider={llm_provider}")
    return compiled

