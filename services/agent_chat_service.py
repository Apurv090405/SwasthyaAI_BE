"""Direct chat with a single compiled clinical agent (LangGraph react agent)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage, HumanMessage

from agents import create_all_agents
from llm_factory import set_llm_provider

# Mirrors agents.create_all_agents keys; used for catalog and validation.
CLINICAL_AGENT_CATALOG: List[Dict[str, str]] = [
    {
        "id": "diagnosis_agent",
        "title": "Diagnosis",
        "summary": "Extracts symptoms and conditions from patient text (Hindi-friendly), maps to English medical terms.",
    },
    {
        "id": "rag_agent",
        "title": "Clinical RAG",
        "summary": "Queries your LanceDB clinical guidelines with natural-language questions.",
    },
    {
        "id": "search_agent",
        "title": "Health search",
        "summary": "Finds recent health news and outbreak context via web search tools.",
    },
    {
        "id": "eval_agent",
        "title": "Evaluation",
        "summary": "Synthesizes findings into structured patient evaluation narrative.",
    },
    {
        "id": "clinical_rules_agent",
        "title": "Clinical rules",
        "summary": "Applies red-flag and escalation rules to clinical inputs.",
    },
    {
        "id": "risk_scoring_agent",
        "title": "Risk scoring",
        "summary": "Computes disease-specific risk scores (e.g. sepsis, cardiac flags).",
    },
    {
        "id": "differential_diagnosis_agent",
        "title": "Differential diagnosis",
        "summary": "Proposes ranked differentials with rationale and guideline citations.",
    },
    {
        "id": "triage_decision_agent",
        "title": "Triage",
        "summary": "Assigns triage level (Emergent / Urgent / Routine) with reasoning.",
    },
    {
        "id": "recommendation_agent",
        "title": "Recommendations",
        "summary": "Doctor actionables and patient instructions (Hindi + English where appropriate).",
    },
    {
        "id": "report_formatting_agent",
        "title": "Report formatting",
        "summary": "Compiles inputs into structured clinical report / JSON-style output.",
    },
]

ALLOWED_AGENT_IDS = frozenset(a["id"] for a in CLINICAL_AGENT_CATALOG)

_agents_by_provider: Dict[str, Dict[str, Any]] = {}


def _get_agents_for_provider(provider: str) -> Dict[str, Any]:
    if provider not in _agents_by_provider:
        _agents_by_provider[provider] = create_all_agents(provider)
    return _agents_by_provider[provider]


def _extract_assistant_reply(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p) for p in content
                )
            text = (content or "").strip()
            if text:
                return text
    return ""


def run_single_agent_chat(
    *,
    agent_id: str,
    user_message: str,
    history: List[Tuple[str, str]],
    provider: str = "openai",
) -> str:
    """
    Run one turn with a single react agent.

    history: list of (role, content) with role in ('user', 'assistant').
    Returns assistant text (last non-empty AIMessage in the trace).
    """
    if agent_id not in ALLOWED_AGENT_IDS:
        raise ValueError(f"Unknown agent_id: {agent_id}")

    text = (user_message or "").strip()
    if not text:
        raise ValueError("Empty message")

    set_llm_provider(provider)
    agents = _get_agents_for_provider(provider)
    app = agents[agent_id]

    lc_messages = []
    for role, content in history:
        c = (content or "").strip()
        if not c:
            continue
        if role == "user":
            lc_messages.append(HumanMessage(content=c))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=c))
    lc_messages.append(HumanMessage(content=text))

    result = app.invoke({"messages": lc_messages})
    out_messages = result.get("messages", [])
    reply = _extract_assistant_reply(out_messages)
    if not reply:
        raise RuntimeError("Agent returned no assistant text (check tools / model output).")
    return reply
