"""LangGraph supervisor clinical report — shared by /analyze_transcript and post-call webhook."""

import json
import random
import re
import string
import time
from datetime import datetime
from typing import Any, Dict, Tuple

from langchain_core.messages import HumanMessage
from loguru import logger

from agents import create_supervisor_workflow
from config import settings
from llm_factory import set_llm_provider

_supervisor_apps: Dict[str, Any] = {}


def get_supervisor_app(provider: str = "openai"):
    if provider not in _supervisor_apps:
        _supervisor_apps[provider] = create_supervisor_workflow(provider)
    return _supervisor_apps[provider]


def _enrich_response(response: Dict[str, Any], userid: str, transcript: str) -> None:
    if "patient_id" not in response or not response.get("patient_id"):
        response["patient_id"] = userid

    if "pseudo_id" not in response or not response.get("pseudo_id"):
        pseudo_id = "anon_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
        response["pseudo_id"] = pseudo_id

    response["language"] = settings.DEFAULT_LANGUAGE

    if "consent_status" not in response:
        response["consent_status"] = settings.DEFAULT_CONSENT_STATUS

    if "input_summary" not in response:
        response["input_summary"] = {}

    if "transcript_excerpt" not in response["input_summary"]:
        excerpt = transcript[:200] + "..." if len(transcript) > 200 else transcript
        response["input_summary"]["transcript_excerpt"] = excerpt

    if "call_date" not in response["input_summary"]:
        response["input_summary"]["call_date"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S+05:30")

    if "transcript_tokens" not in response["input_summary"]:
        response["input_summary"]["transcript_tokens"] = len(transcript) // settings.TOKEN_CHAR_RATIO

    if "report_meta" not in response:
        response["report_meta"] = {}

    if "created_at" not in response["report_meta"]:
        response["report_meta"]["created_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S+05:30")

    if "generated_by" not in response["report_meta"]:
        response["report_meta"]["generated_by"] = "supervisor-agent-v1.2"

    if "audit_log_id" not in response["report_meta"]:
        audit_id = "audit_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
        response["report_meta"]["audit_log_id"] = audit_id


def run_clinical_supervisor_report(
    *,
    transcript: str,
    userid: str,
    location: str = "",
    provider: str = "openai",
) -> Tuple[Dict[str, Any], str]:
    """
    Run full supervisor orchestration; return (parsed_report, raw_model_text_snippet_or_empty).

    Raises:
        RuntimeError: on empty model output, JSON parse failure, or supervisor errors.
    """
    if not (transcript or "").strip():
        raise RuntimeError("Empty transcript")

    t0 = time.time()
    transcript_len = len(transcript)
    logger.info(
        f"[Orchestrator] Starting clinical supervisor report | "
        f"userid={userid} provider={provider} transcript_len={transcript_len} chars"
    )

    set_llm_provider(provider)

    if settings.CLINICAL_MODE != "supervisor":
        from services.clinical_pipeline import run_fast_clinical_report

        t_invoke = time.time()
        content, steps = run_fast_clinical_report(transcript)
        invoke_time = time.time() - t_invoke
        parsed = _parse_report(content, userid, transcript, t0=t0, invoke_time=invoke_time, agents_invoked=9)
        meta = parsed.setdefault("report_meta", {})
        if isinstance(meta, dict):
            meta["pipeline"] = {"mode": "fast", "steps_secs": steps}
        return parsed, content[:2000]

    user_message = f"""
Analyze this patient transcript in Hindi and provide a comprehensive medical report following the exact JSON structure.

Patient Transcript (Hindi):
{transcript}

IMPORTANT:
- Coordinate all agents to extract detailed information
- Ensure report_formatting_agent receives the original transcript for input_summary
- Generate the final report in the exact JSON structure with all required fields
- Preserve Hindi text in symptoms, triage explanations, differential diagnosis, and patient recommendations
"""
    if location:
        user_message += f"\nPatient Location: {location}"

    logger.debug(f"[Orchestrator] Getting/creating supervisor app for provider={provider}")
    supervisor_app = get_supervisor_app(provider)
    logger.info(f"[Orchestrator] Invoking supervisor workflow...")

    t_invoke = time.time()
    result = supervisor_app.invoke({"messages": [HumanMessage(content=user_message)]})
    invoke_time = time.time() - t_invoke

    messages = result.get("messages", [])
    logger.info(
        f"[Orchestrator] Supervisor returned {len(messages)} messages in {invoke_time:.1f}s"
    )

    # Log each agent's contribution
    for i, msg in enumerate(messages):
        role = getattr(msg, "type", "unknown")
        name = getattr(msg, "name", None) or ""
        content_preview = ""
        if hasattr(msg, "content"):
            c = msg.content
            if isinstance(c, str):
                content_preview = c[:120].replace("\n", " ")
            elif isinstance(c, list):
                content_preview = str(c[0])[:120] if c else ""
        tool_calls = getattr(msg, "tool_calls", None)
        tool_info = ""
        if tool_calls:
            tool_names = [tc.get("name", "?") for tc in tool_calls if isinstance(tc, dict)]
            tool_info = f" tools=[{', '.join(tool_names)}]"

        logger.debug(
            f"[Orchestrator] msg[{i}] type={role} name={name}{tool_info} | {content_preview}..."
        )

    final_message = messages[-1] if messages else None
    if not final_message:
        logger.error("[Orchestrator] No response from supervisor workflow — messages list is empty")
        raise RuntimeError("No response from supervisor workflow")

    raw_content = final_message.content if hasattr(final_message, "content") else final_message
    if isinstance(raw_content, list):
        content = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in raw_content
        )
    else:
        content = str(raw_content)

    logger.debug(f"[Orchestrator] Final message length: {len(content)} chars")
    parsed = _parse_report(content, userid, transcript, t0=t0, invoke_time=invoke_time, agents_invoked=len(messages))
    return parsed, content[:2000]


def _parse_report(
    content: str, userid: str, transcript: str, *, t0: float, invoke_time: float, agents_invoked: int
) -> Dict[str, Any]:
    """Extract the report JSON from the final model text and fill in standard fields."""
    try:
        json_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
        if json_match:
            logger.debug("[Orchestrator] Extracted JSON from markdown code block")
            content = json_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                logger.debug("[Orchestrator] Extracted JSON from raw content")
                content = json_match.group(0)

        parsed_response = json.loads(content)
        _enrich_response(parsed_response, userid, transcript)

        # Log key report fields
        triage = parsed_response.get("triage", {})
        triage_level = triage.get("level", "unknown") if isinstance(triage, dict) else str(triage)
        diagnosis = parsed_response.get("diagnosis", "N/A")
        if isinstance(diagnosis, dict):
            diagnosis = diagnosis.get("primary", str(diagnosis))
        diagnosis_preview = str(diagnosis)[:100]

        total_time = time.time() - t0
        logger.info(
            f"[Orchestrator] Report complete | "
            f"userid={userid} triage={triage_level} diagnosis={diagnosis_preview} "
            f"total_time={total_time:.1f}s invoke_time={invoke_time:.1f}s "
            f"agents_invoked={agents_invoked} report_keys={list(parsed_response.keys())[:8]}"
        )

        return parsed_response
    except (json.JSONDecodeError, AttributeError, KeyError) as e:
        total_time = time.time() - t0
        logger.error(
            f"[Orchestrator] JSON parse FAILED after {total_time:.1f}s | "
            f"error={e} response_preview={content[:300]}"
        )
        raise RuntimeError(
            f"Failed to parse supervisor response as JSON: {e}. Response: {content[:500]}"
        ) from e
