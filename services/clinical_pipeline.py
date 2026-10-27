"""
Fast clinical report: the same clinical agents' tools, run in a fixed order with independent steps
in parallel, instead of the LangGraph supervisor deciding every hop with an extra LLM call.

    diagnosis ─▶ guidelines (RAG, no LLM) ─▶ ┌ clinical rules ┐
                                              ├ risk scores    ├─▶ triage ─▶ recommendations ─▶ report
                                              └ differential   ┘

Five LLM calls in sequence (the three in the middle run at once) instead of 20+ supervisor and
ReAct turns. Same prompts and tools as the agents in agents.py, so report content is unchanged.
Selected by CLINICAL_MODE=fast (default); CLINICAL_MODE=supervisor keeps the agentic workflow.
"""

from __future__ import annotations

import contextvars
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Tuple

from loguru import logger

import tools


def _parse(text: str, default: Any) -> Any:
    """Tool outputs are JSON, sometimes wrapped in ```json fences."""
    if not isinstance(text, str):
        return text if text is not None else default
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned[4:] if cleaned.lower().startswith("json") else cleaned
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        return default
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return default


def _keywords(diagnosis: Dict[str, Any]) -> str:
    """English medical keywords for the guideline search, from the diagnosis step."""
    terms: List[str] = []
    for key in ("probable_conditions", "symptoms"):
        for item in diagnosis.get(key) or []:
            if isinstance(item, dict):
                item = item.get("condition") or item.get("name") or item.get("symptom") or item.get("english") or ""
            if item:
                terms.append(str(item))
    return "; ".join(dict.fromkeys(terms))[:500] or str(diagnosis.get("summary") or "")[:300]


class _Timer:
    def __init__(self) -> None:
        self.steps: Dict[str, float] = {}

    def run(self, name: str, fn: Callable[..., str], *args: Any) -> str:
        t0 = time.time()
        try:
            return fn(*args)
        finally:
            self.steps[name] = round(time.time() - t0, 2)


def run_fast_clinical_report(transcript: str) -> Tuple[str, Dict[str, float]]:
    """Return (raw report JSON text from the formatting step, per-step seconds).

    The caller has already set the LLM provider context (llm_factory.set_llm_provider); each
    parallel step runs in a copy of that context so it uses the same provider.
    """
    timer = _Timer()
    t0 = time.time()

    diagnosis_raw = timer.run("diagnosis", tools.diagnose_transcript.func, transcript)
    diagnosis = _parse(diagnosis_raw, {})
    symptoms_raw = json.dumps(diagnosis.get("symptoms") or [], ensure_ascii=False)
    red_flags_raw = json.dumps(diagnosis.get("red_flags") or [], ensure_ascii=False)

    rag_raw = timer.run("guidelines", tools.query_clinical_guidelines.func, _keywords(diagnosis), 5)

    # Rules, risk and differential only need the diagnosis and guidelines: run them together.
    with ThreadPoolExecutor(max_workers=3) as pool:
        def submit(name: str, fn: Callable[..., str], *args: Any):
            ctx = contextvars.copy_context()
            return pool.submit(ctx.run, timer.run, name, fn, *args)

        rules_f = submit("clinical_rules", tools.apply_clinical_rules.func, diagnosis_raw, symptoms_raw)
        risk_f = submit("risk_scores", tools.compute_risk_scores.func, diagnosis_raw, symptoms_raw, "", "")
        diff_f = submit(
            "differential",
            tools.differential_diagnosis.func,
            symptoms_raw,
            rag_raw,
            str(diagnosis.get("summary") or ""),
        )
        rules_raw, risk_raw, diff_raw = rules_f.result(), risk_f.result(), diff_f.result()

    triage_raw = timer.run(
        "triage", tools.assign_triage_level.func, red_flags_raw, risk_raw, diff_raw, rules_raw
    )
    recs_raw = timer.run(
        "recommendations", tools.generate_recommendations.func, diagnosis_raw, triage_raw, diff_raw, risk_raw
    )
    report_raw = timer.run(
        "report",
        tools.format_clinical_report.func,
        diagnosis_raw,
        triage_raw,
        diff_raw,
        risk_raw,
        recs_raw,
        rules_raw,
        rag_raw,
        transcript,
    )

    timer.steps["total"] = round(time.time() - t0, 2)
    logger.info(f"[ClinicalFast] report in {timer.steps['total']}s | steps={timer.steps}")
    return report_raw, timer.steps
