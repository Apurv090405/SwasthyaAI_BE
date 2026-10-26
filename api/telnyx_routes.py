"""Telnyx TeXML integration routes for Pipecat voice pipeline.

Provides:
  - Phone number search / order / listing via Telnyx v2 API
  - Outbound call initiation via TeXML Calls API
  - TeXML answer webhook (returns TwiML-compatible XML with <Stream>)
  - WebSocket handler that runs the Pipecat STT → LLM → TTS pipeline
"""

import uuid
import asyncio
from typing import Dict, Any, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TTSSpeakFrame, InterruptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.serializers.telnyx import TelnyxFrameSerializer

from api.telnyx_serializer import TelnyxMediaStreamSerializer
from services.sarvam_voice import make_stt, make_tts
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.transcriptions.language import Language

from config import settings
from llm_factory import get_openai_compat_params
from api.voice_routes import get_global_session, load_prompt_from_file, send_call_webhook

telnyx_router = APIRouter(prefix="/telnyx", tags=["telnyx"])

TELNYX_API_BASE = "https://api.telnyx.com/v2"

TELNYX_CALLS: Dict[str, Dict[str, Any]] = {}


def _telnyx_managed_account_id() -> Optional[str]:
    """Telnyx managed-account id for x-managed-account-id (sub-account scope)."""
    mid = (
        settings.TELNYX_MANAGED_ACCOUNT_ID
        or settings.TELNYX_SUB_ACCOUNT_ID
        or ""
    ).strip()
    return mid or None


def _telnyx_sends_managed_account_header() -> bool:
    """True when we attach x-managed-account-id (manager key + explicit opt-in only)."""
    mid = _telnyx_managed_account_id()
    return bool(mid and settings.TELNYX_MANAGER_SCOPING)


def _telnyx_headers() -> dict:
    """Standard auth headers for Telnyx v2 REST API.

    **Sub-account API key** (key created inside the managed account): leave
    ``TELNYX_MANAGER_SCOPING=false`` (default). Do not send x-managed-account-id
    or Telnyx returns error 10009.

    **Manager API key** (parent account): set ``TELNYX_MANAGER_SCOPING=true`` and
    ``TELNYX_SUB_ACCOUNT_ID`` (or ``TELNYX_MANAGED_ACCOUNT_ID``) to scope requests.
    """
    if not settings.TELNYX_API_KEY:
        raise ValueError("TELNYX_API_KEY not configured")
    headers = {
        "Authorization": f"Bearer {settings.TELNYX_API_KEY}",
        "Content-Type": "application/json",
    }
    mid = _telnyx_managed_account_id()
    if mid and settings.TELNYX_MANAGER_SCOPING:
        headers["x-managed-account-id"] = mid
    return headers


def _strip_hostname(raw: str) -> str:
    h = (raw or "").strip()
    if not h:
        return "localhost"
    if "://" in h:
        h = h.split("://", 1)[1]
    h = h.split("/")[0]
    if "@" in h:
        h = h.split("@")[-1]
    # drop port for standard wss/https (uncommon on Telnyx-facing URL)
    return h


def _telnyx_voice_hostname() -> str:
    pub = (settings.TELNYX_PUBLIC_HOST or "").strip()
    if pub:
        return _strip_hostname(pub)
    return _strip_hostname(settings.HOST)


def _telnyx_is_local_hostname(host: str) -> bool:
    h = (host or "").lower().split(":")[0]
    return h in ("localhost", "127.0.0.1", "::1", "")


def _telnyx_http_base() -> str:
    """Base URL for /telnyx/answer callbacks (HTTPS in production)."""
    host = _telnyx_voice_hostname()
    if _telnyx_is_local_hostname(host):
        logger.warning(
            "Telnyx cannot reach localhost for TeXML answer URL / Media Streams. "
            "Set TELNYX_PUBLIC_HOST to your tunnel host (e.g. ngrok) and use HTTPS."
        )
        return f"http://{host}"
    return f"https://{host}"


def _get_ws_url(stream_id: str) -> str:
    """WebSocket URL Telnyx Media Stream connects to (must be wss:// and publicly reachable)."""
    host = _telnyx_voice_hostname()
    scheme = "ws" if _telnyx_is_local_hostname(host) else "wss"
    return f"{scheme}://{host}/ws/telnyx/{stream_id}"


# Telnyx <Pause> is capped (typically 1–180s). After each slice we <Redirect> back here so the
# call leg stays up while the media WebSocket runs. Without a blocking verb after <Start>, some
# calls hang up as soon as TeXML finishes (local test clients don't use TeXML, so they keep running).
_TELNYX_STREAM_HOLD_PAUSE_SECS = 180


def _telnyx_stream_hold_url() -> str:
    return f"{_telnyx_http_base()}/telnyx/stream-hold"


def _texml_media_stream_xml(ws_url: str) -> str:
    """TeXML: Start bidirectional Stream, then hold the call open (Pause + Redirect loop)."""
    hold = _telnyx_stream_hold_url()
    pause = _TELNYX_STREAM_HOLD_PAUSE_SECS
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Start>
    <Stream
      url="{ws_url}"
      track="inbound_track"
      bidirectionalMode="rtp"
      bidirectionalCodec="PCMU"
      bidirectionalSamplingRate="8000"
    />
  </Start>
  <Pause length="{pause}"/>
  <Redirect method="POST">{hold}</Redirect>
</Response>"""


def _texml_stream_hold_loop_xml() -> str:
    """Continuation TeXML: only Pause + Redirect (Stream already started on the call)."""
    hold = _telnyx_stream_hold_url()
    pause = _TELNYX_STREAM_HOLD_PAUSE_SECS
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Pause length="{pause}"/>
  <Redirect method="POST">{hold}</Redirect>
</Response>"""


async def _telnyx_merge_request_data(request: Request) -> Dict[str, Any]:
    """Merge query string + body (Telnyx often POSTs TeXML params but keeps ?stream_id= in URL)."""
    merged: Dict[str, Any] = dict(request.query_params)
    if request.method != "POST":
        return merged
    ct = (request.headers.get("content-type") or "").lower()
    try:
        if "application/json" in ct:
            body = await request.json()
            if isinstance(body, dict):
                merged = {**merged, **body}
        else:
            form = dict(await request.form())
            merged = {**merged, **form}
    except Exception:
        pass
    return merged


def _telnyx_call_control_from_payload(data: Dict[str, Any]) -> str:
    for key in ("call_control_id", "CallControlId", "CallControlID"):
        v = data.get(key)
        if v:
            return str(v)
    return ""


# ── Phone number management ──────────────────────────────────────────


class PhoneSearchParams(BaseModel):
    country_code: str = "US"
    phone_number_type: str = "local"
    limit: int = 10
    contains: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None


@telnyx_router.post("/phone-numbers/search")
async def search_phone_numbers(params: PhoneSearchParams):
    """Search Telnyx for available phone numbers."""
    session = await get_global_session()
    headers = _telnyx_headers()

    query: Dict[str, Any] = {
        "filter[country_code]": params.country_code,
        "filter[phone_number_type]": params.phone_number_type,
        "filter[limit]": params.limit,
    }
    if params.contains:
        query["filter[phone_number][contains]"] = params.contains
    if params.state:
        query["filter[administrative_area]"] = params.state
    if params.city:
        query["filter[locality]"] = params.city

    url = f"{TELNYX_API_BASE}/available_phone_numbers"
    async with session.get(url, headers=headers, params=query) as resp:
        if resp.status >= 400:
            text = await resp.text()
            raise HTTPException(status_code=resp.status, detail=f"Telnyx API error: {text}")
        body = await resp.json()

    numbers = [
        {
            "phone_number": n.get("phone_number"),
            "region": n.get("region_information", [{}])[0].get("region_name", ""),
            "type": n.get("phone_number_type"),
            "monthly_cost": n.get("cost_information", {}).get("monthly_cost"),
            "upfront_cost": n.get("cost_information", {}).get("upfront_cost"),
        }
        for n in body.get("data", [])
    ]
    return JSONResponse({"count": len(numbers), "numbers": numbers})


class PhoneOrderParams(BaseModel):
    phone_number: str
    connection_id: Optional[str] = None


@telnyx_router.post("/phone-numbers/order")
async def order_phone_number(params: PhoneOrderParams):
    """Purchase a phone number from Telnyx and bind it to the TeXML app."""
    session = await get_global_session()
    headers = _telnyx_headers()

    payload: Dict[str, Any] = {
        "phone_numbers": [{"phone_number": params.phone_number}],
    }
    if params.connection_id:
        payload["connection_id"] = params.connection_id
    elif settings.TELNYX_TEXML_APP_ID:
        payload["connection_id"] = settings.TELNYX_TEXML_APP_ID

    url = f"{TELNYX_API_BASE}/number_orders"
    async with session.post(url, headers=headers, json=payload) as resp:
        if resp.status >= 400:
            text = await resp.text()
            raise HTTPException(status_code=resp.status, detail=f"Telnyx order error: {text}")
        body = await resp.json()

    return JSONResponse({
        "status": "ordered",
        "order_id": body.get("data", {}).get("id"),
        "phone_numbers": [
            pn.get("phone_number")
            for pn in body.get("data", {}).get("phone_numbers", [])
        ],
    })


@telnyx_router.get("/phone-numbers")
async def list_phone_numbers():
    """List phone numbers owned on this Telnyx account (or managed account if scoped)."""
    session = await get_global_session()
    headers = _telnyx_headers()
    header_active = _telnyx_sends_managed_account_header()

    url = f"{TELNYX_API_BASE}/phone_numbers"
    all_rows: list = []
    page = 1
    page_size = 100
    meta: Dict[str, Any] = {}

    while True:
        params = {"page[number]": page, "page[size]": page_size}
        async with session.get(url, headers=headers, params=params) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise HTTPException(status_code=resp.status, detail=f"Telnyx API error: {text}")
            body = await resp.json()

        all_rows.extend(body.get("data", []))
        meta = body.get("meta", {}) or {}
        total_pages = meta.get("total_pages") or meta.get("page_total")
        if total_pages is not None and page >= int(total_pages):
            break
        if len(body.get("data", [])) < page_size:
            break
        page += 1
        if page > 500:
            break

    numbers = [
        {
            "id": n.get("id"),
            "phone_number": n.get("phone_number"),
            "status": n.get("status"),
            "connection_name": n.get("connection_name"),
        }
        for n in all_rows
    ]
    return JSONResponse(
        {
            "count": len(numbers),
            "numbers": numbers,
            "telnyx_scope": {
                "x_managed_account_id_applied": header_active,
                "TELNYX_MANAGER_SCOPING": settings.TELNYX_MANAGER_SCOPING,
                "note": "Sub-account API key: keep TELNYX_MANAGER_SCOPING unset/false (default). "
                "Manager API key: set TELNYX_MANAGER_SCOPING=true and TELNYX_SUB_ACCOUNT_ID. "
                "Otherwise Telnyx error 10009.",
            },
            "meta": meta,
        }
    )


# ── Outbound calls ───────────────────────────────────────────────────


class OutboundCallParams(BaseModel):
    to: str
    from_number: Optional[str] = None
    llm_provider: str = "openai"
    # Supabase auth user id — links call history & clinical report to dashboard user
    user_id: Optional[str] = None


@telnyx_router.post("/outbound")
async def telnyx_outbound(params: OutboundCallParams, request: Request):
    """Initiate an outbound call via Telnyx TeXML.

    Flow:
      1. POST to Telnyx TeXML Calls API
      2. Telnyx dials the recipient, then hits our /telnyx/answer webhook
      3. Answer returns XML with <Stream> pointing at /ws/telnyx/{stream_id}
      4. WebSocket pipeline handles the conversation
    """
    texml_app_id = settings.TELNYX_TEXML_APP_ID
    api_key = settings.TELNYX_API_KEY
    if not texml_app_id or not api_key:
        raise HTTPException(status_code=500, detail="Telnyx TeXML credentials not configured")

    from_number = params.from_number or settings.TELNYX_PHONE_NUMBER
    if not from_number:
        raise HTTPException(
            status_code=400,
            detail="Missing 'from_number' and TELNYX_PHONE_NUMBER not set",
        )

    stream_id = str(uuid.uuid4())

    answer_url = f"{_telnyx_http_base().rstrip('/')}/telnyx/answer?stream_id={stream_id}"

    TELNYX_CALLS[stream_id] = {
        "direction": "outbound",
        "llm_provider": params.llm_provider,
        "to": params.to,
        "from": from_number,
        "status": "initiating",
        "supabase_user_id": (params.user_id or "").strip() or None,
        "provider": "telnyx",
    }

    session = await get_global_session()
    headers = _telnyx_headers()

    texml_payload = {
        "To": params.to,
        "From": from_number,
        "Url": answer_url,
    }

    url = f"{TELNYX_API_BASE}/texml/calls/{texml_app_id}"
    async with session.post(url, headers=headers, json=texml_payload) as resp:
        if resp.status >= 400:
            text = await resp.text()
            TELNYX_CALLS.pop(stream_id, None)
            logger.error(f"Telnyx outbound failed: {resp.status} — {text}")
            raise HTTPException(status_code=502, detail=f"Telnyx API error: {text}")
        body = await resp.json()

    res = body.get("data") if isinstance(body.get("data"), dict) else body
    call_sid = None
    call_control_id = None
    if isinstance(res, dict):
        call_sid = res.get("call_sid") or res.get("sid") or res.get("call_leg_id")
        call_control_id = res.get("call_control_id") or res.get("CallControlId")

    if not call_sid:
        logger.warning(f"Telnyx outbound: no call_sid in response, full body={body}")

    TELNYX_CALLS[stream_id].update({
        "call_sid": call_sid,
        "call_control_id": call_control_id,
        "status": "initiated",
    })

    logger.info(
        f"Telnyx outbound initiated: stream_id={stream_id}, call_sid={call_sid}, "
        f"answer_url={answer_url}"
    )
    return JSONResponse({
        "status": "initiated",
        "stream_id": stream_id,
        "call_sid": call_sid,
        "to": params.to,
        "from": from_number,
        "llm_provider": params.llm_provider,
        "answer_url": answer_url,
        "media_stream_ws": _get_ws_url(stream_id),
    })


# ── Report delivery callback ────────────────────────────────────────


class ReportDeliveryParams(BaseModel):
    to: str
    call_id: str  # original call ID to fetch report from
    from_number: Optional[str] = None
    llm_provider: str = "openai"
    user_id: Optional[str] = None


@telnyx_router.post("/report-callback")
async def telnyx_report_callback(params: ReportDeliveryParams, request: Request):
    """Call the user back to deliver their clinical report summary.

    Fetches the saved clinical report for the given call_id, formats a summary,
    and initiates an outbound call using the report delivery prompt.
    """
    # Fetch the report from webhook events
    from services.webhook_event_file import load_event_by_call_id

    event = None
    try:
        event = load_event_by_call_id(params.call_id)
    except Exception:
        pass

    report_summary = "Report details could not be loaded."
    original_transcript = ""
    if event:
        report = event.get("report") or {}
        original_transcript = (event.get("request") or {}).get("transcript") or (event.get("transcript") or {}).get("full_text") or ""

        # Give the LLM the FULL report as structured text so it can answer any question
        import json as _json
        parts = []

        # Section 1: Patient's original conversation
        if original_transcript:
            parts.append(f"## Original Patient Conversation\n{original_transcript}")

        # Section 2: Full clinical report - dump all fields the orchestrator generated
        def _format_value(v):
            if isinstance(v, dict):
                return "\n".join(f"  - {k}: {val}" for k, val in v.items() if val)
            if isinstance(v, list):
                return ", ".join(str(i) for i in v)
            return str(v) if v else ""

        report_fields = [
            ("input_summary", "Patient Summary"),
            ("diagnosis", "Diagnosis"),
            ("differential_diagnosis", "Differential Diagnosis"),
            ("triage", "Triage Assessment"),
            ("risk_assessment", "Risk Assessment"),
            ("clinical_rules", "Clinical Rules Applied"),
            ("recommendations", "Recommendations"),
            ("evaluation", "Clinical Evaluation"),
            ("follow_up", "Follow-up Plan"),
        ]

        for key, label in report_fields:
            val = report.get(key)
            if val:
                formatted = _format_value(val)
                if formatted.strip():
                    parts.append(f"## {label}\n{formatted}")

        # Also include any other top-level keys we missed
        covered_keys = {k for k, _ in report_fields} | {"patient_id", "pseudo_id", "language", "consent_status", "report_meta"}
        for key, val in report.items():
            if key not in covered_keys and val:
                formatted = _format_value(val)
                if formatted.strip():
                    parts.append(f"## {key.replace('_', ' ').title()}\n{formatted}")

        if parts:
            report_summary = "\n\n".join(parts)

        logger.info(f"Report delivery: loaded report for call_id={params.call_id}, summary_length={len(report_summary)}")

    # Initiate outbound call with report delivery prompt
    texml_app_id = settings.TELNYX_TEXML_APP_ID
    api_key = settings.TELNYX_API_KEY
    if not texml_app_id or not api_key:
        raise HTTPException(status_code=500, detail="Telnyx TeXML credentials not configured")

    from_number = params.from_number or settings.TELNYX_PHONE_NUMBER
    if not from_number:
        raise HTTPException(status_code=400, detail="Missing from_number")

    stream_id = str(uuid.uuid4())
    answer_url = f"{_telnyx_http_base().rstrip('/')}/telnyx/answer?stream_id={stream_id}"

    TELNYX_CALLS[stream_id] = {
        "direction": "report_delivery",
        "llm_provider": params.llm_provider,
        "to": params.to,
        "from": from_number,
        "status": "initiating",
        "supabase_user_id": (params.user_id or "").strip() or None,
        "provider": "telnyx",
        "report_summary": report_summary,
        "original_call_id": params.call_id,
    }

    session = await get_global_session()
    headers = _telnyx_headers()

    texml_payload = {"To": params.to, "From": from_number, "Url": answer_url}
    url = f"{TELNYX_API_BASE}/texml/calls/{texml_app_id}"

    async with session.post(url, headers=headers, json=texml_payload) as resp:
        if resp.status >= 400:
            text = await resp.text()
            TELNYX_CALLS.pop(stream_id, None)
            raise HTTPException(status_code=502, detail=f"Telnyx API error: {text}")
        body = await resp.json()

    res = body.get("data") if isinstance(body.get("data"), dict) else body
    call_sid = res.get("call_sid") or res.get("sid") if isinstance(res, dict) else None

    TELNYX_CALLS[stream_id].update({"call_sid": call_sid, "status": "initiated"})

    logger.info(f"Report delivery call initiated: {stream_id} -> {params.to} for report {params.call_id}")

    return JSONResponse({
        "status": "initiated",
        "stream_id": stream_id,
        "call_sid": call_sid,
        "to": params.to,
        "original_call_id": params.call_id,
        "message": f"Calling {params.to} to deliver clinical report",
    })


# ── TeXML answer webhook ─────────────────────────────────────────────


@telnyx_router.api_route("/answer", methods=["GET", "POST"])
async def telnyx_answer(request: Request):
    """TeXML answer webhook — returns TeXML with <Start><Stream> for Media Streams.

    Telnyx hits this URL when the callee picks up (outbound) or on inbound ring.
    """
    data = await _telnyx_merge_request_data(request)

    stream_id = (data.get("stream_id") or "").strip() or str(uuid.uuid4())
    call_sid = data.get("CallSid") or data.get("call_sid") or ""
    cc_id = _telnyx_call_control_from_payload(data)

    prior = TELNYX_CALLS.pop(stream_id, {})
    llm_provider = prior.get("llm_provider", "openai")
    direction = prior.get("direction", "inbound")

    if not prior and stream_id:
        logger.warning(
            f"Telnyx answer: no prior state for stream_id={stream_id}; "
            f"query+form keys={list(data.keys())}"
        )

    TELNYX_CALLS[stream_id] = {
        **prior,
        "call_sid": call_sid or prior.get("call_sid", ""),
        "call_control_id": cc_id or prior.get("call_control_id", ""),
        "stream_id": stream_id,
        "direction": direction,
        "llm_provider": llm_provider,
        "status": "answering",
        "texml_callback": data,
    }

    ws_url = _get_ws_url(stream_id)
    xml = _texml_media_stream_xml(ws_url)

    logger.info(f"Telnyx answer: stream_id={stream_id}, ws={ws_url}, call_sid={call_sid}")
    return HTMLResponse(content=xml, media_type="application/xml")


# ── Inbound call webhook ─────────────────────────────────────────────


@telnyx_router.api_route("/inbound", methods=["GET", "POST"])
async def telnyx_inbound(request: Request):
    """Handle inbound call on a Telnyx number.

    This is Voice URL on the TeXML app; same TeXML shape as /answer.
    """
    data = await _telnyx_merge_request_data(request)

    stream_id = str(uuid.uuid4())
    call_sid = data.get("CallSid") or data.get("call_sid") or ""
    from_number = data.get("From") or data.get("from") or "unknown"
    to_number = data.get("To") or data.get("to") or "unknown"
    cc_id = _telnyx_call_control_from_payload(data)

    TELNYX_CALLS[stream_id] = {
        "call_sid": call_sid,
        "call_control_id": cc_id,
        "stream_id": stream_id,
        "direction": "inbound",
        "from": from_number,
        "to": to_number,
        "llm_provider": "openai",
        "status": "answering",
        "texml_callback": data,
    }

    ws_url = _get_ws_url(stream_id)
    xml = _texml_media_stream_xml(ws_url)

    logger.info(f"Telnyx inbound: stream_id={stream_id}, from={from_number}")
    return HTMLResponse(content=xml, media_type="application/xml")


@telnyx_router.api_route("/stream-hold", methods=["GET", "POST"])
async def telnyx_stream_hold(request: Request):
    """TeXML continuation: keep the answered call bridged while the WS media stream is active.

    Invoked only via ``<Redirect>`` from the initial answer document — do not start a second Stream.
    """
    logger.debug("TeXML stream-hold (pause + redirect loop)")
    xml = _texml_stream_hold_loop_xml()
    return HTMLResponse(content=xml, media_type="application/xml")


# ── Status callback ──────────────────────────────────────────────────


@telnyx_router.api_route("/status", methods=["GET", "POST"])
async def telnyx_status_callback(request: Request):
    """Receive call status updates from Telnyx (optional)."""
    try:
        data = await request.json()
    except Exception:
        data = dict(await request.form()) if request.method == "POST" else {}
    logger.info(f"Telnyx status callback: {data}")
    return JSONResponse({"received": True})


# ── Active calls ─────────────────────────────────────────────────────


@telnyx_router.get("/calls")
async def list_telnyx_calls():
    """List active / recently initiated Telnyx calls."""
    return JSONResponse({"calls": TELNYX_CALLS})


# ── Pipecat voice pipeline ───────────────────────────────────────────


async def run_telnyx_pipeline(
    websocket: WebSocket,
    stream_id: str,
    call_data: Dict[str, Any],
):
    """Run Pipecat STT → LLM → TTS pipeline over the Telnyx WebSocket stream."""
    direction = call_data.get("direction", "inbound")
    llm_provider = call_data.get("llm_provider", "openai")
    call_control_id = call_data.get("call_control_id", "")

    logger.info(
        f"Starting Telnyx pipeline: stream={stream_id}, "
        f"direction={direction}, llm={llm_provider}"
    )

    # Pick prompt based on direction / call type
    if direction == "report_delivery":
        prompt_file = "prompts/report_delivery_prompt.md"
        system_prompt = load_prompt_from_file(prompt_file)
        # Inject the actual report summary into the prompt
        report_summary = call_data.get("report_summary", "No report available.")
        system_prompt = system_prompt.replace("{report_summary}", report_summary)
        logger.info(
            f"[ReportDelivery] Injected report into prompt | "
            f"report_length={len(report_summary)} prompt_length={len(system_prompt)}"
        )
    elif direction == "outbound":
        system_prompt = load_prompt_from_file("outbound_prompt.md")
    else:
        system_prompt = load_prompt_from_file("inbound_prompt.md")

    llm_params = get_openai_compat_params(llm_provider)
    llm_service = OpenAILLMService(**llm_params)

    stt_service = make_stt()  # auto-detects the caller's language
    tts_service = make_tts(sample_rate=8000)

    # Dynamic language switcher — updates TTS language when STT detects a new language
    from services.language_detect import LanguageDetectTTSSwitcher
    language_switcher = LanguageDetectTTSSwitcher(default_language=Language.HI)

    messages = [{"role": "system", "content": system_prompt}]
    context = LLMContext(messages=messages)

    telnyx_stream_ready = asyncio.Event()

    serializer = TelnyxMediaStreamSerializer(
        stream_id=stream_id,
        outbound_encoding="PCMU",
        inbound_encoding="PCMU",
        call_control_id=call_control_id or None,
        api_key=settings.TELNYX_API_KEY,
        params=TelnyxFrameSerializer.InputParams(
            telnyx_sample_rate=8000,
            # Pipecat 1.x raises if auto_hang_up is on without a call_control_id (it's empty on
            # some inbound/outbound starts); without one the caller's hangup still ends the stream.
            auto_hang_up=bool(call_control_id),
        ),
        stream_ready=telnyx_stream_ready,
    )

    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(start_secs=0.1, stop_secs=0.2)
    )
    # Pipecat 1.x: VAD lives on the user aggregator; Smart Turn v3 is the default stop strategy.
    context_aggregator = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=vad_analyzer)
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    pipeline = Pipeline([
        transport.input(),
        stt_service,
        language_switcher,
        context_aggregator.user(),
        llm_service,
        tts_service,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline=pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_connect(trans, client):
        logger.info(f"Telnyx client connected: {stream_id}, direction={direction}")

        if direction == "report_delivery":
            greeting = (
                "नमस्ते, मैं SwasthyaAI से बोल रही हूँ। "
                "आपने थोड़ी देर पहले हमसे बात की थी। "
                "आपकी रिपोर्ट डॉक्टर ने रिव्यू कर ली है। मैं आपको रिजल्ट बताती हूँ।"
            )
        elif direction == "outbound":
            greeting = (
                "नमस्ते, मैं SwasthyaAI से बोल रही हूँ। "
                "क्या यह बात करने का सही समय है?"
            )
        else:
            greeting = (
                "नमस्ते, मैं SwasthyaAI से बोल रही हूँ। "
                "आप मुझे अपनी स्वास्थ्य समस्या बता सकते हैं।"
            )

        try:
            await asyncio.wait_for(telnyx_stream_ready.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Telnyx stream not ready in 5s; sending greeting anyway")
        await task.queue_frames([TTSSpeakFrame(greeting)])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnect(trans, client):
        logger.info(f"Telnyx client disconnected: {stream_id}")
        await task.cancel()

    runner = PipelineRunner()
    try:
        await runner.run(task)
    except asyncio.CancelledError:
        logger.info(f"Telnyx pipeline cancelled: {stream_id}")
    except Exception as e:
        logger.info(f"Telnyx pipeline ended: {stream_id}: {type(e).__name__}: {e}")
    finally:
        logger.info(f"Cleaning up Telnyx pipeline: {stream_id}")

        # Cancel the task if it's still running (handles abrupt disconnects)
        try:
            await task.cancel()
        except Exception:
            pass
        try:
            await task.queue_frames([InterruptionFrame()])
        except Exception:
            pass

        call_info = TELNYX_CALLS.get(stream_id, {}).copy()
        TELNYX_CALLS.pop(stream_id, None)

        # Always send webhook to save transcript + trigger clinical report
        try:
            logger.info(f"Sending call webhook for {stream_id}...")
            await send_call_webhook(
                stream_id=stream_id,
                call_id=call_info.get("call_sid", stream_id),
                context=context,
                task=task,
                call_info=call_info,
            )
        except Exception as e:
            logger.warning(f"Webhook send error (non-fatal): {e}")

        try:
            await websocket.close()
        except Exception:
            pass

        logger.info(f"Telnyx pipeline closed: {stream_id}")
