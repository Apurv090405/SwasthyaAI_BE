"""
VoiceLink (India) DID purchase — https://app.voicelink.co.in/documentation/

The platform owns one VoiceLink reseller account (VOICELINK_USERNAME / VOICELINK_PASSWORD).
Signed-in users browse direct-purchase Indian DIDs, get a server-priced quote, then confirm.
Purchases debit the reseller wallet, so order/confirm require a valid Supabase session and
each bought number is recorded in phone_registry (provider='voicelink') for that user.
"""

import time
from typing import Any, Dict, List, Optional

import aiohttp
from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

import auth
from config import settings
from services.supabase_dashboard import add_phone_entry

voicelink_router = APIRouter(prefix="/voicelink", tags=["voicelink"])

VOICELINK_API_BASE = "https://app.voicelink.co.in/api/v1"

# ponytail: in-process token + pending-order map (same as TELNYX_CALLS); move to Supabase if running >1 worker.
_token: Optional[str] = None
_pending_orders: Dict[int, Dict[str, Any]] = {}
_PENDING_TTL_SECONDS = 3600


async def _login() -> str:
    global _token
    if not (settings.VOICELINK_USERNAME and settings.VOICELINK_PASSWORD):
        raise HTTPException(503, "VoiceLink is not configured (VOICELINK_USERNAME / VOICELINK_PASSWORD).")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{VOICELINK_API_BASE}/auth/login",
            json={"username": settings.VOICELINK_USERNAME, "password": settings.VOICELINK_PASSWORD},
            headers={"Accept": "application/json"},
        ) as resp:
            status = resp.status
            body = await resp.json(content_type=None)
    token = body.get("data", {}).get("access_token") if isinstance(body, dict) else None
    if status >= 400 or not token:
        logger.error(f"[VoiceLink] login failed {status}: {str(body)[:300]}")
        raise HTTPException(502, "VoiceLink login failed; check reseller credentials.")
    _token = token
    return token


async def _vl(method: str, path: str, *, params: Optional[Dict] = None, json: Optional[Dict] = None) -> Any:
    """Call VoiceLink and return its `data`. Re-logs in once on 401."""
    for attempt in range(2):
        token = _token or await _login()
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method,
                f"{VOICELINK_API_BASE}{path}",
                params=params,
                json=json,
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            ) as resp:
                status = resp.status
                body = await resp.json(content_type=None)
        if status == 401 and attempt == 0:
            await _login()
            continue
        # Error shape is not documented beyond `message`; pass it through.
        if status >= 400 or not (isinstance(body, dict) and body.get("status")):
            msg = body.get("message") if isinstance(body, dict) else None
            logger.warning(f"[VoiceLink] {method} {path} -> {status}: {str(body)[:400]}")
            raise HTTPException(status if status >= 400 else 502, msg or "VoiceLink request failed")
        return body.get("data")
    raise HTTPException(502, "VoiceLink authentication failed")


@voicelink_router.get("/did-types")
async def did_types():
    """Direct-purchase DID types (request-required series need the approval flow, not supported here)."""
    return {"types": await _vl("GET", "/reseller/did-types", params={"requires_request": 0})}


@voicelink_router.get("/dids")
async def search_dids(
    type: int = Query(..., description="DID type id from /voicelink/did-types"),
    months: int = Query(1, ge=1, le=36),
    number: Optional[str] = Query(None, description="Partial number search"),
    page: int = Query(1, ge=1),
):
    params: Dict[str, Any] = {"country_code": "+91", "type": type, "months": months, "page": page, "per_page": 20}
    if number:
        params["number"] = number
    data = await _vl("GET", "/reseller/purchase/dids", params=params)
    return {"dids": (data or {}).get("data", [])}


class OrderRequest(BaseModel):
    did_ids: List[int] = Field(..., min_length=1, max_length=10)
    months: int = Field(1, ge=1, le=36)


@voicelink_router.post("/order")
async def create_order(req: OrderRequest, user_id: str = Depends(auth.current_user)):
    """Lock DIDs into a draft order and return VoiceLink's price summary. Nothing is charged yet."""
    locked = await _vl("POST", "/reseller/purchase/lock-did", json={"did_ids": req.did_ids, "months": req.months})
    order_id = locked["order_id"]
    summary = await _vl("GET", f"/reseller/purchase/summary/{order_id}")

    now = time.time()
    for oid in [k for k, v in _pending_orders.items() if now - v["created"] > _PENDING_TTL_SECONDS]:
        _pending_orders.pop(oid, None)
    _pending_orders[order_id] = {
        "user_id": user_id,
        "locked_dids": locked.get("locked_dids", []),
        "months": req.months,
        "created": now,
    }
    # wallet_balance is the platform's reseller wallet; never expose it to end users.
    summary = {k: v for k, v in (summary or {}).items() if k != "wallet_balance"}
    return {"order_id": order_id, "locked_dids": locked.get("locked_dids", []), "summary": summary}


class ConfirmRequest(BaseModel):
    order_id: int


@voicelink_router.post("/order/confirm")
async def confirm_order(req: ConfirmRequest, user_id: str = Depends(auth.current_user)):
    """Confirm a draft order created by the same user; debits the reseller wallet."""
    pending = _pending_orders.get(req.order_id)
    if not pending or pending["user_id"] != user_id:
        raise HTTPException(404, "Order not found or expired; get a new quote.")

    result = await _vl("POST", "/reseller/purchase/confirm", json={"order_id": req.order_id}) or {}
    _pending_orders.pop(req.order_id, None)

    numbers = [d["did_number"] for d in pending["locked_dids"] if d.get("did_number")]
    for n in numbers:
        entry = await add_phone_entry(
            user_id=user_id,
            phone_number=n,
            provider="voicelink",
            label=f"VoiceLink · {pending['months']} mo · {result.get('order_number') or req.order_id}",
        )
        if not entry:
            # Paid but not recorded: must be reconciled by hand from the VoiceLink panel.
            logger.error(f"[VoiceLink] PURCHASED BUT NOT REGISTERED user={user_id} number={n} order={req.order_id}")
    logger.info(f"[VoiceLink] purchase confirmed user={user_id} order={req.order_id} numbers={numbers}")
    return {
        "status": "purchased",
        "order_number": result.get("order_number"),
        "total": result.get("total"),
        "phone_numbers": numbers,
    }
