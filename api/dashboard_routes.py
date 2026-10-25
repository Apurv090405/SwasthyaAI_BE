"""
Dashboard API routes for SwasthyaAI.

Provides analytics, report approvals, agent config CRUD, user preferences,
phone registry, and paginated call history — all backed by Supabase.
"""

from typing import Optional

from fastapi import APIRouter, Query
from loguru import logger

from models import (
    AgentConfigRequest,
    ApproveReportRequest,
    PhoneRegistryDeleteRequest,
    PhoneRegistryRequest,
    UserPreferencesRequest,
)
from services.supabase_dashboard import (
    add_phone_entry,
    approve_report,
    delete_phone_entry,
    get_agent_config,
    get_approvals,
    get_call_volume,
    get_calls_history,
    get_dashboard_stats,
    get_phone_registry,
    get_pipeline_stats,
    get_triage_outcomes,
    get_user_preferences,
    revoke_approval,
    save_agent_config,
    save_user_preferences,
)

dashboard_router = APIRouter(tags=["dashboard"])


# ── Analytics ────────────────────────────────────────────────────────

@dashboard_router.get("/analytics/dashboard")
async def analytics_dashboard():
    """KPI stats for overview cards: today's calls, success rate, trends."""
    stats = await get_dashboard_stats()
    return stats


@dashboard_router.get("/analytics/call-volume")
async def analytics_call_volume(days: int = Query(default=7, ge=1, le=90)):
    """Daily call volume by direction (inbound/outbound/web) for bar chart."""
    data = await get_call_volume(days=days)
    return {"data": data, "days": days}


@dashboard_router.get("/analytics/triage-outcomes")
async def analytics_triage_outcomes():
    """Triage outcome distribution for pie chart."""
    data = await get_triage_outcomes()
    return {"data": data}


@dashboard_router.get("/analytics/pipeline-stats")
async def analytics_pipeline_stats():
    """Pipeline completion stats for gauge chart."""
    stats = await get_pipeline_stats()
    return stats


# ── Call History (paginated, filterable) ─────────────────────────────

@dashboard_router.get("/calls/history")
async def calls_history(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    direction: Optional[str] = Query(default=None),
    provider: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
):
    """Paginated call history with optional filters."""
    data = await get_calls_history(
        page=page, limit=limit, direction=direction, provider=provider, status=status
    )
    return data


# ── Report Approvals ─────────────────────────────────────────────────

@dashboard_router.get("/reports/approvals")
async def list_approvals(user_id: str = Query(...)):
    """List all approved reports for a user."""
    approvals = await get_approvals(user_id)
    return {"approvals": approvals, "count": len(approvals)}


@dashboard_router.post("/reports/approve")
async def approve_report_endpoint(req: ApproveReportRequest):
    """Approve a clinical report."""
    ok = await approve_report(req.call_id, req.user_id, req.notes)
    if ok:
        return {"status": "approved", "call_id": req.call_id}
    return {"status": "error", "message": "Failed to approve report"}


@dashboard_router.post("/reports/revoke")
async def revoke_report_endpoint(req: ApproveReportRequest):
    """Revoke approval for a clinical report."""
    ok = await revoke_approval(req.call_id, req.user_id)
    if ok:
        return {"status": "revoked", "call_id": req.call_id}
    return {"status": "error", "message": "Failed to revoke approval"}


# ── Agent Configuration ──────────────────────────────────────────────

@dashboard_router.get("/agent-config")
async def get_agent_config_endpoint(user_id: str = Query(...)):
    """Load saved agent config for a user."""
    config = await get_agent_config(user_id)
    if config:
        return {"found": True, **config}
    return {"found": False, "config": None}


@dashboard_router.post("/agent-config")
async def save_agent_config_endpoint(req: AgentConfigRequest):
    """Save or update agent configuration."""
    config_id = await save_agent_config(req.user_id, req.config, req.name)
    if config_id:
        return {"status": "saved", "id": config_id}
    return {"status": "error", "message": "Failed to save config"}


# ── User Preferences ────────────────────────────────────────────────

@dashboard_router.get("/user-preferences")
async def get_user_prefs(user_id: str = Query(...)):
    """Load user preferences (language, voice, theme)."""
    prefs = await get_user_preferences(user_id)
    return {"user_id": user_id, "preferences": prefs}


@dashboard_router.post("/user-preferences")
async def save_user_prefs(req: UserPreferencesRequest):
    """Save user preferences."""
    ok = await save_user_preferences(req.user_id, req.preferences)
    if ok:
        return {"status": "saved", "user_id": req.user_id}
    return {"status": "error", "message": "Failed to save preferences"}


# ── Phone Registry ──────────────────────────────────────────────────

@dashboard_router.get("/phone-registry")
async def list_phone_registry(user_id: str = Query(...)):
    """List manually registered phone numbers for a user."""
    entries = await get_phone_registry(user_id)
    return {"entries": entries, "count": len(entries)}


@dashboard_router.post("/phone-registry")
async def add_phone_registry(req: PhoneRegistryRequest):
    """Add a phone number to the manual registry."""
    entry_id = await add_phone_entry(
        req.user_id, req.phone_number, req.provider, req.label, req.is_default
    )
    if entry_id:
        return {"status": "added", "id": entry_id}
    return {"status": "error", "message": "Failed to add phone number"}


@dashboard_router.post("/phone-registry/delete")
async def delete_phone_registry(req: PhoneRegistryDeleteRequest):
    """Delete a phone registry entry."""
    ok = await delete_phone_entry(req.entry_id, req.user_id)
    if ok:
        return {"status": "deleted", "id": req.entry_id}
    return {"status": "error", "message": "Failed to delete phone entry"}
