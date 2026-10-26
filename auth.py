"""
Caller identity for routes that act on a user's own data.

The backend talks to Supabase with the service-role key (bypasses RLS), so routes must never
trust a user_id sent by the client. They take it from the caller's Supabase access token instead.
"""

from typing import Optional

import aiohttp
from fastapi import Header, HTTPException
from loguru import logger

from config import settings


async def require_user(authorization: Optional[str]) -> str:
    """Resolve the caller's Supabase user id; reject accounts a super admin has disabled."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Sign in required")
    if not (settings.SUPABASE_URL and settings.SUPABASE_SERVICE_ROLE_KEY):
        raise HTTPException(503, "Supabase is not configured")
    user_id = await _auth_user_id(authorization)
    # Disabling bans sign-in, but issued access tokens live ~1h; this closes that window.
    if not await _profile_active(user_id):
        raise HTTPException(403, "Account is disabled")
    return user_id


async def current_user(authorization: Optional[str] = Header(None)) -> str:
    """FastAPI dependency: `user_id: str = Depends(current_user)`."""
    return await require_user(authorization)


async def _auth_user_id(authorization: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{settings.SUPABASE_URL.rstrip('/')}/auth/v1/user",
            headers={"apikey": settings.SUPABASE_SERVICE_ROLE_KEY, "Authorization": authorization},
        ) as resp:
            status = resp.status
            body = await resp.json(content_type=None)
    if status != 200 or not (isinstance(body, dict) and body.get("id")):
        raise HTTPException(401, "Invalid or expired session")
    return body["id"]


async def _profile_active(user_id: str) -> bool:
    """profiles.is_active (migration 004). Missing row counts as inactive (fail closed)."""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{settings.SUPABASE_URL.rstrip('/')}/rest/v1/profiles",
            params={"id": f"eq.{user_id}", "select": "is_active"},
            headers={
                "apikey": settings.SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
            },
        ) as resp:
            status = resp.status
            rows = await resp.json(content_type=None)
    if status != 200 or not isinstance(rows, list):
        logger.error(f"[auth] profile lookup failed {status}: {str(rows)[:300]}")
        raise HTTPException(503, "Could not verify account status")
    return bool(rows) and rows[0].get("is_active") is True
