"""
Self-check: dashboard routes act on the token's user, never a user_id sent by the client.
No network: Supabase auth and data helpers are stubbed.

    uv run python test_dashboard_auth.py
"""

import importlib.util
import os
import sys
import types

os.environ.setdefault("OPENAI_API_KEY", "test")
seen = []  # (helper, user_id) for every data call the routes make
# Helpers whose user_id is the 2nd positional arg; the rest take it first.
_USER_ID_SECOND = ("approve_report", "revoke_approval", "delete_phone_entry")


def _rec(name, result):
    async def f(*args, **kw):
        seen.append((name, args[1] if name in _USER_ID_SECOND else args[0]))
        return result
    return f


helpers = {
    "add_phone_entry": "row-1", "approve_report": True, "delete_phone_entry": True,
    "get_agent_config": None, "get_approvals": [], "get_phone_registry": [],
    "get_user_preferences": {}, "revoke_approval": True, "save_agent_config": "cfg-1",
    "save_user_preferences": True,
    "get_call_volume": [], "get_calls_history": {}, "get_dashboard_stats": {},
    "get_pipeline_stats": {}, "get_triage_outcomes": [],
}
# Load the route module by path with services stubbed (services/__init__ pulls in PDF + LanceDB).
sys.modules["services"] = types.ModuleType("services")
sys.modules["services.supabase_dashboard"] = types.SimpleNamespace(**{k: _rec(k, v) for k, v in helpers.items()})
_spec = importlib.util.spec_from_file_location(
    "dashboard_routes", os.path.join(os.path.dirname(os.path.abspath(__file__)), "api", "dashboard_routes.py")
)
dr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dr)

import auth  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


async def fake_user(authorization):
    if authorization not in ("Bearer alice", "Bearer carol"):
        raise HTTPException(401, "Invalid or expired session")
    return authorization.split()[1]


async def fake_profile_active(user_id):
    return user_id != "carol"  # carol was disabled by a super admin


auth._auth_user_id = fake_user
auth._profile_active = fake_profile_active
auth.settings.SUPABASE_URL = auth.settings.SUPABASE_URL or "https://example.supabase.co"
auth.settings.SUPABASE_SERVICE_ROLE_KEY = auth.settings.SUPABASE_SERVICE_ROLE_KEY or "service-key"

app = FastAPI()
app.include_router(dr.dashboard_router)
c = TestClient(app)
ALICE = {"Authorization": "Bearer alice"}

# Every user-scoped route, each trying to act as "bob".
routes = [
    ("get", "/reports/approvals?user_id=bob", None),
    ("post", "/reports/approve", {"call_id": "c1", "user_id": "bob"}),
    ("post", "/reports/revoke", {"call_id": "c1", "user_id": "bob"}),
    ("get", "/agent-config?user_id=bob", None),
    ("post", "/agent-config", {"user_id": "bob", "config": {}}),
    ("get", "/user-preferences?user_id=bob", None),
    ("post", "/user-preferences", {"user_id": "bob", "preferences": {}}),
    ("get", "/phone-registry?user_id=bob", None),
    ("post", "/phone-registry", {"user_id": "bob", "phone_number": "+911234567890"}),
    ("post", "/phone-registry/delete", {"user_id": "bob", "entry_id": "e1"}),
]

for method, path, body in routes:
    kw = {"json": body} if body is not None else {}
    assert getattr(c, method)(path, **kw).status_code == 401, f"{path} must require sign-in"
    assert getattr(c, method)(path, headers={"Authorization": "Bearer carol"}, **kw).status_code == 403, \
        f"{path} must reject a disabled account"
assert not seen, "no data helper may run without a valid, active session"

for method, path, body in routes:
    kw = {"json": body} if body is not None else {}
    r = getattr(c, method)(path, headers=ALICE, **kw)
    assert r.status_code == 200, f"{path}: {r.status_code} {r.text}"

assert len(seen) == len(routes), seen
assert all(uid == "alice" for _, uid in seen), f"spoofed user_id reached a helper: {seen}"

print("test_dashboard_auth: ALL PASSED")
