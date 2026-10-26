"""
Self-check for api/voicelink_routes.py (no network: VoiceLink, Supabase auth and registry are stubbed).

    uv run python test_voicelink_routes.py
"""

import importlib.util
import os
import sys
import types

os.environ.setdefault("OPENAI_API_KEY", "test")
# Load the route module by path with services stubbed (services/__init__ pulls in PDF + LanceDB).
registered = []


async def fake_add_phone_entry(**kw):
    registered.append(kw)
    return "row-1"


sys.modules["services"] = types.ModuleType("services")
sys.modules["services.supabase_dashboard"] = types.SimpleNamespace(add_phone_entry=fake_add_phone_entry)
_spec = importlib.util.spec_from_file_location(
    "voicelink_routes", os.path.join(os.path.dirname(os.path.abspath(__file__)), "api", "voicelink_routes.py")
)
vl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vl)

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

calls = []


async def fake_vl(method, path, *, params=None, json=None):
    calls.append((method, path))
    if path == "/reseller/purchase/lock-did":
        return {"order_id": 5, "locked_dids": [{"did_id": 1, "did_number": "+919876543210"}]}
    if path.startswith("/reseller/purchase/summary/"):
        return {"order_id": 5, "order_number": "ORD-1", "total": 90, "did_cost": 90, "wallet_balance": 5000}
    if path == "/reseller/purchase/confirm":
        return {"order_number": "ORD-1", "total": 106.2, "purchase_order_id": 10}
    raise AssertionError(f"unexpected call {path}")


async def fake_user(authorization):
    if authorization not in ("Bearer alice", "Bearer bob", "Bearer carol"):
        raise HTTPException(401, "Invalid or expired session")
    return authorization.split()[1]


async def fake_profile_active(user_id):
    return user_id != "carol"  # carol was disabled by a super admin


vl._vl = fake_vl
# Stub only the two network lookups in auth.py so the real require_user gate runs.
vl.auth._auth_user_id = fake_user
vl.auth._profile_active = fake_profile_active
vl.settings.SUPABASE_URL = vl.settings.SUPABASE_URL or "https://example.supabase.co"
vl.settings.SUPABASE_SERVICE_ROLE_KEY = vl.settings.SUPABASE_SERVICE_ROLE_KEY or "service-key"

app = FastAPI()
app.include_router(vl.voicelink_router)
c = TestClient(app)

assert c.post("/voicelink/order", json={"did_ids": [1], "months": 3}).status_code == 401, "order needs sign-in"

r = c.post("/voicelink/order", json={"did_ids": [1], "months": 3}, headers={"Authorization": "Bearer carol"})
assert r.status_code == 403, "disabled account must not get a quote"
r = c.post("/voicelink/order/confirm", json={"order_id": 5}, headers={"Authorization": "Bearer carol"})
assert r.status_code == 403, "disabled account must not confirm a purchase"
assert not calls, "disabled account must not reach VoiceLink at all"

r = c.post("/voicelink/order", json={"did_ids": [1], "months": 3}, headers={"Authorization": "Bearer alice"})
assert r.status_code == 200, r.text
assert "wallet_balance" not in r.json()["summary"], "reseller wallet must not leak to users"
assert ("POST", "/reseller/purchase/confirm") not in calls, "quote must not charge"

r = c.post("/voicelink/order/confirm", json={"order_id": 5}, headers={"Authorization": "Bearer bob"})
assert r.status_code == 404, "another user must not confirm alice's order"
assert ("POST", "/reseller/purchase/confirm") not in calls

r = c.post("/voicelink/order/confirm", json={"order_id": 5}, headers={"Authorization": "Bearer alice"})
assert r.status_code == 200, r.text
assert r.json()["phone_numbers"] == ["+919876543210"]
assert registered and registered[0]["user_id"] == "alice" and registered[0]["provider"] == "voicelink"

r = c.post("/voicelink/order/confirm", json={"order_id": 5}, headers={"Authorization": "Bearer alice"})
assert r.status_code == 404, "an order must not be confirmed (charged) twice"
assert calls.count(("POST", "/reseller/purchase/confirm")) == 1

print("test_voicelink_routes: ALL PASSED")
