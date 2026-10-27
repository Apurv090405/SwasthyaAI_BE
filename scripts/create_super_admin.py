"""
Create (or promote) the SwasthyaAI super admin account.

Uses the Supabase Admin API with SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY from SwasthyaAI_BE/.env:
  1. creates the auth user with the email + password you choose (email pre-confirmed),
     or, if the email already exists, optionally resets its password;
  2. sets profiles.role = 'super_admin' and is_active = true (needs migration 004).

The password is read from a hidden prompt, so it never lands in shell history or files.

    cd SwasthyaAI_BE
    uv run python scripts/create_super_admin.py admin@yourclinic.org --name "Clinic Admin"
"""

import argparse
import getpass
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)


def _request(method: str, url: str, key: str, body=None, prefer: str = ""):
    headers = {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if prefer:
        headers["Prefer"] = prefer
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode()
            return resp.status, (json.loads(text) if text.strip() else None)
    except urllib.error.HTTPError as e:
        text = e.read().decode()
        try:
            return e.code, json.loads(text)
        except ValueError:
            return e.code, {"message": text}


def _find_user(base: str, key: str, email: str):
    page = 1
    while True:
        status, body = _request("GET", f"{base}/auth/v1/admin/users?page={page}&per_page=200", key)
        if status != 200:
            sys.exit(f"Could not list users ({status}): {body}")
        users = body.get("users", []) if isinstance(body, dict) else []
        for user in users:
            if (user.get("email") or "").lower() == email:
                return user
        if len(users) < 200:
            return None
        page += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or promote the SwasthyaAI super admin.")
    parser.add_argument("email")
    parser.add_argument("--name", default="Super Admin", help="Full name shown in the dashboard")
    args = parser.parse_args()
    email = args.email.strip().lower()

    base = (os.environ.get("SUPABASE_URL") or "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or ""
    if not base or not key:
        sys.exit("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in SwasthyaAI_BE/.env first.")

    user = _find_user(base, key, email)
    if user:
        print(f"User {email} already exists ({user['id']}).")
        if input("Reset its password? [y/N] ").strip().lower() == "y":
            password = _ask_password()
            status, body = _request("PUT", f"{base}/auth/v1/admin/users/{user['id']}", key, {"password": password})
            if status != 200:
                sys.exit(f"Password reset failed ({status}): {body}")
            print("Password updated.")
    else:
        password = _ask_password()
        status, body = _request(
            "POST",
            f"{base}/auth/v1/admin/users",
            key,
            {"email": email, "password": password, "email_confirm": True, "user_metadata": {"full_name": args.name}},
        )
        if status not in (200, 201) or not isinstance(body, dict) or not body.get("id"):
            sys.exit(f"Could not create user ({status}): {body}")
        user = body
        print(f"Created {email} ({user['id']}).")

    # The signup trigger (migration 001) creates the profile row; upsert in case it is missing.
    status, body = _request(
        "POST",
        f"{base}/rest/v1/profiles?on_conflict=id",
        key,
        {"id": user["id"], "email": email, "full_name": args.name, "role": "super_admin", "is_active": True},
        prefer="resolution=merge-duplicates,return=representation",
    )
    if status >= 400:
        hint = " Apply migration 004_super_admin.sql first." if "role" in json.dumps(body) else ""
        sys.exit(f"Could not set the super admin role ({status}): {body}.{hint}")
    print(f"{email} is now a super admin. Sign in at /signin; the Users section appears in the sidebar.")


def _ask_password() -> str:
    while True:
        password = getpass.getpass("New password (min 8 chars): ")
        if len(password) < 8:
            print("Too short.")
            continue
        if getpass.getpass("Repeat password: ") != password:
            print("Passwords don't match.")
            continue
        return password


if __name__ == "__main__":
    main()
