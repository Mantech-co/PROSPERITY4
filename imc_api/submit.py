#!/usr/bin/env python3
"""
Usage:
  python imc_api/submit.py <algo.py> [--round 3]
"""

import argparse
import asyncio
import base64
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

AUTH_FILE = Path(__file__).parent / ".auth.json"
API_BASE = "https://3dzqiahkw1.execute-api.eu-west-1.amazonaws.com/prod"
POLL_INTERVAL = 30
LOGIN_URL = "https://prosperity.imc.com/login"
COGNITO_CLIENT_ID = "5kgp0jm69aeb91paqj1hnps838"
COGNITO_ENDPOINT = "https://cognito-idp.eu-west-1.amazonaws.com/"


def load_token():
    if AUTH_FILE.exists():
        return json.loads(AUTH_FILE.read_text()).get("token")
    return None


def cognito_refresh(refresh_token):
    body = json.dumps({
        "AuthFlow": "REFRESH_TOKEN_AUTH",
        "ClientId": COGNITO_CLIENT_ID,
        "AuthParameters": {"REFRESH_TOKEN": refresh_token},
    }).encode()
    req = urllib.request.Request(
        COGNITO_ENDPOINT,
        data=body,
        headers={
            "Content-Type": "application/x-amz-json-1.1",
            "X-Amz-Target": "AmazonCognitoIdentityProviderService.InitiateAuth",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["AuthenticationResult"]["IdToken"]


def save_token(token):
    AUTH_FILE.write_text(json.dumps({"token": token}))
    AUTH_FILE.chmod(0o600)


def token_valid(token):
    if not token:
        return False
    try:
        seg = token.split(".")[1]
        seg += "=" * (-len(seg) % 4)
        payload = json.loads(base64.b64decode(seg))
        return time.time() < payload.get("exp", 0) - 60
    except Exception:
        return False


async def _playwright_login():
    from playwright.async_api import async_playwright

    token = None
    print("Opening browser for login...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        def handle_request(request):
            nonlocal token
            auth = request.headers.get("authorization", "")
            if auth.startswith("Bearer ") and not token:
                token = auth[len("Bearer "):]

        page.on("request", handle_request)
        await page.goto(LOGIN_URL)
        print("Log in — waiting for authenticated request...")

        while not token:
            await asyncio.sleep(1)

        await browser.close()
        return token


def login_and_get_token():
    print("Auth method: [1] Browser (Playwright)  [2] Paste idToken  [3] Paste refreshToken")
    choice = input("Choose [1/2/3]: ").strip()
    if choice == "2":
        print("Run in browser console on prosperity.imc.com, then navigate any page:")
        print("  (()=>{const o=window.fetch;window.fetch=(...a)=>{const h=(a[1]?.headers||{});const t=h['authorization']||h['Authorization']||'';if(t.startsWith('Bearer '))console.log('TOKEN:',t.slice(7));return o(...a)}})()")
        token = input("Paste idToken: ").strip()
        if token.startswith("Bearer "):
            token = token[len("Bearer "):]
    elif choice == "3":
        print("In DevTools: Network tab -> filter 'cognito' -> find InitiateAuth response -> copy RefreshToken from AuthenticationResult")
        refresh_token = input("Paste refreshToken: ").strip()
        print("Refreshing via Cognito...")
        token = cognito_refresh(refresh_token)
    else:
        token = asyncio.run(_playwright_login())
    if not token:
        raise RuntimeError("No token provided")
    save_token(token)
    print("Token saved.")
    return token


def _headers(token):
    return {
        "authorization": f"Bearer {token}",
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:149.0) Gecko/20100101 Firefox/149.0",
        "Accept": "*/*",
        "Origin": "https://prosperity.imc.com",
        "Referer": "https://prosperity.imc.com/",
    }


def submit_algo(token, file_path):
    url = f"{API_BASE}/submission/algo"
    boundary = b"----prosperiyboundary42"
    filename = Path(file_path).name
    file_bytes = Path(file_path).read_bytes()

    body = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="' + filename.encode() + b'"\r\n'
        b"Content-Type: text/x-python\r\n\r\n"
        + file_bytes
        + b"\r\n--" + boundary + b"--\r\n"
    )

    hdrs = _headers(token)
    hdrs["Content-Type"] = "multipart/form-data; boundary=----prosperiyboundary42"
    hdrs["Content-Length"] = str(len(body))

    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())

    if not data.get("success"):
        raise RuntimeError(f"Submission failed: {data}")

    sub = data["data"]
    print(f"Submitted: id={sub['id']}  round={sub['roundId']}  status={sub['status']}")
    return sub["id"], sub["roundId"]


def poll_status(token, sub_id, round_id):
    url = f"{API_BASE}/submissions/algo/{round_id}?page=1&pageSize=50"
    hdrs = _headers(token)

    while True:
        req = urllib.request.Request(url, headers=hdrs)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())

        items = data["data"]["items"]
        sub = next((x for x in items if x["id"] == sub_id), None)
        if sub is None:
            raise RuntimeError(f"Submission {sub_id} not in response")

        status = sub["status"]
        print(f"[{time.strftime('%H:%M:%S')}] {status}")

        if status not in ACTIVE_STATUSES:
            return status

        time.sleep(POLL_INTERVAL)


ACTIVE_STATUSES = ("SIMULATING", "QUEUED", "PENDING")


def find_active_submission(token, round_id):
    url = f"{API_BASE}/submissions/algo/{round_id}?page=1&pageSize=50"
    req = urllib.request.Request(url, headers=_headers(token))
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return next((x for x in data["data"]["items"] if x["status"] in ACTIVE_STATUSES), None)


def fetch_zip(token, sub_id, logs_dir):
    url = f"{API_BASE}/submissions/algo/{sub_id}/zip"
    req = urllib.request.Request(url, headers=_headers(token))
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())

    zip_url = data["data"]["url"]
    print("Downloading zip...")

    logs_dir = Path(logs_dir)
    logs_dir.mkdir(exist_ok=True)
    zip_path = logs_dir / f"{sub_id}.zip"
    with urllib.request.urlopen(zip_url) as resp2:
        zip_path.write_bytes(resp2.read())
    print(f"Saved zip: {zip_path}")
    return zip_path


def unzip_and_move(zip_path, logviz_dir):
    logviz_dir = Path(logviz_dir)
    logviz_dir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)

        moved = 0
        for lf in tmp.rglob("*.log"):
            dest = logviz_dir / lf.name
            shutil.copy2(str(lf), str(dest))
            print(f"  {lf.name} -> {logviz_dir}/")
            moved += 1

    if moved == 0:
        print("No .log files found in zip.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help=".py algo file to submit")
    parser.add_argument("--round", type=int, default=3, help="Round ID for pre-submit check (default: 3)")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--logviz-dir", default="logviz")
    parser.add_argument("--token", help="Bearer token (skips login prompt)")
    args = parser.parse_args()

    if args.token:
        token = args.token.removeprefix("Bearer ")
    else:
        token = load_token()
    if not token_valid(token):
        print("Token missing/expired, logging in...")
        token = login_and_get_token()

    active = find_active_submission(token, args.round)
    if active:
        print(f"Active submission found: id={active['id']}  status={active['status']}  file={active['filename']}")
        ans = input("Track this instead of submitting new? [y/N]: ").strip().lower()
        if ans == "y":
            sub_id, round_id = active["id"], active["roundId"]
        else:
            print(f"Waiting for submission {active['id']} to finish...")
            poll_status(token, active["id"], active["roundId"])
            sub_id, round_id = submit_algo(token, args.file)
    else:
        sub_id, round_id = submit_algo(token, args.file)

    status = poll_status(token, sub_id, round_id)

    if status not in ("DONE", "FINISHED"):
        print(f"Ended with non-DONE status: {status}")
        sys.exit(1)

    zip_path = fetch_zip(token, sub_id, args.logs_dir)
    unzip_and_move(zip_path, args.logviz_dir)
    print("Done.")

    logviz_dir = Path(args.logviz_dir)
    logs = sorted(logviz_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
    if logs:
        newest = logs[-1]
        print(f"Opening: {newest}")
        visualizer = Path(__file__).parent.parent / "logviz" / "log_visualizer.py"
        if visualizer.exists():
            subprocess.Popen([sys.executable, str(visualizer), str(newest)])


if __name__ == "__main__":
    main()
