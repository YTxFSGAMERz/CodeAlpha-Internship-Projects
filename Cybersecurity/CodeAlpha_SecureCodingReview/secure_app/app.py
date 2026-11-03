"""
NoteBox (hardened edition)
==========================
The same API as vulnerable_app, with every finding in SECURITY_REVIEW.md fixed.
Comments tagged S01 ... S13 match the findings V01 ... V13.

Run:  export NOTEBOX_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
      export NOTEBOX_ADMIN_PASSWORD='choose-a-long-passphrase'
      python app.py
"""
import hmac
import ipaddress
import os
import re
import secrets
import socket
import sqlite3
import subprocess
import sys
import time
from urllib.parse import urlparse

import requests
from flask import Flask, abort, g, jsonify, render_template_string, request, send_from_directory, session
from werkzeug.exceptions import HTTPException
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
HOST_RE = re.compile(r"(?!-)[A-Za-z0-9.-]{1,253}")
USER_RE = re.compile(r"[A-Za-z0-9_]{3,32}")
MAX_FAILS, WINDOW = 5, 60                                  # S13: 5 failures per 60 seconds
DUMMY_HASH = generate_password_hash("not-a-real-password")  # keeps timing equal for unknown users
SCHEMA = (
    "CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT, role TEXT);"
    "CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, owner_id INTEGER, title TEXT, body TEXT);"
)


def is_public_host(hostname):
    """True only if every address the name resolves to is public and routable (S10)."""
    try:
        infos = socket.getaddrinfo(hostname, None)
        return bool(infos) and all(ipaddress.ip_address(i[4][0]).is_global for i in infos)
    except (socket.gaierror, ValueError):
        return False


def create_app(db_path=None, testing=False):
    secret = os.environ.get("NOTEBOX_SECRET_KEY")           # S03: secrets come from the environment
    if not secret:
        raise RuntimeError("NOTEBOX_SECRET_KEY is not set (see the docstring at the top of this file).")
    app = Flask(__name__)
    app.secret_key = secret
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,                       # S12: hidden from page scripts
        SESSION_COOKIE_SAMESITE="Lax",                      # S11/S12: not sent on cross-site POSTs
        SESSION_COOKIE_SECURE=not testing,                  # S12: HTTPS only (tests run over http)
        MAX_CONTENT_LENGTH=64 * 1024,
    )
    db_file = db_path or os.path.join(BASE_DIR, "notebox.db")
    failures = {}   # (ip, username) -> failure times; use Flask-Limiter with Redis in production

    def db():
        if "db" not in g:
            g.db = sqlite3.connect(db_file)
        return g.db

    @app.teardown_appcontext
    def close_db(_exc):
        conn = g.pop("db", None)
        if conn is not None:
            conn.close()

    with app.app_context():
        conn = db()
        conn.executescript(SCHEMA)
        admin_pw = os.environ.get("NOTEBOX_ADMIN_PASSWORD")
        if admin_pw:
            conn.execute("INSERT OR IGNORE INTO users(username, password, role) VALUES ('admin', ?, 'admin')",
                         (generate_password_hash(admin_pw),))      # S04: salted, slow hash
        conn.commit()

    @app.before_request
    def csrf_protect():                                     # S11: token required on every state change
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            sent, expected = request.headers.get("X-CSRF-Token", ""), session.get("csrf", "")
            if not expected or not hmac.compare_digest(sent.encode(), expected.encode()):
                abort(403)

    @app.after_request
    def security_headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.errorhandler(Exception)
    def handle_error(exc):                                  # no stack traces for clients
        if isinstance(exc, HTTPException):
            return jsonify(error=exc.name.lower()), exc.code
        app.logger.exception("unhandled error")
        return jsonify(error="internal error"), 500

    def login_required():
        if session.get("uid") is None:
            abort(401)

    @app.get("/csrf-token")
    def csrf_token():
        return jsonify(token=session.setdefault("csrf", secrets.token_hex(16)))

    @app.post("/register")
    def register():
        data = request.get_json(silent=True) or {}
        username, password = data.get("username"), data.get("password")
        if not (isinstance(username, str) and USER_RE.fullmatch(username)
                and isinstance(password, str) and len(password) >= 12):
            return jsonify(error="username: 3-32 letters, digits or _; password: 12+ characters"), 400
        try:
            db().execute("INSERT INTO users(username, password, role) VALUES (?, ?, 'user')",
                         (username, generate_password_hash(password)))
            db().commit()
        except sqlite3.IntegrityError:
            return jsonify(error="username taken"), 409
        return jsonify(ok=True), 201

    @app.post("/login")
    def login():
        data = request.get_json(silent=True) or {}
        username, password = str(data.get("username", "")), str(data.get("password", ""))
        key, now = (request.remote_addr, username.lower()), time.time()
        recent = [t for t in failures.get(key, []) if now - t < WINDOW]
        if len(recent) >= MAX_FAILS:                        # S13: throttle guessing
            failures[key] = recent
            return jsonify(error="too many attempts, try again later"), 429
        row = db().execute("SELECT id, role, password FROM users WHERE username = ?", (username,)).fetchone()  # S01
        valid = check_password_hash(row[2] if row else DUMMY_HASH, password)
        if not (row and valid):
            failures[key] = recent + [now]
            return jsonify(error="invalid credentials"), 401
        failures.pop(key, None)
        session.clear()
        session.update(uid=row[0], role=row[1], csrf=secrets.token_hex(16))
        return jsonify(ok=True, role=row[1])

    @app.post("/logout")
    def logout():
        session.clear()
        return jsonify(ok=True)

    @app.get("/whoami")
    def whoami():
        return jsonify(uid=session.get("uid"), role=session.get("role"))

    @app.post("/notes")
    def add_note():
        login_required()
        data = request.get_json(silent=True) or {}
        title, body = data.get("title"), data.get("body")
        if not (isinstance(title, str) and isinstance(body, str) and 0 < len(title) <= 200 and len(body) <= 5000):
            abort(400)
        cur = db().execute("INSERT INTO notes(owner_id, title, body) VALUES (?, ?, ?)",
                           (session["uid"], title, body))
        db().commit()
        return jsonify(id=cur.lastrowid), 201

    @app.get("/notes/<int:note_id>")
    def get_note(note_id):
        login_required()
        row = db().execute("SELECT id, owner_id, title, body FROM notes WHERE id = ? AND owner_id = ?",
                           (note_id, session["uid"])).fetchone()                       # S09: owner check
        if not row:
            abort(404)
        return jsonify(id=row[0], owner_id=row[1], title=row[2], body=row[3])

    @app.get("/search")
    def search():
        login_required()
        like = "%" + request.args.get("q", "") + "%"
        rows = db().execute("SELECT id, title FROM notes WHERE owner_id = ? AND title LIKE ?",
                            (session["uid"], like)).fetchall()                         # S02: bound parameters
        return jsonify(results=[{"id": r[0], "title": r[1]} for r in rows])

    @app.get("/greet")
    def greet():                                            # S05: input is data, never template source
        return render_template_string("<h1>Hello {{ name }}!</h1>", name=request.args.get("name", "friend")[:50])

    @app.get("/ping")
    def ping():
        if session.get("role") != "admin":
            abort(403)
        host = request.args.get("host", "127.0.0.1")
        if not HOST_RE.fullmatch(host):                     # S06: allow-list, no leading "-"
            abort(400)
        try:                                                # S06: argument list, no shell
            ping_args = ["ping", "-n", "1", "-w", "2000", host] if sys.platform.startswith("win") else ["ping", "-c", "1", "-W", "2", host]
            result = subprocess.run(ping_args,
                                    capture_output=True, text=True, timeout=5, shell=False)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return jsonify(error="ping unavailable"), 503
        return result.stdout, 200, {"Content-Type": "text/plain; charset=utf-8"}

    @app.get("/download")
    def download():                                         # S07: refuses paths outside uploads/
        return send_from_directory(UPLOAD_DIR, request.args.get("file", "welcome.txt"))

    @app.get("/preview")
    def preview():
        login_required()
        try:
            parsed = urlparse(request.args.get("url", ""))
            allowed = (parsed.scheme in ("http", "https") and parsed.hostname
                       and parsed.port in (None, 80, 443) and is_public_host(parsed.hostname))
        except ValueError:
            allowed = False
        if not allowed:                                     # S10: public http(s) targets only
            abort(400)
        try:
            resp = requests.get(parsed.geturl(), timeout=3, allow_redirects=False, stream=True)
            body = resp.raw.read(2000, decode_content=True)
        except requests.RequestException:
            abort(502)
        return body.decode("utf-8", "replace"), 200, {"Content-Type": "text/plain; charset=utf-8"}

    @app.route("/prefs", methods=["GET", "POST"])
    def prefs():                                            # S08: signed session, no pickle
        if request.method == "POST":
            theme = (request.get_json(silent=True) or {}).get("theme")
            if theme not in ("light", "dark"):
                abort(400)
            session["theme"] = theme
        return jsonify(theme=session.get("theme", "light"))

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5002)           # S12: debug mode off
