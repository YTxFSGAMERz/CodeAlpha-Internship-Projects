"""
NoteBox (vulnerable edition)
============================
A small note-taking API written as the target of the secure code review in this
repository. It deliberately contains mistakes that are common in real code.
Every one is tagged with its finding ID (V01 ... V13) so the review can point at it.

NEVER deploy this. It only listens on 127.0.0.1 and exists for learning.
"""
import base64
import hashlib
import os
import pickle
import sqlite3
import subprocess
import sys
import urllib.request

from flask import Flask, g, jsonify, render_template_string, request, send_file, session

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

SECRET_KEY = "dev-secret-123"      # V03: hard-coded signing key
ADMIN_PASSWORD = "admin123"        # V03: hard-coded default admin password


def md5(text):
    return hashlib.md5(text.encode()).hexdigest()   # V04: fast, unsalted hash used for passwords


def create_app(db_path=None, testing=False):
    app = Flask(__name__)
    app.secret_key = SECRET_KEY
    app.config["SESSION_COOKIE_HTTPONLY"] = False   # V12: page scripts can read the session cookie
    db_file = db_path or os.path.join(BASE_DIR, "notebox.db")

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
        conn.executescript(
            "CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT, role TEXT);"
            "CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, owner_id INTEGER, title TEXT, body TEXT);"
        )
        conn.execute("INSERT OR IGNORE INTO users(username, password, role) VALUES ('admin', ?, 'admin')",
                     (md5(ADMIN_PASSWORD),))
        conn.commit()

    @app.get("/")
    def index():
        return render_template_string("""<!DOCTYPE html>
<html>
<head>
    <title>NoteBox (Vulnerable Edition)</title>
    <style>
        body { font-family: system-ui, -apple-system, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; line-height: 1.5; color: #1c1c1c; }
        .badge { background: #d9381e; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }
        .card { background: #fdf5f5; border: 1px solid #f5c2c7; border-radius: 8px; padding: 16px; margin: 20px 0; }
        code { background: #eee; padding: 2px 6px; border-radius: 4px; }
        ul { line-height: 1.8; }
    </style>
</head>
<body>
    <h1>NoteBox <span class="badge">Vulnerable Edition (Port 5001)</span></h1>
    <div class="card">
        <strong>⚠️ Security Review Target:</strong> This application deliberately contains known security flaws (V01–V13) for audit practice. 
        It is paired with the hardened version running on port 5002.
    </div>
    <h3>Available Endpoints:</h3>
    <ul>
        <li><code>POST /register</code> - Register new account: <code>{"username": "...", "password": "..."}</code></li>
        <li><code>POST /login</code> - Account login (contains SQLi: <code>admin' --</code>)</li>
        <li><code>GET /notes</code> - List authenticated user's notes</li>
        <li><code>POST /notes</code> - Create note: <code>{"title": "...", "body": "..."}</code></li>
        <li><code>GET /search?q=...</code> - Search notes (contains SQLi data extraction)</li>
        <li><code>GET /greet?name=...</code> - Greeter (contains SSTI & XSS)</li>
        <li><code>GET /ping?host=127.0.0.1</code> - Admin ping diagnostic (contains Command Injection)</li>
        <li><code>GET /download?file=...</code> - File download (contains Directory Traversal)</li>
        <li><code>GET /preview?url=...</code> - URL previewer (contains SSRF)</li>
        <li><code>GET/POST /prefs</code> - Theme preferences (contains Insecure Deserialization)</li>
    </ul>
    <p>Run proof-of-concept automated test suite: <code>pytest tests/test_findings.py -v</code></p>
</body>
</html>""")

    @app.post("/register")
    def register():
        data = request.get_json(force=True)
        try:
            db().execute("INSERT INTO users(username, password, role) VALUES (?, ?, 'user')",
                         (data["username"], md5(data["password"])))
            db().commit()
        except sqlite3.IntegrityError:
            return jsonify(error="username taken"), 409
        return jsonify(ok=True), 201

    @app.post("/login")
    def login():
        data = request.get_json(force=True)
        # V01: SQL built by string formatting, so  admin' --  logs in without a password
        query = "SELECT id, role FROM users WHERE username = '%s' AND password = '%s'" % (
            data["username"], md5(data["password"]))
        row = db().execute(query).fetchone()
        if not row:
            return jsonify(error="invalid credentials"), 401      # V13: no limit on failed attempts
        session["uid"], session["role"] = row
        return jsonify(ok=True, role=row[1])

    @app.post("/logout")
    def logout():
        session.clear()
        return jsonify(ok=True)

    @app.get("/whoami")
    def whoami():
        return jsonify(uid=session.get("uid"), role=session.get("role"))

    @app.post("/notes")
    def add_note():                                  # V11: state-changing request with no CSRF token
        if session.get("uid") is None:
            return jsonify(error="login required"), 401
        data = request.get_json(force=True)
        cur = db().execute("INSERT INTO notes(owner_id, title, body) VALUES (?, ?, ?)",
                           (session["uid"], data["title"], data["body"]))
        db().commit()
        return jsonify(id=cur.lastrowid), 201

    @app.get("/notes/<int:note_id>")
    def get_note(note_id):
        if session.get("uid") is None:
            return jsonify(error="login required"), 401
        # V09: any logged-in user can read any note; the owner is never checked (IDOR)
        row = db().execute("SELECT id, owner_id, title, body FROM notes WHERE id = ?", (note_id,)).fetchone()
        if not row:
            return jsonify(error="not found"), 404
        return jsonify(id=row[0], owner_id=row[1], title=row[2], body=row[3])

    @app.get("/search")
    def search():
        if session.get("uid") is None:
            return jsonify(error="login required"), 401
        q = request.args.get("q", "")
        # V02: user input concatenated into SQL, so a UNION query can read other tables
        sql = "SELECT id, title FROM notes WHERE owner_id = %d AND title LIKE '%%%s%%'" % (session["uid"], q)
        return jsonify(results=[{"id": r[0], "title": r[1]} for r in db().execute(sql).fetchall()])

    @app.get("/greet")
    def greet():
        name = request.args.get("name", "friend")
        # V05: user input becomes part of the template source (XSS and template injection)
        return render_template_string("<h1>Hello " + name + "!</h1>")

    @app.get("/ping")
    def ping():
        if session.get("role") != "admin":
            return jsonify(error="admin only"), 403
        host = request.args.get("host", "127.0.0.1")
        # V06: user input is passed to a shell, so ; and && and & run extra commands
        ping_cmd = "ping -n 1 " if sys.platform.startswith("win") else "ping -c 1 "
        result = subprocess.run(ping_cmd + host, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout + result.stderr, 200, {"Content-Type": "text/plain"}

    @app.get("/download")
    def download():
        name = request.args.get("file", "welcome.txt")
        # V07: the path is built from user input with no check that it stays in uploads/
        return send_file(os.path.join(UPLOAD_DIR, name))

    @app.get("/preview")
    def preview():
        if session.get("uid") is None:
            return jsonify(error="login required"), 401
        url = request.args.get("url", "")
        # V10: the server fetches any URL it is given, including internal addresses (SSRF)
        with urllib.request.urlopen(url, timeout=3) as resp:
            return resp.read(2000).decode("utf-8", "replace"), 200, {"Content-Type": "text/plain"}

    @app.route("/prefs", methods=["GET", "POST"])
    def prefs():
        if request.method == "POST":
            theme = request.get_json(force=True).get("theme", "light")
            resp = jsonify(ok=True)
            resp.set_cookie("prefs", base64.b64encode(pickle.dumps({"theme": theme})).decode())
            return resp
        raw = request.cookies.get("prefs") or base64.b64encode(pickle.dumps({"theme": "light"})).decode()
        data = pickle.loads(base64.b64decode(raw))    # V08: untrusted cookie is unpickled (code execution)
        return jsonify(theme=data.get("theme") if isinstance(data, dict) else None)

    return app


if __name__ == "__main__":
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    app = create_app()
    app.run(host="127.0.0.1", port=5001, debug=True)    # V12: debug mode exposes an interactive console
