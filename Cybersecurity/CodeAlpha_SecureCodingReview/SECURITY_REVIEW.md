# Secure Code Review: NoteBox (Python / Flask)

CodeAlpha Cyber Security Internship, Task 3. Review date: October 2026.

## 1. Summary

NoteBox is a small Flask JSON API (registration, login, notes, search and a few utility endpoints) written for this review so that it contains mistakes commonly seen in real code. The review found **13 weaknesses: 3 critical, 6 high and 4 medium.** Several chain together. A login bypass (V01) or a forged session (V03) gives admin access, and the admin-only ping tool (V06) then runs operating-system commands.

Every finding was reproduced with an automated proof-of-concept test, fixed in `secure_app/`, and tested again: each attack that works on `vulnerable_app/` is blocked on `secure_app/`. Bandit reports 3 high, 4 medium and 4 low issues on the vulnerable app and 0 high, 0 medium and 3 low on the hardened one.

Bandit pointed at 8 of the 13 findings. The other five (V05, V07, V09, V11, V13) are logic and design flaws that only manual review and testing found, which is the main lesson of this review.

## 2. Scope and method

| Item | Detail |
|---|---|
| Language and framework | Python 3.12, Flask 3.1 |
| Target | `vulnerable_app/app.py`, 166 lines, 12 routes |
| Out of scope | Deployment (TLS, reverse proxy, operating system), front end, third-party services |
| Static analysis | Bandit 1.9.4, full output in `reports/bandit_vulnerable.txt` |
| Dependency check | pip-audit on the pinned `requirements.txt`: no known vulnerabilities |
| Manual review | Each route traced from user input to where it is used, checked against the OWASP Top 10 (2021) |
| Dynamic testing | One pytest proof of concept per finding, run against both apps with Flask's test client |

Steps followed:

1. List every entry point (route, cookie, header, query parameter) and who may call it.
2. Run Bandit for known dangerous patterns.
3. Read the code route by route, following each input to the place it is used (SQL, shell, template, file path, URL, deserialiser) and checking who is allowed to do what.
4. Prove each suspected weakness with a test, so findings are facts rather than opinions.
5. Fix, then re-run the tests and Bandit.

Severity is a judgment of impact and ease of exploitation in a real deployment. It is not a CVSS calculation. CWE and OWASP labels are the best fit for the weakness, so a few differ from Bandit's own tags.

## 3. Findings at a glance

| ID | Finding | Severity | CWE | OWASP 2021 | Location (`vulnerable_app/app.py`) |
|---|---|---|---|---|---|
| V01 | SQL injection in login | Critical | CWE-89 | A03 Injection | 72-75 |
| V06 | Command injection in `/ping` | Critical | CWE-78 | A03 Injection | 130-131 |
| V08 | Insecure deserialization of a cookie (pickle) | Critical | CWE-502 | A08 Integrity Failures | 157 |
| V02 | SQL injection in `/search` | High | CWE-89 | A03 Injection | 115-117 |
| V03 | Hard-coded secret key and admin password | High | CWE-798, CWE-321 | A07 Authentication / A02 Cryptographic Failures | 23-24, 33 |
| V05 | Template injection and XSS in `/greet` | High | CWE-1336, CWE-79 | A03 Injection | 122-123 |
| V07 | Path traversal in `/download` | High | CWE-22 | A01 Broken Access Control | 137-138 |
| V09 | Insecure direct object reference on notes | High | CWE-639 | A01 Broken Access Control | 104-106 |
| V10 | Server-side request forgery in `/preview` | High | CWE-918 | A10 SSRF | 145-147 |
| V04 | Unsalted MD5 password hashes | Medium | CWE-328, CWE-916 | A02 Cryptographic Failures | 27-28 |
| V11 | No CSRF protection | Medium | CWE-352 | A01 Broken Access Control | 91 and every POST route |
| V12 | Debug mode on, weak cookie flags | Medium | CWE-489, CWE-1004 | A05 Misconfiguration | 34, 166 |
| V13 | No limit on login attempts | Medium | CWE-307 | A07 Authentication Failures | 77 |

## 4. Findings in detail

Each entry gives the problem, the proof (the test that reproduces it, in `tests/test_findings.py`) and the fix (in `secure_app/app.py`).

### V01: SQL injection in login (Critical)

**Problem.** The username and password hash are formatted into the SQL text, so a quote character changes the query. Bandit: B608, line 73.
**Proof.** Logging in as `admin' --` with any password returns 200 and role `admin`. The `--` turns the password check into a comment.

```python
# before
query = "SELECT id, role FROM users WHERE username = '%s' AND password = '%s'" % (username, md5(password))
# after
row = db().execute("SELECT id, role, password FROM users WHERE username = ?", (username,)).fetchone()
valid = check_password_hash(row[2] if row else DUMMY_HASH, password)
```

**Fix.** Bound parameters keep data separate from SQL. The password is verified in Python, and a dummy hash is checked for unknown users so response times do not reveal which usernames exist.

### V06: Command injection in /ping (Critical)

**Problem.** The `host` parameter is appended to a command run with `shell=True`. Bandit: B602, line 131. The route is admin-only, but V01 and V03 both give admin access.
**Proof.** `host=127.0.0.1;echo INJECTED_V06` made the server run the extra `echo`.

```python
# before
subprocess.run("ping -c 1 " + host, shell=True, capture_output=True, text=True, timeout=10)
# after
if not HOST_RE.fullmatch(host):          # letters, digits, dots, hyphens; no leading hyphen
    abort(400)
subprocess.run(["ping", "-c", "1", "-W", "2", host], capture_output=True, text=True, timeout=5, shell=False)
```

**Fix.** Allow-list the input, pass arguments as a list so no shell is involved, and set a timeout. The no-leading-hyphen rule stops option injection such as `-f`.

### V08: Insecure deserialization (Critical)

**Problem.** The `prefs` cookie is a base64 pickle that the server loads. Unpickling can run code chosen by whoever built the cookie, and no login is needed. Bandit: B301, line 157.
**Proof.** A forged cookie ran a harmless payload that set an environment variable inside the server process.

```python
# before
data = pickle.loads(base64.b64decode(request.cookies["prefs"]))
# after: store a validated value in the signed session
if theme not in ("light", "dark"):
    abort(400)
session["theme"] = theme
```

**Fix.** Never unpickle data an attacker can touch. Use a plain data format (JSON) or the signed session, and validate every value.

### V02: SQL injection in /search (High)

**Problem.** The search term is concatenated into a `LIKE` clause. Bandit: B608, line 116.
**Proof.** The term `x%' UNION SELECT id, username || ':' || password FROM users --` returned `admin:<password hash>` in place of a note title, so any logged-in user can read the users table.

```python
# before
sql = "SELECT id, title FROM notes WHERE owner_id = %d AND title LIKE '%%%s%%'" % (uid, q)
# after
db().execute("SELECT id, title FROM notes WHERE owner_id = ? AND title LIKE ?", (uid, "%" + q + "%"))
```

**Fix.** Bound parameters, as in V01.

### V03: Hard-coded secrets (High)

**Problem.** `SECRET_KEY = "dev-secret-123"` and `ADMIN_PASSWORD = "admin123"` are in the source. Flask signs session cookies with the key, so anyone who can read the code (or the repository history) can forge any session. Bandit: B105, lines 23 and 24.
**Proof.** A cookie signed with the published key made `/whoami` report `role: admin`, with no account and no password.

```python
# before
SECRET_KEY = "dev-secret-123"
# after
secret = os.environ.get("NOTEBOX_SECRET_KEY")
if not secret:
    raise RuntimeError("NOTEBOX_SECRET_KEY is not set")
```

**Fix.** Read secrets from the environment (or a secrets manager) and refuse to start without them. Treat any secret that has been committed as leaked and rotate it.

### V05: Template injection and XSS in /greet (High)

**Problem.** The name is concatenated into the template source passed to `render_template_string`. Jinja2 then evaluates `{{ ... }}` (server-side template injection, which commonly escalates to code execution) and does not escape HTML (reflected XSS). Bandit did not flag it.
**Proof.** `{{7*7}}` came back as `49`, and `<script>alert(1)</script>` was returned unescaped. Only expression evaluation was tested; no code execution was attempted.

```python
# before
render_template_string("<h1>Hello " + name + "!</h1>")
# after
render_template_string("<h1>Hello {{ name }}!</h1>", name=name)
```

**Fix.** Pass user input as a template variable, which is auto-escaped, and never build template source from it.

### V07: Path traversal in /download (High)

**Problem.** `os.path.join(UPLOAD_DIR, name)` accepts `..` segments, so the path can leave the uploads folder. Not flagged by Bandit.
**Proof.** `file=../secret.txt` returned the contents of `secret.txt`.

```python
# before
return send_file(os.path.join(UPLOAD_DIR, name))
# after
return send_from_directory(UPLOAD_DIR, name)
```

**Fix.** `send_from_directory` rejects any path that resolves outside the given directory. For production, also allow-list file names or extensions.

### V09: Insecure direct object reference (High)

**Problem.** `/notes/<id>` checks that the caller is logged in but not that the note belongs to them, and ids are sequential. Not flagged by Bandit.
**Proof.** A second user read the first user's private note by requesting its id.

```python
# before
"SELECT id, owner_id, title, body FROM notes WHERE id = ?"
# after
"SELECT id, owner_id, title, body FROM notes WHERE id = ? AND owner_id = ?"
```

**Fix.** Enforce ownership on the server for every object access, and return 404 (not 403) so the existence of other users' notes is not revealed.

### V10: Server-side request forgery in /preview (High)

**Problem.** The server fetches any URL it is given, including internal services, cloud metadata addresses and `file://` URLs. Bandit: B310, line 146.
**Proof.** The server returned the content of a service listening only on loopback.

```python
# before
urllib.request.urlopen(url, timeout=3)
# after: public http(s) only, ports 80/443, host must resolve to public addresses, no redirects
requests.get(parsed.geturl(), timeout=3, allow_redirects=False, stream=True)
```

**Fix.** Check scheme, port and the resolved IP addresses (`ipaddress.is_global`), turn off redirects, and cap the response size. Remaining gap: a name can resolve differently between the check and the fetch (DNS rebinding). Close it by connecting to the checked IP address or by forcing outbound requests through an egress proxy.

### V04: Unsalted MD5 password hashes (Medium)

**Problem.** MD5 is fast and unsalted, so identical passwords give identical hashes and attackers can test billions of guesses per second. Combined with V02, hashes can be dumped and cracked offline. Bandit: B324, line 28.
**Proof.** The stored value for a test user equals the plain MD5 of the password.

**Fix.** `generate_password_hash` and `check_password_hash` from Werkzeug: salted and deliberately slow.

### V11: No CSRF protection (Medium)

**Problem.** Authentication relies on a cookie that browsers send automatically, so a malicious page can make a logged-in user's browser send requests. Because the routes use `get_json(force=True)`, which ignores the Content-Type, even a plain HTML form can trigger them. Not flagged by Bandit.
**Proof.** A logged-in POST to `/notes` without any token returned 201.

**Fix.** A random per-session token must arrive in the `X-CSRF-Token` header on every POST, PUT, PATCH and DELETE, and is compared in constant time (`hmac.compare_digest`). `SameSite=Lax` cookies add a second layer.

### V12: Debug mode on, weak cookie flags (Medium)

**Problem.** `debug=True` (Bandit: B201, line 166) enables Werkzeug's interactive debugger and detailed stack traces, which become code execution if the port is ever reachable. The session cookie is readable by page scripts (`HttpOnly` turned off) and has no `Secure` or `SameSite` setting.
**Proof.** The login response sets a cookie without `HttpOnly`, and the source starts the app with `debug=True`.

**Fix.** Debug off, `HttpOnly`, `SameSite=Lax` and `Secure` on the session cookie, a generic JSON error handler that never returns stack traces, and security headers (`Content-Security-Policy`, `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`).

### V13: No limit on login attempts (Medium)

**Problem.** Unlimited password guesses against any account. Not flagged by Bandit.
**Proof.** Eight wrong passwords in a row were all answered with 401 and none was throttled.

**Fix.** After 5 failures in 60 seconds for the same (IP address, username), the app answers 429. The counters here live in memory, which is fine for a demo. In production use Flask-Limiter with a shared store, add IP-wide limits and backoff or CAPTCHA (a per-account lockout alone can be abused to lock victims out), and offer MFA.

## 5. What Bandit found and what it missed

| Bandit rule | Line | Finding it points at |
|---|---|---|
| B608 (SQL string building) | 73, 116 | V01, V02 |
| B105 (hard-coded password) | 23, 24 | V03 |
| B324 (weak MD5 hash) | 28 | V04 |
| B602 (`shell=True`) | 131 | V06 |
| B301, B403 (pickle) | 157, 13 | V08 |
| B310 (URL open) | 146 | V10 |
| B201 (Flask debug) | 166 | V12, debug part only |
| B404 (subprocess import) | 15 | Informational |

Missed entirely: V05, V07, V09, V11, V13, and the cookie-flag part of V12. Bandit matches dangerous calls and patterns. It cannot know that a note should belong to one user, that a form needs a token, or that a path leaves a folder. Good practice is to use static analysis, manual review and tests together, and to run the scanner on every commit (for example in CI).

## 6. Results after the fixes

Output of `pytest -v` (26 tests, all passing), also saved in `reports/poc_results.txt`:

```
Finding                                       vulnerable_app   secure_app
V01 SQL injection: login bypass               EXPLOITED        blocked
V02 SQL injection: data theft via search      EXPLOITED        blocked
V03 Hard-coded key: forged admin session      EXPLOITED        blocked
V04 Weak hash: unsalted MD5 passwords         EXPLOITED        blocked
V05 Template injection and XSS in /greet      EXPLOITED        blocked
V06 Command injection in /ping                EXPLOITED        blocked
V07 Path traversal in /download               EXPLOITED        blocked
V08 Insecure deserialization (pickle cookie)  EXPLOITED        blocked
V09 IDOR: reading another user's note         EXPLOITED        blocked
V10 SSRF: server fetches internal URLs        EXPLOITED        blocked
V11 CSRF: state change without a token        EXPLOITED        blocked
V12 Debug mode and weak cookie flags          EXPLOITED        blocked
V13 No lockout: unlimited password guesses    EXPLOITED        blocked
```

Bandit on `secure_app/` (`reports/bandit_secure.txt`): 0 high, 0 medium, 3 low. All three concern the ping feature: B404 (importing `subprocess`), B603 (a subprocess call) and B607 (running `ping` by name instead of full path). They are accepted: the input is allow-listed, no shell is used, and the call has a timeout. Using an absolute path to `ping` would clear B607.

## 7. Secure coding checklist

- **Input:** validate type, length and format on the server with allow-lists. Treat every header, cookie, query parameter and file name as untrusted.
- **Databases:** bound parameters or an ORM, never string building.
- **Output:** let the template engine escape. Never create template source from user input.
- **Commands and files:** avoid shells, pass argument lists, and resolve and check file paths against an allowed directory.
- **Authentication:** hash passwords with a slow salted algorithm, throttle attempts, offer MFA, and give the same response for unknown users and wrong passwords.
- **Authorisation:** check ownership and role on the server for every object and action. Never rely on hidden fields or sequential ids being secret.
- **Sessions:** `HttpOnly`, `Secure`, `SameSite` cookies, CSRF tokens on state changes, and a fresh session on login.
- **Secrets:** environment variables or a secrets manager. Never commit them, and rotate any that leak.
- **Deserialization:** never `pickle`, `marshal` or unsafe `yaml.load` on untrusted data. Use JSON or signed values.
- **Outbound requests:** allow-list destinations and block internal address ranges.
- **Errors and configuration:** no debug mode in production, generic error messages, security headers.
- **Dependencies:** pin versions, run `pip-audit` regularly, update promptly.
- **Process:** peer review with a security checklist, Bandit or Semgrep in CI, and tests for security behaviour.

## 8. Limitations

- NoteBox is a small training app. Real applications have far more surface (file uploads, third-party integrations, background jobs).
- Findings were verified in-process through Flask's test client, not over a live network.
- The dependency audit covers known vulnerabilities in the pinned packages at the time it was run.
- Logging and monitoring (OWASP A09) were not part of the fixes. A production system should log logins, failures and access-control denials and alert on them.
- The in-memory rate limiter and the SSRF resolve-then-fetch check have the caveats described above.

## 9. Reproduce

```bash
pip install -r requirements-dev.txt
pytest -v                                         # 26 tests: 13 attacks x 2 apps
bandit -r vulnerable_app                          # static analysis of the target
bandit -r secure_app                              # and of the fixed version
pip-audit -r requirements.txt                     # dependency check
```
