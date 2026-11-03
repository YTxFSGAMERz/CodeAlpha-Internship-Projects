# CodeAlpha_SecureCodingReview

A secure code review of a small Python / Flask application, built for **Task 3 (Secure Coding Review)** of the CodeAlpha Cyber Security Internship.

The audit target, **NoteBox**, is a note-taking API written for this review. It deliberately contains mistakes that are common in real code, so every finding can be shown safely and reproduced. The repository holds the review, the evidence, and a fixed version of the app.

> **Warning:** `vulnerable_app/` is insecure on purpose. Run it only on your own machine (it binds to 127.0.0.1) and never deploy it.

**Read the review first: [SECURITY_REVIEW.md](SECURITY_REVIEW.md).**

## Result

13 weaknesses found (3 critical, 6 high, 4 medium), each reproduced with a test, fixed, and re-tested.

| | Vulnerable app | Hardened app |
|---|---|---|
| Proof-of-concept attacks that work | 13 of 13 | 0 of 13 |
| Bandit high / medium / low | 3 / 4 / 4 | 0 / 0 / 3 |
| `pip-audit` on dependencies | | no known vulnerabilities |

Bandit flagged 8 of the 13 findings. The other five needed manual review and testing.

## Repository layout

| Path | Contents |
|---|---|
| `SECURITY_REVIEW.md` | The report: method, findings, evidence, fixes, checklist |
| `vulnerable_app/app.py` | The audit target. Every flaw is tagged `V01` to `V13` in a comment |
| `secure_app/app.py` | The fixed version. Fixes are tagged `S01` to `S13` |
| `tests/test_findings.py` | One proof-of-concept attack per finding, run against both apps |
| `reports/` | Bandit output for both apps and the test results |

## Run it

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt

pytest -v                  # 13 attacks x 2 apps; a summary table prints at the end
bandit -r vulnerable_app   # static analysis of the target
bandit -r secure_app       # and of the fixed version
pip-audit -r requirements.txt
```

The tests call the apps in-process through Flask's test client, so no network port is opened. A test passes when the vulnerable app **is** exploitable and the hardened app **is not**.

To try the apps by hand:

```bash
python vulnerable_app/app.py        # http://127.0.0.1:5001, debug mode on (that is finding V12)

export NOTEBOX_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
export NOTEBOX_ADMIN_PASSWORD='choose-a-long-passphrase'
python secure_app/app.py            # http://127.0.0.1:5002
```

Example attack on the vulnerable app (login as admin without the password):

```bash
curl -c jar -X POST http://127.0.0.1:5001/login \
     -H 'Content-Type: application/json' \
     -d '{"username": "admin'"'"' --", "password": "x"}'
```

## Task 3 requirements and where they are covered

| Requirement | Where |
|---|---|
| Select a language and application to audit | Python / Flask, NoteBox (`vulnerable_app/`), see section 2 of the report |
| Review the code for vulnerabilities | Manual review plus Bandit, 13 findings with CWE and OWASP mapping |
| Use static analysers or manual inspection | Both, and a comparison of what each caught (report section 5) |
| Recommendations and best practices | Per-finding fixes and the checklist in report section 7 |
| Document findings and remediation | `SECURITY_REVIEW.md`, `secure_app/`, `reports/` |
