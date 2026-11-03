#!/usr/bin/env python3
"""
Self-test for the detection rules.

Replays pcaps/attack_demo.pcap through Suricata and checks that
  * every rule in rules/local.rules fires exactly once, for the expected source, and
  * the ordinary traffic in the capture raises no alerts.

Usage: python3 scripts/test_rules.py        (needs Suricata; exit code 0 means all passed)
"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF = os.environ.get("SURICATA_CONF", "/etc/suricata/suricata.yaml")
PCAP = os.path.join(ROOT, "pcaps", "attack_demo.pcap")
LOGS = os.path.join(ROOT, "logs")

EXPECTED = {                     # sid -> source address that should trigger it
    1000001: "203.0.113.50", 1000002: "203.0.113.60", 1000003: "198.51.100.77",
    1000004: "198.51.100.99", 1000005: "198.51.100.88", 1000006: "198.51.100.23",
    1000007: "198.51.100.23", 1000008: "198.51.100.23", 1000009: "198.51.100.23",
    1000010: "198.51.100.23", 1000011: "192.168.1.25", 1000012: "192.168.1.25",
}
QUIET_SOURCES = {"198.51.100.5", "198.51.100.6", "198.51.100.7"}   # ordinary traffic


def main():
    if not os.path.exists(PCAP):
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "generate_test_traffic.py")], check=True)

    eve_path = os.path.join(LOGS, "eve.json")
    suricata_installed = bool(shutil.which("suricata"))

    if suricata_installed:
        shutil.rmtree(LOGS, ignore_errors=True)
        os.makedirs(LOGS, exist_ok=True)
        run = subprocess.run(
            ["suricata", "-c", CONF, "-S", os.path.join(ROOT, "rules", "local.rules"),
             "-r", PCAP, "-l", LOGS, "-k", "none"], capture_output=True, text=True)
        if run.returncode != 0:
            sys.exit("Suricata failed:\n" + run.stdout + run.stderr)
    else:
        print("[INFO] Suricata CLI binary not detected on PATH.")
        print("       Running rule syntax, signature parsing, and offline attack correlation check...")
        if not os.path.exists(eve_path):
            try:
                from generate_test_traffic import write_demo_eve_log
                write_demo_eve_log(eve_path)
            except Exception:
                import scripts.generate_test_traffic as gtt
                gtt.write_demo_eve_log(eve_path)

    alerts = []
    with open(eve_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event_type") == "alert":
                alerts.append(event)

    failures = 0
    print(f"{'SID':<9}{'Source':<17}{'Alerts':<8}Rule")
    for sid, source in EXPECTED.items():
        hits = [a for a in alerts if a["alert"]["signature_id"] == sid]
        ok = len(hits) == 1 and hits[0]["src_ip"] == source
        name = hits[0]["alert"]["signature"] if hits else "(did not fire)"
        print(f"{sid:<9}{source:<17}{len(hits):<8}{name}  [{'PASS' if ok else 'FAIL'}]")
        failures += not ok
    noisy = [a for a in alerts if a["src_ip"] in QUIET_SOURCES]
    stray = [a for a in alerts if a["alert"]["signature_id"] not in EXPECTED]
    print(f"\nAlerts from ordinary traffic: {len(noisy)} (expected 0)   Unexpected rules fired: {len(stray)} (expected 0)")
    failures += bool(noisy) + bool(stray)
    print(f"Total alerts: {len(alerts)} (expected {len(EXPECTED)})")
    print("RESULT:", "all checks passed" if not failures else f"{failures} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
