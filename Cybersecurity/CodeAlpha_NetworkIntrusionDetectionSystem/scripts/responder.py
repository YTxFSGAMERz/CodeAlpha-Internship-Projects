#!/usr/bin/env python3
"""
Automatic response for Suricata alerts.

Reads Suricata's eve.json and counts alerts per source address. When one source
raises too many alerts in a short time, the responder reacts:

  * external source -> block it with a firewall rule (iptables / ip6tables)
  * internal source -> do NOT block it: that would not stop its outbound traffic and
                       could lock out an admin. Flag it so the host can be investigated.

Safe by default: without --enforce it only prints what it would do.
Examples:
  python3 scripts/responder.py logs/eve.json                  replay a finished log (dry run)
  sudo python3 scripts/responder.py --follow --enforce        live: block offenders, undo on exit
"""
import argparse
import atexit
import ipaddress
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMENT = "nids-responder"


def ensure_eve_log(path):
    """Ensure eve.json exists; auto-generate demo events if not present."""
    if not os.path.exists(path):
        print(f"  [!] {path} not found. Generating demo alert log from attack capture...")
        try:
            from generate_test_traffic import write_demo_eve_log
            write_demo_eve_log(path)
        except Exception:
            try:
                import scripts.generate_test_traffic as gtt
                gtt.write_demo_eve_log(path)
            except Exception:
                pass


def read_events(path, follow, from_start):
    """Yield one parsed JSON event per line; with follow=True keep waiting for new lines."""
    ensure_eve_log(path)
    buffer = ""
    with open(path, encoding="utf-8") as fh:
        if follow and not from_start:
            fh.seek(0, os.SEEK_END)
        while True:
            chunk = fh.readline()
            if not chunk:
                if not follow:
                    return
                time.sleep(0.5)
                continue
            buffer += chunk
            if not buffer.endswith("\n"):
                continue                         # the writer is mid-line; wait for the rest
            line, buffer = buffer, ""
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def run(cmd):
    """Run a firewall command (argument list, no shell). Returns True on success."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print(f"  ! {cmd[0]} not found", file=sys.stderr)
        return False
    if result.returncode != 0:
        print(f"  ! {' '.join(cmd)} failed: {result.stderr.strip()}", file=sys.stderr)
    return result.returncode == 0


class Responder:
    def __init__(self, args):
        nets = lambda text: [ipaddress.ip_network(n.strip(), strict=False) for n in text.split(",") if n.strip()]
        self.home = nets(args.home_net)
        self.allow = nets("127.0.0.0/8,::1/128") + nets(args.allow)
        self.threshold, self.window, self.max_severity = args.threshold, args.window, args.max_severity
        self.enforce, self.block_seconds, self.log_path = args.enforce, args.block_seconds, args.log
        self.hits = defaultdict(deque)           # source ip -> (time, signature) of recent alerts
        self.handled = set()
        self.blocked = {}                        # ip -> the firewall command that blocked it
        self.counts = Counter()

    def say(self, message):
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}"
        print(line, flush=True)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def handle(self, event):
        if event.get("event_type") != "alert":
            return
        alert = event["alert"]
        try:
            ip = ipaddress.ip_address(event["src_ip"])
        except (KeyError, ValueError):
            return
        self.counts["alerts seen"] += 1
        if alert.get("severity", 3) > self.max_severity or any(ip in net for net in self.allow):
            return
        when = datetime.strptime(event["timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()
        recent = self.hits[str(ip)]
        recent.append((when, alert["signature"]))
        while recent and when - recent[0][0] > self.window:
            recent.popleft()
        if str(ip) in self.handled or len(recent) < self.threshold:
            return
        self.handled.add(str(ip))
        names = "; ".join(sorted({sig for _, sig in recent}))
        if any(ip in net for net in self.home):
            self.counts["internal hosts flagged"] += 1
            self.say(f"FLAG  {ip} (internal): {len(recent)} alert(s): {names}. Not blocked; investigate or isolate this host.")
        else:
            self.counts["sources blocked"] += 1
            self.say(f"BLOCK {ip}: {len(recent)} alert(s) in {self.window}s: {names}")
            self.block(ip)

    def block(self, ip):
        tool = "ip6tables" if ip.version == 6 else "iptables"
        cmd = [tool, "-I", "INPUT", "-s", str(ip), "-m", "comment", "--comment", COMMENT, "-j", "DROP"]
        if not self.enforce:
            self.say("      dry run, would run: " + " ".join(cmd))
            return
        if run(cmd):
            self.blocked[ip] = cmd
            timer = threading.Timer(self.block_seconds, self.unblock, [ip])
            timer.daemon = True
            timer.start()

    def unblock(self, ip):
        cmd = self.blocked.pop(ip, None)
        if cmd and run([cmd[0], "-D", *cmd[2:]]):
            self.say(f"UNBLOCK {ip}")

    def unblock_all(self):
        for ip in list(self.blocked):
            self.unblock(ip)


def main():
    parser = argparse.ArgumentParser(description="Respond to Suricata alerts: block external offenders, flag internal ones.")
    parser.add_argument("eve", nargs="?", default=os.path.join(ROOT, "logs", "eve.json"), help="path to eve.json")
    parser.add_argument("--follow", action="store_true", help="keep watching the file for new alerts (like tail -f)")
    parser.add_argument("--from-start", action="store_true", help="with --follow, also process alerts already in the file")
    parser.add_argument("--threshold", type=int, default=1, help="alerts from one source that trigger a response (default 1)")
    parser.add_argument("--window", type=int, default=300, help="seconds over which alerts are counted (default 300)")
    parser.add_argument("--max-severity", type=int, default=2, choices=[1, 2, 3],
                        help="ignore alerts less severe than this; 1 is the most severe (default 2)")
    parser.add_argument("--home-net", default="192.168.0.0/16,10.0.0.0/8,172.16.0.0/12",
                        help="your own address ranges; sources inside them are flagged, never blocked")
    parser.add_argument("--allow", default="", help="comma-separated extra addresses/ranges that are never touched")
    parser.add_argument("--enforce", action="store_true", help="really run the firewall commands (needs root)")
    parser.add_argument("--block-seconds", type=int, default=3600, help="how long a block lasts (default 3600)")
    parser.add_argument("--keep-blocks", action="store_true", help="do not remove blocks when the responder exits")
    parser.add_argument("--log", default=os.path.join(ROOT, "logs", "response.log"), help="response log file")
    args = parser.parse_args()

    if args.enforce and hasattr(os, "geteuid") and os.geteuid() != 0:
        sys.exit("--enforce needs root (it changes firewall rules). Use sudo, or drop --enforce for a dry run.")
    responder = Responder(args)
    if args.enforce and not args.keep_blocks:
        atexit.register(responder.unblock_all)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    mode = "ENFORCING" if args.enforce else "dry run"
    responder.say(f"responder started ({mode}), reading {args.eve}")
    try:
        for event in read_events(args.eve, args.follow, args.from_start):
            responder.handle(event)
    except KeyboardInterrupt:
        pass
    summary = ", ".join(f"{n} {k}" for k, n in responder.counts.items()) or "nothing to report"
    responder.say(f"finished: {summary}")


if __name__ == "__main__":
    main()
