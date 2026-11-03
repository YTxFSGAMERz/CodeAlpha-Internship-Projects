#!/usr/bin/env python3
"""Run Suricata IDS engine with local rules or provide offline demo navigation."""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Run Suricata IDS with local rules.")
    parser.add_argument("iface", nargs="?", default="eth0", help="network interface to monitor")
    parser.add_argument(
        "home_net",
        nargs="?",
        default="192.168.0.0/16,10.0.0.0/8,172.16.0.0/12",
        help="home network CIDRs",
    )
    args = parser.parse_args()

    suricata_bin = shutil.which("suricata")
    if not suricata_bin:
        print("[!] Suricata binary was not found in your system PATH.")
        print("    This project can be fully tested and demonstrated without Suricata:")
        print("      1. Generate attack packets and demo alert logs:")
        print("         python scripts/generate_test_traffic.py")
        print("      2. Run automated incident response (dry run):")
        print("         python scripts/responder.py")
        print("      3. Visualize detection analytics dashboard:")
        print("         python scripts/visualize.py")
        print("      4. Verify rule syntax and attack correlation:")
        print("         python scripts/test_rules.py")
        print()
        print("    To run live packet inspection on Linux/WSL with Suricata installed:")
        print(f"      sudo bash scripts/run_ids.sh {args.iface} {args.home_net}")
        return 0

    conf = os.environ.get("SURICATA_CONF", "/etc/suricata/suricata.yaml")
    logs = os.path.join(ROOT, "logs")
    os.makedirs(logs, exist_ok=True)
    cmd = [
        suricata_bin,
        "-c",
        conf,
        "-i",
        args.iface,
        "-S",
        os.path.join(ROOT, "rules", "local.rules"),
        "-l",
        logs,
        "--set",
        f"vars.address-groups.HOME_NET=[{args.home_net}]",
    ]
    print(f"[*] Starting Suricata on {args.iface} (protecting {args.home_net}). Logs: {logs}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
