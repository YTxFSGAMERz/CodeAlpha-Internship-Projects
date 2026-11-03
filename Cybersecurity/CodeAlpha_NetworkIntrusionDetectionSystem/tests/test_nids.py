"""
Automated unit tests for NIDS rules and traffic generator.
"""
import os
import re
from pathlib import Path
import pytest
from scapy.all import rdpcap

ROOT = Path(__file__).resolve().parents[1]
RULES_FILE = ROOT / "rules" / "local.rules"
PCAP_FILE = ROOT / "pcaps" / "attack_demo.pcap"


def test_rules_file_exists_and_valid():
    assert RULES_FILE.exists(), "local.rules must exist"
    lines = [
        line.strip() for line in RULES_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(lines) >= 10, f"Expected at least 10 active rules, found {len(lines)}"

    sids = []
    sid_re = re.compile(r"sid:\s*(\d+);")
    for line in lines:
        assert line.startswith("alert "), f"Rule must start with alert: {line}"
        match = sid_re.search(line)
        assert match is not None, f"Rule missing sid: {line}"
        sids.append(int(match.group(1)))

    # Verify uniqueness of SIDs
    assert len(sids) == len(set(sids)), "All rule SIDs must be unique"
    # Verify SIDs are in local range
    for sid in sids:
        assert 1000001 <= sid <= 1099999, f"SID {sid} outside local rule range"


def test_pcap_structure():
    assert PCAP_FILE.exists(), "attack_demo.pcap should be generated"
    packets = rdpcap(str(PCAP_FILE))
    assert len(packets) > 100, f"Expected >100 packets in demo capture, got {len(packets)}"

    # Check for presence of TCP and ICMP
    has_tcp = any(pkt.haslayer("TCP") for pkt in packets)
    has_icmp = any(pkt.haslayer("ICMP") for pkt in packets)
    assert has_tcp, "Capture must contain TCP packets"
    assert has_icmp, "Capture must contain ICMP packets"
