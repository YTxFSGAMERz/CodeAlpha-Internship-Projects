"""
Unit and parsing tests for Basic Network Sniffer.
Tests parsing of IP, TCP, UDP, ICMP, DNS, and HTTP without needing raw socket privileges.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from scapy.all import IP, TCP, UDP, ICMP, Ether, Raw, DNS, DNSQR
from sniffer import parse_packet, tcp_flags, http_first_line, describe_tls


def test_tcp_flags_decoding():
    # Test SYN flag
    pkt = TCP(flags="S")
    assert tcp_flags(pkt) == "SYN"

    # Test SYN-ACK flag
    pkt = TCP(flags="SA")
    assert tcp_flags(pkt) == "SYN,ACK"


def test_parse_tcp_packet():
    pkt = Ether() / IP(src="192.168.1.10", dst="192.168.1.1") / TCP(sport=12345, dport=80, flags="S")
    info = parse_packet(pkt)
    assert info.protocol == "TCP"
    assert info.src == "192.168.1.10"
    assert info.dst == "192.168.1.1"
    assert info.src_port == 12345
    assert info.dst_port == 80
    assert "SYN" in info.info


def test_parse_udp_packet():
    pkt = Ether() / IP(src="10.0.0.5", dst="10.0.0.1") / UDP(sport=5353, dport=5353) / Raw(load=b"test-payload")
    info = parse_packet(pkt)
    assert info.protocol == "UDP"
    assert info.src == "10.0.0.5"
    assert info.dst == "10.0.0.1"
    assert info.src_port == 5353
    assert info.dst_port == 5353


def test_parse_icmp_ping():
    pkt = Ether() / IP(src="192.168.1.50", dst="8.8.8.8") / ICMP(type=8, id=1, seq=1)
    info = parse_packet(pkt)
    assert info.protocol == "ICMP"
    assert "Echo request" in info.info


def test_http_detection():
    http_payload = b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n"
    pkt = Ether() / IP(src="192.168.1.10", dst="93.184.216.34") / TCP(sport=54321, dport=80) / Raw(load=http_payload)
    info = parse_packet(pkt)
    assert info.protocol == "HTTP"
    assert "GET /index.html HTTP/1.1" in info.info


def test_describe_tls():
    # Valid TLS Handshake header (ContentType=22, Version=3.3 TLS 1.2, Type=1 ClientHello)
    tls_payload = bytes([22, 3, 3, 0, 5, 1])
    assert describe_tls(tls_payload) == "Handshake: ClientHello"
