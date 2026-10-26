#!/usr/bin/env python3
"""
Basic Network Sniffer - CodeAlpha Cyber Security Internship, Task 1

Captures packets with Scapy (live from a network interface, or from a saved
.pcap file), decodes them layer by layer and prints what is inside:

    Ethernet frame -> IP packet -> TCP segment / UDP datagram -> payload

For every packet you get the time, protocol, source and destination
(IP address and port), size and a short description. Optional flags add a
hex/ASCII payload preview (-p) and a layer-by-layer breakdown (-v). When the
capture ends, a summary shows which protocols and hosts were talking.

Only capture traffic on networks you own or have permission to monitor.
"""

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, List, Optional

from scapy.all import (
    ARP, DNS, DNSQR, DNSRR, ICMP, IP, TCP, UDP, Ether, IPv6, PcapWriter, Raw, conf, sniff,
)
from scapy.error import Scapy_Exception
from scapy.packet import NoPayload, Packet

# --------------------------------------------------------------------------
# Reference tables (port numbers, flag names, ... used to label packets)
# --------------------------------------------------------------------------
WELL_KNOWN_PORTS = {
    20: "FTP-DATA", 21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS",
    67: "DHCP", 68: "DHCP", 80: "HTTP", 110: "POP3", 123: "NTP", 143: "IMAP",
    161: "SNMP", 443: "HTTPS", 445: "SMB", 465: "SMTPS", 587: "SMTP",
    993: "IMAPS", 995: "POP3S", 1900: "SSDP", 3306: "MySQL", 3389: "RDP",
    5353: "mDNS", 5432: "PostgreSQL", 8080: "HTTP-alt", 8443: "HTTPS-alt",
}
TCP_FLAG_NAMES = {
    "F": "FIN", "S": "SYN", "R": "RST", "P": "PSH",
    "A": "ACK", "U": "URG", "E": "ECE", "C": "CWR", "N": "NS",
}
ICMP_TYPES = {
    0: "Echo reply (ping)", 3: "Destination unreachable", 5: "Redirect",
    8: "Echo request (ping)", 11: "Time exceeded",
}
DNS_TYPES = {
    1: "A", 2: "NS", 5: "CNAME", 6: "SOA", 12: "PTR", 15: "MX",
    16: "TXT", 28: "AAAA", 33: "SRV", 65: "HTTPS",
}
DNS_RCODES = {0: "NOERROR", 1: "FORMERR", 2: "SERVFAIL", 3: "NXDOMAIN", 5: "REFUSED"}
TLS_RECORDS = {20: "Change Cipher Spec", 21: "Alert", 22: "Handshake", 23: "Application Data"}
TLS_HANDSHAKES = {
    1: "ClientHello", 2: "ServerHello", 11: "Certificate",
    12: "ServerKeyExchange", 14: "ServerHelloDone", 16: "ClientKeyExchange", 20: "Finished",
}
HTTP_STARTS = (b"GET ", b"POST ", b"HEAD ", b"PUT ", b"DELETE ", b"OPTIONS ", b"PATCH ", b"HTTP/1.")

# Layers whose header fields are printed one by one in verbose mode (-v)
DETAILED_LAYERS = (Ether, IP, IPv6, TCP, UDP, ICMP, ARP)

COLORS = {
    "TCP": "\033[36m", "UDP": "\033[35m", "DNS": "\033[33m", "HTTP": "\033[32m",
    "TLS": "\033[34m", "ICMP": "\033[31m", "ICMPv6": "\033[31m", "ARP": "\033[93m",
}
RESET = "\033[0m"


# --------------------------------------------------------------------------
# Decoding: turn a raw Scapy packet into something easy to print and count
# --------------------------------------------------------------------------
@dataclass
class PacketInfo:
    """The few facts about a packet that we print and count."""
    timestamp: float
    length: int
    protocol: str = "Other"
    src: str = "?"
    dst: str = "?"
    src_port: Optional[int] = None
    dst_port: Optional[int] = None
    info: str = ""
    payload: bytes = b""


def iter_layers(pkt: Packet) -> Iterator[Packet]:
    """Yield every protocol layer in a packet, outermost first."""
    layer = pkt
    while not isinstance(layer, NoPayload):
        yield layer
        layer = layer.payload


def tcp_flags(tcp: Packet) -> str:
    """Spell out the TCP flags, e.g. 'SA' -> 'SYN,ACK'."""
    names = [TCP_FLAG_NAMES.get(ch, ch) for ch in str(tcp.flags)]
    return ",".join(names) or "NONE"


def describe_dns(dns: Packet) -> str:
    """One-line description of a DNS query or response."""
    name, qtype = "", ""
    try:
        if dns.qd:
            question = dns.qd[0] if isinstance(dns.qd, list) else dns.qd
            qname = getattr(question, "qname", b"")
            if isinstance(qname, bytes):
                qname = qname.decode(errors="replace")
            name = qname.rstrip(".")
            qtype = DNS_TYPES.get(getattr(question, "qtype", 0), str(getattr(question, "qtype", "")))
    except (AttributeError, IndexError, TypeError):
        pass

    if dns.qr == 0:
        return f"Query {qtype} {name}".strip() or "DNS query"
    rcode = DNS_RCODES.get(dns.rcode, str(dns.rcode))
    ancount = dns.ancount if dns.ancount is not None else (len(dns.an) if dns.an else 0)
    target = f"{qtype} {name}".strip()
    return f"Response {target} \u2192 {rcode}, {ancount} answer(s)" if target else f"Response \u2192 {rcode}, {ancount} answer(s)"


def http_first_line(payload: bytes) -> Optional[str]:
    """Return the request/status line if the payload looks like plain HTTP."""
    if payload.startswith(HTTP_STARTS):
        return payload.split(b"\r\n", 1)[0][:100].decode("ascii", errors="replace")
    return None


def describe_tls(payload: bytes) -> Optional[str]:
    """Recognise a TLS record header: content type, version 3.x, length."""
    if len(payload) < 5 or payload[0] not in TLS_RECORDS or payload[1] != 3 or payload[2] > 4:
        return None
    text = TLS_RECORDS[payload[0]]
    if payload[0] == 22 and len(payload) > 5:       # byte 5 is the handshake message type
        text += f": {TLS_HANDSHAKES.get(payload[5], 'message')}"
    elif payload[0] == 23:
        text += " (encrypted)"
    return text


def parse_packet(pkt: Packet) -> PacketInfo:
    """Decode one Scapy packet into a PacketInfo."""
    p = PacketInfo(timestamp=float(pkt.time), length=len(pkt))

    # ---- Who is talking to whom (ARP / IPv4 / IPv6) ----------------------
    if pkt.haslayer(ARP):
        arp = pkt[ARP]
        p.protocol, p.src, p.dst = "ARP", arp.psrc, arp.pdst
        if arp.op == 1:
            p.info = f"Who has {arp.pdst}? Tell {arp.psrc}"
        elif arp.op == 2:
            p.info = f"{arp.psrc} is at {arp.hwsrc}"
        else:
            p.info = f"ARP opcode {arp.op}"
        return p

    if pkt.haslayer(IP):
        p.protocol, p.src, p.dst = "IPv4", pkt[IP].src, pkt[IP].dst
    elif pkt.haslayer(IPv6):
        p.protocol, p.src, p.dst = "IPv6", pkt[IPv6].src, pkt[IPv6].dst
    else:
        # Not IP or ARP (LLDP, STP, ...): fall back to the MAC addresses.
        if pkt.haslayer(Ether):
            eth = pkt[Ether]
            p.src, p.dst = eth.src, eth.dst
            p.protocol = eth.sprintf("%Ether.type%")      # e.g. 'LLDP', or '0x1234' if unknown
        p.info = pkt.summary()
        return p

    # Application data carried by TCP/UDP/ICMP. (Ethernet padding is a separate
    # layer in Scapy, so short frames do not show fake payload bytes.)
    payload = bytes(pkt[Raw].load) if pkt.haslayer(Raw) else b""

    # ---- Transport layer: protocol and ports ------------------------------
    if pkt.haslayer(ICMP):
        icmp = pkt[ICMP]
        p.protocol = "ICMP"
        p.info = ICMP_TYPES.get(icmp.type, f"type {icmp.type}")
        if icmp.type in (0, 8):
            p.info += f" id={icmp.id} seq={icmp.seq}"
    elif pkt.haslayer(TCP):
        tcp = pkt[TCP]
        p.protocol = "TCP"
        p.src_port, p.dst_port = tcp.sport, tcp.dport
        p.info = (f"[{tcp_flags(tcp)}] seq={tcp.seq} ack={tcp.ack} "
                  f"win={tcp.window} len={len(payload)}")
    elif pkt.haslayer(UDP):
        p.protocol = "UDP"
        p.src_port, p.dst_port = pkt[UDP].sport, pkt[UDP].dport
        p.info = f"len={len(payload)}"
    else:
        icmp6 = next((l for l in iter_layers(pkt) if l.name.startswith("ICMPv6")), None)
        if icmp6 is not None:
            p.protocol = "ICMPv6"
            p.info = icmp6.name.replace("ICMPv6 ", "", 1)

    # ---- Application layer: recognise a few well-known protocols ----------
    if p.protocol in ("TCP", "UDP"):
        service = WELL_KNOWN_PORTS.get(p.dst_port) or WELL_KNOWN_PORTS.get(p.src_port) or ""
        http_line = http_first_line(payload)
        tls = describe_tls(payload) if p.protocol == "TCP" else None
        if pkt.haslayer(DNS):
            p.protocol, p.info = "DNS", describe_dns(pkt[DNS])
        elif http_line:
            p.protocol, p.info = "HTTP", http_line
        elif tls:
            p.protocol, p.info = "TLS", tls
        elif service:
            p.info = f"{service}  {p.info}"

    p.payload = payload
    return p


# --------------------------------------------------------------------------
# Output formatting
# --------------------------------------------------------------------------
def endpoint(addr: str, port: Optional[int]) -> str:
    """'1.2.3.4:80', or '[::1]:80' for IPv6, or just the address if no port."""
    if port is None:
        return addr
    return f"[{addr}]:{port}" if ":" in addr else f"{addr}:{port}"


def header_line() -> str:
    return (f"{'#':>5}  {'TIME':<12}  {'PROTO':<7} {'SOURCE':<24}   "
            f"{'DESTINATION':<24} {'LEN':>5}  INFO")


def format_line(number: int, p: PacketInfo, color: bool) -> str:
    when = datetime.fromtimestamp(p.timestamp).strftime("%H:%M:%S.%f")[:-3]
    proto = f"{p.protocol:<7}"
    if color:
        proto = f"{COLORS.get(p.protocol, '')}{proto}{RESET}"
    src = endpoint(p.src, p.src_port)
    dst = endpoint(p.dst, p.dst_port)
    return f"{number:>5}  {when}  {proto} {src:<24} \u2192 {dst:<24} {p.length:>5}  {p.info}"


def layer_breakdown(pkt: Packet) -> List[str]:
    """One line per protocol layer, with the header fields that were on the wire."""
    lines = []
    for layer in iter_layers(pkt):
        if isinstance(layer, Raw):                  # Raw and Padding
            lines.append(f"      {layer.name:<10}{len(layer.load)} bytes")
        elif isinstance(layer, DETAILED_LAYERS):
            parts = []
            for name, value in layer.fields.items():
                text = layer.get_field(name).i2repr(layer, value)
                if text not in ("", "[]"):
                    parts.append(f"{name}={text if len(text) <= 60 else text[:57] + '...'}")
            lines.append(f"      {layer.name:<10}" + "  ".join(parts))
        else:
            lines.append(f"      {layer.mysummary() or layer.name}")
    return lines


def hexdump(data: bytes, limit: int) -> List[str]:
    """Classic hex + ASCII view of the first `limit` bytes."""
    shown = data[:limit]
    lines = []
    for offset in range(0, len(shown), 16):
        chunk = shown[offset:offset + 16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        text = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"      {offset:04x}  {hex_part:<47}  |{text}|")
    if len(data) > limit:
        lines.append(f"      ... {len(data) - limit} more bytes not shown")
    return lines


# --------------------------------------------------------------------------
# Statistics shown when the capture ends
# --------------------------------------------------------------------------
class Stats:
    """Counts packets per protocol and per pair of hosts."""

    def __init__(self) -> None:
        self.packets = 0
        self.bytes = 0
        self.first: Optional[float] = None
        self.last: Optional[float] = None
        self.protocols: Counter = Counter()
        self.talkers: Counter = Counter()           # packets per host pair
        self.talker_bytes: Counter = Counter()

    def add(self, p: PacketInfo) -> None:
        self.packets += 1
        self.bytes += p.length
        if self.first is None:
            self.first = p.timestamp
        self.last = p.timestamp
        self.protocols[p.protocol] += 1
        pair = tuple(sorted((p.src, p.dst)))
        self.talkers[pair] += 1
        self.talker_bytes[pair] += p.length

    def report(self) -> None:
        print("\n" + "\u2500" * 78)
        if not self.packets:
            print("Capture summary: no packets captured.")
            return
        print("Capture summary")
        print(f"  Packets : {self.packets:,}")
        print(f"  Data    : {self.bytes:,} bytes")
        print(f"  Duration: {self.last - self.first:.2f} s")
        print("\n  Protocols")
        for proto, count in self.protocols.most_common():
            share = count / self.packets
            bar = "\u2588" * max(1, round(share * 30))
            print(f"    {proto:<8}{count:>7}  {share:6.1%}  {bar}")
        print("\n  Top conversations")
        for pair, count in self.talkers.most_common(5):
            label = f"{pair[0]} \u2194 {pair[1]}"
            print(f"    {label:<46}{count:>6} pkts{self.talker_bytes[pair]:>10,} bytes")


# --------------------------------------------------------------------------
# Demonstration packets (for testing without root or live traffic)
# --------------------------------------------------------------------------
def generate_demo_packets() -> List[Packet]:
    """Generate a realistic sample flow of packets for testing and demonstration."""
    import time
    t = time.time() - 3.0
    packets = []

    # 1. ARP Request & Reply
    p1 = Ether(src="aa:bb:cc:00:00:05", dst="ff:ff:ff:ff:ff:ff") / ARP(
        op=1, hwsrc="aa:bb:cc:00:00:05", psrc="192.168.1.5",
        hwdst="00:00:00:00:00:00", pdst="192.168.1.1"
    )
    p1.time = t
    packets.append(p1)

    p2 = Ether(src="aa:bb:cc:00:00:01", dst="aa:bb:cc:00:00:05") / ARP(
        op=2, hwsrc="aa:bb:cc:00:00:01", psrc="192.168.1.1",
        hwdst="aa:bb:cc:00:00:05", pdst="192.168.1.5"
    )
    p2.time = t + 0.01
    packets.append(p2)

    # 2. DNS Query & Response
    p3 = Ether() / IP(src="192.168.1.5", dst="8.8.8.8") / UDP(sport=53124, dport=53) / DNS(
        rd=1, qd=DNSQR(qname="example.com")
    )
    p3.time = t + 0.05
    packets.append(p3)

    p4 = Ether() / IP(src="8.8.8.8", dst="192.168.1.5") / UDP(sport=53, dport=53124) / DNS(
        qr=1, ancount=1, qd=DNSQR(qname="example.com"),
        an=DNSRR(rrname="example.com", rdata="93.184.216.34")
    )
    p4.time = t + 0.08
    packets.append(p4)

    # 3. TCP Handshake (SYN, SYN-ACK, ACK)
    p5 = Ether() / IP(src="192.168.1.5", dst="93.184.216.34") / TCP(
        sport=51234, dport=80, flags="S", seq=1000, ack=0, window=64240
    )
    p5.time = t + 0.15
    packets.append(p5)

    p6 = Ether() / IP(src="93.184.216.34", dst="192.168.1.5") / TCP(
        sport=80, dport=51234, flags="SA", seq=5000, ack=1001, window=65535
    )
    p6.time = t + 0.18
    packets.append(p6)

    p7 = Ether() / IP(src="192.168.1.5", dst="93.184.216.34") / TCP(
        sport=51234, dport=80, flags="A", seq=1001, ack=5001, window=64240
    )
    p7.time = t + 0.19
    packets.append(p7)

    # 4. HTTP Request & Response
    http_req = b"GET /index.html HTTP/1.1\r\nHost: example.com\r\nUser-Agent: Mozilla/5.0\r\nAccept: */*\r\n\r\n"
    p8 = Ether() / IP(src="192.168.1.5", dst="93.184.216.34") / TCP(
        sport=51234, dport=80, flags="PA", seq=1001, ack=5001, window=64240
    ) / Raw(http_req)
    p8.time = t + 0.20
    packets.append(p8)

    http_resp = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 48\r\n\r\n<!doctype html><html><body>Hello!</body></html>"
    p9 = Ether() / IP(src="93.184.216.34", dst="192.168.1.5") / TCP(
        sport=80, dport=51234, flags="PA", seq=5001, ack=1001 + len(http_req), window=65535
    ) / Raw(http_resp)
    p9.time = t + 0.25
    packets.append(p9)

    # 5. TLS Handshake ClientHello
    tls_payload = bytes([0x16, 0x03, 0x03, 0x00, 0x30, 0x01]) + b"\x00" * 43
    p10 = Ether() / IP(src="192.168.1.5", dst="93.184.216.34") / TCP(
        sport=51300, dport=443, flags="PA", seq=100, ack=200, window=64240
    ) / Raw(tls_payload)
    p10.time = t + 0.35
    packets.append(p10)

    # 6. ICMP Ping Request & Echo Reply
    p11 = Ether() / IP(src="192.168.1.5", dst="192.168.1.1") / ICMP(type=8, id=1, seq=1)
    p11.time = t + 0.45
    packets.append(p11)

    p12 = Ether() / IP(src="192.168.1.1", dst="192.168.1.5") / ICMP(type=0, id=1, seq=1)
    p12.time = t + 0.46
    packets.append(p12)

    return packets


# --------------------------------------------------------------------------
# Capture loop and command line
# --------------------------------------------------------------------------
def run_capture(args: argparse.Namespace, color: bool) -> Stats:
    """Sniff (or read a pcap), print each packet and return the statistics."""
    stats = Stats()
    writer = None
    number = 0

    def handle(pkt: Packet) -> None:
        nonlocal number
        number += 1
        if number == 1:
            print("\n" + header_line(), flush=True)
        if writer is not None:
            writer.write(pkt)
        try:
            info = parse_packet(pkt)
        except Exception as exc:                    # one odd packet must not stop the sniffer
            print(f"{number:>5}  [!] could not decode packet: {exc}", flush=True)
            return
        stats.add(info)
        print(format_line(number, info, color), flush=True)
        if args.verbose:
            print("\n".join(layer_breakdown(pkt)), flush=True)
        if args.payload and info.payload:
            print("\n".join(hexdump(info.payload, args.max_bytes)), flush=True)

    options = {"prn": handle, "store": False, "count": args.count, "filter": args.filter}
    if args.read:
        options["offline"] = args.read
    else:
        options["iface"] = args.interface

    try:
        if args.write:
            writer = PcapWriter(args.write, sync=True)
        if getattr(args, "demo", False):
            for pkt in generate_demo_packets():
                handle(pkt)
                if args.count and number >= args.count:
                    break
        else:
            sniff(**options)
    except KeyboardInterrupt:
        pass                                        # Ctrl+C is the normal way to stop
    except PermissionError:
        sys.exit("[!] Permission denied. Capturing packets needs elevated rights: run with "
                 "sudo (Linux/macOS) or from an Administrator terminal (Windows).")
    except (OSError, ValueError, Scapy_Exception) as exc:
        hint = ""
        if "filter" in str(exc).lower():
            hint = ("\n    -f needs libpcap: install tcpdump (Linux/macOS) or Npcap (Windows, "
                    "https://npcap.com), or run without -f.")
        elif not args.read:
            hint = "\n    Use --list-interfaces to see valid interface names."
            if os.name == "nt":
                hint += "\n    On Windows, Npcap must be installed (https://npcap.com)."
        sys.exit(f"[!] Capture failed: {exc}{hint}")
    finally:
        if writer is not None:
            writer.close()
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Basic network sniffer (CodeAlpha Cyber Security internship, Task 1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  sudo python3 sniffer.py                       capture until Ctrl+C\n"
            "  sudo python3 sniffer.py -i eth0 -c 50 -p      50 packets with payload preview\n"
            "  sudo python3 sniffer.py -f \"tcp port 80\" -v   HTTP only, full layer breakdown\n"
            "  sudo python3 sniffer.py -w capture.pcap       also save a pcap (open in Wireshark)\n"
            "  python3 sniffer.py -r capture.pcap            analyse a saved capture\n"
            "  python3 sniffer.py --demo -p                  test run with sample packets\n"
        ),
    )
    parser.add_argument("-i", "--interface", help="interface to sniff on (default: system default)")
    parser.add_argument("-c", "--count", type=int, default=0,
                        help="stop after N packets (default: until Ctrl+C)")
    parser.add_argument("-f", "--filter", help='BPF filter, e.g. "tcp port 80" or "udp port 53"')
    parser.add_argument("-p", "--payload", action="store_true",
                        help="show a hex/ASCII preview of each packet's payload")
    parser.add_argument("--max-bytes", type=int, default=64,
                        help="payload bytes to preview with -p (default: 64)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print a layer-by-layer breakdown of every packet")
    parser.add_argument("-w", "--write", metavar="FILE",
                        help="save captured packets to a .pcap file")
    parser.add_argument("-r", "--read", metavar="FILE",
                        help="analyse a saved .pcap file instead of sniffing live")
    parser.add_argument("--demo", action="store_true",
                        help="run a demonstration with sample packets (no root or network needed)")
    parser.add_argument("--list-interfaces", action="store_true",
                        help="show the available network interfaces and exit")
    parser.add_argument("--no-color", action="store_true", help="disable coloured output")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.count < 0 or args.max_bytes < 1:
        parser.error("--count must be >= 0 and --max-bytes must be >= 1")
    if args.read and args.interface:
        parser.error("--read and --interface cannot be used together")
    if args.demo and (args.read or args.interface):
        parser.error("--demo cannot be combined with --read or --interface")
    if args.read and not os.path.isfile(args.read):
        parser.error(f"cannot find the capture file: {args.read}")

    if hasattr(sys.stdout, "reconfigure"):          # keep arrows/bars printable when redirected
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if os.name == "nt":
        os.system("")                               # enables ANSI colours in the Windows console

    if args.list_interfaces:
        conf.ifaces.show()
        return

    color = sys.stdout.isatty() and not args.no_color and "NO_COLOR" not in os.environ

    if args.demo:
        print("[*] Running demonstration mode with synthetic traffic...")
    elif args.read:
        print(f"[*] Reading packets from {args.read}")
    else:
        extra = f" | filter: {args.filter}" if args.filter else ""
        print(f"[*] Sniffing on {args.interface or conf.iface}{extra}  (Ctrl+C to stop)")
        print("[*] Capture only on networks you own or are authorised to monitor.")

    stats = run_capture(args, color)
    stats.report()
    if args.write:
        print(f"\n[*] Saved {stats.packets} packets to {args.write}")


if __name__ == "__main__":
    main()
