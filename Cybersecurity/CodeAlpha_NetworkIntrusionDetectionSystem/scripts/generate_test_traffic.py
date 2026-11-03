#!/usr/bin/env python3
"""
Write a demo capture (pcaps/attack_demo.pcap) with one example of every behaviour
the rules in rules/local.rules are meant to detect, plus ordinary traffic that
must NOT raise alerts.

Nothing is sent on the network. The packets are only written to a file, which
Suricata can replay safely with "suricata -r".
"""
import argparse
import os

from scapy.all import DNS, DNSQR, ICMP, IP, TCP, UDP, Ether, Raw, wrpcap

BASE = 1_761_566_400.0                       # fixed start time, so every run is identical
VICTIM, CLIENT, RESOLVER = "192.168.1.10", "192.168.1.25", "8.8.8.8"
packets = []


def mac(ip):
    return "02:00:00:00:00:%02x" % (int(ip.split(".")[-1]) % 256)


def frame(src, dst, t, layer4):
    """Add one Ethernet/IP packet to the capture at time BASE + t."""
    pkt = Ether(src=mac(src), dst=mac(dst)) / IP(src=src, dst=dst) / layer4
    pkt.time = BASE + t
    packets.append(pkt)


def syn(t, src, dst, sport, dport):
    frame(src, dst, t, TCP(sport=sport, dport=dport, flags="S", seq=1000))


def tcp_session(t, src, dst, sport, dport, convo=()):
    """A complete TCP connection: handshake, the (side, payload) exchanges in convo, then teardown."""
    seq, clock = {"c": 1000, "s": 5000}, [t]

    def send(side, flags, payload=b""):
        a, b = (src, dst) if side == "c" else (dst, src)
        sp, dp = (sport, dport) if side == "c" else (dport, sport)
        other = "s" if side == "c" else "c"
        layer = TCP(sport=sp, dport=dp, flags=flags, seq=seq[side], ack=0 if flags == "S" else seq[other])
        frame(a, b, clock[0], layer / Raw(payload) if payload else layer)
        clock[0] += 0.01
        seq[side] += len(payload) + (1 if flags in ("S", "SA", "FA") else 0)

    send("c", "S"), send("s", "SA"), send("c", "A")
    for side, payload in convo:
        send(side, "PA", payload)
        send("s" if side == "c" else "c", "A")
    send("c", "FA"), send("s", "FA"), send("c", "A")


def http(t, sport, path, src="198.51.100.23", ua="Mozilla/5.0"):
    request = f"GET {path} HTTP/1.1\r\nHost: shop.example\r\nUser-Agent: {ua}\r\nAccept: */*\r\n\r\n".encode()
    response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"
    tcp_session(t, src, VICTIM, sport, 80, [("c", request), ("s", response)])


def build():
    # ---- attacks (each should raise the alert named in the comment) ----
    for i in range(30):                                               # 1000001 SYN port scan
        syn(10 + i * 0.1, "203.0.113.50", VICTIM, 40000, 1000 + i * 7)
    for i in range(15):                                               # 1000002 ping sweep
        frame("203.0.113.60", f"192.168.1.{i + 1}", 20 + i * 0.2, ICMP(type=8, id=7, seq=i) / Raw(b"sweep"))
    for i in range(8):                                                # 1000003 SSH brute force
        syn(30 + i * 2, "198.51.100.77", VICTIM, 50000 + i, 22)
    syn(50, "198.51.100.99", VICTIM, 51000, 23)                       # 1000004 Telnet
    tcp_session(55, "198.51.100.88", VICTIM, 52000, 21,               # 1000005 FTP login in cleartext
                [("s", b"220 FTP ready\r\n"), ("c", b"USER anonymous\r\n"), ("s", b"331 Password required\r\n")])
    http(60, 53001, "/products?id=1%20UNION%20SELECT%20username,password%20FROM%20users--")   # 1000006
    http(62, 53002, "/login?user=admin%27%20OR%20%271%27%3D%271")                              # 1000007
    http(64, 53003, "/download?file=../../etc/passwd")                                         # 1000008
    http(66, 53004, "/search?q=%3Cscript%3Ealert(1)%3C/script%3E")                             # 1000009
    http(68, 53005, "/index.php?id=1", ua="sqlmap/1.7.2#stable (https://sqlmap.org)")          # 1000010
    frame(CLIENT, RESOLVER, 75, UDP(sport=53000, dport=53) /                                   # 1000011
          DNS(rd=1, qd=DNSQR(qname="a" * 60 + ".tunnel.example")))
    syn(80, CLIENT, "203.0.113.99", 54000, 4444)                      # 1000012 reverse-shell port

    # ---- ordinary traffic (must NOT alert) ----
    http(5, 55001, "/index.html", src="198.51.100.5")
    frame(CLIENT, RESOLVER, 6, UDP(sport=53001, dport=53) / DNS(rd=1, qd=DNSQR(qname="www.example.com")))
    syn(7, "198.51.100.6", VICTIM, 56000, 22)
    for i in range(3):
        frame("198.51.100.7", VICTIM, 8 + i, ICMP(type=8, id=9, seq=i) / Raw(b"ping"))


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Write a demo attack capture for testing the IDS rules.")
    parser.add_argument("-o", "--output", default=os.path.join(root, "pcaps", "attack_demo.pcap"))
    args = parser.parse_args()
    build()
    packets.sort(key=lambda p: float(p.time))
    wrpcap(args.output, packets)
    print(f"Wrote {len(packets)} packets to {args.output}")
