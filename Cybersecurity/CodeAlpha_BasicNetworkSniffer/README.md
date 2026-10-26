# CodeAlpha_BasicNetworkSniffer

A command-line network sniffer written in Python with [Scapy](https://scapy.net), built for **Task 1 (Basic Network Sniffer)** of the CodeAlpha Cyber Security Internship.

It captures packets, takes them apart layer by layer, and shows who is talking to whom, over which protocol, and what the payload looks like.

> **Use it responsibly.** Only capture traffic on networks you own or have explicit permission to monitor. Sniffing other people's traffic without permission is illegal in most places.

## Features

- Live capture from any interface, or offline analysis of a `.pcap` file (`-r`)
- Per packet: time, protocol, source and destination IP and port, size, and a readable description
- Recognises ARP, IPv4/IPv6, TCP (with flag names), UDP, ICMP/ICMPv6, DNS, plain HTTP and TLS
- Hex + ASCII payload preview (`-p`) and a layer-by-layer header breakdown (`-v`)
- BPF capture filters (`-f "tcp port 80"`) and saving to `.pcap` (`-w`), which opens in Wireshark
- End-of-capture summary: protocol share and busiest host pairs
- A malformed packet prints a warning instead of stopping the sniffer

## Requirements

- Python 3.8 or newer (tested on 3.12)
- Scapy: `pip install -r requirements.txt`
- Windows: [Npcap](https://npcap.com) installed
- Linux/macOS: root privileges, to open the network interface

## Usage

```bash
# Linux / macOS (root needed)
sudo python3 sniffer.py                      # capture until Ctrl+C
sudo python3 sniffer.py -i eth0 -c 50 -p     # 50 packets on eth0, with payload preview
sudo python3 sniffer.py -f "tcp port 80" -v  # only HTTP traffic, full layer breakdown
sudo python3 sniffer.py -w capture.pcap      # also save the capture

# Windows (run the terminal as Administrator if capture fails)
python sniffer.py --list-interfaces
python sniffer.py -i "Wi-Fi" -c 50 -p

# Quick test / demo (no root or network interface needed)
python sniffer.py --demo -p

# Analyse a saved capture
python sniffer.py -r capture.pcap
```

If Scapy is installed in a virtual environment, plain `sudo` will not find it. Use `sudo venv/bin/python sniffer.py` instead.

| Option | Meaning |
|---|---|
| `-i`, `--interface` | Interface to sniff on (default: system default). List them with `--list-interfaces` |
| `-c`, `--count` | Stop after N packets (default: until Ctrl+C) |
| `-f`, `--filter` | BPF filter, e.g. `tcp port 80`, `udp port 53`, `host 192.168.1.1` |
| `-p`, `--payload` | Hex/ASCII preview of each packet's payload |
| `--max-bytes` | How many payload bytes `-p` shows (default: 64) |
| `-v`, `--verbose` | Header fields of every layer for each packet |
| `-w`, `--write` | Save the capture to a `.pcap` file |
| `-r`, `--read` | Analyse a `.pcap` file instead of sniffing live |
| `--demo` | Run with built-in sample packets to test parsing instantly |
| `--no-color` | Plain output without colours |

## Example output

A DNS lookup, a TCP handshake, an HTTP request and response, a TLS hello and a ping:

```
    #  TIME          PROTO   SOURCE                     DESTINATION                LEN  INFO
    1  10:48:25.535  DNS     192.168.1.5:53124        → 8.8.8.8:53                  71  Query A example.com
    2  10:48:25.565  DNS     8.8.8.8:53               → 192.168.1.5:53124           98  Response A example.com → NOERROR, 1 answer(s)
    3  10:48:25.635  TCP     192.168.1.5:51234        → 93.184.216.34:80            54  HTTP  [SYN] seq=1000 ack=0 win=64240 len=0
    4  10:48:25.665  TCP     93.184.216.34:80         → 192.168.1.5:51234           54  HTTP  [SYN,ACK] seq=5000 ack=1001 win=65535 len=0
    5  10:48:25.666  TCP     192.168.1.5:51234        → 93.184.216.34:80            54  HTTP  [ACK] seq=1001 ack=5001 win=64240 len=0
    6  10:48:25.675  HTTP    192.168.1.5:51234        → 93.184.216.34:80           119  GET /index.html HTTP/1.1
    7  10:48:25.735  HTTP    93.184.216.34:80         → 192.168.1.5:51234          131  HTTP/1.1 200 OK
    8  10:48:25.835  TLS     192.168.1.5:51300        → 93.184.216.34:443           97  Handshake: ClientHello
    9  10:48:26.035  ICMP    192.168.1.5              → 192.168.1.1                 58  Echo request (ping) id=1 seq=1

──────────────────────────────────────────────────────────────────────────────
Capture summary
  Packets : 9
  Data    : 736 bytes
  Duration: 0.50 s

  Protocols
    TCP           3   33.3%  ██████████
    DNS           2   22.2%  ███████
    HTTP          2   22.2%  ███████
    TLS           1   11.1%  ███
    ICMP          1   11.1%  ███

  Top conversations
    192.168.1.5 ↔ 93.184.216.34                        6 pkts       509 bytes
    192.168.1.5 ↔ 8.8.8.8                              2 pkts       169 bytes
    192.168.1.1 ↔ 192.168.1.5                          1 pkts        58 bytes
```

The same HTTP request (packet 6) with `-v -p`:

```
    6  10:48:25.675  HTTP    192.168.1.5:51234        → 93.184.216.34:80           119  GET /index.html HTTP/1.1
      Ethernet  dst=aa:bb:cc:00:00:01  src=aa:bb:cc:00:00:05  type=IPv4
      IP        version=4  ihl=5  tos=0x0  len=105  id=1  frag=0  ttl=64  proto=tcp  chksum=0x8306  src=192.168.1.5  dst=93.184.216.34
      TCP       sport=51234  dport=http  seq=1001  ack=5001  dataofs=5  reserved=0  flags=PA  window=8192  chksum=0x8c86  urgptr=0
      Raw       65 bytes
      0000  47 45 54 20 2f 69 6e 64 65 78 2e 68 74 6d 6c 20  |GET /index.html |
      0010  48 54 54 50 2f 31 2e 31 0d 0a 48 6f 73 74 3a 20  |HTTP/1.1..Host: |
      0020  65 78 61 6d 70 6c 65 2e 63 6f 6d 0d 0a 55 73 65  |example.com..Use|
      ... 17 more bytes not shown
```

## How it works

Every packet is a set of envelopes, one inside the other. The sniffer opens them in order:

```
Ethernet frame            MAC addresses   who is next on this network link
 └─ IP packet             IP addresses    who the end hosts are
     └─ TCP / UDP         ports           which program on each host
         └─ payload       the data        an HTTP request, a DNS question, ...
```

In the code (`sniffer.py`):

1. `sniff()` from Scapy captures frames and calls `handle()` for each one. Nothing is stored in memory, so long captures stay light.
2. `parse_packet()` walks the layers (ARP, IPv4/IPv6, then ICMP/TCP/UDP) and turns them into a small `PacketInfo`. Well-known ports and payload patterns add DNS, HTTP and TLS labels.
3. `format_line()`, `layer_breakdown()` and `hexdump()` print the one-line summary, the layer fields (`-v`) and the payload preview (`-p`).
4. `Stats` counts packets per protocol and per pair of hosts, and prints the summary when the capture ends.

## Reading the INFO column

| You see | It means |
|---|---|
| `[SYN]`, `[SYN,ACK]`, `[ACK]` | The TCP three-way handshake that opens a connection |
| `[PSH,ACK]` | The sender is pushing data now |
| `[FIN,ACK]` / `[RST]` | Connection closing politely / being aborted |
| `len=N` | Bytes of payload in this packet (0 for pure acknowledgements) |
| `Who has X? Tell Y` | ARP: Y asks the local network which device owns IP address X |
| `Query A example.com` | DNS: asking for the IPv4 address of a name |
| `Echo request (ping)` | ICMP ping; the matching reply is `Echo reply (ping)` |
| `Handshake: ClientHello` | TLS (HTTPS) is starting. After the handshake, content shows as `Application Data (encrypted)` |

Plain HTTP payloads are readable with `-p`. HTTPS payloads are encrypted, so you can see who talks to whom and when, but not what is inside. When a packet has no decoded protocol but uses a well-known port, the port's service name (`HTTP`, `SSH`, `HTTPS`, ...) is shown at the start of INFO.

## Task requirements and where they are covered

| Task 1 requirement | Where |
|---|---|
| Capture network traffic packets | `sniff()` in `run_capture()`; `-r` reads saved captures |
| Analyse packet structure and content | `parse_packet()`, `-v` layer breakdown, `-p` payload |
| Learn how data flows and the basics of protocols | Capture summary and the tables above |
| Use `scapy` or `socket` | Scapy |
| Show source/destination IPs, protocols and payloads | Main output line and `-p` |

## Notes and troubleshooting

- **Permission denied:** packet capture needs `sudo` (Linux/macOS) or an Administrator terminal (Windows).
- **Windows, capture fails with a pcap/winpcap error:** install [Npcap](https://npcap.com).
- **`-f` says libpcap is not available:** on Debian/Ubuntu/Kali run `sudo apt install tcpdump`, which installs libpcap. Windows needs Npcap.
- **Every packet appears twice on `-i lo` (Linux):** Linux hands loopback traffic to the capture socket once for each direction. Real interfaces such as `eth0` or `wlan0` do not do this.
- **Unknown interface name:** run `--list-interfaces` and copy the name exactly.

## Files

- `sniffer.py`: the sniffer
- `requirements.txt`: Python dependency (Scapy)
- `README.md`: this file
