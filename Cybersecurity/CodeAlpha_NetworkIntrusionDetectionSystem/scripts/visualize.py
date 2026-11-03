#!/usr/bin/env python3
"""
Turn Suricata's eve.json into a one-page dashboard (PNG) of the alerts it recorded:
alerts over time, top rules, top sources and severity.

Usage: python3 scripts/visualize.py [logs/eve.json] [-o docs/alerts_dashboard.png]
Needs matplotlib for the picture (pip install matplotlib); the text summary always prints.
"""
import argparse
import json
import os
from collections import Counter
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEVERITY = {1: "High (1)", 2: "Medium (2)", 3: "Low (3)"}


def load_alerts(path):
    alerts = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event_type") == "alert":
                event["_t"] = datetime.strptime(event["timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z")
                alerts.append(event)
    return alerts


def short(name):
    name = name.replace("LOCAL ", "")
    return name if len(name) <= 46 else name[:43] + "..."


def text_summary(alerts):
    print(f"{len(alerts)} alerts")
    for title, counter in (("Top rules", Counter(short(a["alert"]["signature"]) for a in alerts)),
                           ("Top sources", Counter(a["src_ip"] for a in alerts))):
        print(f"\n{title}")
        for name, count in counter.most_common(8):
            print(f"  {count:>4}  {name}")


def draw(alerts, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    times = [a["_t"] for a in alerts]
    first, last = min(times), max(times)
    span = (last - first).total_seconds()
    width = 5 if span <= 300 else 60 if span <= 7200 else 3600          # seconds per bar
    bins = Counter(int((t - first).total_seconds() // width) for t in times)
    n = max(bins) + 1
    step = max(1, n // 8)

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"Suricata alerts: {len(alerts)} between {first:%H:%M:%S} and {last:%H:%M:%S} UTC", fontsize=14)

    ax[0][0].bar(range(n), [bins.get(i, 0) for i in range(n)], color="#0f6b73")
    ax[0][0].set_xticks(range(0, n, step))
    ax[0][0].set_xticklabels([datetime.fromtimestamp(first.timestamp() + i * width, first.tzinfo).strftime("%H:%M:%S")
                              for i in range(0, n, step)], rotation=30, ha="right")
    ax[0][0].set_title(f"Alerts over time ({width}-second bars)")
    ax[0][0].set_ylabel("alerts")

    rules = Counter(short(a["alert"]["signature"]) for a in alerts).most_common(8)[::-1]
    ax[0][1].barh([r for r, _ in rules], [c for _, c in rules], color="#b3261e")
    ax[0][1].set_title("Top rules")

    sources = Counter(a["src_ip"] for a in alerts).most_common(8)[::-1]
    ax[1][0].barh([s for s, _ in sources], [c for _, c in sources], color="#8a5a00")
    ax[1][0].set_title("Top source addresses")

    severity = Counter(a["alert"]["severity"] for a in alerts)
    keys = sorted(severity)
    ax[1][1].bar([SEVERITY.get(k, str(k)) for k in keys], [severity[k] for k in keys],
                 color=["#b3261e", "#e08a00", "#c9b800"][:len(keys)])
    ax[1][1].set_title("Alerts by severity")

    for axis, count_axis in ((ax[0][0], "y"), (ax[0][1], "x"), (ax[1][0], "x"), (ax[1][1], "y")):
        getattr(axis, count_axis + "axis").set_major_locator(MaxNLocator(integer=True))   # whole alerts only
    for axis in ax.flat:
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"Saved dashboard to {out}")


def main():
    parser = argparse.ArgumentParser(description="Draw a dashboard from Suricata's eve.json.")
    parser.add_argument("eve", nargs="?", default=os.path.join(ROOT, "logs", "eve.json"))
    parser.add_argument("-o", "--output", default=os.path.join(ROOT, "docs", "alerts_dashboard.png"))
    args = parser.parse_args()
    alerts = load_alerts(args.eve)
    if not alerts:
        raise SystemExit("No alerts found in " + args.eve)
    text_summary(alerts)
    try:
        draw(alerts, args.output)
    except ImportError:
        print("\nmatplotlib is not installed, so no picture was drawn (pip install matplotlib).")


if __name__ == "__main__":
    main()
