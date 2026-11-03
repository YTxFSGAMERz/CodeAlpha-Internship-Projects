#!/usr/bin/env bash
# Run Suricata live on a network interface with this project's rules.
#
#   sudo ./scripts/run_ids.sh <interface> [home_net]
#
# home_net is the network you protect (default: the three private ranges).
# By default ONLY rules/local.rules are loaded, so results are easy to follow.
# To add the Emerging Threats rules that "suricata-update" installed, run with WITH_ET=1.
set -euo pipefail

IFACE="${1:?usage: sudo $0 <interface> [home_net, e.g. 192.168.1.0/24]}"
HOME_NET="${2:-192.168.0.0/16,10.0.0.0/8,172.16.0.0/12}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONF="${SURICATA_CONF:-/etc/suricata/suricata.yaml}"
RULES_FLAG="-S"                       # -S: load only our rules, -s: add them to the configured rule set
[ "${WITH_ET:-0}" = "1" ] && RULES_FLAG="-s"

mkdir -p "$ROOT/logs"
echo "[*] Suricata on $IFACE, protecting $HOME_NET. Logs: $ROOT/logs (Ctrl+C to stop)"
exec suricata -c "$CONF" -i "$IFACE" "$RULES_FLAG" "$ROOT/rules/local.rules" -l "$ROOT/logs" \
     --set "vars.address-groups.HOME_NET=[$HOME_NET]"
