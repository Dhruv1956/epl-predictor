#!/usr/bin/env bash
# One-off download of Championship (second tier) results from football-data.co.uk into data/raw_e1/.
#   bash scripts/get_championship.sh
# Same rules as refresh.sh: use a phone hotspot if you see "SSL certificate problem"; certificate checking is never turned off.
set -u
cd "$(dirname "$0")/.."
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
CURL=/usr/bin/curl; [ -x "$CURL" ] || CURL=curl
dl() { "$CURL" -4 -sS -L -m 30 -A "$UA" --resolve football-data.co.uk:443:217.160.0.118 -o "$2" "$1"; }
mkdir -p data/raw_e1
ok=0
for s in 1516 1617 1718 1819 1920 2021 2122 2223 2324 2425 2526 2627; do
  tmp=$(mktemp)
  if dl "https://football-data.co.uk/mmz4281/$s/E1.csv" "$tmp" && head -1 "$tmp" | grep -q "^.\{0,3\}Div,Date"; then
    mv "$tmp" "data/raw_e1/E1_$s.csv"; echo "E1_$s.csv: $(($(wc -l < data/raw_e1/E1_$s.csv) - 1)) matches"; ok=$((ok+1))
  else
    rm -f "$tmp"; echo "E1_$s.csv: FAILED (nothing saved)"
  fi
done
echo; echo "Downloaded $ok of 12 files. Paste the output above to Claude."
