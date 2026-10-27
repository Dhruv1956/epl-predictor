#!/usr/bin/env bash
# Weekly post-match refresh: download new results, rebuild the tables, score the live log, rebuild the site.
#   bash scripts/refresh.sh
# Safe to run any time: if football-data.co.uk has not published new results yet, nothing is changed.
# If you get "SSL certificate problem", you are on a network that inspects HTTPS (e.g. a college firewall):
# switch to a phone hotspot and rerun. Do not turn certificate checking off.
set -u
cd "$(dirname "$0")/.."
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
CURL=/usr/bin/curl; [ -x "$CURL" ] || CURL=curl
dl() { "$CURL" -4 -sS -L -m 30 -A "$UA" --resolve football-data.co.uk:443:217.160.0.118 -o "$2" "$1"; }

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
dl https://football-data.co.uk/mmz4281/2627/E0.csv "$tmp/E0_2627.csv" || { echo "Download failed (see message above)."; exit 1; }
dl https://football-data.co.uk/fixtures.csv "$tmp/fixtures.csv" || { echo "Fixtures download failed."; exit 1; }

head -1 "$tmp/E0_2627.csv" | grep -q "^.\{0,3\}Div,Date" || { echo "Downloaded file does not look like football-data results; nothing changed."; exit 1; }
old=$(wc -l < data/raw/E0_2627.csv); new=$(wc -l < "$tmp/E0_2627.csv")
echo "Results file: $old lines now, $new lines on football-data.co.uk"
if [ "$new" -lt "$old" ]; then echo "The new file is SHORTER than ours; nothing changed. Please tell Claude."; exit 1; fi
cp "$tmp/fixtures.csv" data/live/fixtures.csv
if [ "$new" -eq "$old" ]; then
  echo "No new results published yet. Try again later (they usually appear within a day or two)."; exit 0
fi
cp "$tmp/E0_2627.csv" data/raw/E0_2627.csv

python3 -m src.data && python3 -m src.add_shots | tail -2 && python3 -m src.live score && python3 -m src.backtest_weeks --out reports/backtest_season.csv | head -3 && python3 -m src.site
echo; echo "Done. Paste the output above to Claude."
