"""Build the website from the frozen files.

    python -m src.site                # writes docs/index.html (static site) and docs/artifact.html

`docs/index.html` is one self-contained file (data embedded), ready for GitHub Pages or any web host.
`docs/artifact.html` is the same page without the <html> wrapper, for publishing from Claude.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .site_data import build_data

TEMPLATE = Path(__file__).with_name("site_template.html")


def render(data):
    """Embed the data in the template. Returns the page fragment (title, style, body, script)."""
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8")
    assert "__DATA__" in html
    return html.replace("__DATA__", payload)


def wrap(fragment):
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width,initial-scale=1">\n</head>\n<body>\n'
            + fragment + "\n</body>\n</html>\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default="live/predictions_log.csv")
    ap.add_argument("--explanations", default="live/explanations.jsonl")
    ap.add_argument("--matches", default="data/processed/matches.csv")
    ap.add_argument("--backtest-dir", default="backtest", help="output of src.backtest_weeks; skipped if missing")
    ap.add_argument("--out-dir", default="docs")
    args = ap.parse_args(argv)
    bt = Path(args.backtest_dir) if (Path(args.backtest_dir) / "predictions_log.csv").exists() else None
    data = build_data(args.log, args.explanations, args.matches, backtest_dir=bt)
    frag = render(data)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(wrap(frag), encoding="utf-8")
    (out / "artifact.html").write_text(frag, encoding="utf-8")
    (out / ".nojekyll").write_text("")
    n = sum(w["n"] for w in data["weeks"])
    print(f"Wrote {out}/index.html and {out}/artifact.html: {len(data['weeks'])} week(s), {n} match(es), "
          f"{data['scoreboard']['n']} scored.")
    if data.get("backtest"):
        b = data["backtest"]
        print(f"Backtest included: {len(b['weeks'])} round(s), {b['scoreboard']['n']} scored.")


if __name__ == "__main__":
    main()
