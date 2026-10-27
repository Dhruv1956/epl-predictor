"""The Findings page must not carry numbers that differ from the reports it cites."""
import json
import math

import pandas as pd
import pytest

from src import findings

REPORTS = "reports"


@pytest.fixture(scope="module")
def F():
    return findings.build(REPORTS)


def test_no_nan_and_json_safe(F):
    json.dumps(F, allow_nan=False)


def test_season_gaps_match_csv(F):
    for v in findings.VERSIONS:
        csv = pd.read_csv(f"{REPORTS}/gap_by_season_{v}.csv")
        for rec in F["seasons"]:
            row = csv[csv["season"] == rec["season"]].iloc[0]
            assert rec[v][0] == pytest.approx(row["gap_B365"], abs=5e-5)
            assert rec[v][1] == pytest.approx(row["se_B365"], abs=5e-5)
            assert rec["n"] == row["n"]


def test_gap_is_model_minus_book(F):
    csv = pd.read_csv(f"{REPORTS}/gap_by_season_v2-shots.csv")
    for _, r in csv.iterrows():
        assert r["gap_B365"] == pytest.approx(r["LL_model"] - r["LL_B365"], abs=1e-9)


def test_ladder_is_ordered_and_consistent(F):
    L = F["ladder"]
    models = [r for r in L if r["kind"] == "model"]
    assert [r["kind"] for r in L] == ["baseline"] * 2 + ["model"] * len(models) + ["book"]
    assert [r["label"] for r in models][0] == "Poisson goals model"
    assert models[-1]["label"].startswith("+ shots")
    for r in models:  # gap to Avg odds = model log loss - 0.9650 within rounding of the report
        assert r["gap"] == pytest.approx(r["ll"] - F["book_test_ll"], abs=2e-4)
    assert models[-1]["ll"] < models[0]["ll"] < findings.BASELINES[1][1]


def test_shots_pooled_is_about_the_mean_of_windows():
    ch = [w[2] for w in findings.SHOTS_WINDOWS]  # xi_s = 1.0
    assert sum(ch) / len(ch) == pytest.approx(findings.SHOTS_POOLED[0], abs=1.5e-3)
    assert abs(findings.SHOTS_POOLED[0] / findings.SHOTS_POOLED[1]) == pytest.approx(1.9, abs=0.2)


def test_site_embeds_findings(tmp_path):
    from src import site
    out = tmp_path / "docs"
    site.main(["--out-dir", str(out)])
    html = (out / "index.html").read_text(encoding="utf-8")
    assert '"findings"' in html and 'id="pg-find"' in html and 'id="pg-method"' in html
    assert "</script><script" not in html.split('id="data">')[1].split("</script>")[0]
