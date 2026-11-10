import numpy as np
import pandas as pd

from src import extra_stats as E
from src import walkforward as W


def _df(n=240, T=6, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.integers(0, T, n)
    a = (h + rng.integers(1, T, n)) % T
    d = pd.DataFrame({"Date": pd.to_datetime("2024-08-01") + pd.to_timedelta(np.arange(n), "D"), "h": h, "a": a,
                      "HST": rng.poisson(4.5, n), "AST": rng.poisson(3.5, n), "HS": rng.poisson(13, n), "AS": rng.poisson(10, n),
                      "HC": rng.poisson(5.5, n), "AC": rng.poisson(4.5, n), "FTHG": rng.poisson(1.5, n), "FTAG": rng.poisson(1.2, n)})
    return d


def test_covariate_names_and_columns_grow_with_extras():
    spec = E.with_extras(W.ModelSpec(shots=True, shots_xi=1.0), (("HS", "AS", 1.0, 5.0), ("HC", "AC", 1.0, 5.0)))
    assert spec.covariate_names() == ["shots_strength", "extra_HS_AS", "extra_HC_AC"]
    df = _df()
    aux = W.fit_shot_strengths(df.iloc[:150], df["Date"].iloc[150], 6, spec)
    X, Xm = W._covariates(df.iloc[150:], spec, aux)
    assert X.shape == (90, 3) and Xm.shape == (90, 3)
    assert len(aux["extra"]) == 2


def test_default_spec_is_unchanged_by_the_new_field():
    spec = W.ModelSpec(shots=True)
    assert spec.extra_stats == () and spec.covariate_names() == ["shots_strength"]
    df = _df()
    aux = W.fit_shot_strengths(df.iloc[:150], df["Date"].iloc[150], 6, spec)
    assert aux["extra"] == [] and set(aux) == {"a", "b", "extra"}


def test_stage0_uses_only_earlier_matches():
    spec = E.with_extras(W.ModelSpec(shots=True), (("HC", "AC", 1.0, 5.0),))
    df = _df()
    ref = df["Date"].iloc[150]
    a = W.fit_shot_strengths(df[df["Date"] < ref], ref, 6, spec)
    df2 = df.copy()
    late = df2["Date"] >= ref
    df2.loc[late, ["HC", "AC", "HST", "AST"]] = 40
    b = W.fit_shot_strengths(df2[df2["Date"] < ref], ref, 6, spec)
    assert np.allclose(a["extra"][0]["a"], b["extra"][0]["a"]) and np.allclose(a["a"], b["a"])


def test_extra_only_spec_without_shots_works():
    spec = E.with_extras(W.ModelSpec(), (("HS", "AS", 1.0, 5.0),))
    df = _df()
    res = W.fit_before(df, df["Date"].iloc[200], 6, 1.0, 5.0, spec=spec)
    assert res.aux["extra"][0]["a"].shape == (6,)
    probs = W.predict_block(res.theta, df.iloc[200:210], 6, spec, res.aux)
    assert np.allclose(probs.sum(axis=1), 1)
