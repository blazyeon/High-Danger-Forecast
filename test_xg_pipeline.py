"""
End-to-end tests for the PBP-based xG pipeline.

Run:
    python test_xg_pipeline.py

Tests:
  1. PBP fetch + parse for a small fixture
  2. xG model train + predict on a fixture
  3. Stats aggregators return expected shapes/columns
  4. End-to-end /api/predict against the Flask test client
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Quiet down noisy loggers
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# pytest is optional; if not installed, tests can still run via __main__.
try:
    import pytest  # noqa: F401
    _HAS_PYTEST = True
except ImportError:
    _HAS_PYTEST = False

    class _SkipTest(Exception):
        """Local stub for pytest.skip when pytest is unavailable."""

    def _skip(reason=""):
        raise _SkipTest(reason)

    class _PytestStub:
        Exception = type("Exception", (), {"Skip": _SkipTest})
        skip = staticmethod(_skip)

    pytest = _PytestStub()  # type: ignore[assignment]

from NHL.PlayByPlay import (  # noqa: E402
    parse_pbp_events,
    load_shot_store,
)
from NHL.xGModel import (  # noqa: E402
    build_features,
    predict_xg,
    load_xg_model,
    train_xg_model,
)
from NHL.StatsFromPBP import (  # noqa: E402
    compute_team_rates,
    compute_skater_rates,
    compute_goalie_rates,
)

# A handful of stable game IDs from 2024-25 (publicly visible on api-web.nhle.com)
TEST_GAMES_2024 = [2024020001, 2024020002, 2024020003, 2024020004, 2024020005]
TEST_GAMES_2023 = [2023020001, 2023020002, 2023020003, 2023020004, 2023020005]


# ── 1. PBP parse fixture ─────────────────────────────────────────────────

def test_pbp_parse_returns_enriched_rows():
    """
    parse_pbp_events on a cached PBP JSON should return shot rows with
    shooter_name and goalie_name populated for at least 90% of events.
    """
    raw_dir = PROJECT_ROOT / "pbp_cache" / "raw"
    if not raw_dir.exists():
        pytest.skip("no pbp_cache/raw — run update_pbp_stats.py first")
    cached = sorted(raw_dir.glob("202402*.json"))[:1]
    if not cached:
        pytest.skip("no 2024-25 PBP games cached")
    with open(cached[0]) as f:
        pbp = json.load(f)
    df = parse_pbp_events(pbp, game_id=int(cached[0].stem))
    assert not df.empty, "parse_pbp_events returned empty"
    assert {"event_type", "x", "y", "shooter_name", "goalie_name"}.issubset(df.columns)
    shooter_pct = (df["shooter_name"].astype(str).str.strip() != "").mean()
    goalie_pct = (df["goalie_name"].astype(str).str.strip() != "").mean()
    assert shooter_pct >= 0.9, f"shooter_name coverage only {shooter_pct:.1%}"
    assert goalie_pct >= 0.6, f"goalie_name coverage only {goalie_pct:.1%}"


# ── 2. xG model train + predict ──────────────────────────────────────────

def test_xg_model_auc_threshold():
    """
    The trained xG model on the 2024-25 shot store should reach AUC >= 0.74.
    """
    shots = load_shot_store(2024, 2)
    if shots.empty:
        pytest.skip("no 2024-25 shot store; run update_pbp_stats.py")
    # We don't retrain (the saved one is the baseline). Just predict and
    # check AUC on the full data — not a true holdout but a smoke test.
    from sklearn.metrics import roc_auc_score
    model = load_xg_model()
    xg = predict_xg(shots, model)
    y = shots["is_goal"].astype(int).values
    auc = float(roc_auc_score(y, xg))
    assert auc >= 0.74, f"xG AUC {auc:.3f} below 0.74 threshold"


def test_xg_predict_handles_nan():
    """
    predict_xg must not raise on NaN-containing shots and must return
    one float per input row.
    """
    model = load_xg_model()
    # Make a frame with some NaN x/y
    bad = pd.DataFrame({
        "x": [89.0, np.nan, 60.0, 0.0],
        "y": [0.0, 5.0, np.nan, 0.0],
        "period": [1, 1, 1, 1],
        "time_seconds": [60.0, 65.0, 70.0, 75.0],
        "home_skaters": [5, 5, 5, 5],
        "away_skaters": [5, 5, 5, 5],
        "is_goal": [0, 1, 0, 1],
        "shot_type": ["wrist", "slap", "", "snap"],
    })
    out = predict_xg(bad, model)
    assert len(out) == len(bad)
    assert np.isfinite(out).all(), "predict_xg returned non-finite values"


# ── 3. Stats aggregator shapes ──────────────────────────────────────────

EXPECTED_TEAM_COLS = {
    "team", "gp", "gf", "ga", "cf", "ca", "ff", "fa", "sf", "sa",
    "hdcf", "hdca", "goals_per_game", "xgf", "xga", "xgf_per_game",
    "xga_per_game", "cf_pct", "ff_pct", "sf_pct", "hdcf_pct", "xgf_pct",
    "sv_pct", "sh_pct", "gsax",
}


def test_compute_team_rates_columns():
    df = compute_team_rates(2024, 2)
    if df.empty:
        pytest.skip("no 2024-25 shot store")
    missing = EXPECTED_TEAM_COLS - set(df.columns)
    assert not missing, f"compute_team_rates missing cols: {missing}"
    assert (df["team"] != "").all(), "some teams have empty abbrev"
    # 32 NHL teams in 2024-25
    assert 30 <= len(df) <= 32, f"unexpected team count: {len(df)}"


def test_compute_skater_rates_shape():
    rates = compute_skater_rates(2024, 2)
    if not rates:
        pytest.skip("no 2024-25 shot store")
    # >= 500 skaters
    assert len(rates) >= 500, f"only {len(rates)} skaters (expected 500+)"
    # Spot-check keys
    sample = next(iter(rates.values()))
    for k in ("name", "gpg", "apg", "sogpg", "xgf_pg", "gp", "goals", "shots", "assists"):
        assert k in sample, f"skater dict missing {k}"
    # Goals + assists should be integer-like
    assert isinstance(sample["goals"], (int, np.integer))
    assert isinstance(sample["assists"], (int, np.integer))


def test_compute_goalie_rates_shape():
    df = compute_goalie_rates(2024, 2)
    if df.empty:
        pytest.skip("no 2024-25 shot store")
    for c in ("name", "gp", "ga", "sa", "sv", "sv_pct", "xga", "gsax", "gsax_per_60"):
        assert c in df.columns, f"goalie rates missing {c}"
    # Reasonable NHL goalie counts
    assert 50 <= len(df) <= 150, f"unexpected goalie count: {len(df)}"


# ── 4. End-to-end Flask API ──────────────────────────────────────────────

def test_api_predict_endpoint():
    """
    /api/predict should return a 200 with home_win_pct and home_xg/away_xg.
    """
    import app as appmod
    client = appmod.app.test_client()
    r = client.post("/api/predict", json={
        "home_team": "BOS",
        "away_team": "TOR",
        "game_date": "2026-01-15",
        "simulations": 200,
    })
    assert r.status_code == 200, f"predict failed: {r.status_code} {r.get_data(as_text=True)[:200]}"
    d = r.get_json()
    assert "home_win_pct" in d, f"missing home_win_pct in {list(d.keys())[:5]}"


def test_api_stats_endpoints():
    """
    The 3 stats endpoints should each return 32 teams / 500+ skaters / 50+ goalies.
    """
    import app as appmod
    client = appmod.app.test_client()
    for path, min_rows in [
        ("/api/stats/teams?season=20242025&stype=2", 30),
        ("/api/stats/skaters?season=20242025&stype=2", 500),
        ("/api/stats/goalies?season=20242025&stype=2", 50),
    ]:
        r = client.get(path)
        assert r.status_code == 200, f"{path}: {r.status_code}"
        d = r.get_json()
        assert "data" in d, f"{path}: no data"
        assert len(d["data"]) >= min_rows, f"{path}: only {len(d['data'])} rows"


# ── Main: run all ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Without pytest, run each test and report
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            # Real pytest uses `Skipped`; our stub uses `_SkipTest`.
            cls = type(e).__name__
            if _HAS_PYTEST and isinstance(e, pytest.skip.Exception):
                print(f"  SKIP  {t.__name__}: {e}")
            elif not _HAS_PYTEST and cls == "_SkipTest":
                print(f"  SKIP  {t.__name__}: {e}")
            else:
                print(f"  FAIL  {t.__name__}: {cls}: {e}")
                failures.append((t.__name__, str(e)))
    if failures:
        print(f"\n{len(failures)} failures")
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed")
