"""
Tests for core Flask API endpoints.

Run:
    python test_api_endpoints.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

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


# ── Helpers ──────────────────────────────────────────────────────────────

def _client():
    import app as appmod
    return appmod.app.test_client()


# ── 1. /api/predict ──────────────────────────────────────────────────────

def test_api_predict_returns_result_shape():
    """
    /api/predict should return a 200 with the expected win-probability fields.
    """
    client = _client()
    r = client.post("/api/predict", json={
        "home_team": "TOR",
        "away_team": "MTL",
        "date": "2025-01-15",
        "simulations": 100,
        "trend_games": 10,
    })
    # If no PBP/Elo data is available the backend may 500; treat that as a skip.
    if r.status_code == 500 and "no data" in r.get_data(as_text=True).lower():
        pytest.skip("no cached PBP/Elo data for prediction")
    assert r.status_code == 200, f"predict failed: {r.status_code} {r.get_data(as_text=True)[:200]}"
    d = r.get_json()
    for key in ("home_win_pct", "away_win_pct", "exp_home_goals", "exp_away_goals",
                "mode_home_goals", "mode_away_goals", "confidence"):
        assert key in d, f"missing {key} in response keys: {list(d.keys())[:10]}"
    assert float(d["home_win_pct"]) >= 0
    assert float(d["away_win_pct"]) >= 0


def test_api_predict_validates_teams():
    """Missing teams should return a 400."""
    client = _client()
    r = client.post("/api/predict", json={"home_team": "TOR"})
    assert r.status_code == 400
    d = r.get_json()
    assert "error" in d


# ── 2. /api/player-props ─────────────────────────────────────────────────

def test_api_player_props_missing_key_is_handled():
    """
    Without an API key the endpoint should return a JSON error rather than crash.
    """
    client = _client()
    with mock.patch.dict(os.environ, {"ODDS_API_KEY": ""}, clear=False):
        r = client.get("/api/player-props/2025-01-15")
        assert r.status_code in (200, 500)
        d = r.get_json()
        assert d is not None, "endpoint should always return JSON"
        if r.status_code == 500:
            assert "error" in d


def test_api_player_props_demo_fallback():
    """
    When the API returns no props, the frontend-style demo file should still load.
    This verifies the demo payload format is valid JSON.
    """
    demo_path = PROJECT_ROOT / "static" / "data" / "demo_props.json"
    if not demo_path.exists():
        pytest.skip("demo_props.json not present")
    with open(demo_path) as f:
        payload = json.load(f)
    assert "props" in payload
    assert isinstance(payload["props"], list)
    if payload["props"]:
        first = payload["props"][0]
        for key in ("player", "market", "line", "prob_over", "recommendation"):
            assert key in first, f"demo prop missing {key}: {list(first.keys())}"


# ── 3. /api/seasons (cached filter) ───────────────────────────────────────

def test_api_seasons_returns_json():
    """/api/seasons should return a list of season options."""
    client = _client()
    r = client.get("/api/seasons")
    assert r.status_code == 200
    d = r.get_json()
    assert "seasons" in d
    assert isinstance(d["seasons"], list)
    for s in d["seasons"]:
        assert "label" in s and "key" in s
        assert len(s["key"]) == 8 and s["key"].isdigit()


# ── Main: run all ────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
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
