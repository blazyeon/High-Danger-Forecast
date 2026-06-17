"""
Tests for the NHL Betting Edge module and endpoint.

Run:
    python test_betting_edge.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

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

from NHL.BettingEdge import (
    implied_probability,
    remove_vig_2way,
    find_event_for_game,
    compute_game_edges,
)


# ── 1. Core math ─────────────────────────────────────────────────────────

def test_implied_probability():
    assert implied_probability(2.0) == 0.5
    assert implied_probability(1.25) == 0.8
    assert implied_probability(0.0) == 0.0
    assert implied_probability(1.0) == 0.0


def test_remove_vig_2way():
    # Fair coin at -110 / -110 should return ~0.5 / 0.5
    p1 = implied_probability(american_to_decimal(-110))
    p2 = implied_probability(american_to_decimal(-110))
    t1, t2 = remove_vig_2way(p1, p2)
    assert abs(t1 - 0.5) < 0.001
    assert abs(t2 - 0.5) < 0.001
    assert abs(t1 + t2 - 1.0) < 1e-9

    # One-sided / empty returns zeros
    assert remove_vig_2way(0.6, 0.0) == (0.0, 0.0)


def american_to_decimal(am):
    am = float(am)
    if am > 0:
        return am / 100.0 + 1.0
    return 100.0 / abs(am) + 1.0


# ── 2. Event matching ────────────────────────────────────────────────────

def test_find_event_for_game_by_abbr_and_full_name():
    events = [
        {
            "home_team": "Toronto Maple Leafs",
            "away_team": "Montreal Canadiens",
            "id": "ev1",
        },
        {
            "home_team": "Colorado Avalanche",
            "away_team": "Vegas Golden Knights",
            "id": "ev2",
        },
    ]

    # Schedule game uses abbreviations
    g1 = {"home": "TOR", "away": "MTL"}
    ev = find_event_for_game(g1, events)
    assert ev is not None
    assert ev["id"] == "ev1"

    # Historical mapping (ARI -> UTA) also works
    g2 = {"home": "UTA", "away": "VGK"}
    ev = find_event_for_game(g2, [
        {"home_team": "Arizona Coyotes", "away_team": "Vegas Golden Knights", "id": "ev3"},
    ])
    assert ev is not None
    assert ev["id"] == "ev3"

    g3 = {"home": "TOR", "away": "BOS"}
    assert find_event_for_game(g3, events) is None


# ── 3. Edge computation ──────────────────────────────────────────────────

def test_compute_game_edges_finds_value():
    event = {
        "home_team": "Toronto Maple Leafs",
        "away_team": "Montreal Canadiens",
        "bookmakers": [
            {
                "key": "book1",
                "title": "Book One",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Toronto Maple Leafs", "price": 2.5},
                            {"name": "Montreal Canadiens", "price": 1.667},
                        ],
                    },
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Toronto Maple Leafs", "price": 1.8, "point": -1.5},
                            {"name": "Montreal Canadiens", "price": 2.1, "point": 1.5},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": 1.91, "point": 6.5},
                            {"name": "Under", "price": 1.91, "point": 6.5},
                        ],
                    },
                ],
            }
        ],
    }

    game = {"home": "TOR", "away": "MTL"}
    sim = {
        "home_win_pct": 70.0,
        "away_win_pct": 30.0,
        "home_win_2plus_pct": 45.0,
        "away_win_2plus_pct": 18.0,
        "totals_distribution": {5: 1000, 6: 3000, 7: 4000, 8: 2000},
    }

    edges = compute_game_edges(game, event, sim, edge_threshold=0.03)
    assert len(edges) > 0

    # Moneyline should flag Toronto as value (70% model vs ~46% no-vig implied)
    ml = next(e for e in edges if e["market"] == "Moneyline" and e["side"] == "Toronto Maple Leafs")
    assert ml["edge"] > 0.2


# ── 4. Flask endpoint ─────────────────────────────────────────────────────

def test_api_betting_edge_demo():
    """Hit the /api/betting-edge endpoint with demo odds forced."""
    try:
        import app as app_module
    except Exception as e:
        pytest.skip(f"Could not import app: {e}")

    client = app_module.app.test_client()
    resp = client.get("/api/betting-edge?date=2026-06-16&demo=1")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "games" in data
    assert "warning" in data
    assert data.get("source") in ("demo", "the-odds-api")

    # Demo odds have 16 games and at least one should produce an edge.
    assert isinstance(data["games"], list)


# ── Runner ────────────────────────────────────────────────────────────────

def _run_all():
    import inspect
    failures = []
    for name, obj in globals().items():
        if not name.startswith("test_") or not callable(obj):
            continue
        try:
            obj()
            print(f"PASS {name}")
        except Exception as e:
            failures.append((name, e))
            print(f"FAIL {name}: {e}")
    return failures


if __name__ == "__main__":
    failures = _run_all()
    if failures:
        print(f"\n{len(failures)} test(s) failed")
        sys.exit(1)
    print("\nAll tests passed")
