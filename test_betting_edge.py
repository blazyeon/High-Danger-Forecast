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


def test_compute_game_edges_matches_book_short_team_names():
    """
    A book that labels its own markets "<ABBR> <Nickname>" must still resolve.

    The feed disagrees with itself inside a single event: the event fields carry
    "<City> <Nickname>" ("Toronto Maple Leafs") while that event's DraftKings
    markets carry "TOR Maple Leafs" / "CHI Blackhawks". Comparing the strings
    directly resolved only the sides that happened to coincide between the two
    forms, so the moneyline -- which needs BOTH sides before it is emitted at
    all -- disappeared from every game on the board, and each puck line silently
    lost whichever side missed, leaving the surviving side to be shown even when
    the model preferred the other one.
    """
    event = {
        "home_team": "Toronto Maple Leafs",
        "away_team": "Chicago Blackhawks",
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "TOR Maple Leafs", "price": -120},
                            {"name": "CHI Blackhawks", "price": 100},
                        ],
                    },
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "TOR Maple Leafs", "price": -110, "point": -1.5},
                            {"name": "CHI Blackhawks", "price": -110, "point": 1.5},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": -110, "point": 6.5},
                            {"name": "Under", "price": -110, "point": 6.5},
                        ],
                    },
                ],
            }
        ],
    }

    game = {"home": "TOR", "away": "CHI"}
    sim = {
        "home_win_pct": 62.0,
        "away_win_pct": 38.0,
        "home_win_2plus_pct": 42.0,
        "away_win_2plus_pct": 20.0,
        "totals_distribution": {5: 1000, 6: 3000, 7: 4000, 8: 2000},
    }

    # edge_threshold=None -- the Game Bet board keeps every line.
    edges = compute_game_edges(game, event, sim, edge_threshold=None)
    markets = {e["market"] for e in edges}
    assert markets == {"Moneyline", "Puck Line (1.5)", "Total 6.5"}, markets

    # The moneyline only exists at all when both sides resolved.
    ml = next(e for e in edges if e["market"] == "Moneyline")
    assert ml["side"] == "Toronto Maple Leafs"
    assert abs(ml["model_prob"] - 0.62) < 1e-6

    # Both sides of the puck line resolved, so the surviving row is the model's
    # preferred side: TOR wins by 2+ only 42% of the time, so +1.5 on the
    # underdog is the side to keep, not -1.5 on the favourite.
    pl = next(e for e in edges if e["market"].startswith("Puck Line"))
    assert pl["side"] == "CHI Blackhawks", pl
    assert pl["edge"] > 0


# ── 4. Flask endpoint ─────────────────────────────────────────────────────

def _run_all():
    import inspect
    failures = []
    for name, obj in globals().items():
        if not name.startswith("test_") or not callable(obj):
            continue
        try:
            obj()
            print(f"PASS {name}")
        except BaseException as e:
            # pytest's `Skipped` subclasses BaseException, not Exception, so a
            # plain `except Exception` lets any skip abort the whole run.
            cls = type(e).__name__
            if _HAS_PYTEST and isinstance(e, pytest.skip.Exception):
                print(f"SKIP {name}: {e}")
            elif not _HAS_PYTEST and cls == "_SkipTest":
                print(f"SKIP {name}: {e}")
            elif not isinstance(e, Exception):
                raise  # KeyboardInterrupt, SystemExit, genuine aborts
            else:
                failures.append((name, e))
                print(f"FAIL {name}: {e}")
    return failures


if __name__ == "__main__":
    failures = _run_all()
    if failures:
        print(f"\n{len(failures)} test(s) failed")
        sys.exit(1)
    print("\nAll tests passed")
