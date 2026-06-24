"""
Unit tests for the defensive-injury impact model.

Run:
    py -3 test_defensive_injury_impact.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from NHL.Simulation import calculate_automatic_injury_impact


def _make_def_scores(**kwargs):
    """Build a defensive_scores dict for a single injured player."""
    base = {
        "name": kwargs.get("name", "Player"),
        "team": kwargs.get("team", "NYI"),
        "position": kwargs.get("position", "D"),
        "situation": "5on5",
        "gp": kwargs.get("gp", 50),
        "icetime_hours": kwargs.get("icetime_hours", 20.0),
        "onice_xga60": 2.5,
        "delta_xg_pct": kwargs.get("delta_xg_pct", 0.0),
        "onice_xg_pct": 0.5,
        "onice_ca60": 50.0,
        "blocks60": 3.0,
        "takeaways60": 1.0,
        "giveaways60": 1.0,
        "hits60": 3.0,
        "defensive_score": kwargs.get("defensive_score", 0.0),
        "defensive_percentile": kwargs.get("defensive_percentile", 50.0),
        "raw_defensive_score": kwargs.get("defensive_score", 0.0),
    }
    key = kwargs.get("key", base["name"].lower().replace(" ", ""))
    return {key: base}


def _make_player_stats(name: str, points: int, gp: int = 50):
    key = name.lower().replace(" ", "")
    return {
        key: {
            "name": name,
            "team": "NYI",
            "gp": gp,
            "goals": 0,
            "assists": points,
            "points": points,
        }
    }


def _run_case(injuries, player_stats, defensive_scores, team_stats=None):
    """Patch the live imports inside calculate_automatic_injury_impact and run it."""
    team_stats = team_stats or {"NYI": {"total_goals": 200, "total_points": 500}}
    with mock.patch("NHL.ApiScrape.get_team_injuries", return_value=injuries):
        with mock.patch(
            "NHL.DefensiveImpact.compute_defensive_impact_scores",
            return_value=defensive_scores,
        ):
            return calculate_automatic_injury_impact(
                team_abbr="NYI",
                season="20252026",
                player_stats_cache=player_stats,
                team_stats_cache=team_stats,
            )


def test_elite_top_pair_defenseman_out():
    """Losing an elite, high-TOI defenseman should materially raise opponent goals."""
    name = "Ryan Pulock"
    key = name.lower().replace(" ", "")
    result = _run_case(
        injuries={key: "Out"},
        player_stats=_make_player_stats(name, points=15),
        defensive_scores=_make_def_scores(
            name=name,
            key=key,
            defensive_score=2.5,
            icetime_hours=22.0,
        ),
    )
    assert result["defense_impact"] > 0.12, (
        f"Expected elite D impact >12%, got {result['defense_impact']*100:.1f}%"
    )


def test_depth_liability_defenseman_out_can_help():
    """A below-replacement defenseman's absence should not raise opponent goals."""
    name = "Depth D"
    key = name.lower().replace(" ", "")
    result = _run_case(
        injuries={key: "Out"},
        player_stats=_make_player_stats(name, points=5),
        defensive_scores=_make_def_scores(
            name=name,
            key=key,
            defensive_score=-2.5,
            icetime_hours=8.0,
        ),
    )
    # Negative score, low TOI: the absence is a small defensive help or neutral.
    assert result["defense_impact"] <= 0.02, (
        f"Expected liability absence to be neutral/helpful, got "
        f"{result['defense_impact']*100:+.1f}%"
    )


def test_pair_disruption_compounds():
    """Two top-pair D out should cost more than the sum of either individually."""
    d1, d2 = "Top D One", "Top D Two"
    k1, k2 = d1.lower().replace(" ", ""), d2.lower().replace(" ", "")

    def one_out(k):
        return _run_case(
            injuries={k: "Out"},
            player_stats=_make_player_stats(d1 if k == k1 else d2, points=10),
            defensive_scores=_make_def_scores(
                name=d1 if k == k1 else d2,
                key=k,
                defensive_score=2.0,
                icetime_hours=20.0,
            ),
        )["defense_impact"]

    both_out = _run_case(
        injuries={k1: "Out", k2: "Out"},
        player_stats={**_make_player_stats(d1, points=10), **_make_player_stats(d2, points=10)},
        defensive_scores={
            **_make_def_scores(name=d1, key=k1, defensive_score=2.0, icetime_hours=20.0),
            **_make_def_scores(name=d2, key=k2, defensive_score=2.0, icetime_hours=20.0),
        },
    )["defense_impact"]

    single_sum = one_out(k1) + one_out(k2)
    assert both_out > single_sum, (
        f"Pair disruption expected: both={both_out*100:.1f}% > "
        f"sum={single_sum*100:.1f}%"
    )


def test_forward_counts_less_than_defenseman():
    """A forward with the same defensive score should have a smaller impact."""
    d_name, f_name = "Same Score D", "Same Score F"
    d_key, f_key = d_name.lower().replace(" ", ""), f_name.lower().replace(" ", "")

    def case(name, key, pos):
        return _run_case(
            injuries={key: "Out"},
            player_stats=_make_player_stats(name, points=20),
            defensive_scores=_make_def_scores(
                name=name,
                key=key,
                position=pos,
                defensive_score=2.0,
                icetime_hours=18.0,
            ),
        )["defense_impact"]

    d_impact = case(d_name, d_key, "D")
    f_impact = case(f_name, f_key, "LW")
    assert f_impact < d_impact, (
        f"Forward impact {f_impact*100:.1f}% should be smaller than "
        f"D impact {d_impact*100:.1f}%"
    )


def test_team_cap_limits_extreme_cases():
    """Three elite top-pair D out should hit the team cap, not go infinite."""
    players = ["Elite D 1", "Elite D 2", "Elite D 3"]
    keys = [p.lower().replace(" ", "") for p in players]
    stats = {}
    for p in players:
        stats.update(_make_player_stats(p, points=12))

    result = _run_case(
        injuries={k: "Out" for k in keys},
        player_stats=stats,
        defensive_scores={
            **_make_def_scores(name=players[0], key=keys[0], defensive_score=4.0, icetime_hours=24.0),
            **_make_def_scores(name=players[1], key=keys[1], defensive_score=4.0, icetime_hours=24.0),
            **_make_def_scores(name=players[2], key=keys[2], defensive_score=4.0, icetime_hours=24.0),
        },
    )
    from NHL.Config import DEFENSIVE_INJURY_PARAMS
    assert result["defense_impact"] <= DEFENSIVE_INJURY_PARAMS["team_cap"] + 0.001, (
        f"Team cap breached: {result['defense_impact']*100:.1f}%"
    )
    assert result["defense_impact"] >= 0.30, (
        f"Three elite D should at least reach 30%, got "
        f"{result['defense_impact']*100:.1f}%"
    )


def test_missing_defensive_record_counts_zero_defense():
    """A player with no defensive score still gets offense impact but no defense impact."""
    name = "No Def Data"
    key = name.lower().replace(" ", "")
    result = _run_case(
        injuries={key: "Out"},
        player_stats=_make_player_stats(name, points=25),
        defensive_scores={},
    )
    assert result["offense_impact"] < -0.03, "Missing defensive record should still produce offense impact"
    assert result["defense_impact"] == 0.0, (
        f"Missing defensive record should yield 0 defense impact, got "
        f"{result['defense_impact']}"
    )


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failures.append((t.__name__, str(e)))
        except Exception as e:
            print(f"  ERR   {t.__name__}: {type(e).__name__}: {e}")
            failures.append((t.__name__, str(e)))
    if failures:
        print(f"\n{len(failures)} failures")
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed")
