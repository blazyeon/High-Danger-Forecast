"""
Today's Picks module.

Pre-computes the full slate of games (simulations), betting edges, and player
props for a date and caches them so the "Today's Picks" tab opens instantly
without running a simulation on every click.

Run during the daily update (right after odds/edges are refreshed) via
``update_todays_picks.py``.
"""
from __future__ import annotations

import logging
import math
from datetime import date as _date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from NHL.BettingEdge import (
    compute_game_edges,
    find_event_for_game,
    load_cached_odds,
    load_demo_odds,
    DEFAULT_DEMO_PATH,
    EDGE_THRESHOLD,
)
from NHL.Simulation import simulate_slate
from NHL.Lookup import get_team_full_name, display_abbr_for_game
from NHL.ApiScrape import get_games_on_date
from NHL.Errors import safe_api_call
from NHL.Utils import atomic_write_json, read_json_robust

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent / "static" / "data" / "todays_picks_cache.json"
DEFAULT_SIMS = 10000


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    if isinstance(obj, bool):
        return obj
    return obj


def _load_cache_index(cache_path: Path) -> Dict[str, Any]:
    """Load the multi-date picks cache, normalizing a legacy single-date payload."""
    if not cache_path.exists():
        return {}
    try:
        data = read_json_robust(cache_path)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    if "dates" in data:
        return data
    if "date" in data and "games" in data:
        return {"dates": {data["date"]: data}}
    return {}


def list_cached_todays_picks_dates(cache_path: Optional[Path] = None) -> List[str]:
    """Return the sorted list of dates that have cached picks."""
    cache_path = Path(cache_path or DEFAULT_CACHE_PATH)
    return sorted(_load_cache_index(cache_path).get("dates", {}).keys())


def load_cached_todays_picks(
    day: _date,
    cache_path: Optional[Path] = None,
    max_age_hours: float = 24.0,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Load pre-computed picks for a date. Returns (payload, warning).
    warning is set if the cache is missing, has no entry for the date, or is stale.
    """
    cache_path = Path(cache_path or DEFAULT_CACHE_PATH)
    if not cache_path.exists():
        return None, f"No cached picks found. Run `python update_todays_picks.py --date {day.isoformat()}`."

    index = _load_cache_index(cache_path)
    payload = index.get("dates", {}).get(day.isoformat())
    if payload is None:
        return None, f"No cached picks for {day.isoformat()}. Run `python update_todays_picks.py --date {day.isoformat()}`."

    computed_at = payload.get("computed_at")
    if computed_at:
        try:
            computed_dt = datetime.fromisoformat(computed_at.replace("Z", "+00:00"))
            age = datetime.now(timezone.utc) - computed_dt
            if age > timedelta(hours=max_age_hours):
                return payload, f"Picks cache is {age.total_seconds() / 3600:.1f} hours old."
        except Exception:
            pass

    return payload, None


def resolve_next_game_date(start: _date, max_lookahead: int = 14) -> _date:
    """
    Return the next date with NHL games, starting from `start` (inclusive).

    Used so the "Today's Picks" tab always points at a real game day: during the
    season this is just today, but in the preseason gap (or on an off day) it
    rolls forward to the next scheduled game.
    """
    for offset in range(max_lookahead + 1):
        d = start + timedelta(days=offset)
        try:
            games = safe_api_call(
                get_games_on_date, d.isoformat(),
                service_name="NHL Schedule API", fallback=[],
            )
        except Exception:
            games = []
        if games:
            return d
    return start


def compute_and_cache_todays_picks(
    day: _date,
    sims: int = DEFAULT_SIMS,
    cache_path: Optional[Path] = None,
    odds_payload: Optional[Dict[str, Any]] = None,
    include_props: bool = True,
) -> Dict[str, Any]:
    """
    Pre-compute the full slate of games (simulations), betting edges, and player
    props for a date and write them to the local cache.

    Designed to run during the daily update so the "Today's Picks" tab opens
    instantly and never re-runs a simulation on click.
    """
    cache_path = Path(cache_path or DEFAULT_CACHE_PATH)

    # 1. Load odds (cached or demo) for edge computation.
    warning = None
    if odds_payload is None:
        odds_payload, warning = load_cached_odds(day, max_age_hours=24.0)
        if odds_payload is None:
            odds_payload = load_demo_odds(DEFAULT_DEMO_PATH)
            warning = "Using demo odds (no live odds cached)."
    events = odds_payload.get("events", [])

    # 2. Load schedule for the date.
    schedule_games = safe_api_call(
        get_games_on_date, day.isoformat(),
        service_name="NHL Schedule API", fallback=[],
    )

    # 3. Build slate matchups and match each to an odds event.
    slate_matchups: List[Tuple[str, str]] = []
    slate_games: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for game in schedule_games or []:
        home_team = game.get("homeTeam") or {}
        away_team = game.get("awayTeam") or {}
        home_abbr = display_abbr_for_game(home_team.get("abbrev", home_team.get("name", "")))
        away_abbr = display_abbr_for_game(away_team.get("abbrev", away_team.get("name", "")))
        if not home_abbr or not away_abbr:
            continue
        schedule_game = {
            "home": home_abbr,
            "away": away_abbr,
            "home_name": get_team_full_name(home_team),
            "away_name": get_team_full_name(away_team),
            "startTime": game.get("startTimeUTC", game.get("gameDate", "")),
        }
        key = (home_abbr, away_abbr)
        slate_matchups.append(key)
        slate_games[key] = {
            "schedule_game": schedule_game,
            "event": find_event_for_game(schedule_game, events),
        }

    # 4. Simulate the slate with the full simulation count so the UI never
    #    re-runs a simulation on click.
    slate_results = simulate_slate(
        game_date=day,
        matchups=slate_matchups,
        stype=2,
        sims=sims,
        trend_games=25,
        use_recent_window_days=14,
    )

    # 5. Build per-game entries (full sim + edges).
    games: List[Dict[str, Any]] = []
    for res in slate_results:
        home_abbr = res["home"]
        away_abbr = res["away"]
        sim = res.get("sim")
        if sim is None:
            logger.warning(f"Simulation failed for {home_abbr} v {away_abbr}: {res.get('error', '')}")
            continue

        entry = slate_games.get((home_abbr, away_abbr), {})
        schedule_game = entry.get("schedule_game", {
            "home": home_abbr, "away": away_abbr,
            "home_name": home_abbr, "away_name": away_abbr, "startTime": "",
        })
        event = entry.get("event")

        edges: List[Dict[str, Any]] = []
        if event:
            edges = compute_game_edges(schedule_game, event, sim, edge_threshold=EDGE_THRESHOLD)
        best_edge = max((abs(e.get("edge", 0.0)) for e in edges), default=0.0)

        # Trim the raw goal distributions (large numpy arrays the UI never reads)
        # and make the rest JSON-safe before caching.
        sim_clean = dict(sim)
        sim_clean.pop("dist_home", None)
        sim_clean.pop("dist_away", None)
        sim_clean = _json_safe(sim_clean)

        games.append({
            "home": home_abbr,
            "away": away_abbr,
            "home_name": schedule_game["home_name"],
            "away_name": schedule_game["away_name"],
            "start_time": schedule_game["startTime"],
            "best_edge": best_edge,
            "edges": edges,
            "sim": sim_clean,
        })

    # 6. Pre-compute player props for the date.
    props: List[Dict[str, Any]] = []
    if include_props:
        try:
            from NHL.PlayerLinePredictor import compute_player_props_for_date
            props, props_warning = compute_player_props_for_date(day)
            props = _json_safe(props)
            if props_warning:
                warning = (warning + " " if warning else "") + props_warning
        except Exception as e:
            logger.warning(f"Props pre-computation failed for {day}: {e}")

    payload = {
        "date": day.isoformat(),
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "source": odds_payload.get("source", "unknown"),
        "warning": warning,
        "no_games": len(games) == 0,
        "games": games,
        "props": props,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    index = _load_cache_index(cache_path)
    index.setdefault("dates", {})[day.isoformat()] = payload
    atomic_write_json(cache_path, index, indent=2)

    logger.info(f"Cached today's picks for {day}: {len(games)} games, {len(props)} props -> {cache_path}")
    return payload
