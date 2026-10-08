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
import sqlite3
from datetime import date as _date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from NHL.BettingEdge import (
    compute_game_edges,
    fetch_and_cache_odds,
    find_event_for_game,
    load_cached_odds,
    EDGE_THRESHOLD,
)
from NHL.Simulation import simulate_slate
from NHL.Lookup import get_team_full_name, display_abbr_for_game
from NHL.ApiScrape import get_games_on_date
from NHL.Errors import safe_api_call
from NHL.Utils import atomic_write_json, read_json_robust

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent / "static" / "data" / "todays_picks_cache.json"
HISTORY_DIR = Path(__file__).resolve().parent.parent / "static" / "data" / "picks_history"
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


def _slim_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """History keeps only what the picks tab shows: games + value-screened props.

    ``all_props`` (the full priced board) is large and only ever served for the
    current day's props board, so it is dropped from archived dates.
    """
    return {
        "date": payload.get("date"),
        "computed_at": payload.get("computed_at"),
        "games": payload.get("games", []),
        "props": payload.get("props", []),
    }


def _history_path(day: _date) -> Path:
    return HISTORY_DIR / f"{day.isoformat()}.json"


def write_picks_history(day: _date, payload: Dict[str, Any]) -> None:
    """Archive one date's picks as a slim per-date file (written once, never grown)."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(_history_path(day), _slim_payload(payload), indent=2)


def list_picks_history_dates() -> List[str]:
    """Sorted dates that have an archived picks file."""
    if not HISTORY_DIR.exists():
        return []
    out = []
    for p in HISTORY_DIR.glob("*.json"):
        stem = p.stem
        if len(stem) == 10 and stem[4] == "-" and stem[7] == "-":
            out.append(stem)
    return sorted(out)


def load_picks_history(day: _date) -> Optional[Dict[str, Any]]:
    """Load an archived picks payload for a date, or None."""
    path = _history_path(day)
    if not path.exists():
        return None
    try:
        data = read_json_robust(path)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def list_cached_todays_picks_dates(cache_path: Optional[Path] = None) -> List[str]:
    """Return the sorted list of dates with cached picks (live cache + archive).

    Preseason dates are excluded — the picks tab browses regular-season games
    only, so the handful of preseason picks cached before opening night don't
    show up.
    """
    cache_path = Path(cache_path or DEFAULT_CACHE_PATH)
    dates = set(_load_cache_index(cache_path).get("dates", {}).keys())
    dates.update(list_picks_history_dates())
    season_start = _regular_season_start()
    if season_start is not None:
        start_iso = season_start.isoformat()
        dates = {d for d in dates if d >= start_iso}
    return sorted(dates)


def load_cached_todays_picks(
    day: _date,
    cache_path: Optional[Path] = None,
    max_age_hours: float = 24.0,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Load pre-computed picks for a date. Returns (payload, warning).
    warning is set if the cache is missing, has no entry for the date, or is stale.

    The live cache holds the current day's full payload (including the props
    board); older dates are served from the slim per-date archive under
    ``picks_history/``.
    """
    cache_path = Path(cache_path or DEFAULT_CACHE_PATH)
    payload = None

    if cache_path.exists():
        index = _load_cache_index(cache_path)
        payload = index.get("dates", {}).get(day.isoformat())

    if payload is None:
        payload = load_picks_history(day)

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

    # 1. Load odds for edge computation. The on-disk cache is only usable when
    #    it is actually for this date; Render never receives the gitignored
    #    odds_cache.json, so fall back to a live fetch before giving up.
    warning = None
    if odds_payload is None:
        odds_payload, warning = load_cached_odds(day, max_age_hours=24.0)
        if not odds_payload or odds_payload.get("date") != day.isoformat():
            try:
                odds_payload = fetch_and_cache_odds(day)
                warning = None
            except Exception as e:
                logger.warning(f"Live odds fetch failed for {day}: {e}")
                odds_payload = {"events": [], "source": "none"}
                warning = "Live odds unavailable; no value bets computed."
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
    #
    # One walk, two products. Today's Picks wants only the rows where the model
    # disagrees with the book -- a list of bets to make. The props board is a
    # browser over what is on offer, so it wants every priced row, including the
    # ones where the book is right. Computing both from a single pass keeps the
    # second free: the value screen is a filter on the full set, not a rerun.
    props: List[Dict[str, Any]] = []
    all_props: List[Dict[str, Any]] = []
    if include_props:
        try:
            from NHL.PlayerLinePredictor import compute_player_props_for_date
            all_props, props_warning = compute_player_props_for_date(
                day, require_positive_edge=False
            )
            # Mirrors the screen inside compute_player_props_for_date: skater rows
            # are forced to Over, so only saves can come back as an Under.
            props = [
                r for r in all_props
                if r.get("recommendation") == "Under" or (r.get("edge") or 0) > 0
            ]
            all_props = _json_safe(all_props)
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
        "all_props": all_props,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Archive this date's picks as a slim per-date file (games + props, no
    # all_props). One file per day keeps git from rewriting a growing blob and
    # lets old picks be browsed without shipping the whole props board forever.
    write_picks_history(day, payload)

    # The live cache holds only the current day's full payload (including the
    # props board). Overwrite rather than accumulate so it never grows into a
    # multi-megabyte file rewritten every run.
    atomic_write_json(cache_path, {"dates": {day.isoformat(): payload}}, indent=2)

    logger.info(f"Cached today's picks for {day}: {len(games)} games, {len(props)} props -> {cache_path}")
    return payload


# ── ML track record ────────────────────────────────────────────────────

_ARIZONA_TO_UTAH = {"ARI": "UTA"}


def _norm_abbr(abbr: Any) -> str:
    return _ARIZONA_TO_UTAH.get(str(abbr or "").strip().upper(), str(abbr or "").strip().upper())


def _load_game_results() -> Dict[Tuple[str, str, str], Tuple[int, int]]:
    """{(game_date, home_abbr, away_abbr): (home_score, away_score)} from Elo DB."""
    db_path = Path(__file__).resolve().parent.parent / "elo_ratings.db"
    out: Dict[Tuple[str, str, str], Tuple[int, int]] = {}
    if not db_path.exists():
        return out
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT game_date, home_team, away_team, home_score, away_score "
                "FROM game_results WHERE home_score IS NOT NULL AND away_score IS NOT NULL"
            )
            for gd, h, a, hs, aws in cur.fetchall():
                out[(str(gd), _norm_abbr(h), _norm_abbr(a))] = (int(hs), int(aws))
        finally:
            conn.close()
    except sqlite3.Error as e:
        logger.warning(f"Could not read game_results for picks record: {e}")
    return out


def _regular_season_start() -> Optional[_date]:
    """First regular-season game date, or None if the Elo DB is unavailable.

    The NHL encodes the game type in ``game_id`` (digits 5-6): ``01`` is
    preseason, ``02`` is regular season. Preseason games are excluded from the
    ML track record because they don't measure the regular-season predictor.
    """
    db_path = Path(__file__).resolve().parent.parent / "elo_ratings.db"
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT MIN(game_date) FROM game_results "
                "WHERE SUBSTR(game_id, 5, 2) = '02' "
                "AND season = (SELECT MAX(season) FROM game_results)"
            )
            row = cur.fetchone()
            if row and row[0]:
                return _date.fromisoformat(str(row[0]))
        finally:
            conn.close()
    except (sqlite3.Error, ValueError) as e:
        logger.warning(f"Could not determine regular-season start: {e}")
    return None


def compute_picks_record() -> Dict[str, Any]:
    """
    Grade every cached ML pick (the "HOME WIN / AWAY WIN" call) against final
    scores. Preseason games are excluded (they don't measure the regular-season
    predictor), and games without a final score yet are counted as ``pending``
    and excluded from the accuracy percentage.
    """
    results = _load_game_results()
    season_start = _regular_season_start()
    dates = list_cached_todays_picks_dates()

    correct = incorrect = pending = 0
    by_date: List[Dict[str, Any]] = []
    for ds in dates:
        try:
            day = _date.fromisoformat(ds)
        except ValueError:
            continue
        if season_start is not None and day < season_start:
            continue  # preseason — not graded
        payload, _warn = load_cached_todays_picks(day, max_age_hours=10**6)
        if not payload:
            continue

        d_correct = d_incorrect = d_pending = 0
        for g in payload.get("games", []):
            sim = g.get("sim") or {}
            try:
                h_pct = float(sim.get("home_win_pct"))
                a_pct = float(sim.get("away_win_pct"))
            except (TypeError, ValueError):
                continue
            if h_pct == a_pct:
                continue  # no winner called, nothing to grade
            score = results.get((ds, _norm_abbr(g.get("home")), _norm_abbr(g.get("away"))))
            if score is None:
                d_pending += 1
                continue
            predicted_home = h_pct > a_pct
            actual_home_win = score[0] > score[1]
            if predicted_home == actual_home_win:
                d_correct += 1
            else:
                d_incorrect += 1

        if d_correct or d_incorrect or d_pending:
            by_date.append({
                "date": ds,
                "correct": d_correct,
                "incorrect": d_incorrect,
                "pending": d_pending,
            })
        correct += d_correct
        incorrect += d_incorrect
        pending += d_pending

    graded = correct + incorrect
    accuracy_pct = round(100.0 * correct / graded, 1) if graded else None
    return {
        "graded": graded,
        "correct": correct,
        "incorrect": incorrect,
        "pending": pending,
        "accuracy_pct": accuracy_pct,
        "by_date": by_date,
    }
