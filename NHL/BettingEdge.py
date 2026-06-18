"""
Betting Edge module.

Fetches NHL odds from The Odds API, caches them locally, removes vig,
and compares no-vig implied probabilities to model probabilities to
surface value bets.

Markets covered:
  - h2h (moneyline)
  - spreads (puck line, typically +/- 1.5)
  - totals (over/under)
"""
from __future__ import annotations

import json
import logging
from datetime import date as _date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from NHL.Config import TEAM_ABBR_MAPPING, NST_ABBR_TO_FULL
from NHL.OddsAPI import fetch_nhl_odds_by_date, OddsAPIError
from NHL.PlayerLinePredictor import american_to_decimal
from NHL.Simulation import simulate_slate
from NHL.Lookup import get_team_full_name, display_abbr_for_game
from NHL.ApiScrape import get_games_on_date
from NHL.Errors import safe_api_call

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent / "static" / "data" / "odds_cache.json"
DEFAULT_DEMO_PATH = Path(__file__).resolve().parent.parent / "static" / "data" / "demo_odds.json"
DEFAULT_EDGE_CACHE_PATH = Path(__file__).resolve().parent.parent / "static" / "data" / "betting_edge_cache.json"
DEFAULT_REGIONS = "us"
DEFAULT_MARKETS = ["h2h", "spreads", "totals"]
EDGE_THRESHOLD = 0.03

# Reverse map from full team name (and common variants) to canonical abbreviation.
_FULL_TO_ABBR: Dict[str, str] = {}
for _abbr, _full in NST_ABBR_TO_FULL.items():
    _key = str(_full).upper().strip()
    if _key not in _FULL_TO_ABBR:
        _FULL_TO_ABBR[_key] = _abbr


def implied_probability(decimal_odds: float) -> float:
    """Decimal odds -> implied probability in [0, 1]."""
    try:
        if not decimal_odds or decimal_odds <= 1.0:
            return 0.0
        return 1.0 / decimal_odds
    except Exception:
        return 0.0


def remove_vig_2way(p1: float, p2: float) -> Tuple[float, float]:
    """
    Normalize two implied probabilities so they sum to 1.0.
    Returns (true_p1, true_p2). If one side is missing or the total is zero,
    returns (0, 0).
    """
    p1 = max(0.0, p1)
    p2 = max(0.0, p2)
    if p1 == 0.0 or p2 == 0.0:
        return 0.0, 0.0
    total = p1 + p2
    if total <= 0:
        return 0.0, 0.0
    return p1 / total, p2 / total


def _normalize_abbr(abbr: str) -> str:
    """Return canonical team abbreviation, accepting either abbr or full name."""
    raw = str(abbr).upper().strip()
    # Try abbreviation directly (including historical mappings).
    mapped = TEAM_ABBR_MAPPING.get(raw, raw)
    # Try full-team-name reverse lookup.
    full_abbr = _FULL_TO_ABBR.get(mapped, mapped)
    # Re-apply historical mapping in case reverse lookup returned an old abbr.
    return TEAM_ABBR_MAPPING.get(full_abbr, full_abbr)


def _schedule_from_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build a minimal schedule from odds events (used as an offseason fallback)."""
    games: List[Dict[str, Any]] = []
    for ev in events or []:
        home = ev.get("home_team")
        away = ev.get("away_team")
        if not home or not away:
            continue
        home_key = str(home).upper().strip()
        away_key = str(away).upper().strip()
        home_abbr = _FULL_TO_ABBR.get(home_key, home_key)
        away_abbr = _FULL_TO_ABBR.get(away_key, away_key)
        games.append({
            "id": ev.get("id"),
            "homeTeam": {"abbrev": home_abbr, "name": {"default": home}},
            "awayTeam": {"abbrev": away_abbr, "name": {"default": away}},
            "startTimeUTC": ev.get("commence_time"),
            "gameState": "FUT",
        })
    return games


def find_event_for_game(game: Dict[str, Any], events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Match a schedule game dict {home, away} to an Odds API event."""
    home = _normalize_abbr(game.get("home") or game.get("home_team", ""))
    away = _normalize_abbr(game.get("away") or game.get("away_team", ""))
    for ev in events or []:
        ev_home = _normalize_abbr(ev.get("home_team") or "")
        ev_away = _normalize_abbr(ev.get("away_team") or "")
        if home and away and ev_home == home and ev_away == away:
            return ev
    return None


def _best_outcome(outcomes: List[Dict[str, Any]], side: str) -> Optional[Dict[str, Any]]:
    for o in outcomes or []:
        if str(o.get("name", "")).strip().lower() == side.lower():
            return o
    return None


def _decimal_price(outcome: Optional[Dict[str, Any]]) -> Optional[float]:
    """Return decimal odds for an outcome, accepting either decimal or American prices."""
    if not outcome:
        return None
    price = outcome.get("price")
    if price is None:
        return None
    try:
        p = float(price)
    except Exception:
        return None
    # Odds API returns American-style integers (e.g. -135, +220). Convert them.
    # Use is_integer() to avoid float precision issues with values like 100.0.
    if abs(p) >= 100.0 and p.is_integer():
        return american_to_decimal(p)
    # Otherwise treat as decimal odds.
    if p <= 1.0:
        return None
    return p


def _best_book_market(event: Dict[str, Any], market_key: str) -> Optional[Dict[str, Any]]:
    """Return the bookmaker market with the lowest total vig."""
    best = None
    best_vig = float("inf")
    for bk in event.get("bookmakers", []) or []:
        for m in bk.get("markets", []) or []:
            if m.get("key") != market_key:
                continue
            outcomes = m.get("outcomes", []) or []
            decs = [_decimal_price(o) for o in outcomes]
            decs = [d for d in decs if d]
            if len(decs) < 2:
                continue
            total_impl = sum(1.0 / d for d in decs)
            if total_impl < best_vig:
                best = {
                    "book_key": bk.get("key"),
                    "book_title": bk.get("title"),
                    "market": m,
                }
                best_vig = total_impl
    return best


def _home_outcome_name(event: Dict[str, Any]) -> str:
    """The Odds API uses team names as outcome names; map home_team to its outcome."""
    return str(event.get("home_team") or "Home").strip()


def _away_outcome_name(event: Dict[str, Any]) -> str:
    return str(event.get("away_team") or "Away").strip()


def _model_prob_for_total(totals_dist: Dict[int, int], line: float) -> Tuple[float, float]:
    """Model probability of over / under the given total line."""
    total_sims = max(1, sum(totals_dist.values()))
    over_count = sum(c for t, c in totals_dist.items() if t > line)
    under_count = sum(c for t, c in totals_dist.items() if t < line)
    push_count = sum(c for t, c in totals_dist.items() if t == line)
    if push_count:
        over_count += push_count / 2.0
        under_count += push_count / 2.0
    return over_count / total_sims, under_count / total_sims


def _model_prob_for_spread(
    sim: Dict[str, Any],
    target_point: float,
    is_home: bool,
) -> float:
    """
    Model probability of covering the puck line / spread.

    If the simulation output contains a full margin distribution we use it
    directly. Otherwise we fall back to the existing win-by-2+ proxy.
    """
    margin_dist = sim.get("margin_distribution")
    if not margin_dist:
        fallback = "home_win_2plus_pct" if is_home else "away_win_2plus_pct"
        return float(sim.get(fallback, 25.0)) / 100.0

    total = max(1, sum(margin_dist.values()))
    # Puck lines in NHL are typically +/- 1.5. A home team covers -1.5 when
    # home goals - away goals >= 2, and +1.5 when home goals - away goals >= -1.
    cover_count = 0
    for margin, count in margin_dist.items():
        try:
            margin = float(margin)
        except Exception:
            continue
        if target_point < 0:
            # Favored team must win by more than the absolute spread.
            cover_count += count if margin > abs(target_point) else 0
        else:
            # Underdog covers if margin is better than the spread (i.e. > -spread).
            cover_count += count if margin > -target_point else 0
    return cover_count / total


def _edge_dict(**kwargs) -> Dict[str, Any]:
    return {
        "market": kwargs.get("market"),
        "side": kwargs.get("side"),
        "pick": kwargs.get("pick"),
        "team": kwargs.get("team"),
        "odds": kwargs.get("odds"),
        "odds_decimal": kwargs.get("odds_decimal"),
        "model_prob": round(kwargs.get("model_prob", 0.0), 4),
        "implied_prob": round(kwargs.get("implied_prob", 0.0), 4),
        "edge": round(kwargs.get("edge", 0.0), 4),
        "book": kwargs.get("book"),
    }


def compute_game_edges(
    game: Dict[str, Any],
    event: Dict[str, Any],
    sim: Dict[str, Any],
    edge_threshold: float = EDGE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Compare model probabilities to no-vig implied probabilities for one game.
    Returns edge dicts sorted by absolute edge descending.
    """
    edges: List[Dict[str, Any]] = []

    home = _normalize_abbr(game.get("home") or game.get("home_team", ""))
    away = _normalize_abbr(game.get("away") or game.get("away_team", ""))
    home_name = _home_outcome_name(event)
    away_name = _away_outcome_name(event)

    home_win_pct = float(sim.get("home_win_pct", 50.0)) / 100.0
    away_win_pct = float(sim.get("away_win_pct", 50.0)) / 100.0
    home_win_2plus = float(sim.get("home_win_2plus_pct", 25.0)) / 100.0
    away_win_2plus = float(sim.get("away_win_2plus_pct", 25.0)) / 100.0
    totals_dist = sim.get("totals_distribution") or {}

    # ── Moneyline ──────────────────────────────────────────────────────
    h2h = _best_book_market(event, "h2h")
    if h2h:
        m = h2h["market"]
        home_out = _best_outcome(m.get("outcomes", []), home_name)
        away_out = _best_outcome(m.get("outcomes", []), away_name)
        home_dec = _decimal_price(home_out)
        away_dec = _decimal_price(away_out)
        if home_dec and away_dec:
            home_imp, away_imp = remove_vig_2way(implied_probability(home_dec), implied_probability(away_dec))
            home_edge = home_win_pct - home_imp
            away_edge = away_win_pct - away_imp
            for side, edge, model_p, imp_p, out, team in (
                (home_name, home_edge, home_win_pct, home_imp, home_out, home),
                (away_name, away_edge, away_win_pct, away_imp, away_out, away),
            ):
                if edge > edge_threshold:
                    dec = _decimal_price(out)
                    edges.append(_edge_dict(
                        market="Moneyline",
                        side=side,
                        pick=side,
                        team=team,
                        odds=out.get("price"),
                        odds_decimal=dec,
                        model_prob=model_p,
                        implied_prob=imp_p,
                        edge=edge,
                        book=h2h.get("book_key"),
                    ))

    # ── Puck Line / Spreads ──────────────────────────────────────────────
    spreads = _best_book_market(event, "spreads")
    if spreads:
        m = spreads["market"]
        for out in m.get("outcomes", []) or []:
            point = out.get("point")
            price = out.get("price")
            if point is None or price is None:
                continue
            if abs(float(point)) != 1.5:
                continue
            side_name = str(out.get("name", "")).strip()
            is_home = side_name.lower() == home_name.lower()
            is_away = side_name.lower() == away_name.lower()
            if not is_home and not is_away:
                continue
            model_p = _model_prob_for_spread(sim, float(point), is_home)
            dec = _decimal_price(out) or american_to_decimal(price)
            if not dec:
                continue
            other_name = away_name if is_home else home_name
            other_out = _best_outcome(m.get("outcomes", []), other_name)
            other_dec = _decimal_price(other_out)
            if other_dec:
                true_p, _ = remove_vig_2way(implied_probability(dec), implied_probability(other_dec))
            else:
                true_p = implied_probability(dec)
            edge = model_p - true_p
            if edge > edge_threshold:
                edges.append(_edge_dict(
                    market=f"Puck Line ({point})",
                    side=side_name,
                    pick=side_name,
                    team=home if is_home else away,
                    odds=price,
                    odds_decimal=dec,
                    model_prob=model_p,
                    implied_prob=true_p,
                    edge=edge,
                    book=spreads.get("book_key"),
                ))

    # ── Totals ───────────────────────────────────────────────────────────
    totals = _best_book_market(event, "totals")
    if totals and totals_dist:
        m = totals["market"]
        # Some books offer multiple total lines; evaluate every distinct line
        # that has both Over and Under prices available.
        by_line: Dict[float, Dict[str, Any]] = {}
        for out in m.get("outcomes", []) or []:
            if out.get("point") is None:
                continue
            line_val = float(out["point"])
            side = str(out.get("name", "")).strip()
            if side not in ("Over", "Under"):
                continue
            if line_val not in by_line:
                by_line[line_val] = {"Over": None, "Under": None}
            by_line[line_val][side] = out

        for line, sides in by_line.items():
            over_out = sides.get("Over")
            under_out = sides.get("Under")
            if not over_out or not under_out:
                continue
            over_dec = _decimal_price(over_out) or american_to_decimal(over_out.get("price"))
            under_dec = _decimal_price(under_out) or american_to_decimal(under_out.get("price"))
            if not over_dec or not under_dec:
                continue
            over_imp, under_imp = remove_vig_2way(implied_probability(over_dec), implied_probability(under_dec))
            model_over, model_under = _model_prob_for_total(totals_dist, line)
            over_edge = model_over - over_imp
            under_edge = model_under - under_imp
            for side, edge, model_p, imp_p, out in (
                ("Over", over_edge, model_over, over_imp, over_out),
                ("Under", under_edge, model_under, under_imp, under_out),
            ):
                if edge > edge_threshold:
                    dec = _decimal_price(out) or american_to_decimal(out.get("price"))
                    edges.append(_edge_dict(
                        market=f"Total {line}",
                        side=side,
                        pick=side,
                        team=None,
                        odds=out.get("price"),
                        odds_decimal=dec,
                        model_prob=model_p,
                        implied_prob=imp_p,
                        edge=edge,
                        book=totals.get("book_key"),
                    ))

    edges.sort(key=lambda e: abs(e["edge"]), reverse=True)
    return edges


def fetch_and_cache_odds(
    day: _date,
    cache_path: Optional[Path] = None,
    regions: str = DEFAULT_REGIONS,
    markets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fetch featured NHL odds for the date and write them to the local cache."""
    if markets is None:
        markets = list(DEFAULT_MARKETS)
    cache_path = Path(cache_path or DEFAULT_CACHE_PATH)

    data = fetch_nhl_odds_by_date(
        day=day,
        regions=regions,
        markets=markets,
        odds_format="american",
    )

    payload = {
        "date": day.isoformat(),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "the-odds-api",
        "events": data,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(payload, f, default=str, indent=2)

    logger.info(f"Cached odds for {day}: {len(data)} events -> {cache_path}")
    return payload


def load_cached_odds(
    day: _date,
    cache_path: Optional[Path] = None,
    max_age_hours: float = 6.0,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Load cached odds. Returns (payload, warning_message).
    warning_message is set if the cache is missing, wrong date, or stale.
    """
    cache_path = Path(cache_path or DEFAULT_CACHE_PATH)
    if not cache_path.exists():
        return None, f"No cached odds found. Run `python update_odds.py --date {day.isoformat()}`."

    try:
        with open(cache_path, "r") as f:
            payload = json.load(f)
    except Exception as e:
        return None, f"Could not read cached odds: {e}"

    if payload.get("date") != day.isoformat():
        return payload, f"Cached odds are for {payload.get('date')}, not {day.isoformat()}."

    fetched_at = payload.get("fetched_at")
    if fetched_at:
        try:
            fetched_dt = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
            age = datetime.now(timezone.utc) - fetched_dt
            if age > timedelta(hours=max_age_hours):
                return payload, f"Odds cache is {age.total_seconds() / 3600:.1f} hours old."
        except Exception:
            pass

    return payload, None


def load_demo_odds(demo_path: Optional[Path] = None) -> Dict[str, Any]:
    demo_path = Path(demo_path or DEFAULT_DEMO_PATH)
    if not demo_path.exists():
        return {"date": _date.today().isoformat(), "fetched_at": None, "source": "demo", "events": []}
    with open(demo_path, "r") as f:
        return json.load(f)


# Path to the shared offseason demo props/odds fixture.
DEFAULT_DEMO_FIXTURE_PATH = Path(__file__).resolve().parent.parent / "static" / "data" / "demo_props.json"


def load_demo_schedule(demo_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return a synthetic NHL schedule from the demo fixture (offseason testing)."""
    demo_path = Path(demo_path or DEFAULT_DEMO_FIXTURE_PATH)
    if not demo_path.exists():
        return []
    try:
        with open(demo_path, "r") as f:
            data = json.load(f)
    except Exception:
        return []

    games = []
    for g in data.get("games", []):
        games.append({
            "id": g.get("event_id"),
            "homeTeam": {"abbrev": g.get("home_team"), "name": {"default": g.get("home_team")}},
            "awayTeam": {"abbrev": g.get("away_team"), "name": {"default": g.get("away_team")}},
            "startTimeUTC": g.get("commence_time"),
            "gameState": "FUT",
        })
    return games


def compute_and_cache_edges(
    day: _date,
    odds_payload: Optional[Dict[str, Any]] = None,
    edge_threshold: float = EDGE_THRESHOLD,
    cache_path: Optional[Path] = None,
    sims: int = 1000,
    use_events_schedule: bool = False,
) -> Dict[str, Any]:
    """
    Pre-compute betting edges for a date and write them to a local JSON cache.
    This is designed to run during the daily update so the UI opens instantly.
    """
    cache_path = Path(cache_path or DEFAULT_EDGE_CACHE_PATH)

    # 1. Load odds (use provided payload, then cache, then demo).
    warning = None
    if odds_payload is None:
        odds_payload, warning = load_cached_odds(day, max_age_hours=24.0)
        if odds_payload is None:
            odds_payload = load_demo_odds(DEFAULT_DEMO_PATH)
            warning = "Using demo odds (no live odds cached)."

    events = odds_payload.get("events", [])

    # 2. Load schedule for the date.
    if use_events_schedule:
        warning = warning or "Using odds event matchups."
        schedule_games = _schedule_from_events(events)
    else:
        schedule_games = safe_api_call(
            get_games_on_date, day.isoformat(),
            service_name="NHL Schedule API", fallback=[],
        )
        if not schedule_games:
            warning = warning or "No live schedule found; using odds event matchups."
            schedule_games = load_demo_schedule()
            if not schedule_games:
                schedule_games = _schedule_from_events(events)

    # 3. Build slate matchups.
    slate_matchups: List[Tuple[str, str]] = []
    slate_games: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for game in schedule_games or []:
        home_team = game.get("homeTeam") or {}
        away_team = game.get("awayTeam") or {}
        home_abbr = display_abbr_for_game(
            home_team.get("abbrev", home_team.get("name", ""))
        )
        away_abbr = display_abbr_for_game(
            away_team.get("abbrev", away_team.get("name", ""))
        )
        if not home_abbr or not away_abbr:
            continue

        schedule_game = {
            "home": home_abbr,
            "away": away_abbr,
            "home_name": get_team_full_name(home_team),
            "away_name": get_team_full_name(away_team),
            "startTime": game.get("startTimeUTC", game.get("gameDate", "")),
        }

        event = find_event_for_game(schedule_game, events)
        if not event:
            continue

        key = (home_abbr, away_abbr)
        slate_matchups.append(key)
        slate_games[key] = {"schedule_game": schedule_game, "event": event}

    # 4. Simulate slate.
    slate_results = simulate_slate(
        game_date=day,
        matchups=slate_matchups,
        stype=2,
        sims=sims,
        trend_games=25,
        use_recent_window_days=14,
    )

    # 5. Compute edges.
    value_games = []
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
        if not event:
            continue

        edges = compute_game_edges(schedule_game, event, sim, edge_threshold=edge_threshold)
        if not edges:
            continue

        edges.sort(key=lambda e: e.get("edge", 0.0), reverse=True)
        best_edge = max(edges, key=lambda e: e.get("edge", 0.0))
        value_games.append({
            "home": home_abbr,
            "away": away_abbr,
            "home_name": schedule_game["home_name"],
            "away_name": schedule_game["away_name"],
            "start_time": schedule_game["startTime"],
            "best_edge": best_edge.get("edge", 0.0),
            "edges": edges,
        })

    value_games.sort(key=lambda g: abs(g.get("best_edge", 0.0)), reverse=True)

    payload = {
        "date": day.isoformat(),
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "source": odds_payload.get("source", "unknown"),
        "warning": warning,
        "games": value_games,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(payload, f, default=str, indent=2)

    logger.info(f"Cached betting edges for {day}: {len(value_games)} games -> {cache_path}")
    return payload


def load_cached_edges(
    day: _date,
    cache_path: Optional[Path] = None,
    max_age_hours: float = 24.0,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Load pre-computed betting edge cache. Returns (payload, warning_message).
    warning_message is set if the cache is missing, wrong date, or stale.
    """
    cache_path = Path(cache_path or DEFAULT_EDGE_CACHE_PATH)
    if not cache_path.exists():
        return None, f"No cached edges found. Run `python update_odds.py --date {day.isoformat()}`."

    try:
        with open(cache_path, "r") as f:
            payload = json.load(f)
    except Exception as e:
        return None, f"Could not read cached edges: {e}"

    if payload.get("date") != day.isoformat():
        return payload, f"Cached edges are for {payload.get('date')}, not {day.isoformat()}."

    computed_at = payload.get("computed_at")
    if computed_at:
        try:
            computed_dt = datetime.fromisoformat(computed_at.replace("Z", "+00:00"))
            age = datetime.now(timezone.utc) - computed_dt
            if age > timedelta(hours=max_age_hours):
                return payload, f"Edge cache is {age.total_seconds() / 3600:.1f} hours old."
        except Exception:
            pass

    return payload, None
