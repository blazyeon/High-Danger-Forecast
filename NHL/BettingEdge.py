"""
Betting Edge module.

Fetches NHL odds from SharpAPI, caches them locally, removes vig,
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
import unicodedata
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
from NHL.Utils import atomic_write_json, read_json_robust

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent / "static" / "data" / "odds_cache.json"
DEFAULT_EDGE_CACHE_PATH = Path(__file__).resolve().parent.parent / "static" / "data" / "betting_edge_cache.json"
DEFAULT_REGIONS = "us"
DEFAULT_MARKETS = ["h2h", "spreads", "totals"]
EDGE_THRESHOLD = 0.03
# Age at which the underlying odds snapshot is considered too old to power a
# "bet tonight" recommendation without a prominent staleness warning.
ODDS_STALENESS_HOURS = 8.0

# Reverse map from full team name (and common variants) to canonical abbreviation.
_FULL_TO_ABBR: Dict[str, str] = {}
for _abbr, _full in NST_ABBR_TO_FULL.items():
    _key = str(_full).upper().strip()
    if _key not in _FULL_TO_ABBR:
        _FULL_TO_ABBR[_key] = _abbr

# Canonical abbreviations, for spotting one embedded in a longer name.
_KNOWN_ABBRS = set(_FULL_TO_ABBR.values())


def _iso_age_hours(iso_str: Optional[str]) -> Optional[float]:
    """Return the age in hours of an ISO timestamp, or None if unparseable."""
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(str(iso_str).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
    except Exception:
        return None


def odds_staleness_warning(
    payload: Dict[str, Any],
    threshold_hours: float = ODDS_STALENESS_HOURS,
) -> Optional[str]:
    """
    Warn when the odds snapshot underlying a payload is older than `threshold_hours`.

    Edges are computed against the daily odds snapshot; lines move toward game
    time, so an edge measured against hours-old lines is systematically
    overstated. This gives the UI a string to surface before the user bets on it.
    """
    fetched_at = payload.get("odds_fetched_at") or payload.get("computed_at")
    age = _iso_age_hours(fetched_at)
    if age is not None and age > threshold_hours:
        return f"Odds snapshot is {age:.1f} hours old; lines may have moved since."
    return None


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
    # Strip diacritics so "Montréal" matches "Montreal" in the team map.
    raw = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    # Try abbreviation directly (including historical mappings).
    mapped = TEAM_ABBR_MAPPING.get(raw, raw)
    # Try full-team-name reverse lookup.
    full_abbr = _FULL_TO_ABBR.get(mapped, mapped)
    # The odds feed is not consistent about how it names a team: most events carry
    # the full name, but some carry "<ABBR> <Nickname>" -- "MTL Canadiens" and
    # "VGK Golden Knights" both appear alongside "Montreal Canadiens" and "Vegas
    # Golden Knights". Neither form resolves above, and an unresolved name means
    # the game silently never matches an event, so it drops off the board with no
    # explanation. Fall back to whichever word is itself a known abbreviation.
    if " " in full_abbr:
        for token in full_abbr.split():
            if token in _KNOWN_ABBRS:
                return TEAM_ABBR_MAPPING.get(token, token)
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


def _team_key(name: Any) -> str:
    """The nickname of a team name, lowercased and stripped of punctuation.

    The feed names one club several different ways inside a single event: the
    event's own ``home_team``/``away_team`` fields carry "<City> <Nickname>"
    ("Toronto Maple Leafs", "Chicago Blackhawks"), while that same event's
    DraftKings markets label the outcomes "<City-ish> <Nickname>" -- "TOR Maple
    Leafs", "NY Rangers", "CHI Blackhawks". Comparing the strings directly
    matched only the clubs that happened to coincide between the two forms
    (MTL Canadiens, VGK Golden Knights), so the moneyline -- which needs BOTH
    sides to resolve before it is emitted at all -- disappeared from every game
    on the board, and the puck line silently lost whichever side missed.

    The nickname alone is a safe key: it is unique within the league, so
    "Rangers" only ever means New York and "Leafs" only ever means Toronto.
    """
    tokens = "".join(c if c.isalnum() else " " for c in str(name or "")).lower().split()
    return tokens[-1] if tokens else ""


def _same_team(a: Any, b: Any) -> bool:
    """True when two team-name strings refer to the same club -- see _team_key."""
    if str(a or "").strip().lower() == str(b or "").strip().lower():
        return True
    ka = _team_key(a)
    return bool(ka) and ka == _team_key(b)


def _best_outcome(outcomes: List[Dict[str, Any]], side: str) -> Optional[Dict[str, Any]]:
    for o in outcomes or []:
        if _same_team(o.get("name", ""), side):
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

    A favorite (-1.5) covers when it wins by 2+ goals; an underdog (+1.5)
    covers when it does NOT lose by 2+ goals. ``margin_distribution``, when
    present, is keyed by the *home* goal margin (home_goals - away_goals).
    """
    margin_dist = sim.get("margin_distribution")
    if margin_dist:
        total = max(1, sum(margin_dist.values()))
        cover_count = 0
        for margin, count in margin_dist.items():
            try:
                margin = float(margin)
            except Exception:
                continue
            if target_point < 0:
                # This side is the favorite: must win by more than |point|.
                covers = margin > abs(target_point) if is_home else margin < -abs(target_point)
            else:
                # This side is the underdog: covers unless it loses by > point.
                covers = margin > -target_point if is_home else margin < target_point
            cover_count += count if covers else 0
        return cover_count / total

    # No margin distribution: fall back to the win-by-2+ summary probabilities.
    home_win_2plus = float(sim.get("home_win_2plus_pct", 25.0)) / 100.0
    away_win_2plus = float(sim.get("away_win_2plus_pct", 25.0)) / 100.0
    if target_point < 0:
        # Favorite must win by 2+.
        return home_win_2plus if is_home else away_win_2plus
    # Underdog covers unless it loses by 2+.
    return (1.0 - away_win_2plus) if is_home else (1.0 - home_win_2plus)


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


def _passes_screen(edge: float, threshold: Optional[float]) -> bool:
    """``None`` means keep everything -- see ``compute_game_edges``."""
    return threshold is None or edge > threshold


def compute_game_edges(
    game: Dict[str, Any],
    event: Dict[str, Any],
    sim: Dict[str, Any],
    edge_threshold: Optional[float] = EDGE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Compare model probabilities to no-vig implied probabilities for one game.
    Returns edge dicts sorted by absolute edge descending.

    ``edge_threshold=None`` disables the screen entirely and keeps every line.
    The Game Bet board is a comparison of model vs. implied probability for every
    market, not a list of bets, so it passes ``None``; Today's Picks passes
    ``EDGE_THRESHOLD`` and keeps the value screen.
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
            # Moneyline is only shown for the side the model gives over 50% to,
            # even when the underdog carries the larger edge.
            if home_win_pct > 0.5:
                side, model_p, imp_p, out, team = home_name, home_win_pct, home_imp, home_out, home
            elif away_win_pct > 0.5:
                side, model_p, imp_p, out, team = away_name, away_win_pct, away_imp, away_out, away
            else:
                side = None
            if side is not None:
                edge = model_p - imp_p
                if _passes_screen(edge, edge_threshold):
                    edges.append(_edge_dict(
                        market="Moneyline",
                        side=side,
                        pick=side,
                        team=team,
                        odds=out.get("price"),
                        odds_decimal=_decimal_price(out),
                        model_prob=model_p,
                        implied_prob=imp_p,
                        edge=edge,
                        book=h2h.get("book_key"),
                    ))

    # ── Puck Line / Spreads ──────────────────────────────────────────────
    spreads = _best_book_market(event, "spreads")
    if spreads:
        m = spreads["market"]
        # Compute a candidate row for both sides, then keep only the larger
        # edge. The two sides are exact complements (model probs sum to 1,
        # no-vig probs sum to 1), so max-edge is the model's preferred side.
        best_puck_line = None
        for out in m.get("outcomes", []) or []:
            point = out.get("point")
            price = out.get("price")
            if point is None or price is None:
                continue
            if abs(float(point)) != 1.5:
                continue
            side_name = str(out.get("name", "")).strip()
            # Same "<ABBR> <Nickname>" mismatch as _best_outcome: an unparsed
            # side is skipped outright, so the surviving side of the pair was
            # whichever one the feed happened to spell the long way.
            is_home = _same_team(side_name, home_name)
            is_away = _same_team(side_name, away_name)
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
            row = _edge_dict(
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
            )
            if best_puck_line is None or row["edge"] > best_puck_line["edge"]:
                best_puck_line = row
        if best_puck_line is not None and _passes_screen(best_puck_line["edge"], edge_threshold):
            edges.append(best_puck_line)

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
            # Keep only the larger-edge side: one row per line, on the side the
            # model prefers (Over and Under are exact complements).
            side, edge, model_p, imp_p, out = max(
                (
                    ("Over", over_edge, model_over, over_imp, over_out),
                    ("Under", under_edge, model_under, under_imp, under_out),
                ),
                key=lambda cand: cand[1],
            )
            if _passes_screen(edge, edge_threshold):
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
        "source": "sharpapi",
        "events": data,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(cache_path, payload, indent=2)

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
        payload = read_json_robust(cache_path)
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


def compute_and_cache_edges(
    day: _date,
    odds_payload: Optional[Dict[str, Any]] = None,
    edge_threshold: Optional[float] = None,
    cache_path: Optional[Path] = None,
    sims: int = 1000,
    use_events_schedule: bool = False,
) -> Dict[str, Any]:
    """
    Pre-compute betting edges for a date and write them to a local JSON cache.
    This is designed to run during the daily update so the UI opens instantly.

    The default is the full board: every matched game with every line. Callers
    pass ``EDGE_THRESHOLD`` if they want the value screen instead.

    There is no fixture fallback: with no live odds this raises rather than
    inventing a slate. Edges are recommendations to stake money, so anything
    that reaches a caller has to trace back to real prices.
    """
    cache_path = Path(cache_path or DEFAULT_EDGE_CACHE_PATH)

    # 1. Load odds (use provided payload, then the on-disk cache).
    warning = None
    if odds_payload is None:
        odds_payload, warning = load_cached_odds(day, max_age_hours=24.0)
        if odds_payload is None:
            raise OddsAPIError(f"No live odds available for {day.isoformat()}.")

    events = odds_payload.get("events", [])

    # 2. Load schedule for the date. On a no-games day we keep the empty slate
    #    rather than fabricating one from the odds events.
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
            schedule_games = _schedule_from_events(events)

    # 3. Build slate matchups. Track how many schedule games we scanned and how
    #    many matched an odds event, so silent drops are visible in the payload.
    scheduled_count = 0
    matched_count = 0
    no_odds_games: List[Dict[str, Any]] = []
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
        scheduled_count += 1

        schedule_game = {
            "home": home_abbr,
            "away": away_abbr,
            "home_name": get_team_full_name(home_team),
            "away_name": get_team_full_name(away_team),
            "startTime": game.get("startTimeUTC", game.get("gameDate", "")),
        }

        event = find_event_for_game(schedule_game, events)
        if not event:
            logger.warning(
                f"No odds event matched schedule game {home_abbr} v {away_abbr} "
                f"({schedule_game.get('away_name')} @ {schedule_game.get('home_name')}); dropped."
            )
            no_odds_games.append({
                "home": home_abbr,
                "away": away_abbr,
                "home_name": schedule_game["home_name"],
                "away_name": schedule_game["away_name"],
                "start_time": schedule_game["startTime"],
            })
            continue

        matched_count += 1
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
    board_games = []
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
        # A game with no disagreement between model and market still belongs on
        # the board -- the board compares the two, it is not a list of bets.
        edges.sort(key=lambda e: e.get("edge", 0.0), reverse=True)
        best_edge = max(edges, key=lambda e: e.get("edge", 0.0), default={"edge": 0.0})
        board_games.append({
            "home": home_abbr,
            "away": away_abbr,
            "home_name": schedule_game["home_name"],
            "away_name": schedule_game["away_name"],
            "start_time": schedule_game["startTime"],
            "best_edge": best_edge.get("edge", 0.0),
            "edges": edges,
        })

    board_games.sort(key=lambda g: abs(g.get("best_edge", 0.0)), reverse=True)

    payload = {
        "date": day.isoformat(),
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "source": odds_payload.get("source", "unknown"),
        "odds_fetched_at": odds_payload.get("fetched_at"),
        "warning": warning,
        "no_games": scheduled_count == 0,
        "scanned": scheduled_count,
        "matched": matched_count,
        "with_edges": sum(1 for g in board_games if any((e.get("edge") or 0.0) > 0 for e in g.get("edges", []))),
        "dropped": scheduled_count - matched_count,
        "games": board_games,
        "no_odds_games": no_odds_games,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    index = _load_edge_cache_index(cache_path)
    index.setdefault("dates", {})[day.isoformat()] = payload
    atomic_write_json(cache_path, index, indent=2)

    logger.info(f"Cached betting edges for {day}: {len(board_games)} games -> {cache_path}")
    return payload


def _load_edge_cache_index(cache_path: Path) -> Dict[str, Any]:
    """
    Load the multi-date edge cache file, normalizing a legacy single-date
    payload into the ``{"dates": {date: payload}}`` shape.
    """
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
    # Legacy single-date payload: wrap it so it can be merged into.
    if "date" in data and "games" in data:
        return {"dates": {data["date"]: data}}
    return {}


def list_cached_edge_dates(cache_path: Optional[Path] = None) -> List[str]:
    """Return the sorted list of dates that have cached betting edges."""
    cache_path = Path(cache_path or DEFAULT_EDGE_CACHE_PATH)
    index = _load_edge_cache_index(cache_path)
    return sorted(index.get("dates", {}).keys())


def load_cached_edges(
    day: _date,
    cache_path: Optional[Path] = None,
    max_age_hours: float = 24.0,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Load pre-computed betting edge cache for a date. Returns (payload, warning).
    warning_message is set if the cache is missing, has no entry for the date,
    or the entry is stale.
    """
    cache_path = Path(cache_path or DEFAULT_EDGE_CACHE_PATH)
    if not cache_path.exists():
        return None, f"No cached edges found. Run `python update_odds.py --date {day.isoformat()}`."

    index = _load_edge_cache_index(cache_path)
    payload = index.get("dates", {}).get(day.isoformat())
    if payload is None:
        return None, f"No cached edges for {day.isoformat()}. Run `python update_odds.py --date {day.isoformat()}`."

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


def _game_started(start_time: Optional[str], now: Optional[datetime] = None) -> bool:
    """Return True if a game's start time is in the past (market closed)."""
    if not start_time:
        return False
    try:
        dt = datetime.fromisoformat(str(start_time).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = now or datetime.now(timezone.utc)
        return dt < now
    except Exception:
        return False


def drop_started_games(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """
    Remove games whose start time has already passed from an edge payload.

    Edges are computed against pre-game odds; once a game puck-drops those lines are
    no longer bettable. Returns ``(filtered_payload, started_count)``.
    """
    games = payload.get("games", []) or []
    started = [g for g in games if _game_started(g.get("start_time"))]
    if not started:
        return payload, 0
    remaining = [g for g in games if not _game_started(g.get("start_time"))]
    out = dict(payload)
    out["games"] = remaining
    return out, len(started)
