"""
Player Line Predictor — hit probability calculations:
- Fetches NHL player prop lines via SharpAPI
- Calculates hit probability using player Elo + NST stats
- Sorts by most likely to hit
- Returns recommended bets (Over/Under)

Pure computation module; no Streamlit dependency.
"""
from __future__ import annotations

import functools
import json
import logging
import math
import pandas as pd
from datetime import date as _date
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import difflib

from NHL.OddsAPI import fetch_nhl_player_props_by_date, OddsAPIError
from NHL.Utils import normalize_name_key, season_from_date
from NHL.StatsFromPBP import load_skater_rates_from_json, load_goalie_rates_from_json
from NHL.Config import NST_ABBR_TO_FULL, TEAM_ABBR_MAPPING
from EloMl.Database import EloDatabase
# NST import removed — see get_player_pbp_stats below for the new source.

# Reverse map from full team name to canonical abbreviation (same pattern as BettingEdge).
_FULL_TO_ABBR: Dict[str, str] = {}
for _abbr, _full in NST_ABBR_TO_FULL.items():
    _key = str(_full).upper().strip()
    if _key not in _FULL_TO_ABBR:
        _FULL_TO_ABBR[_key] = _abbr


def _normalize_team_abbr(value: str) -> str:
    """Return canonical team abbreviation, accepting either abbr or full name."""
    raw = str(value).upper().strip()
    mapped = TEAM_ABBR_MAPPING.get(raw, raw)
    full_abbr = _FULL_TO_ABBR.get(mapped, mapped)
    return TEAM_ABBR_MAPPING.get(full_abbr, full_abbr)


logger = logging.getLogger(__name__)

# Default player markets to fetch
DEFAULT_PLAYER_MARKETS = [
    "player_points",
    "player_assists",
    # Shots-on-goal is a two-sided market (DraftKings quotes Over AND Under on
    # its main line), so the model can price it honestly -- unlike player_goals.
    "player_shots_on_goal",
    # Goals are priced as "anytime goal scorer", not as an Over/Under line.
    # The plain player_goals market is an Over-only 2+/3+ ladder with no
    # two-sided rung anywhere (verified: 0 of 179 lines carry an Under), and
    # DraftKings still stamps is_main_line=True on the 1.5 rung, so that flag
    # cannot filter it. Its fitted tails are also noise -- median edge ~0.9%
    # against ~1.7% for anytime scorer, with depth defencemen showing as
    # 2-goal threats -- so it is left off entirely.
    "anytime_goal_scorer",
    # Blocked shots and power-play points are left off, for different reasons.
    # DraftKings does post blocked shots (87 rows on a 5-game slate), but the PBP
    # shot store carries no blocked-shot data, so every row prices as the 50.0
    # "No Data" sentinel and is filtered out before ranking -- it can never show.
    # Power-play points is not posted at all (0 rows from DraftKings and FanDuel
    # across every slate checked), so requesting it only spends rate limit.
    "player_total_saves",
]


def american_to_decimal(price: float) -> Optional[float]:
    try:
        p = float(price)
        if p > 0:
            return 1.0 + (p / 100.0)
        elif p < 0:
            return 1.0 + (100.0 / abs(p))
        else:
            return None
    except Exception:
        return None


def decimal_to_american(dec: float) -> Optional[int]:
    try:
        d = float(dec)
        if d <= 1.0:
            return None
        if d >= 2.0:
            return int(round((d - 1.0) * 100.0))
        return int(round(-100.0 / (d - 1.0)))
    except Exception:
        return None


def implied_probability(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability (0-100%)."""
    try:
        return (1.0 / decimal_odds) * 100.0
    except (ZeroDivisionError, TypeError, ValueError):
        return 50.0


@functools.lru_cache(maxsize=None)
def _load_player_props_for_day_uncached(
    day: _date,
    regions: str,
    markets: Tuple[str, ...],
    bookmakers_csv: Optional[str],
    odds_format: str = "american",
) -> Tuple[Tuple[Dict[str, Any], ...], Tuple[str, ...]]:
    """
    Fetch player props for exactly the requested league day.

    This used to fetch ``day + 1`` as well, as a timezone guard: if a late local
    game were the API's "tomorrow", today's window would miss it. That guard is
    obsolete and now actively harmful. The fetch is already windowed to Eastern
    midnight boundaries (``_day_window``), so a 10 PM ET game belongs to its own
    league day and tomorrow's games belong to tomorrow. Fetching the next day
    anyway spliced the next slate into today's board -- the props tab listed
    games with no matchup on the date it claimed to cover.

    Returns ``(results, errors)`` so callers can distinguish "no props for this date"
    from "live odds were unavailable" (quota exhausted / bad key / network failure).

    Note: ``markets`` is accepted as a tuple so that lru_cache can hash it.
    Callers passing a list should convert via ``tuple(markets)``.
    """
    try:
        props = fetch_nhl_player_props_by_date(
            day=day,
            regions=regions,
            markets=list(markets),
            bookmakers_csv=bookmakers_csv,
            odds_format=odds_format,
        )
    except OddsAPIError as e:
        logger.warning("Odds API error for %s: %s", day, e)
        return (), (str(e),)
    except Exception as e:
        logger.warning("Error fetching props for %s: %s", day, e)
        return (), (str(e),)

    return tuple(props), ()


def load_player_props_for_day(
    day: _date,
    regions: str,
    markets: Tuple[str, ...],
    bookmakers_csv: Optional[str],
    odds_format: str = "american",
) -> Tuple[Dict[str, Any], ...]:
    """Fetch player props for the requested day (results only)."""
    results, _ = _load_player_props_for_day_uncached(
        day, regions, markets, bookmakers_csv, odds_format
    )
    return results


def load_player_props_for_day_with_status(
    day: _date,
    regions: str,
    markets: Tuple[str, ...],
    bookmakers_csv: Optional[str],
    odds_format: str = "american",
) -> Tuple[Tuple[Dict[str, Any], ...], Tuple[str, ...]]:
    """Fetch player props and also report any odds-fetch errors encountered."""
    return _load_player_props_for_day_uncached(
        day, regions, markets, bookmakers_csv, odds_format
    )


@functools.lru_cache(maxsize=128)
def get_player_elo_ratings(season: str) -> Dict[str, Dict]:
    """Get player Elo ratings from database."""
    try:
        db = EloDatabase("elo_ratings.db")
        cursor = db.conn.cursor()

        cursor.execute("""
            SELECT player_name, position, team_abbr, rating
            FROM player_elo
            WHERE season = ?
            GROUP BY player_name
            HAVING id = MAX(id)
        """, (season,))

        players = {}
        for name, pos, team, rating in cursor.fetchall():
            name_key = normalize_name_key(name)
            players[name_key] = {
                'name': name,
                'position': pos,
                'team': team,
                'elo': rating
            }

        db.close()
        return players
    except Exception as e:
        logger.warning("Could not load player Elo: %s", e)
        return {}


@functools.lru_cache(maxsize=64)
def get_player_nst_stats(season: str) -> Dict[str, Dict]:
    """
    Get player stats. Backed by NHL API PBP (was NST HTML scrape).

    `season` is the YYYYYYYY form ("20242025"). The first 4 chars are
    the start year; we call `compute_skater_rates(start_year, stype=2)`.
    """
    try:
        start_year = int(str(season)[:4])
    except (ValueError, TypeError):
        logger.warning("Invalid season format %r, expected YYYYYYYY", season)
        return {}
    try:
        rates = load_skater_rates_from_json(start_year, 2)
    except Exception as e:
        logger.warning("Could not load PBP stats for %s: %s", season, e)
        return {}

    stats: Dict[str, Dict] = {}
    for name_key, d in rates.items():
        gp = d.get("gp", 0)
        if gp == 0:
            continue
        goals = d.get("goals", 0)
        assists = d.get("assists", 0)
        shots = d.get("shots", 0)
        stats[name_key] = {
            "name": d.get("name", ""),
            "gp": gp,
            "goals": goals,
            "assists": assists,
            "points": goals + assists,
            "shots": shots,
            "goals_pg": goals / gp,
            "assists_pg": assists / gp,
            "points_pg": (goals + assists) / gp,
            "shots_pg": shots / gp,
            "position": d.get("position", ""),
        }

    # Goal rates are left at their raw per-game values (goals / gp). Shrinking
    # them toward the positional mean (previously K=40 games) compressed the whole
    # board -- Connor McDavid's 0.585 goals/game and a depth forward's 0.10 both
    # landed near 0.40 -- and the Anytime Goal Scorer market priced into a useless
    # ~30-35% cluster where every star looked alike. Base the rate on the player's
    # own production instead: last year's full season, or this year's pace once
    # the season selector below starts returning the current year.

    # Merge goalie save rates so `player_total_saves` props can be priced.
    try:
        goalie_rates = load_goalie_rates_from_json(start_year, 2)
        if goalie_rates is not None and not goalie_rates.empty:
            for _, g in goalie_rates.iterrows():
                gname = str(g.get("name", "") or "")
                gkey = normalize_name_key(gname)
                if not gkey or gkey in stats:
                    continue
                ggp = int(g.get("gp", 0) or 0)
                if ggp == 0:
                    continue
                sv = g.get("saves_per_game", None)
                if sv is None or pd.isna(sv):
                    sv = (int(g.get("sv", 0) or 0)) / ggp
                stats[gkey] = {
                    "name": gname,
                    "gp": ggp,
                    "saves_pg": float(sv or 0.0),
                }
    except Exception as e:
        logger.warning("Could not load goalie rates for %s: %s", season, e)

    return stats


# New canonical name; the old name stays as a thin alias so all
# existing callers (app.py, etc.) keep working.
get_player_pbp_stats = get_player_nst_stats


# A per-game rate is only trusted once the season has played this many games for
# its leaders. A veteran a couple games into October shows goals_pg = 0 (or 1),
# which reads as "No Data" and drops every star off the props board. Until the
# new season's own rates clear this bar, price off the previous one.
_TRUSTED_SAMPLE_GP = 10


def _season_with_player_stats(season: str) -> str:
    """
    Season to price props from.

    A September date resolves to the *upcoming* season, whose PBP data does not
    exist until games are played. With no stats every player abstains, so props
    collapse to the handful of prospects carrying a rookie projection. Until the
    season has rates, price off the previous one instead.

    Early in a season "has rates" is not enough: the first couple of games leave
    every player with goals_pg 0 or 1, so the model abstains on the whole board.
    The new season's own rates are only usable once its best-sampled player has
    reached a trusted sample; before then, fall back to last season's full rates.
    """
    if _season_rates_mature(season):
        return season
    try:
        previous = f"{int(season[:4]) - 1}{season[:4]}"
    except (ValueError, TypeError):
        return season
    if get_player_pbp_stats(previous):
        logger.info("Season %s rates not mature yet; pricing props from %s.", season, previous)
        return previous
    return season


def _season_rates_mature(season: str) -> bool:
    """True once the season's own per-game rates are trustworthy enough to price off."""
    stats = get_player_pbp_stats(season)
    if not stats:
        return False
    gps = [int(d.get("gp", 0) or 0) for d in stats.values()]
    return bool(gps) and max(gps) >= _TRUSTED_SAMPLE_GP


def _display_position(player_stats: Dict[str, Dict], player_key: str, market: str) -> str:
    """
    F / D / G for the props board.

    Goalies are not in the skater rates at all, and saves is the only market
    they are priced on, so that market implies G. Otherwise the PBP position
    code decides; wings and centres are all just "forward" to a reader.
    """
    if "save" in str(market).lower():
        return "G"
    pos = str((player_stats or {}).get(player_key, {}).get("position") or "").upper()
    if pos in ("C", "L", "R", "W", "LW", "RW", "F"):
        return "F"
    if pos in ("D", "LD", "RD"):
        return "D"
    # Prospects carrying an NHLe projection have no PBP position to read.
    return ""


# Variance-to-mean ratio per market family, fitted against de-vigged
# DraftKings main-line prices. 1.0 would be a pure Poisson process.
_COUNT_DISPERSION = {
    "save": 1.15,
    "shot": 1.20,
    "point": 1.20,
    "goal": 1.20,
    "assist": 1.20,
}
_DEFAULT_DISPERSION = 1.20


def _dispersion_for_market(market: str) -> float:
    """Variance-to-mean ratio for a count market; 1.0 would be Poisson."""
    market_lower = market.lower()
    for key, phi in _COUNT_DISPERSION.items():
        if key in market_lower:
            return phi
    return _DEFAULT_DISPERSION


def _nb_cdf(mean: float, dispersion: float, k: int) -> float:
    """
    P(X <= k) for a count with the given mean and variance-to-mean ratio.

    A count prop is a count, not a normal deviate. The previous normal
    approximation gave the tail far too much weight at low lines: Radko Gudas
    (0.20 assists/game) came out at 39% to record one, where Poisson says 17.8%
    and the book said 16.7%. That single error was the source of most of the
    implausible "edges" on the props board. The negative binomial falls back to
    Poisson as dispersion approaches 1 and carries the mild over-dispersion NHL
    counting stats actually show.
    """
    if mean <= 0:
        return 1.0
    if k < 0:
        return 0.0
    if dispersion <= 1.0:
        term = math.exp(-mean)
        total = term
        for i in range(1, k + 1):
            term *= mean / i
            total += term
        return min(1.0, total)
    r = mean / (dispersion - 1.0)
    p = 1.0 / dispersion
    term = math.exp(r * math.log(p))
    total = term
    for i in range(1, k + 1):
        term *= (i + r - 1) / i * (1.0 - p)
        total += term
    return min(1.0, total)


def _count_prob_over(mean: float, line: float, dispersion: float) -> float:
    """P(X > line) in percent for a count with the given per-game mean."""
    if mean <= 0:
        return 0.0
    return 100.0 * (1.0 - _nb_cdf(mean, dispersion, int(math.floor(line))))


def _std_for_market(avg: float, market: str) -> float:
    """
    Market-appropriate standard deviation for a per-game average.

    Still used for shots and saves: fitted against de-vigged book prices, the
    normal path beats the negative binomial there (shots 6.25pp vs 9.01pp RMSE),
    so only the 0.5-line markets use the count distribution.
    """
    market_lower = market.lower()
    if 'save' in market_lower:
        dispersion = 1.1
    elif 'shot' in market_lower:
        dispersion = 1.2
    elif 'point' in market_lower:
        dispersion = 1.4
    elif 'goal' in market_lower or 'assist' in market_lower:
        dispersion = 1.6
    else:
        dispersion = 1.4
    return max(math.sqrt(max(avg, 0.1)) * dispersion, 0.5)


def _elo_rate_multiplier(elo_rating: Optional[float]) -> float:
    """
    Convert a player Elo rating into a small rate multiplier.

    This is more accurate than a flat percentage adjustment because a 3%
    bump matters far more for a 0.4-goal scorer than a 4.5-shot shooter.
    Multipliers are capped to avoid extreme projections for sparse data.
    """
    if elo_rating is None:
        return 1.0
    try:
        r = float(elo_rating)
    except (TypeError, ValueError):
        return 1.0
    if r >= 1700:
        return 1.06
    if r >= 1600:
        return 1.03
    if r >= 1500:
        return 1.0
    return 0.97


_ROOKIE_PROJECTIONS: Optional[Dict[str, Dict[str, Any]]] = None


def _get_rookie_projection(name_key: str) -> Optional[Dict[str, Any]]:
    """Return a cached rookie projection for a player, or None."""
    global _ROOKIE_PROJECTIONS
    if _ROOKIE_PROJECTIONS is None:
        try:
            from NHL.Rosters import load_rookie_projections
            _ROOKIE_PROJECTIONS = load_rookie_projections()
        except Exception as e:
            logger.warning(f"Could not load rookie projections: {e}")
            _ROOKIE_PROJECTIONS = {}
    return _ROOKIE_PROJECTIONS.get(name_key)


# Projected power-play deployment is a mild signal that a player's scoring rate
# (goals/assists/points) should run above their historical per-game mean, which
# is averaged over seasons of varying deployment. Deliberately small heuristic
# multipliers -- a placeholder until the effect is calibrated against PP vs ES
# production rather than guessed.
PP1_SCORING_BOOST = 1.08
PP2_SCORING_BOOST = 1.03

_PP_UNITS: Optional[Dict[str, str]] = None


def _load_pp_role_lookup() -> Dict[str, str]:
    """Return {normalized_name_key: 'pp1'|'pp2'} from lineups.json.

    A name that collides across teams with conflicting roles is dropped, so a
    shared name (e.g. two Elias Petterssons) never gets the wrong boost.
    """
    global _PP_UNITS
    if _PP_UNITS is None:
        _PP_UNITS = {}
        try:
            path = Path(__file__).resolve().parent.parent / "lineups.json"
            if not path.exists():
                return _PP_UNITS
            data = json.loads(path.read_text(encoding="utf-8"))
            roles: Dict[str, str] = {}
            for team in (data.get("teams") or {}).values():
                for role in ("pp1", "pp2"):
                    for name in team.get(role) or []:
                        key = normalize_name_key(name)
                        if not key:
                            continue
                        roles[key] = "conflict" if (key in roles and roles[key] != role) else role
            _PP_UNITS = {k: v for k, v in roles.items() if v in ("pp1", "pp2")}
        except Exception as e:
            logger.warning(f"Could not load lineups.json: {e}")
            _PP_UNITS = {}
    return _PP_UNITS


# Matchup funnel: shotpropz.com "goals allowed by position" tells us how leaky
# each team is to opposing centres / wings / defence. A goal scorer facing a
# team that bleeds goals to his position gets a lift; one facing a shutdown
# team gets a haircut. Clamped so a ~5-game home/away sample can't swing a rate
# wildly; a placeholder until it is validated against actual outcomes.
_MATCHUP_MIN = 0.75
_MATCHUP_MAX = 1.35

# Raw PBP position code -> shotpropz bucket (their tables are C/LW/RW/D).
_SHOTPROPZ_POSITION = {
    "C": "C",
    "L": "LW",
    "R": "RW",
    "LW": "LW",
    "RW": "RW",
    "D": "D",
    "LD": "D",
    "RD": "D",
}

_SHOTPROPZ: Optional[Dict] = None


def _load_shotpropz() -> Dict:
    """Return the parsed shotpropz.json payload, cached for the process."""
    global _SHOTPROPZ
    if _SHOTPROPZ is None:
        _SHOTPROPZ = {}
        try:
            path = Path(__file__).resolve().parent.parent / "shotpropz.json"
            if path.exists():
                _SHOTPROPZ = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Could not load shotpropz.json: {e}")
            _SHOTPROPZ = {}
    return _SHOTPROPZ


def _matchup_multiplier(
    player_team: Optional[str],
    home_abbr: Optional[str],
    away_abbr: Optional[str],
    position: Optional[str],
    metric: str = "goals",
) -> Optional[float]:
    """Opponent's allowed-by-position funnel, as a rate multiplier.

    `metric` is "goals" (goal-scoring markets) or "sog" (shots-on-goal), and
    selects the matching shotpropz table. Returns None when the matchup cannot
    be resolved (unknown team, venue, or position), which the caller treats as
    "no adjustment".
    """
    bucket_key = _SHOTPROPZ_POSITION.get(str(position or "").upper())
    if bucket_key is None:
        return None

    team = _normalize_team_abbr(player_team) if player_team else None
    opponent = None
    defending_location = None
    if team and away_abbr and team == home_abbr:
        # Player's team is home, so the opponent defends on the road.
        opponent = away_abbr
        defending_location = "Away"
    elif team and home_abbr and team == away_abbr:
        opponent = home_abbr
        defending_location = "Home"
    if not opponent or not defending_location:
        return None

    data_key = "sog_against" if metric == "sog" else "goals_against"
    locations = _load_shotpropz().get(data_key) or {}
    bucket = (locations.get(defending_location) or {}).get(bucket_key) or {}
    if opponent not in bucket:
        # A team with no recent home/away sample is absent from that split;
        # fall back to the all-situations number rather than skipping it.
        bucket = (locations.get("All") or {}).get(bucket_key) or {}
    opponent_value = bucket.get(opponent)
    if opponent_value is None or not bucket:
        return None
    league_avg = sum(bucket.values()) / len(bucket)
    if league_avg <= 0:
        return None
    return max(_MATCHUP_MIN, min(_MATCHUP_MAX, opponent_value / league_avg))


def calculate_hit_probability(
    player_name: str,
    market: str,
    line: float,
    player_elo: Dict[str, Dict],
    player_stats: Dict[str, Dict],
    matchup_multiplier: Optional[float] = None,
) -> Tuple[float, str]:
    """
    Calculate probability of hitting the line and recommend Over/Under.

    Uses a per-game rate + market-appropriate over-dispersion estimate, then
    maps the distance from the line to an over probability via a logistic CDF.
    Player Elo is applied as a rate multiplier rather than a flat probability
    shift so it scales correctly across different prop markets.

    Returns:
        (probability_pct, recommendation)
        - probability_pct: 0-100, probability of OVER hitting
        - recommendation: "Over", "Under", "Pass", or "No Data". "No Data"
          means no model was produced at all (the 50.0 above is a sentinel,
          not an estimate); "Pass" means a real estimate near 50%.
    """
    name_key = normalize_name_key(player_name)

    # Get player data
    elo_data = player_elo.get(name_key, {})
    stats_data = player_stats.get(name_key, {})

    gp = int(stats_data.get("gp", 0) or 0)

    # Rookies: no trusted NHL sample yet. Fall back to a projected per-game
    # rate (non-NHL stats proxy) for their first ~10 games.
    if gp < 10:
        projection = _get_rookie_projection(name_key)
        if projection:
            stats_data = {
                "gp": max(gp, 1),
                "points_pg": float(projection.get("points_pg") or 0.0),
                "goals_pg": float(projection.get("goals_pg") or 0.0),
                "assists_pg": float(projection.get("assists_pg") or 0.0),
                "shots_pg": float(projection.get("shots_pg") or 0.0),
            }
        elif gp < 5:
            # Not enough games to trust the per-game rate.
            return 50.0, "No Data"

    if not stats_data:
        return 50.0, "No Data"  # No data

    # Get stat average based on market
    market_lower = market.lower()
    if 'save' in market_lower:
        avg = float(stats_data.get('saves_pg', 0) or 0)
    elif 'blocked' in market_lower:
        # Per-player blocked shots are not in the PBP shot store (team-level only).
        return 50.0, "No Data"
    elif 'power_play' in market_lower:
        # Per-player power-play points are not in the PBP shot store.
        return 50.0, "No Data"
    elif 'point' in market_lower:
        avg = float(stats_data.get('points_pg', 0) or 0)
    elif 'assist' in market_lower:
        avg = float(stats_data.get('assists_pg', 0) or 0)
    elif 'shot' in market_lower:
        # Check before 'goal': "player_shots_on_goal" contains both "shot"
        # and "goal", and must map to shots_pg, not goals_pg.
        avg = float(stats_data.get('shots_pg', 0) or 0)
    elif 'goal' in market_lower:
        avg = float(stats_data.get('goals_pg', 0) or 0)
    else:
        return 50.0, "No Data"

    if avg <= 0:
        return 50.0, "No Data"

    # Apply Elo as a rate multiplier instead of a flat probability shift.
    elo_rating = elo_data.get('elo', 1500)
    adjusted_avg = avg * _elo_rate_multiplier(elo_rating)

    # Power-play deployment: a projected PP1/PP2 role creates more scoring than
    # the historical mean suggests. Boost only the scoring markets, never shots
    # or saves.
    if "shot" not in market_lower and any(k in market_lower for k in ("goal", "assist", "point")):
        pp_role = _load_pp_role_lookup().get(name_key)
        if pp_role == "pp1":
            adjusted_avg *= PP1_SCORING_BOOST
        elif pp_role == "pp2":
            adjusted_avg *= PP2_SCORING_BOOST

    # Matchup funnel (goals allowed by the opponent to this player's position).
    if matchup_multiplier is not None:
        adjusted_avg *= matchup_multiplier

    # The count distribution wins clearly on the 0.5-line markets, where the
    # normal tail is far too heavy; on shots and saves the fitted normal beats it.
    market_lower = market.lower()
    if any(k in market_lower for k in ("goal", "assist", "point")):
        base_prob = _count_prob_over(adjusted_avg, float(line), _dispersion_for_market(market))
    else:
        std = _std_for_market(adjusted_avg, market)
        base_prob = 100.0 / (1.0 + math.exp((line - adjusted_avg) / std))

    # Floor/ceiling; never claim 0% or 100% from a noisy per-game estimate.
    prob_over = max(1.0, min(99.0, base_prob))

    # Recommendation logic
    # Over if probability > 55%
    # Under if probability < 45%
    # Pass otherwise

    if prob_over >= 55.0:
        recommendation = "Over"
    elif prob_over <= 45.0:
        recommendation = "Under"
    else:
        recommendation = "Pass"

    return prob_over, recommendation


def _shape_player_df(
    raw: List[Dict[str, Any]],
    fetched_odds_format: str,
    player_elo: Dict[str, Dict],
    player_stats: Dict[str, Dict]
) -> pd.DataFrame:
    """Flatten /events/{id}/odds payloads with hit probability."""
    rows: List[Dict[str, Any]] = []
    fmt = (fetched_odds_format or "american").lower()

    for ev in raw or []:
        ev_id = ev.get("id")
        home = ev.get("home_team")
        away = ev.get("away_team")
        ctime = ev.get("commence_time")
        books = ev.get("bookmakers", []) or []
        for bk in books:
            book_key = bk.get("key")
            for m in bk.get("markets", []) or []:
                mkey = m.get("key")
                last_upd = m.get("last_update")
                outs = m.get("outcomes", []) or []

                by_player: Dict[tuple, Dict[str, Any]] = {}
                for o in outs:
                    side = o.get("name")
                    player = o.get("description")
                    line = o.get("point")
                    price = o.get("price")

                    if not player or side not in ("Over", "Under"):
                        continue

                    key = (player, line)
                    if key not in by_player:
                        by_player[key] = {
                            "player": player,
                            "line": line,
                            "over_american": None,
                            "under_american": None,
                            "over_decimal": None,
                            "under_decimal": None,
                            # None when the feed does not flag main vs alternate.
                            "is_main_line": None,
                            # Set when two selections collide on this name -- see
                            # the conflict check below.
                            "ambiguous": False,
                        }

                    flagged = o.get("is_main_line")
                    if flagged is not None:
                        by_player[key]["is_main_line"] = bool(by_player[key]["is_main_line"]) or bool(flagged)

                    if fmt == "american":
                        amer = None if price is None else int(round(float(price)))
                        dec = american_to_decimal(amer) if amer is not None else None
                    else:
                        dec = None if price is None else float(price)
                        amer = decimal_to_american(dec) if dec is not None else None

                    # A second, differently-priced selection on the same side and
                    # line means two players share this name. Both land in this
                    # one group, so the later price would silently overwrite the
                    # earlier one.
                    if side == "Over":
                        prev = by_player[key]["over_decimal"]
                        if prev is not None and dec is not None and prev != dec:
                            by_player[key]["ambiguous"] = True
                        by_player[key]["over_american"] = amer
                        by_player[key]["over_decimal"] = dec
                    else:
                        prev = by_player[key]["under_decimal"]
                        if prev is not None and dec is not None and prev != dec:
                            by_player[key]["ambiguous"] = True
                        by_player[key]["under_american"] = amer
                        by_player[key]["under_decimal"] = dec

                for (_, _), rec in by_player.items():
                    # Two players can share a name -- Vancouver dresses two Elias
                    # Pettersson, and DraftKings posts both at +290 and +2000 in
                    # the same anytime-goal-scorer market. The model resolves the
                    # name the same way this grouping does, so it would price
                    # whichever selection survived against whichever player's
                    # rate it looked up first: here a depth defenceman's +2000
                    # against the forward's scoring rate, a 12.9% edge that does
                    # not exist. There is no way to tell the two apart from the
                    # feed, so drop the row rather than guess.
                    if rec.get("ambiguous"):
                        continue

                    # An alternate line ("2+ points") is quoted Over-only, so it
                    # is not a two-sided market and cannot be recommended as an
                    # Under. Drop it at the source.
                    if rec.get("is_main_line") is False:
                        continue

                    player_key = normalize_name_key(rec["player"])
                    player_team = player_elo.get(player_key, {}).get("team") if player_elo else None
                    home_abbr = _normalize_team_abbr(home) if home else None
                    away_abbr = _normalize_team_abbr(away) if away else None

                    # Matchup funnel: the opponent's allowed-by-position rate vs
                    # the league average, applied to goal-scoring and shots-on-goal
                    # markets (the two shotpropz tables). Points/assists have no
                    # matching table, so they get no matchup adjustment.
                    mkey_lower = str(mkey).lower()
                    if "shot" in mkey_lower:
                        metric = "sog"
                    elif "goal" in mkey_lower:
                        metric = "goals"
                    else:
                        metric = None
                    matchup_multiplier = None
                    if metric:
                        raw_pos = str((player_stats or {}).get(player_key, {}).get("position") or "")
                        matchup_multiplier = _matchup_multiplier(
                            player_team, home_abbr, away_abbr, raw_pos, metric=metric
                        )

                    # Calculate hit probability
                    prob_over, recommendation = calculate_hit_probability(
                        rec["player"],
                        mkey,
                        rec["line"],
                        player_elo,
                        player_stats,
                        matchup_multiplier=matchup_multiplier,
                    )

                    rows.append({
                        "event_id": ev_id,
                        "commence_time": ctime,
                        "home_team": home,
                        "away_team": away,
                        "home_abbr": home_abbr,
                        "away_abbr": away_abbr,
                        "player_team": player_team,
                        "book_key": book_key,
                        "market": mkey,
                        "market_last_update": last_upd,
                        "player": rec["player"],
                        "position": _display_position(player_stats, player_key, mkey),
                        "line": rec["line"],
                        "over_american": rec["over_american"],
                        "over_decimal": rec["over_decimal"],
                        "under_american": rec["under_american"],
                        "under_decimal": rec["under_decimal"],
                        "implied_over": implied_probability(rec["over_decimal"]) if rec["over_decimal"] else None,
                        "implied_under": implied_probability(rec["under_decimal"]) if rec["under_decimal"] else None,
                        "prob_over": prob_over,
                        "recommendation": recommendation,
                    })

    df = pd.DataFrame(rows)
    if not df.empty:
        for col in ("line", "over_decimal", "under_decimal", "prob_over"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("over_american", "under_american"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        df["market"] = df["market"].astype(str).str.replace("_", " ").str.title()
        df["_player_key"] = df["player"].astype(str).map(normalize_name_key)

        try:
            df["commence_time"] = pd.to_datetime(df["commence_time"])
        except Exception:
            pass

    return df


# Props are priced off one book, not a best-price-per-side sweep across books.
# FanDuel posts Over-only rungs and no Unders at all, so shopping the two sides
# independently stitched its long Over onto a DraftKings Under -- a market that
# does not exist (the "pair" summed to ~72% where a real two-way sums to ~105%)
# -- and every edge was then measured against the longest Over in the market,
# which understated the real price and inflated the edge. DraftKings is the only
# book here quoting both sides, and the dispersion constants below were fitted
# against its main lines.
_PROP_BOOK = "draftkings"


def _book_for_market(market: str) -> str:
    """
    Preferred book for a market.

    Most props price off DraftKings, but anytime-goal-scorer does not: SharpAPI
    serves DraftKings' 2+-goal ladder under that market type (Connor McDavid at
    +450 instead of ~-105), while FanDuel's feed is the real anytime-goal
    market. It is a one-sided "does he score" market, so the FanDuel-posts-no-
    Unders problem that keeps the two-sided markets on DraftKings does not apply.
    """
    if "goal scorer" in market.lower():
        return "fanduel"
    return _PROP_BOOK


def _best_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Price each (player, market, line) off a single book."""
    if df.empty:
        return df

    if "book_key" in df.columns:
        # Filter each market to its preferred book, falling back to whatever is
        # on offer when that book has no rows for the market.
        kept = []
        for market, sub in df.groupby("market", sort=False):
            primary = sub[sub["book_key"] == _book_for_market(str(market))]
            kept.append(primary if not primary.empty else sub)
        df = pd.concat(kept, ignore_index=True)

    agg_rows: List[Dict[str, Any]] = []
    group_cols = ["player", "market", "line"]

    for keys, sub in df.groupby(group_cols):
        player, market, line = keys

        # Get probability (same for all books)
        prob_over = sub["prob_over"].iloc[0] if "prob_over" in sub.columns else 50.0
        recommendation = sub["recommendation"].iloc[0] if "recommendation" in sub.columns else "Pass"

        # Best Over
        sub_over = sub.dropna(subset=["over_decimal"])
        if sub_over.empty:
            sub_over = sub.dropna(subset=["over_american"]).copy()
            if not sub_over.empty:
                sub_over["over_decimal"] = sub_over["over_american"].map(american_to_decimal)

        best_over_row = None
        if not sub_over.empty:
            sub_over = sub_over.sort_values(["over_decimal", "market_last_update"], ascending=[False, True])
            best_over_row = sub_over.iloc[0]

        # Best Under
        sub_under = sub.dropna(subset=["under_decimal"])
        if sub_under.empty:
            sub_under = sub.dropna(subset=["under_american"]).copy()
            if not sub_under.empty:
                sub_under["under_decimal"] = sub_under["under_american"].map(american_to_decimal)

        best_under_row = None
        if not sub_under.empty:
            sub_under = sub_under.sort_values(["under_decimal", "market_last_update"], ascending=[False, True])
            best_under_row = sub_under.iloc[0]

        # Carry through event context from any row (all rows share the same event).
        ctx_row = sub.iloc[0]

        row: Dict[str, Any] = {
            "event_id": ctx_row.get("event_id") if "event_id" in ctx_row else None,
            "commence_time": ctx_row.get("commence_time") if "commence_time" in ctx_row else None,
            "home_team": ctx_row.get("home_team") if "home_team" in ctx_row else None,
            "away_team": ctx_row.get("away_team") if "away_team" in ctx_row else None,
            "home_abbr": ctx_row.get("home_abbr") if "home_abbr" in ctx_row else None,
            "away_abbr": ctx_row.get("away_abbr") if "away_abbr" in ctx_row else None,
            "player_team": ctx_row.get("player_team") if "player_team" in ctx_row else None,
            "player": player,
            "position": ctx_row.get("position") if "position" in ctx_row else None,
            "market": market,
            "line": line,
            "prob_over": prob_over,
            "recommendation": recommendation
        }

        if best_over_row is not None:
            row["over_decimal"] = float(best_over_row.get("over_decimal")) if pd.notna(best_over_row.get("over_decimal")) else None
            oa = best_over_row.get("over_american")
            if pd.isna(oa) and row["over_decimal"] is not None:
                oa = decimal_to_american(row["over_decimal"])
            row["over_american"] = int(oa) if oa is not None and not pd.isna(oa) else None
            row["implied_over"] = float(best_over_row.get("implied_over")) if pd.notna(best_over_row.get("implied_over")) else None

        if best_under_row is not None:
            row["under_decimal"] = float(best_under_row.get("under_decimal")) if pd.notna(best_under_row.get("under_decimal")) else None
            ua = best_under_row.get("under_american")
            if pd.isna(ua) and row["under_decimal"] is not None:
                ua = decimal_to_american(row["under_decimal"])
            row["under_american"] = int(ua) if ua is not None and not pd.isna(ua) else None
            row["implied_under"] = float(best_under_row.get("implied_under")) if pd.notna(best_under_row.get("implied_under")) else None

        agg_rows.append(row)

    out = pd.DataFrame(agg_rows)
    if not out.empty:
        out = out.sort_values(["player", "market", "line"]).reset_index(drop=True)
        for c in ("over_american", "under_american"):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    return out


def _filter_by_player(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Filter DataFrame by player query with fuzzy fallback."""
    if df.empty or not query:
        return df
    q = query.strip()
    if not q:
        return df

    mask_contains = df["player"].astype(str).str.contains(q, case=False, na=False)
    qkey = normalize_name_key(q)
    mask_key = df["_player_key"].astype(str).str.contains(qkey, case=False, na=False)

    out = df[mask_contains | mask_key]
    if not out.empty:
        return out

    players = df["player"].dropna().astype(str).unique().tolist()
    close = difflib.get_close_matches(q, players, n=5, cutoff=0.6)
    if close:
        return df[df["player"].isin(close)]

    return out


def _current_filters(day: _date, regions: str, markets: List[str], bookmakers_csv: str, odds_format: str) -> Dict[str, Any]:
    mk = tuple(sorted([m.strip() for m in (markets or [])]))
    bks = ",".join(sorted([s.strip() for s in (bookmakers_csv or "").split(",") if s.strip()]))
    return {
        "day": day,
        "regions": regions,
        "markets": mk,
        "bookmakers_csv": bks,
        "odds_format": odds_format,
    }


def compute_player_props_for_date(
    game_date: _date,
    markets: Optional[Tuple[str, ...]] = None,
    regions: str = "us",
    bookmakers_csv: Optional[str] = None,
    odds_format: str = "american",
    require_positive_edge: bool = True,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Compute shaped player props (with model edge) for a date.

    Returns ``(records, warning)``. ``records`` is the list of prop dicts the UI
    consumes; ``warning`` is set when live odds are unavailable. Shared by the
    player-props endpoint and the Today's Picks pre-computation.

    ``require_positive_edge`` drops every row the book prices shorter than the
    model. That is right for Today's Picks, which is a list of bets to make, and
    wrong for the props board, which is a browser over what is on offer: the
    screen is invisible from the UI, so a missing prop reads as a missing
    market. Top-of-market names are exactly the ones it removes -- Auston
    Matthews at +145 is a 40.8% implied against a 27.4% model, so he is absent
    from a board whose cheapest surviving row is +450.
    """
    if markets is None:
        markets = tuple(DEFAULT_PLAYER_MARKETS)
    markets = tuple(m.strip().lower().replace(" ", "_") for m in markets)

    season = _season_with_player_stats(season_from_date(game_date.isoformat()))
    player_elo = get_player_elo_ratings(season)
    player_stats = get_player_pbp_stats(season)

    raw, odds_errors = load_player_props_for_day_with_status(
        day=game_date,
        regions=regions,
        markets=markets,
        bookmakers_csv=bookmakers_csv,
        odds_format=odds_format,
    )

    df = _shape_player_df(raw, odds_format, player_elo, player_stats)
    if df.empty:
        if odds_errors:
            return [], "Live odds unavailable: " + "; ".join(odds_errors)
        return [], None

    df = _best_prices(df)

    # Rows the model could not price at all carry a 50.0 sentinel ("No Data"),
    # not an estimate. Forced to "Over" below, they read as a huge edge against a
    # long-shot book price and crowd out every real pick, so drop them first.
    df = df[df["recommendation"] != "No Data"].copy()

    # Skater props are shown Over-only: the board is for "this player does the
    # thing", and an Under on a 0.5 line is not what it is for. Goalie saves keep
    # both sides, since backing a starter under his line is a normal bet.
    def _side(row):
        if "save" in str(row.get("market", "")).lower():
            return row.get("recommendation")
        return "Over"

    df["side"] = df.apply(_side, axis=1)

    # A pick is only actionable when the side chosen above is actually priced.
    # Over-only alternate ladders have no Under, so this also drops the rows that
    # used to appear as Unders with no book behind them.
    priced = (
        ((df["side"] == "Over") & df["over_decimal"].notna() & df["implied_over"].notna())
        | ((df["side"] == "Under") & df["under_decimal"].notna() & df["implied_under"].notna())
    )
    df = df[priced].copy()

    # Model edge vs. book-implied probability (same convention as Betting Edge).
    def _edge(row):
        if row["side"] == "Over":
            return row["prob_over"] / 100.0 - float(row["implied_over"]) / 100.0
        return (100.0 - row["prob_over"]) / 100.0 - float(row["implied_under"]) / 100.0

    df["edge"] = df.apply(_edge, axis=1)
    df["recommendation"] = df["side"]

    # An Over that the book already prices above the model is not a pick. Only a
    # value screen, though -- see ``require_positive_edge``.
    if require_positive_edge:
        df = df[(df["side"] == "Under") | (df["edge"] > 0)].copy()
    df = df.drop(columns=["side"])
    df = df.sort_values(["edge", "prob_over"], ascending=False)
    df = df.reset_index(drop=True)

    # Rename for the UI and convert to records.
    out_df = df[[
        "player", "position", "market", "line", "prob_over", "recommendation",
        "over_american", "under_american", "over_decimal", "under_decimal",
        "edge", "home_abbr", "away_abbr", "player_team",
        "implied_over", "implied_under",
    ]].copy()
    out_df["market"] = out_df["market"].str.replace("Player ", "")

    records = out_df.to_dict(orient="records")
    return records, None