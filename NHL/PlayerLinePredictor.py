"""
Player Line Predictor — hit probability calculations:
- Fetches NHL player prop lines via The Odds API
- Calculates hit probability using player Elo + NST stats
- Sorts by most likely to hit
- Returns recommended bets (Over/Under)

Pure computation module; no Streamlit dependency.
"""
from __future__ import annotations

import functools
import logging
import math
import pandas as pd
from datetime import date as _date, timedelta
from typing import Dict, Any, List, Optional, Tuple
import difflib

from NHL.OddsAPI import fetch_nhl_player_props_by_date, OddsAPIError
from NHL.Utils import normalize_name_key
from NHL.StatsFromPBP import load_skater_rates_from_json
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
    "player_goals",
    "player_shots_on_goal",
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
def load_player_props_multi_day(
    day: _date,
    regions: str,
    markets: Tuple[str, ...],
    bookmakers_csv: Optional[str],
    odds_format: str = "american",
) -> Tuple[Dict[str, Any], ...]:
    """
    Fetch player props for the selected day AND the next day to handle timezone issues.
    This ensures games don't get missed when it's late at night in your timezone
    but the API considers them "tomorrow" in UTC.

    Note: ``markets`` is accepted as a tuple so that lru_cache can hash it.
    Callers passing a list should convert via ``tuple(markets)``.
    """
    results: List[Dict[str, Any]] = []

    # Fetch today
    try:
        today_props = fetch_nhl_player_props_by_date(
            day=day,
            regions=regions,
            markets=list(markets),
            bookmakers_csv=bookmakers_csv,
            odds_format=odds_format,
        )
        results.extend(today_props)
    except OddsAPIError as e:
        logger.warning("Odds API error for %s: %s", day, e)
    except Exception as e:
        logger.warning("Error fetching props for %s: %s", day, e)

    # Fetch tomorrow to catch timezone edge cases
    try:
        tomorrow = day + timedelta(days=1)
        tomorrow_props = fetch_nhl_player_props_by_date(
            day=tomorrow,
            regions=regions,
            markets=list(markets),
            bookmakers_csv=bookmakers_csv,
            odds_format=odds_format,
        )
        results.extend(tomorrow_props)
    except OddsAPIError as e:
        logger.warning("Odds API error for %s: %s", day + timedelta(days=1), e)
    except Exception as e:
        logger.warning("Error fetching props for %s: %s", day + timedelta(days=1), e)

    return tuple(results)


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
        }
    return stats


# New canonical name; the old name stays as a thin alias so all
# existing callers (app.py, etc.) keep working.
get_player_pbp_stats = get_player_nst_stats


def _std_for_market(avg: float, market: str) -> float:
    """
    Return a market-appropriate standard deviation for a per-game average.

    Count stats (goals, assists, points, shots) are over-dispersed compared
    to a pure Poisson process. We start with sqrt(mean) (Poisson baseline) and
    multiply by a market-specific dispersion factor derived from typical NHL
    season-to-season variance:
        - Goals/assists are the most volatile (dispersion ~1.6)
        - Points are slightly more stable than their components (~1.4)
        - Shots are the most repeatable (~1.2)
    A floor keeps low-volume players from collapsing to zero variance.
    """
    market_lower = market.lower()
    if 'shot' in market_lower:
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


def calculate_hit_probability(
    player_name: str,
    market: str,
    line: float,
    player_elo: Dict[str, Dict],
    player_stats: Dict[str, Dict]
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
        - recommendation: "Over", "Under", or "Pass"
    """
    name_key = normalize_name_key(player_name)

    # Get player data
    elo_data = player_elo.get(name_key, {})
    stats_data = player_stats.get(name_key, {})

    if not stats_data:
        return 50.0, "Pass"  # No data

    gp = int(stats_data.get("gp", 0) or 0)
    if gp < 5:
        # Not enough games to trust the per-game rate.
        return 50.0, "Pass"

    # Get stat average based on market
    market_lower = market.lower()
    if 'point' in market_lower:
        avg = float(stats_data.get('points_pg', 0) or 0)
    elif 'assist' in market_lower:
        avg = float(stats_data.get('assists_pg', 0) or 0)
    elif 'goal' in market_lower:
        avg = float(stats_data.get('goals_pg', 0) or 0)
    elif 'shot' in market_lower:
        avg = float(stats_data.get('shots_pg', 0) or 0)
    else:
        return 50.0, "Pass"

    if avg <= 0:
        return 50.0, "Pass"

    # Apply Elo as a rate multiplier instead of a flat probability shift.
    elo_rating = elo_data.get('elo', 1500)
    adjusted_avg = avg * _elo_rate_multiplier(elo_rating)

    # Standard deviation that respects count-stat over-dispersion.
    std = _std_for_market(adjusted_avg, market)

    # Z-score: how many standard deviations away is the line
    z_score = (line - adjusted_avg) / std

    # Logistic CDF approximation of over probability.
    base_prob = 100.0 / (1.0 + math.exp(z_score))

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
                        }

                    if fmt == "american":
                        amer = None if price is None else int(round(float(price)))
                        dec = american_to_decimal(amer) if amer is not None else None
                    else:
                        dec = None if price is None else float(price)
                        amer = decimal_to_american(dec) if dec is not None else None

                    if side == "Over":
                        by_player[key]["over_american"] = amer
                        by_player[key]["over_decimal"] = dec
                    else:
                        by_player[key]["under_american"] = amer
                        by_player[key]["under_decimal"] = dec

                for (_, _), rec in by_player.items():
                    # Calculate hit probability
                    prob_over, recommendation = calculate_hit_probability(
                        rec["player"],
                        mkey,
                        rec["line"],
                        player_elo,
                        player_stats
                    )

                    player_key = normalize_name_key(rec["player"])
                    player_team = player_elo.get(player_key, {}).get("team") if player_elo else None

                    rows.append({
                        "event_id": ev_id,
                        "commence_time": ctime,
                        "home_team": home,
                        "away_team": away,
                        "home_abbr": _normalize_team_abbr(home) if home else None,
                        "away_abbr": _normalize_team_abbr(away) if away else None,
                        "player_team": player_team,
                        "book_key": book_key,
                        "market": mkey,
                        "market_last_update": last_upd,
                        "player": rec["player"],
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


def _best_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Compute best Over/Under price per (player, market, line)."""
    if df.empty:
        return df

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