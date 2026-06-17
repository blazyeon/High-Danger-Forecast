"""
Feature utilities for NHL simulations:
- Rest/travel fatigue and scheduling effects (IMPROVED)
- Penalty differential approximation
- Score effects modeling (IMPROVED)
- Correlated goal modeling (shared factor)
- Diagnostics helpers
"""
from __future__ import annotations

import math
from typing import Dict, Tuple, Optional, Any, List
from datetime import date as _date, datetime, timedelta

import numpy as np
import pandas as pd

from NHL.Utils import season_from_date
from NHL.Config import (
    NHL_API_BASE, REQUEST_HEADERS, DEFAULT_TIMEOUT,
    REST_TRAVEL_PARAMS
)
from NHL.Errors import retry_on_failure
from NHL.TeamsMeta import TEAM_META

import requests
import logging

logger = logging.getLogger(__name__)

@retry_on_failure(max_attempts=3, backoff_base=0.6)
def _get_json(url: str) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(url, headers=REQUEST_HEADERS, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.debug(f"Features: GET failed {url[:120]}: {e}")
        return None

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    try:
        R = 6371.0
        p1 = math.radians(lat1)
        p2 = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2
        c = 2*math.asin(math.sqrt(a))
        return R*c
    except Exception:
        return 0.0

def _team_loc(abbr: str) -> Tuple[float, float]:
    m = TEAM_META.get(abbr.upper())
    if not m:
        return (0.0, 0.0)
    return float(m["lat"]), float(m["lon"])


def _team_tz(abbr: str) -> float:
    """Return the venue time-zone offset for a team (UTC). Returns 0 if unknown."""
    m = TEAM_META.get(abbr.upper())
    if not m:
        return 0.0
    return float(m.get("tz", 0.0))

def _prev_game_for(team_abbr: str, season: str, until: _date) -> Optional[Dict[str, Any]]:
    sched_url = f"{NHL_API_BASE}/club-schedule-season/{team_abbr}/{season}"
    js = _get_json(sched_url)
    if not js:
        return None
    games = js.get("games", [])
    if not isinstance(games, list):
        return None
    chosen = None
    for g in games:
        raw = g.get("gameDate") or g.get("startTimeUTC") or g.get("startTime")
        dt = None
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00")).date()
        except Exception:
            try:
                dt = datetime.fromisoformat(str(raw)).date()
            except Exception:
                dt = None
        if dt and dt < until:
            if (chosen is None) or (dt > chosen["_d"]):
                g["_d"] = dt
                chosen = g
    return chosen

def compute_rest_travel_features(team_abbr: str, opponent_abbr: str, game_date: _date) -> Dict[str, Any]:
    """
    Compute rest/travel fatigue proxy:
    - is_b2b, rest_days
    - opp_rest_days
    - travel_km since last game
    - timezone difference
    """
    out = {
        "is_b2b": False,
        "rest_days": 3.0,
        "opp_rest_days": 3.0,
        "travel_km": 0.0,
        "tz_diff": 0.0
    }
    try:
        season = season_from_date(game_date.isoformat())
        prev_team = _prev_game_for(team_abbr, season, game_date)
        prev_opp = _prev_game_for(opponent_abbr, season, game_date)

        if prev_team:
            last_date = prev_team["_d"]
            out["is_b2b"] = (game_date - last_date) == timedelta(days=1)
            out["rest_days"] = max(0.0, (game_date - last_date).days - 1)
            lat1, lon1 = _team_loc(team_abbr)
            lat2, lon2 = _team_loc(opponent_abbr)
            out["travel_km"] = _haversine_km(lat1, lon1, lat2, lon2)

            # Use real venue time-zone offsets instead of the rough km/1500 proxy.
            out["tz_diff"] = _team_tz(team_abbr) - _team_tz(opponent_abbr)

        if prev_opp:
            last_date_o = prev_opp["_d"]
            out["opp_rest_days"] = max(0.0, (game_date - last_date_o).days - 1)

    except Exception as e:
        logger.debug(f"Rest/travel feature failure for {team_abbr} vs {opponent_abbr}: {e}")

    return out


def _prev_game_from_completed(
    team_abbr: str,
    game_date: _date,
    completed_games: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Find the most recent game for a team before game_date from an in-memory list."""
    chosen = None
    target = team_abbr.upper()
    for g in completed_games:
        gd = g.get("game_date")
        if not gd:
            continue
        if isinstance(gd, str):
            try:
                gd = datetime.fromisoformat(gd).date()
            except Exception:
                continue
        if gd >= game_date:
            continue
        if g.get("home_team", "").upper() == target or g.get("away_team", "").upper() == target:
            if chosen is None or gd > chosen["_date"]:
                g = dict(g)
                g["_date"] = gd
                chosen = g
    return chosen


def compute_rest_travel_features_fast(
    team_abbr: str,
    opponent_abbr: str,
    game_date: _date,
    completed_games: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Fast, offline rest/travel features using already-played games.
    Useful during historical training where API calls are too slow.
    """
    out = {
        "is_b2b": False,
        "rest_days": 3.0,
        "opp_rest_days": 3.0,
        "travel_km": 0.0,
        "tz_diff": 0.0,
    }
    try:
        prev_team = _prev_game_from_completed(team_abbr, game_date, completed_games)
        prev_opp = _prev_game_from_completed(opponent_abbr, game_date, completed_games)

        if prev_team:
            last_date = prev_team["_date"]
            out["is_b2b"] = (game_date - last_date) == timedelta(days=1)
            out["rest_days"] = max(0.0, (game_date - last_date).days - 1)
            lat1, lon1 = _team_loc(team_abbr)
            lat2, lon2 = _team_loc(opponent_abbr)
            out["travel_km"] = _haversine_km(lat1, lon1, lat2, lon2)
            out["tz_diff"] = _team_tz(team_abbr) - _team_tz(opponent_abbr)

        if prev_opp:
            last_date_o = prev_opp["_date"]
            out["opp_rest_days"] = max(0.0, (game_date - last_date_o).days - 1)
    except Exception as e:
        logger.debug(f"Fast rest/travel feature failure: {e}")

    return out


def fatigue_multiplier(features: Dict[str, Any]) -> float:
    """
    ✅ Compute fatigue multiplier for expected goals.

    Penalties:
    - Back-to-back: -14%
    - Road B2B (travel >= 1000 km): extra -3%
    - Rest disadvantage: -2% per day
    - Travel: -0.02% per km
    - Cross-country (2500+ km): Additional -4%
    """
    is_b2b = 1.0 if features.get("is_b2b") else 0.0
    rest_days = float(features.get("rest_days", 3.0))
    opp_rest_days = float(features.get("opp_rest_days", 3.0))
    travel_km = float(features.get("travel_km", 0.0))
    tz_diff = float(features.get("tz_diff", 0.0))

    W_B2B = REST_TRAVEL_PARAMS["back_to_back_penalty"]  # -0.14
    W_REST_DIFF = REST_TRAVEL_PARAMS["rest_diff_penalty"]  # -0.02
    W_TRAVEL = REST_TRAVEL_PARAMS["travel_penalty_per_km"]  # -0.0002

    rest_diff = max(-3.0, min(3.0, opp_rest_days - rest_days))

    # ✅ NEW: Additional cross-country penalty
    cross_country_penalty = 0.0
    if travel_km > 2500 or tz_diff >= 2.5:
        cross_country_penalty = REST_TRAVEL_PARAMS["cross_country_penalty"]  # -0.04

    # Cap stacking: B2B already includes the fatigue of a same-city second game.
    # Travel penalties are capped so a road B2B does not double-penalize brutally.
    travel_delta = W_TRAVEL * min(travel_km, 4000.0)
    cross_delta = cross_country_penalty
    non_b2b_delta = W_REST_DIFF * rest_diff + travel_delta + cross_delta

    # B2B gets its full penalty only if travel is light; heavy travel mostly
    # replaces part of the B2B penalty instead of stacking on top.
    b2b_delta = W_B2B * is_b2b * max(0.0, 1.0 - 0.0002 * min(travel_km, 1500.0))

    # Road B2B: extra penalty when a team played yesterday and traveled >=1000 km.
    road_b2b_delta = 0.0
    if is_b2b and travel_km >= 1000.0:
        road_b2b_delta = REST_TRAVEL_PARAMS.get("road_b2b_extra_penalty", -0.03)

    delta = b2b_delta + non_b2b_delta + road_b2b_delta

    # Hard ceiling on total fatigue so pathological schedules don't break mu.
    # Raised to -0.30 to allow the stronger B2B signal to show through.
    delta = max(-0.30, delta)

    result = float(max(
        REST_TRAVEL_PARAMS["min_multiplier"],  # 0.70
        min(REST_TRAVEL_PARAMS["max_multiplier"], 1.0 + delta)  # 1.08
    ))

    logger.debug(
        f"Fatigue: B2B={is_b2b}, RestDiff={rest_diff:.1f}, Travel={travel_km:.0f}km, "
        f"TZ={tz_diff:.1f}h → Multiplier={result:.3f}"
    )

    return result

def penalty_diff_per60(all_df: pd.DataFrame, abbr: str) -> float:
    """
    Approximate penalties drawn - penalties taken per 60 for a team.
    If columns are missing, returns 0.0.
    """
    if not isinstance(all_df, pd.DataFrame) or all_df.empty:
        return 0.0
    try:
        tcol = None
        for c in all_df.columns:
            if str(c).strip().lower() in ("team", "tm", "squad"):
                tcol = c
                break
        if not tcol:
            return 0.0
        row = all_df[all_df[tcol].astype(str).str.contains(abbr, case=False, na=False)]
        if row.empty:
            return 0.0
        r = row.iloc[0]
        pd60 = float(r.get("Pen Drawn/60", 0.0) or 0.0)
        pt60 = float(r.get("Pen Taken/60", 0.0) or 0.0)
        val = pd60 - pt60
        if not math.isfinite(val):
            return 0.0
        return float(max(-2.0, min(2.0, val)))
    except Exception:
        return 0.0

def shared_correlation_factor(sims: int, rho: float = 0.38, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    ✅ IMPROVED: Generate a per-simulation shared multiplicative factor.

    Default rho increased from 0.25 to 0.38 for better correlation.
    Pass rng for deterministic output.
    """
    rho = float(max(0.0, min(0.8, rho)))
    if sims <= 0:
        return np.array([], dtype=float)
    sigma = 0.30 * rho
    mu = -0.5 * (sigma ** 2)
    gen = rng if rng is not None else np.random
    return np.exp(gen.normal(loc=mu, scale=sigma, size=sims))

def score_effect_scaler(mu_for: float, mu_against: float) -> float:
    """
    ✅ IMPROVED: Score-effects approximation with non-linear scaling.
    
    Teams expected to lead play more defensive.
    Teams expected to trail play more aggressive.
    
    - Leading by 2+: -15% (defensive shell)
    - Leading by 1: -8% (slight defensive)
    - Trailing by 1: +6% (aggressive)
    - Trailing by 2+: +12% (desperate)
    """
    diff = float(mu_for - mu_against)
    
    # ✅ IMPROVED: Non-linear scaling based on expected differential
    if diff >= 1.5:  # Expected to lead by 2+
        return 0.85  # -15%
    elif diff >= 0.8:  # Expected to lead by 1
        return 0.92  # -8%
    elif diff <= -1.5:  # Expected to trail by 2+
        return 1.12  # +12%
    elif diff <= -0.8:  # Expected to trail by 1
        return 1.06  # +6%
    else:
        # Close game: smooth tanh curve
        return float(max(0.85, min(1.12, 1.0 - 0.08 * np.tanh(diff / 0.8))))

def component_breakdown(
    base_xgf: float,
    xga_adj: float,
    pp_pk: float,
    goalie_adj: float,
    recent_adj: float,
    lineup_adj: float,
    rest_adj: float,
    score_adj: float,
    final_mu: float,
    momentum_adj: float = 0.0
) -> Dict[str, float]:
    return {
        "Base xGF": base_xgf,
        "xGA adjustment": xga_adj,
        "Special teams (PP/PK)": pp_pk,
        "Goalie adj (mult-1)": goalie_adj,
        "Recent form adj": recent_adj,
        "Lineup delta": lineup_adj,
        "Momentum adj (mult-1)": momentum_adj,
        "Rest/Travel adj (mult-1)": rest_adj,
        "Score-effects adj (mult-1)": score_adj,
        "Final mu": final_mu
    }