"""
Prediction layer with FIXED season data selection, improved error handling,
GOALIE-AWARE match probability, and DETERMINISTIC predictions.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import math
import logging
import hashlib

import numpy as np
import pandas as pd
import functools
import requests

from NHL.Lookup import sanitize_text, format_initial_last
from NST.Cache import get_nst_table_from_url
from NHL.Config import (
    PLAYER_PARAMS,
    GOALIE_PARAMS,
    MODEL_WEIGHTS,
    LEAGUE_AVERAGES,
    COLUMN_VARIATIONS,
    REQUEST_HEADERS,
    DEFAULT_TIMEOUT,
    NHL_API_BASE,
    NHL_STATS_API_BASE
)
from NHL.Errors import (
    safe_api_call,
    retry_on_failure,
    safe_division,
    validate_dataframe,
    DataValidationError
)
from NHL.Utils import (
    normalize_name_key,
    last_token_norm,
    format_initial_last as util_format_initial_last,
    sanitize_text as util_sanitize_text,
    get_column_safe,
    normalize_sv_column,
    prev_season_key,
    get_data_season_for_game
)

logger = logging.getLogger(__name__)

# ===================== CACHE & DETERMINISTIC SEEDING =====================

_PREDICTION_CACHE = {}


def _deterministic_seed(home_team: str, away_team: str, game_date: str) -> int:
    """
    Generate deterministic seed from game parameters.
    Same teams + date = same seed = same random outcomes.
    """
    key = f"{home_team}|{away_team}|{game_date}"
    hash_obj = hashlib.md5(key.encode())
    return int(hash_obj.hexdigest()[:8], 16)


def clear_prediction_cache():
    """Clear the prediction cache (useful for forcing refresh)."""
    global _PREDICTION_CACHE
    count = len(_PREDICTION_CACHE)
    _PREDICTION_CACHE = {}
    logger.info(f"Prediction cache cleared ({count} entries removed)")


# ===================== GOALIE-AWARE MATCH PROBABILITY =====================

def get_starting_goalie_for_game(team_abbr: str, game_date: str, game_id: Optional[str] = None) -> Optional[str]:
    """
    Return the predicted starting goalie's display name for a team on a given date.
    Uses get_confirmed_or_predicted_lineup from NHL.ApiScrape.
    """
    try:
        from NHL.ApiScrape import get_confirmed_or_predicted_lineup
        
        lineup = get_confirmed_or_predicted_lineup(team_abbr, game_date, game_id=game_id)
        goalies = lineup.get("goalies", []) or []
        if not goalies:
            logger.debug(f"No goalies found for {team_abbr} on {game_date}")
            return None
        
        g = goalies[0]
        goalie_name = g.get("name") if isinstance(g.get("name"), str) else None
        
        if goalie_name:
            logger.debug(f"Predicted starter for {team_abbr}: {goalie_name}")
        
        return goalie_name
    except Exception as e:
        logger.debug(f"get_starting_goalie_for_game error for {team_abbr} {game_date}: {e}")
        return None


def get_player_elo(player_name: str, season: str, db_path: str = "elo_ratings.db") -> Optional[float]:
    """
    Fetch the latest Elo rating for a player (by name and season).
    Uses shared AppState when running in Streamlit to avoid opening new DB connections.
    Falls back to direct DB query outside Streamlit.
    """
    if not player_name:
        return None

    # Try shared AppState first (avoids opening a new DB connection per call)
    try:
        from NHL.AppState import get_app_state
        state = get_app_state()
        if not state.get('is_fallback'):
            rating = state['player_elo'].get_player_rating(player_name)
            logger.debug(f"Player Elo for {player_name} (via AppState): {rating}")
            return float(rating)
    except Exception:
        pass  # Not in Streamlit or state unavailable

    # Fallback: direct DB query
    try:
        from EloMl.Database import EloDatabase

        with EloDatabase(db_path) as db:
            cursor = db.conn.cursor()

            cursor.execute("""
                SELECT rating FROM player_elo
                WHERE player_name = ? AND season = ?
                ORDER BY id DESC LIMIT 1
            """, (player_name, season))

            r = cursor.fetchone()
            if r and r[0] is not None:
                logger.debug(f"Found Elo for {player_name}: {r[0]}")
                return float(r[0])

            nk = normalize_name_key(player_name)
            cursor.execute("""
                SELECT player_name, rating FROM player_elo
                WHERE season = ? AND position = 'G'
                ORDER BY id DESC
            """, (season,))

            for pname, rating in cursor.fetchall():
                if normalize_name_key(pname) == nk:
                    logger.debug(f"Found Elo for {player_name} via normalized match: {rating}")
                    return float(rating)

        logger.debug(f"No Elo found for goalie: {player_name}")
        return None
    except Exception as e:
        logger.debug(f"get_player_elo error for {player_name}: {e}")
        return None


def get_team_elo(team_abbr: str, season: str, db_path: str = "elo_ratings.db") -> float:
    """
    Return latest team Elo for team_abbr, season.
    Uses shared AppState when running in Streamlit to avoid opening new DB connections.
    Falls back to direct DB query outside Streamlit.
    """
    # Try shared AppState first (avoids opening a new DB connection per call)
    try:
        from NHL.AppState import get_app_state
        state = get_app_state()
        if not state.get('is_fallback'):
            rating = state['team_elo'].get_team_rating(team_abbr.upper())
            logger.debug(f"Team Elo for {team_abbr} (via AppState): {rating}")
            return float(rating)
    except Exception:
        pass  # Not in Streamlit or state unavailable

    # Fallback: direct DB query
    try:
        from EloMl.Database import EloDatabase

        with EloDatabase(db_path) as db:
            cursor = db.conn.cursor()
            cursor.execute("""
                SELECT rating FROM team_elo
                WHERE team_abbr = ? AND season = ?
                ORDER BY id DESC LIMIT 1
            """, (team_abbr, season))

            r = cursor.fetchone()

        if r and r[0] is not None:
            logger.debug(f"Team Elo for {team_abbr}: {r[0]}")
            return float(r[0])
    except Exception as e:
        logger.debug(f"get_team_elo error for {team_abbr}: {e}")

    from EloMl.Ratings import EloConfig
    default_rating = float(EloConfig().initial_team_rating)
    logger.debug(f"Using default team Elo for {team_abbr}: {default_rating}")
    return default_rating


def elo_to_prob(home_rating: float, away_rating: float) -> float:
    """
    Convert Elo difference to home-win probability (0..1).
    Standard Elo logistic function.
    """
    diff = home_rating - away_rating
    prob = 1.0 / (1.0 + 10 ** (-diff / 400.0))
    return prob


def combined_match_probability(
    home_team: str,
    away_team: str,
    game_date: str,
    season: str,
    *,
    w_team: float = 0.85,
    w_goalie: float = 0.15,
    db_path: str = "elo_ratings.db",
    game_id: Optional[str] = None,
) -> dict:
    """
    Compute goalie-aware match win probability with DETERMINISTIC results.
    
    Returns a dict with:
      {
        "home_team": home_team,
        "away_team": away_team,
        "home_team_elo": ...,
        "away_team_elo": ...,
        "home_goalie": name_or_None,
        "away_goalie": name_or_None,
        "home_goalie_elo": ...,
        "away_goalie_elo": ...,
        "home_combined": ...,
        "away_combined": ...,
        "prob_home": 0..1,
        "prob_away": 0..1,
        "seed": deterministic_seed
      }
    
    Results are CACHED per (home, away, date, season) to ensure consistency.
    """
    cache_key = f"{home_team}|{away_team}|{game_date}|{season}|{w_team}|{w_goalie}"
    if cache_key in _PREDICTION_CACHE:
        logger.debug(f"Using cached prediction for {home_team} vs {away_team} on {game_date}")
        return _PREDICTION_CACHE[cache_key]
    
    seed = _deterministic_seed(home_team, away_team, game_date)
    logger.info(f"Computing prediction for {home_team} vs {away_team} on {game_date} (seed: {seed})")
    
    home_team_elo = get_team_elo(home_team, season, db_path)
    away_team_elo = get_team_elo(away_team, season, db_path)

    home_goalie = get_starting_goalie_for_game(home_team, game_date, game_id)
    away_goalie = get_starting_goalie_for_game(away_team, game_date, game_id)

    home_goalie_elo = get_player_elo(home_goalie, season, db_path) if home_goalie else None
    away_goalie_elo = get_player_elo(away_goalie, season, db_path) if away_goalie else None

    if home_goalie_elo is None:
        logger.debug(f"No goalie Elo for {home_goalie}, using team Elo: {home_team_elo}")
        home_goalie_elo = home_team_elo
    if away_goalie_elo is None:
        logger.debug(f"No goalie Elo for {away_goalie}, using team Elo: {away_team_elo}")
        away_goalie_elo = away_team_elo

    home_combined = w_team * home_team_elo + w_goalie * home_goalie_elo
    away_combined = w_team * away_team_elo + w_goalie * away_goalie_elo

    prob_home = elo_to_prob(home_combined, away_combined)

    result = {
        "home_team": home_team,
        "away_team": away_team,
        "home_team_elo": round(home_team_elo, 1),
        "away_team_elo": round(away_team_elo, 1),
        "home_goalie": home_goalie,
        "away_goalie": away_goalie,
        "home_goalie_elo": round(home_goalie_elo, 1),
        "away_goalie_elo": round(away_goalie_elo, 1),
        "home_combined": round(home_combined, 1),
        "away_combined": round(away_combined, 1),
        "prob_home": round(prob_home, 4),
        "prob_away": round(1.0 - prob_home, 4),
        "seed": seed,
    }
    
    logger.info(
        f"Prediction: {home_team} ({result['prob_home']*100:.1f}%) vs "
        f"{away_team} ({result['prob_away']*100:.1f}%)"
    )
    logger.info(
        f"  Goalies: {home_goalie} (Elo: {result['home_goalie_elo']}) vs "
        f"{away_goalie} (Elo: {result['away_goalie_elo']})"
    )
    
    _PREDICTION_CACHE[cache_key] = result
    
    return result


# ===================== FUZZY NAME MATCHING =====================

def fuzzy_match_player_name(
    search_name: str,
    candidate_names: List[str],
    threshold: float = 0.7
) -> Optional[str]:
    """
    Fuzzy match player name against candidates.
    
    Args:
        search_name: Name to search for
        candidate_names: List of candidate names
        threshold: Minimum similarity score (0-1)
    
    Returns:
        Best matching candidate name or None
    """
    if not search_name or not candidate_names:
        return None
    
    search_norm = normalize_name_key(search_name)
    search_last = last_token_norm(search_name)
    
    best_match = None
    best_score = 0.0
    
    for candidate in candidate_names:
        if not candidate:
            continue
        
        cand_norm = normalize_name_key(candidate)
        cand_last = last_token_norm(candidate)
        
        # Exact normalized match
        if search_norm == cand_norm:
            return candidate
        
        # Last name match
        if search_last and cand_last and search_last == cand_last:
            if len(search_last) >= 4:  # At least 4 chars for reliability
                score = 0.9
                if score > best_score:
                    best_score = score
                    best_match = candidate
        
        # Substring match
        if search_norm in cand_norm or cand_norm in search_norm:
            score = 0.8
            if score > best_score:
                best_score = score
                best_match = candidate
    
    if best_score >= threshold:
        return best_match
    
    return None


# ===================== LINE/USAGE MODELING =====================

def lineup_shooting_factor(
    df_lineup: pd.DataFrame,
    season_skill: Dict[str, Dict[str, float]]
) -> float:
    """
    Calculate lineup shooting factor vs baseline.
    
    Args:
        df_lineup: Lineup DataFrame with Name and Position columns
        season_skill: Player skill dictionary
    
    Returns:
        Factor adjustment (-0.30 to +0.30)
    """
    if df_lineup is None or df_lineup.empty:
        logger.debug("Empty lineup for shooting factor")
        return 0.0
    
    try:
        validate_dataframe(df_lineup, required_columns=["Name"], min_rows=1)
    except DataValidationError as e:
        logger.warning(f"Invalid lineup DataFrame: {e}")
        return 0.0
    
    names = df_lineup["Name"].tolist()
    sogpg_sum = 0.0
    
    for nm in names:
        key = normalize_name_key(nm)
        sogpg_sum += float(season_skill.get(key, {}).get("sogpg", 0.0))
    
    baseline = LEAGUE_AVERAGES["shots_per_game"]
    ratio = safe_division(sogpg_sum, baseline, 1.0)
    
    return max(-0.30, min(0.30, ratio - 1.0))


def assign_lines(df_lineup: pd.DataFrame) -> Dict[str, int]:
    """
    Assign line numbers to players.
    
    Forwards: 3 per line (lines 1-4)
    Defense: 2 per pair (pairs 1-3)
    
    Args:
        df_lineup: Lineup DataFrame
    
    Returns:
        Dictionary mapping normalized names to line numbers
    """
    if df_lineup is None or df_lineup.empty:
        return {}
    
    try:
        validate_dataframe(df_lineup, required_columns=["Name", "Position"])
    except DataValidationError:
        return {}
    
    pos_up = df_lineup["Position"].map(str).str.upper()
    
    forwards_idx = [
        i for i, p in enumerate(pos_up)
        if p in ("C", "LW", "RW")
    ]
    defense_idx = [
        i for i, p in enumerate(pos_up)
        if p in ("D", "LD", "RD")
    ]
    
    mapping: Dict[str, int] = {}
    
    # Assign forward lines (3 per line)
    for k, i in enumerate(forwards_idx):
        line_num = min(4, (k // 3) + 1)
        mapping[normalize_name_key(df_lineup["Name"].iloc[i])] = line_num
    
    # Assign defense pairs (2 per pair)
    for k, i in enumerate(defense_idx):
        pair_num = min(3, (k // 2) + 1)
        mapping[normalize_name_key(df_lineup["Name"].iloc[i])] = pair_num
    
    return mapping


# ===================== GOAL SCORING ALLOCATION =====================

def goal_scorer_weights(
    df_lineup: pd.DataFrame,
    season_skill_cur: Dict[str, Dict[str, float]],
    recent_form: Dict[str, Dict[str, float]],
    last_game_goals: Dict[str, int],
    season_skill_prev: Optional[Dict[str, Dict[str, float]]] = None,
    early_season: bool = False,
) -> Dict[str, float]:
    """
    Calculate per-player goal scoring propensities.
    
    Args:
        df_lineup: Lineup DataFrame
        season_skill_cur: Current season skills
        recent_form: Recent form metrics
        last_game_goals: Goals in last game
        season_skill_prev: Previous season skills (for early season)
        early_season: Whether to blend with previous season
    
    Returns:
        Dictionary of normalized name -> weight
    """
    if df_lineup is None or df_lineup.empty:
        logger.debug("Empty lineup for goal weights")
        return {}
    
    try:
        validate_dataframe(df_lineup, required_columns=["Name", "Position"])
    except DataValidationError as e:
        logger.warning(f"Invalid lineup for goal weights: {e}")
        return {}
    
    line_map = assign_lines(df_lineup)
    w: Dict[str, float] = {}
    
    for _, row in df_lineup.iterrows():
        name = str(row["Name"])
        pos = str(row["Position"]).upper()
        key = normalize_name_key(name)
        
        # Get skill metrics
        cur = season_skill_cur.get(key, {})
        prev = (season_skill_prev or {}).get(key, {}) if early_season else {}
        rec = recent_form.get(key, {})
        
        gpg = float(cur.get("gpg", 0.0))
        sogpg = float(cur.get("sogpg", 0.0))
        gpg_prev = float(prev.get("gpg", 0.0))
        gpg_recent = float(rec.get("gpg", 0.0))
        lg = float(last_game_goals.get(key, 0))
        
        # Base position weight
        base_pos = 1.0 if pos in ("C", "LW", "RW") else 0.45
        
        # Line/pair weight
        li = line_map.get(key, 4 if pos in ("C", "LW", "RW") else 3)
        if pos in ("C", "LW", "RW"):
            line_w = PLAYER_PARAMS["line_weights"]["forward"].get(li, 0.52)
        else:
            line_w = PLAYER_PARAMS["line_weights"]["defense"].get(li, 0.45)
        
        # Hot/cold streaks
        hot = PLAYER_PARAMS["hot_streak_boost"] if lg >= 2 else (
            PLAYER_PARAMS["hot_streak_boost"] * 0.6 if lg == 1 else 0.0
        )
        cold = PLAYER_PARAMS["cold_streak_penalty"] if (lg == 0 and gpg_recent == 0.0) else 0.0
        
        # Early season blend with previous year
        prev_blend = (0.25 * gpg_prev) if early_season else 0.0
        
        # Compute score
        score = (
            0.50 * gpg +
            0.25 * gpg_recent +
            0.12 * safe_division(sogpg, 3.0, 0.0) +
            prev_blend
        )
        score = max(0.0, score) * base_pos * line_w
        score *= (1.0 + hot + cold)
        
        # Defense penalty
        if pos in ("D", "LD", "RD"):
            score *= 0.9
        
        w[key] = score
    
    # Ensure non-zero weights
    if sum(w.values()) == 0:
        logger.debug("All weights are zero, defaulting to uniform")
        for k in w:
            w[k] = 1.0
    
    return w


def allocate_goals_to_players(
    df_lineup: pd.DataFrame,
    weights: Dict[str, float],
    team_goals: int
) -> pd.DataFrame:
    """
    Allocate team goals to players using largest remainder method.
    
    Args:
        df_lineup: Lineup DataFrame
        weights: Player weights
        team_goals: Total team goals to allocate
    
    Returns:
        DataFrame with goal allocations
    """
    if df_lineup is None or df_lineup.empty:
        return pd.DataFrame(columns=["Name", "Position", "xG_allocation", "Assigned Goals"])
    
    try:
        validate_dataframe(df_lineup, required_columns=["Name", "Position"])
    except DataValidationError:
        return pd.DataFrame(columns=["Name", "Position", "xG_allocation", "Assigned Goals"])
    
    names = df_lineup["Name"].tolist()
    pos = [str(p).upper() for p in df_lineup["Position"].tolist()]
    
    # Get weights
    ws = [max(0.0, float(weights.get(normalize_name_key(n), 0.0))) for n in names]
    total = sum(ws) or 1.0
    
    # Calculate shares
    shares = [w / total for w in ws]
    quotas = [team_goals * s for s in shares]
    
    # Largest remainder method
    floors = [int(math.floor(q)) for q in quotas]
    remainder = team_goals - sum(floors)
    
    # Allocate remainder
    fracs = sorted(
        [(q - f, i) for i, (q, f) in enumerate(zip(quotas, floors))],
        reverse=True
    )
    
    for k in range(max(0, remainder)):
        if k < len(fracs):
            floors[fracs[k][1]] += 1
    
    # Build result
    rows = [
        {
            "Name": n,
            "Position": p,
            "xG_allocation": q,
            "Assigned Goals": g
        }
        for n, p, q, g in zip(names, pos, quotas, floors)
    ]
    
    return pd.DataFrame(rows)


def scorer_picks(df_alloc: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate scorer picks from goal allocations.
    
    Args:
        df_alloc: DataFrame with goal allocations
    
    Returns:
        Dictionary with first_goal and anytime picks
    """
    out = {"first_goal": "TBD", "anytime": []}
    
    if df_alloc is None or df_alloc.empty:
        return out
    
    try:
        validate_dataframe(df_alloc, required_columns=["Name", "Assigned Goals", "xG_allocation"])
    except DataValidationError:
        return out
    
    df_sorted = df_alloc.sort_values(
        ["Assigned Goals", "xG_allocation"],
        ascending=[False, False]
    ).reset_index(drop=True)
    
    # First goal scorer
    out["first_goal"] = str(df_sorted.iloc[0]["Name"])
    
    # Anytime scorers
    picks = []
    for _, r in df_sorted.head(5).iterrows():
        if float(r["xG_allocation"]) <= 0.0 and int(r["Assigned Goals"]) <= 0:
            continue
        
        picks.append({
            "Name": str(r["Name"]),
            "ExpGoals": round(float(r["xG_allocation"]), 2)
        })
    
    out["anytime"] = picks[:3]
    
    return out


# ===================== PLAYER RATES (PBP-derived) =====================
# As of v2, these come from NHL API play-by-play (via NHL.PlayByPlay +
# NHL.StatsFromPBP) rather than the old NST HTML scraper. The return
# shape is preserved: {name_key: {gpg, apg, sogpg, ...}} so callers in
# this file don't have to change.

@functools.lru_cache(maxsize=32)
def season_skater_rates_from_nst(
    season: str,
    stype: int,
    fd: str = "",
    td: str = ""
) -> Dict[str, Dict[str, float]]:
    """
    Per-player rates for a season, with optional date-window filtering.

    Backed by NHL/StatsFromPBP.compute_skater_rates (PBP-derived). The
    `season` arg is a YYYYZZZZ key (e.g. "20242025"); we convert to the
    start year for the shot store.

    Returns: {name_key: {gpg, apg, sogpg, xgf_pg, gp, goals, shots}}
    """
    try:
        season_start = int(season[:4]) if len(season) >= 4 else 2024
    except (ValueError, TypeError):
        season_start = 2024
    try:
        from NHL.StatsFromPBP import compute_skater_rates as _compute
        return _compute(season_start, stype, fd=fd, td=td)
    except Exception as e:
        logger.warning(f"compute_skater_rates failed: {e}")
        return {}


# Old NST internals are kept as a private alias for any code that imports
# them by name. New code should call season_skater_rates_from_nst directly.
_load_nst_players_df = None  # type: ignore[assignment]  # removed in v2



# ===================== SHOT & ASSIST MODELING =====================

def estimate_player_toi_minutes(df_lineup: pd.DataFrame) -> Dict[str, float]:
    """Estimate TOI minutes for each player based on line assignment"""
    if df_lineup is None or df_lineup.empty:
        return {}
    
    try:
        validate_dataframe(df_lineup, required_columns=["Name", "Position"])
    except DataValidationError:
        return {}
    
    line_map = assign_lines(df_lineup)
    out = {}
    
    for _, row in df_lineup.iterrows():
        name = str(row["Name"])
        pos = str(row["Position"]).upper()
        key = normalize_name_key(name)
        
        li = line_map.get(key, 4 if pos in ("C", "LW", "RW") else 3)
        
        if pos in ("C", "LW", "RW"):
            minutes = PLAYER_PARAMS["toi_estimates"]["forward"].get(li, 12.0)
        elif pos in ("LD", "RD", "D"):
            minutes = PLAYER_PARAMS["toi_estimates"]["defense"].get(li, 16.0)
        else:
            minutes = 12.0
        
        out[key] = minutes
    
    return out


def generate_expected_shots_by_player(
    df_lineup: pd.DataFrame,
    season_skill: Dict[str, Dict[str, float]],
    expected_team_shots: float,
    pp_bias: float = 1.0,
    weight_top6: float = None
) -> Dict[str, float]:
    """Generate expected shots per player normalized to team total"""
    if weight_top6 is None:
        weight_top6 = PLAYER_PARAMS["top6_boost"]
    
    if df_lineup is None or df_lineup.empty:
        logger.debug("Empty lineup for shot generation")
        return {}
    
    try:
        validate_dataframe(df_lineup, required_columns=["Name", "Position"])
    except DataValidationError:
        return {}
    
    toi_map = estimate_player_toi_minutes(df_lineup)
    base = {}
    total_base = 0.0
    
    for _, row in df_lineup.iterrows():
        name = str(row["Name"])
        pos = str(row["Position"]).upper()
        key = normalize_name_key(name)
        
        sogpg = float(season_skill.get(key, {}).get("sogpg", 0.0))
        toi = toi_map.get(key, 14.0)
        
        # Scale by TOI and PP bias
        est = sogpg * safe_division(toi, 60.0, 1.0) * pp_bias
        
        # Boost top-6 forwards
        if pos in ("C", "LW", "RW") and toi >= 17.0:
            est *= weight_top6
        
        base[key] = max(0.0, est)
        total_base += base[key]
    
    # Normalize to team total
    if total_base <= 0:
        n = len(base) or 1
        return {k: safe_division(expected_team_shots, n, 0.0) for k in base}
    
    scale = safe_division(expected_team_shots, total_base, 1.0)
    return {k: v * scale for k, v in base.items()}


def simulate_player_shots_and_goals(
    df_lineup: pd.DataFrame,
    season_skill: Dict[str, Dict[str, float]],
    expected_team_shots: float,
    rng_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate shots and goals for each player with deterministic seeding"""
    if df_lineup is None or df_lineup.empty:
        return pd.DataFrame(columns=["Name", "Position", "ExpShots", "SimShots", "SimGoals", "ShotPct"])
    
    # Use deterministic seed if provided
    rng = np.random.default_rng(rng_seed)
    expected_shots = generate_expected_shots_by_player(df_lineup, season_skill, expected_team_shots)
    
    rows = []
    
    for _, row in df_lineup.iterrows():
        name = str(row["Name"])
        pos = str(row["Position"]).upper()
        key = normalize_name_key(name)
        
        exp_sh = float(expected_shots.get(key, 0.0))
        
        # Poisson draw for shots
        sim_sh = int(rng.poisson(exp_sh)) if exp_sh > 0 else 0
        
        # Shooting percentage from skills
        sk = season_skill.get(key, {})
        sogpg = float(sk.get("sogpg", 0.0))
        gpg = float(sk.get("gpg", 0.0))
        
        if sogpg > 0:
            sh_pct = min(
                PLAYER_PARAMS["max_shooting_pct"],
                max(PLAYER_PARAMS["min_shooting_pct"], safe_division(gpg, sogpg, 0.092))
            )
        else:
            sh_pct = LEAGUE_AVERAGES["shooting_pct"]
        
        # Binomial draw for goals
        sim_goals = int(rng.binomial(sim_sh, sh_pct)) if sim_sh > 0 else 0
        
        rows.append({
            "Name": name,
            "Position": pos,
            "ExpShots": exp_sh,
            "SimShots": sim_sh,
            "SimGoals": sim_goals,
            "ShotPct": round(sh_pct, 4),
        })
    
    return pd.DataFrame(rows)


def _linemate_weights_for_assists(
    df_lineup: pd.DataFrame,
    season_skill: Dict[str, Dict[str, float]],
    team_GFpg: float,
    line_favor_top6: float = None,
    pp_favor: float = None
) -> Dict[str, float]:
    """Create per-player assist propensity weights"""
    if line_favor_top6 is None:
        line_favor_top6 = PLAYER_PARAMS["top6_boost"]
    if pp_favor is None:
        pp_favor = PLAYER_PARAMS["pp_boost"]
    
    if df_lineup is None or df_lineup.empty:
        return {}
    
    try:
        validate_dataframe(df_lineup, required_columns=["Name", "Position"])
    except DataValidationError:
        return {}
    
    toi = estimate_player_toi_minutes(df_lineup)
    out = {}
    
    for _, r in df_lineup.iterrows():
        name = r["Name"]
        key = normalize_name_key(name)
        
        apg = float(season_skill.get(key, {}).get("apg", 0.0))
        
        # Normalize to team context
        if team_GFpg and team_GFpg > 0:
            ast_pct = safe_division(apg, team_GFpg, 0.0)
        else:
            ast_pct = apg
        
        # Weight by TOI
        minutes = toi.get(key, 14.0)
        w = ast_pct * safe_division(minutes, 18.0, 1.0)
        
        # Boost top-6 forwards
        if r["Position"] in ("C", "LW", "RW") and minutes >= 17.0:
            w *= line_favor_top6
        
        # PP bias for high-TOI players
        if minutes >= 17.0:
            w *= pp_favor
        
        out[key] = max(0.0, w)
    
    # Ensure non-zero
    if sum(out.values()) <= 0:
        n = len(out) or 1
        return {k: 1.0 for k in out}
    
    return out


def assign_assists_for_goals(
    df_goals_by_scorer: pd.DataFrame,
    df_lineup: pd.DataFrame,
    season_skill: Dict[str, Dict[str, float]],
    team_GFpg: float = None,
    rng_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Assign primary and secondary assists for goals with deterministic seeding"""
    if team_GFpg is None:
        team_GFpg = LEAGUE_AVERAGES["goals_per_game"]
    
    if df_goals_by_scorer is None or df_goals_by_scorer.empty:
        return []
    
    if df_lineup is None or df_lineup.empty:
        return []
    
    # Use deterministic seed if provided
    rng = np.random.default_rng(rng_seed)
    events: List[Dict[str, Any]] = []
    
    assist_weights = _linemate_weights_for_assists(df_lineup, season_skill, team_GFpg)
    name_map = {normalize_name_key(r["Name"]): r["Name"] for _, r in df_lineup.iterrows()}
    
    names = list(assist_weights.keys())
    weights = np.array([assist_weights[n] for n in names], dtype=float)
    
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    
    probs = safe_division(weights, weights.sum(), 1.0 / len(weights))
    
    for _, row in df_goals_by_scorer.iterrows():
        scorer_key = normalize_name_key(row["Name"])
        goals = int(row.get("SimGoals", 0))
        
        for _ in range(goals):
            # Primary assist (exclude scorer)
            mask = np.ones_like(probs)
            try:
                idx_scorer = names.index(scorer_key)
                mask[idx_scorer] = 0.0
            except ValueError:
                idx_scorer = None
            
            p = probs * mask
            if p.sum() <= 0:
                p = probs.copy()
            p = safe_division(p, p.sum(), 1.0 / len(p))
            
            prim_idx = rng.choice(len(names), p=p)
            primary = name_map.get(names[prim_idx], "")
            
            # Secondary assist (exclude scorer)
            mask2 = np.ones_like(probs)
            if idx_scorer is not None:
                mask2[idx_scorer] = 0.0
            
            p2 = probs * mask2
            if p2.sum() <= 0:
                p2 = probs.copy()
            p2 = safe_division(p2, p2.sum(), 1.0 / len(p2))
            
            sec_idx = rng.choice(len(names), p=p2)
            secondary = name_map.get(names[sec_idx], "")
            
            events.append({
                "scorer": row["Name"],
                "primary": primary,
                "secondary": secondary
            })
    
    return events


# ===================== GOALTENDER UTILITIES =====================

def _goalie_row_by_name(
    goalie_df: pd.DataFrame,
    name: str,
    team_abbr: Optional[str] = None
) -> Optional[pd.Series]:
    """Find goalie row by name with fuzzy matching"""
    if goalie_df is None or goalie_df.empty or not name:
        return None
    
    df = normalize_sv_column(goalie_df.copy())
    
    # Get player column
    player_col = get_column_safe(df, COLUMN_VARIATIONS, "player")
    if player_col is None:
        return None
    
    # Filter by team if provided
    if team_abbr:
        tcol = get_column_safe(df, COLUMN_VARIATIONS, "team")
        if tcol:
            aliases = [team_abbr.upper()]
            u = df[tcol].astype(str).str.upper()
            mask = u.apply(lambda s: any(a == s or a in s for a in aliases))
            df = df[mask] if mask.any() else df
    
    # Normalize names for matching
    df = df.assign(
        _player=df[player_col].astype(str),
        _norm=lambda d: d["_player"].map(normalize_name_key),
        _last=lambda d: d["_player"].map(last_token_norm)
    )
    
    norm_in = normalize_name_key(name)
    last_in = last_token_norm(name)
    
    # Try exact normalized match
    cand = df[df["_norm"] == norm_in]
    
    # Try last name match
    if cand.empty and last_in:
        cand = df[df["_last"] == last_in]
        
        # Try substring match
        if cand.empty:
            cand = df[df["_player"].str.contains(last_in, case=False, na=False)]
    
    # Try fuzzy match as last resort
    if cand.empty:
        candidates = df["_player"].tolist()
        match = fuzzy_match_player_name(name, candidates, threshold=0.7)
        if match:
            cand = df[df["_player"] == match]
    
    if cand.empty:
        logger.debug(f"No match found for goalie: {name}")
        return None
    
    # Sort by games started, games played, save percentage
    gs_col = get_column_safe(cand, COLUMN_VARIATIONS, "games_started")
    gp_col = get_column_safe(cand, COLUMN_VARIATIONS, "games_played")
    sv_col = get_column_safe(cand, COLUMN_VARIATIONS, "save_pct")
    
    def tonum(s):
        return pd.to_numeric(s, errors="coerce").fillna(0)
    
    cand = cand.assign(
        _gs=tonum(cand[gs_col]) if gs_col else 0,
        _gp=tonum(cand[gp_col]) if gp_col else 0,
        _sv=tonum(cand[sv_col]) if sv_col else 0.0
    ).sort_values(by=["_gs", "_gp", "_sv"], ascending=[False, False, False])
    
    return cand.iloc[0]


def _goalie_stats_from_row(r: pd.Series) -> Dict[str, Any]:
    """Extract goalie stats from DataFrame row"""
    if r is None or not isinstance(r, pd.Series):
        return {}
    
    out: Dict[str, Any] = {}
    
    def g(cands, default=None):
        c = get_column_safe(pd.DataFrame([r]), COLUMN_VARIATIONS, cands) if isinstance(cands, str) else None
        if not c:
            for cand in (cands if isinstance(cands, list) else [cands]):
                if cand in r.index:
                    return r.get(cand, default)
        if c:
            return r.get(c, default)
        return default
    
    sv = g("save_pct")
    dsv = g(["dSV%", "dSv%"])
    gp = g("games_played")
    gs = g("games_started")
    ga = g("goals_against")
    sa = g("shots_against")
    svs = g(["SV", "Saves"])
    toi = g(["TOI", "TOI (All)", "TOI(All)"])
    
    # Calculate GAA if possible
    gaa = None
    try:
        ga_v = float(pd.to_numeric(ga, errors="coerce"))
        toi_v = None
        
        if isinstance(toi, str) and ":" in toi:
            parts = str(toi).split(":")
            if len(parts) == 2:
                m, s = int(parts[0]), int(parts[1])
                toi_v = m + s / 60.0
            elif len(parts) == 3:
                h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
                toi_v = h * 60.0 + m + s / 60.0
        else:
            toi_v = float(pd.to_numeric(toi, errors="coerce"))
        
        if ga_v >= 0 and toi_v and toi_v > 0:
            hours = safe_division(toi_v, 60.0, 0.0)
            if hours > 0:
                gaa = safe_division(ga_v, hours, None)
    except Exception:
        gaa = None
    
    def fnum(x):
        try:
            return float(pd.to_numeric(x, errors="coerce"))
        except Exception:
            return None
    
    out["Sv%"] = fnum(sv) if sv is not None else None
    out["dSV%"] = fnum(dsv) if dsv is not None else None
    out["GP"] = int(pd.to_numeric(gp, errors="coerce")) if gp is not None and str(gp) != "" else None
    out["GS"] = int(pd.to_numeric(gs, errors="coerce")) if gs is not None and str(gs) != "" else None
    
    if gaa is not None and np.isfinite(gaa):
        out["GAA"] = float(gaa)
    else:
        out["GAA"] = None
    
    out["SA"] = int(pd.to_numeric(sa, errors="coerce")) if sa is not None and str(sa) != "" else None
    out["SV"] = int(pd.to_numeric(svs, errors="coerce")) if svs is not None and str(svs) != "" else None
    out["TOI"] = str(toi) if toi is not None and str(toi).strip() else None
    
    return out


def _top_two_team_goalies_from_pbp(
    pbp_goalie_df: pd.DataFrame,
    team_abbr_raw: str
) -> List[str]:
    """
    Get top 2 goalies for a team from PBP-derived goalie rates.

    The team_abbr match is best-effort: the goalie row doesn't carry a
    team abbrev, so we look up the goalie's most-recent team via the
    PBP shot store. For the current implementation we just return the
    top 2 by GP and let the caller filter further if needed.
    """
    if pbp_goalie_df is None or pbp_goalie_df.empty:
        return []
    df = pbp_goalie_df.copy()
    if "gp" in df.columns:
        df = df.sort_values("gp", ascending=False)
    names = df["name"].astype(str).tolist()
    names_fmt = [format_initial_last(sanitize_text(nm)) for nm in names]
    seen = set()
    uniq = []
    for nm in names_fmt:
        if nm not in seen:
            seen.add(nm)
            uniq.append(nm)
        if len(uniq) >= 2:
            break
    return uniq


def _pbp_players_url(season: int, stype: int) -> str:
    """Placeholder kept for backwards compat — no longer used (data is on disk)."""
    return f"pbp://skaters/{season}/{stype}"


def _pbp_goalie_url(season: int, stype: int) -> str:
    """Placeholder kept for backwards compat — no longer used."""
    return f"pbp://goalies/{season}/{stype}"


@retry_on_failure(max_attempts=3, backoff_base=0.75)
def get_player_and_goalie_names(
    team_abbr: str,
    season: str,
    stype: int = 2
) -> Dict[str, List[str]]:
    """
    Get player and goalie names for a team with intelligent fallback.

    As of v2, data comes from the PBP shot store (NHL/StatsFromPBP) rather
    than scraping NST HTML. Fallback order is the same (current-season
    regular → preseason → previous-season regular → playoffs).
    """
    players: List[str] = []
    goalies: List[str] = []

    prev_season = prev_season_key(season)
    fallback_order = [
        (season, 2, "current season regular"),
        (season, 1, "current season preseason"),
        (prev_season, 2, "previous season regular"),
        (prev_season, 3, "previous season playoffs"),
    ]

    for try_season, try_stype, desc in fallback_order:
        try:
            season_start = int(try_season[:4]) if len(str(try_season)) >= 4 else 2024
        except (ValueError, TypeError):
            continue
        logger.debug(f"Trying {desc} data for {team_abbr} (season={try_season}, stype={try_stype})")

        # Players from PBP-derived rates
        try:
            from NHL.StatsFromPBP import compute_skater_rates
            skater_rates = compute_skater_rates(season_start, try_stype)
            for name_key, info in skater_rates.items():
                nm_fmt = format_initial_last(sanitize_text(info.get("name", "")))
                if nm_fmt and nm_fmt not in players:
                    players.append(nm_fmt)
        except Exception as e:
            logger.debug(f"Could not fetch players from PBP ({desc}): {e}")

        # Goalies from PBP-derived rates
        try:
            from NHL.StatsFromPBP import compute_goalie_rates
            goalie_df = compute_goalie_rates(season_start, try_stype)
            if not goalie_df.empty and "name" in goalie_df.columns:
                for nm in goalie_df["name"].astype(str).tolist():
                    nm_fmt = format_initial_last(sanitize_text(nm))
                    if nm_fmt and nm_fmt not in goalies:
                        goalies.append(nm_fmt)
        except Exception as e:
            logger.debug(f"Could not fetch goalies from PBP ({desc}): {e}")

        if goalies:
            logger.info(f"Found {len(goalies)} goalies for {team_abbr} using {desc} data")
            break

    # NHL API roster fallback (only if still no goalies)
    if not goalies:
        logger.info(f"No goalies found in PBP, trying NHL API for {team_abbr}")
        try:
            teams_js = safe_api_call(
                lambda: requests.get(
                    f"{NHL_STATS_API_BASE}/teams",
                    headers=REQUEST_HEADERS,
                    timeout=DEFAULT_TIMEOUT
                ).json(),
                service_name="NHL Teams API",
                fallback=None
            )
            
            if teams_js:
                team_id = None
                for t in teams_js.get("teams", []):
                    if (str(t.get("abbreviation", "")).upper() == team_abbr.upper() or
                        str(t.get("teamName", "")).upper().find(team_abbr.upper()) >= 0):
                        team_id = t.get("id")
                        break
                
                if team_id is not None:
                    # Try current season first, then previous
                    for api_season in [season, prev_season]:
                        roster_js = safe_api_call(
                            lambda: requests.get(
                                f"{NHL_STATS_API_BASE}/teams/{team_id}?expand=team.roster&season={api_season}",
                                headers=REQUEST_HEADERS,
                                timeout=DEFAULT_TIMEOUT
                            ).json(),
                            service_name="NHL Roster API",
                            fallback=None
                        )
                        
                        if roster_js:
                            roster = roster_js.get("teams", [])[0].get("roster", {}).get("roster", []) if roster_js.get("teams") else []
                            for person in roster:
                                pn = (person.get("person") or {}).get("fullName")
                                pos = (person.get("position") or {}).get("code", "")
                                
                                if pn:
                                    nm_fmt = format_initial_last(sanitize_text(pn))
                                    if (pos or "").upper() in ("G", "GK"):
                                        if nm_fmt not in goalies:
                                            goalies.append(nm_fmt)
                                    else:
                                        if nm_fmt not in players:
                                            players.append(nm_fmt)
                            
                            if goalies:
                                logger.info(f"Found {len(goalies)} goalies from NHL API (season {api_season})")
                                break
        except Exception as e:
            logger.debug(f"Could not fetch from NHL API: {e}")
    
    logger.info(f"Final result for {team_abbr}: {len(players)} players, {len(goalies)} goalies")
    return {"players": players, "goalies": goalies}


# ===================== PREDICT PLAYER/GOALIE STATS =====================

def predict_player_stats(
    player_name: str,
    season_current: str,
    season_prev: str,
    stype: int = 2,
    weight_current: float = None,
    recent_games_stats: Optional[dict] = None
) -> Dict[str, Any]:
    """Predict player stats blending current and previous season"""
    if weight_current is None:
        weight_current = MODEL_WEIGHTS["season_blend_current"]
    
    cur_rates = season_skater_rates_from_nst(season_current, stype)
    prev_rates = season_skater_rates_from_nst(season_prev, stype)
    
    key = normalize_name_key(format_initial_last(sanitize_text(player_name)))
    
    cur = cur_rates.get(key, {"gpg": 0.0, "apg": 0.0, "sogpg": 0.0})
    prev = prev_rates.get(key, {"gpg": 0.0, "apg": 0.0, "sogpg": 0.0})
    
    w = max(0.0, min(1.0, float(weight_current)))
    
    pred_gpg = w * float(cur.get("gpg", 0.0)) + (1.0 - w) * float(prev.get("gpg", 0.0))
    pred_apg = w * float(cur.get("apg", 0.0)) + (1.0 - w) * float(prev.get("apg", 0.0))
    pred_sogpg = w * float(cur.get("sogpg", 0.0)) + (1.0 - w) * float(prev.get("sogpg", 0.0))
    
    # Blend with recent form if available
    if recent_games_stats:
        pred_gpg = 0.7 * pred_gpg + 0.3 * recent_games_stats.get("gpg", pred_gpg)
        pred_apg = 0.7 * pred_apg + 0.3 * recent_games_stats.get("apg", pred_apg)
        pred_sogpg = 0.7 * pred_sogpg + 0.3 * recent_games_stats.get("sogpg", pred_sogpg)
    
    return {
        "Name": format_initial_last(sanitize_text(player_name)),
        "Pred_gpg": round(pred_gpg, 3),
        "Pred_apg": round(pred_apg, 3),
        "Pred_sogpg": round(pred_sogpg, 3),
    }


def predict_goalie_stats(
    goalie_name: str,
    team_abbr: str,
    season_current: str,
    season_prev: str,
    stype: int = 2,
    weight_current: float = None,
    nst_goalie_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Predict goalie stats blending seasons"""
    if weight_current is None:
        weight_current = 1.0  # Heavier current season weight for goalies