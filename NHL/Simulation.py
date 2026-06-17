"""
Simulation and analytics module with:
- PURE data-driven home/away advantage (no artificial floors)
- Automatic injury impact calculation from player stats
- Current season Elo integration (2025-2026 only) with SMART ML GUARD
- Rest/travel fatigue and penalty differential
- Goalie impact with optional GSAA-like adjustment
- Correlated goal modeling (shared factor)
- Improved empty net model
- Calibration hook
- Expanded outputs: totals distribution, reg/OT/SO probs, confidence, component breakdown

UPDATED: Home advantage purely from actual home/away records (no floors/ceilings).
UPDATED: Rangers-style road warriors can have advantage over weak home teams.
UPDATED: ML Guard Rail blends predictions instead of overriding.
UPDATED: Reduced venue advantage scaling for more realistic predictions.
UPDATED: Score effects removed from pre-simulation expected goals (in-game state only).
UPDATED: Added sanity checks for extreme predictions.
FIXED: Team matching with proper NST mapping and reverse lookup.
FIXED: Safe default handling when schedule data unavailable.
FIXED: safe_division call with correct parameters.
FIXED: Score effects no longer applied to pre-simulation mus.
FIXED: Variable name error in goalie function (df_goalie -> goalie_df).
"""
from __future__ import annotations
import math
import re
import time
from collections import Counter
from datetime import date as _date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# NST.Cache no longer imported — get_player_and_goalie_names and the
# injury stats function have been migrated to PBP-based equivalents.
from NHL.Config import (
    RATE_LIMIT_SLEEP_SECONDS,
    RATE_LIMIT_JITTER_SECONDS,
    SIMULATION_PARAMS,
    LEAGUE_AVERAGES,
    MODEL_WEIGHTS,
    NHL_API_BASE,
    REQUEST_HEADERS,
    DEFAULT_TIMEOUT,
    COLUMN_VARIATIONS,
    GOALIE_PARAMS,
    SPECIAL_TEAMS_PARAMS,
    NST_ABBR_TO_FULL,
    VENUE_ADV_PARAMS,
)
from NHL.Errors import (
    retry_on_failure,
    safe_division,
    SimulationError,
    log_performance
)
from NHL.Utils import (
    season_from_date,
    prev_season_key,
    normalize_name_key,
    get_column_safe,
    normalize_sv_column,
    last_token_norm,
)

# Feature helpers and calibration
from NHL.Features import (
    compute_rest_travel_features_fast,
    fatigue_multiplier,
    penalty_diff_per60,
    shared_correlation_factor,
    component_breakdown,
)

from Calibration import calibrate_prob, Calibrator  # Imported for calibration
from NHL.StatsFromPBP import TEAM_ID_TO_ABBR

# Import persistent app state instead of creating new instances
from NHL.AppState import get_app_state

# Goalie selection from each team's most recent completed game
from NHL.ApiScrape import get_roster_goalies_for_override
from NHL.GoaliePrediction import predict_starting_goalie

import logging
logger = logging.getLogger(__name__)

# ── In-process caches for expensive PBP computations ─────────────────
_RATES_CACHE: Dict[str, Tuple[float, Any]] = {}
_RATES_CACHE_TTL = 300  # seconds
_INJURY_CACHE: Dict[str, Tuple[float, Tuple[Dict, Dict]]] = {}
_INJURY_CACHE_TTL = 300  # seconds


def _rate_limit_sleep():
    jitter = float(np.random.uniform(0, RATE_LIMIT_JITTER_SECONDS)) if RATE_LIMIT_JITTER_SECONDS > 0 else 0.0
    time.sleep(RATE_LIMIT_SLEEP_SECONDS + jitter)

def _norm_team(s: str) -> str:
    """
    Normalize team name with NST mapping first, including reverse lookup.
    Handles both abbreviations and full names.
    """
    s_upper = str(s).upper().strip()
    
    # First, try direct abbreviation lookup
    if s_upper in NST_ABBR_TO_FULL:
        s_upper = NST_ABBR_TO_FULL[s_upper].upper()
    else:
        # Try reverse lookup for full names
        for abbr, full_name in NST_ABBR_TO_FULL.items():
            if s_upper == full_name.upper():
                s_upper = full_name.upper()
                break
    
    # Return alphanumeric version
    return "".join(ch for ch in s_upper if ch.isalnum())

def _match_team(df: pd.DataFrame, abbr: str, team_col: str) -> pd.DataFrame:
    """
    Helper to find team rows with strict matching first, then fallback to contains.
    Falls back to NHL team_id when the team name column is unreliable.
    Prevents picking up wrong teams or failing silently.
    """
    target = _norm_team(abbr)
    norm_col = df[team_col].astype(str).map(_norm_team)

    # 1. Exact match
    subset = df[norm_col == target]
    if not subset.empty:
        return subset

    # 2. Contains match (fallback)
    subset = df[norm_col.str.contains(target, na=False, regex=False)]
    if not subset.empty:
        return subset

    # 3. Try matching original abbr directly (for edge cases)
    target_orig = str(abbr).upper().strip()
    for nst_abbr, nst_full in NST_ABBR_TO_FULL.items():
        if target_orig in (nst_abbr, nst_full.upper()):
            norm_target = "".join(ch for ch in nst_full.upper() if ch.isalnum())
            subset = df[norm_col == norm_target]
            if not subset.empty:
                return subset

    # 4. Fallback: match by NHL team_id if available
    team_id_col = get_column_safe(df, {"team_id": ["team_id", "TeamID", "id", "teamId"]}, "team_id")
    if team_id_col:
        abbr_to_id = {v: k for k, v in TEAM_ID_TO_ABBR.items()}
        from NHL.Config import TEAM_ABBR_MAPPING
        lookup_abbr = TEAM_ABBR_MAPPING.get(abbr.upper(), abbr.upper())
        target_id = abbr_to_id.get(lookup_abbr)
        if target_id is not None:
            subset = df[df[team_id_col].astype(str) == str(target_id)]
            if not subset.empty:
                logger.debug(f"Matched {abbr} by team_id {target_id}")
                return subset

    return pd.DataFrame()

def _parse_game_date(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        try:
            return datetime.fromisoformat(str(raw))
        except Exception:
            return None

@retry_on_failure(max_attempts=3, backoff_base=0.75)
def _safe_get_json(url: str) -> Optional[Dict[str, Any]]:
    _rate_limit_sleep()
    try:
        r = requests.get(url, headers=REQUEST_HEADERS, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"Failed to fetch JSON from {url[:80]}: {e}")
        return None

def nst_team_url(season: str, stype: int, sit: str, rate: str = "n", fd: str = "", td: str = "") -> str:
    """
    DEPRECATED in v2. Use NHL/StatsFromPBP.compute_team_rates instead.
    Kept as a stub so any code that imports it by name still resolves.
    """
    return ""


def nst_goalie_url(season: str, stype: int, sit: str = "all", fd: str = "", td: str = "", gpfilt: str = "none") -> str:
    """
    DEPRECATED in v2. Use NHL/StatsFromPBP.compute_goalie_rates instead.
    """
    return ""


def _fetch_nst_df(url: str) -> pd.DataFrame:
    """DEPRECATED in v2. Returns empty DataFrame; callers should use
    NHL/StatsFromPBP functions directly."""
    return pd.DataFrame()


def get_team_rates_all(season: str, stype: int, fd: str = "", td: str = "") -> pd.DataFrame:
    """Return team rates from the lightweight exported JSON cache only.

    We intentionally do NOT fall back to compute_team_rates here because that
    loads the full PBP parquet and can exceed Render's memory on a web request.
    Run `python update_pbp_stats.py` to refresh the cache.
    """
    try:
        season_start = int(season[:4]) if len(str(season)) >= 4 else 2024
    except (ValueError, TypeError):
        season_start = 2024
    key = f"all_{season_start}_{stype}_{fd}_{td}"
    now = time.time()
    cached = _RATES_CACHE.get(key)
    if cached and (now - cached[0]) < _RATES_CACHE_TTL:
        return cached[1].copy() if isinstance(cached[1], pd.DataFrame) else cached[1]

    try:
        from NHL.StatsFromPBP import load_team_rates_from_json
        df = load_team_rates_from_json(season_start, stype)
        if not df.empty:
            _RATES_CACHE[key] = (now, df.copy())
            return df.copy()
    except Exception as e:
        logger.debug(f"JSON team rates failed: {e}")

    logger.warning(
        f"Team rates cache missing for {season_start}-{season_start + 1}; "
        "returning empty DataFrame. Run python update_pbp_stats.py to rebuild."
    )
    return pd.DataFrame()


def get_team_rates_ev(season: str, stype: int, fd: str = "", td: str = "") -> pd.DataFrame:
    """PBP-backed. (ev = even-strength slice; future work to filter.)"""
    return get_team_rates_all(season, stype, fd=fd, td=td)



def get_team_rates_pp_per60(season: str, stype: int, fd: str = "", td: str = "") -> pd.DataFrame:
    """PBP-backed. (PP slice; future work to filter by situation.)"""
    return get_team_rates_all(season, stype, fd=fd, td=td)


def get_team_rates_pk_per60(season: str, stype: int, fd: str = "", td: str = "") -> pd.DataFrame:
    """PBP-backed. (PK slice; future work to filter by situation.)"""
    return get_team_rates_all(season, stype, fd=fd, td=td)

def get_goalie_table(season: str, stype: int, fd: str = "", td: str = "") -> pd.DataFrame:
    """Return goalie rates from the lightweight exported JSON cache only.

    We intentionally do NOT fall back to compute_goalie_rates here because that
    loads the full PBP parquet and can exceed Render's memory on a web request.
    """
    try:
        season_start = int(season[:4]) if len(str(season)) >= 4 else 2024
    except (ValueError, TypeError):
        season_start = 2024
    key = f"goalie_{season_start}_{stype}_{fd}_{td}"
    now = time.time()
    cached = _RATES_CACHE.get(key)
    if cached and (now - cached[0]) < _RATES_CACHE_TTL:
        return cached[1].copy() if isinstance(cached[1], pd.DataFrame) else cached[1]

    try:
        from NHL.StatsFromPBP import load_goalie_rates_from_json
        df = load_goalie_rates_from_json(season_start, stype)
        if not df.empty:
            _RATES_CACHE[key] = (now, df.copy())
            return df.copy()
    except Exception as e:
        logger.debug(f"JSON goalie rates failed: {e}")

    logger.warning(
        f"Goalie rates cache missing for {season_start}-{season_start + 1}; "
        "returning empty DataFrame. Run python update_pbp_stats.py to rebuild."
    )
    return pd.DataFrame()

def _extract_pp_pk_rates(pp_df: pd.DataFrame, pk_df: pd.DataFrame, abbr: str) -> Tuple[float, float]:
    pp_gf60 = 0.0
    pk_ga60 = 0.0
    
    # 1. PP Extraction
    if isinstance(pp_df, pd.DataFrame) and not pp_df.empty:
        try:
            team_col = get_column_safe(pp_df, COLUMN_VARIATIONS, "team")
            if team_col:
                subset = _match_team(pp_df, abbr, team_col)
                if not subset.empty:
                    r = subset.iloc[0]
                    gf60_col = get_column_safe(pp_df, {"gf60": ["GF/60", "GF60", "GF per 60", "GF60.0"]}, "gf60")
                    if gf60_col:
                        pp_gf60 = float(r.get(gf60_col) or 0.0)
        except Exception as e:
            logger.debug(f"Could not extract PP rate for {abbr}: {e}")

    # 2. PK Extraction
    if isinstance(pk_df, pd.DataFrame) and not pk_df.empty:
        try:
            team_col = get_column_safe(pk_df, COLUMN_VARIATIONS, "team")
            if team_col:
                subset = _match_team(pk_df, abbr, team_col)
                if not subset.empty:
                    r = subset.iloc[0]
                    ga60_col = get_column_safe(pk_df, {"ga60": ["GA/60", "GA60", "GA per 60", "GA60.0"]}, "ga60")
                    if ga60_col:
                        pk_ga60 = float(r.get(ga60_col) or 0.0)
        except Exception as e:
            logger.debug(f"Could not extract PK rate for {abbr}: {e}")
            
    return pp_gf60, pk_ga60

def derive_all_pg_metrics(all_df: pd.DataFrame, abbr: str) -> Dict[str, float]:
    """
    Derive per-game team metrics from a team-rates DataFrame.
    Supports both the old NST uppercase columns and the PBP lowercase columns.
    Falls back to league averages for any missing fields.
    """
    defaults = {
        "xGFpg": LEAGUE_AVERAGES["goals_per_game"],
        "xGApG": LEAGUE_AVERAGES["goals_per_game"],
        "GFpg": LEAGUE_AVERAGES["goals_per_game"],
        "GApG": LEAGUE_AVERAGES["goals_per_game"],
        "SFpg": LEAGUE_AVERAGES["shots_per_game"],
        "SApg": LEAGUE_AVERAGES["shots_per_game"],
        "SvPct": LEAGUE_AVERAGES["sv_pct"],
        "FFpg": LEAGUE_AVERAGES["shots_per_game"],
        "CFpg": LEAGUE_AVERAGES["shots_per_game"],
        "xGF%": 50.0,
        "SCF%": 50.0,
        "HDCF%": 50.0,
        "PDO": 100.0,
    }
    if not isinstance(all_df, pd.DataFrame) or all_df.empty:
        return defaults
    try:
        team_col = get_column_safe(all_df, COLUMN_VARIATIONS, "team")
        if not team_col:
            return defaults

        subset = _match_team(all_df, abbr, team_col)
        if subset.empty:
            logger.warning(
                f"⚠️ MISSING STATS: Could not find '{abbr}' in team rates DataFrame. "
                f"Available teams: {list(all_df[team_col].unique())[:5]}..."
            )
            return defaults

        r = subset.iloc[0]

        # Helper to read a field trying several possible column names.
        def _get(candidates: List[str], default: float = 0.0) -> float:
            for c in candidates:
                if c in subset.columns:
                    try:
                        v = r.get(c)
                        if v is None or (isinstance(v, float) and np.isnan(v)):
                            continue
                        return float(v)
                    except Exception:
                        continue
            return default

        gp = max(1.0, _get(["GP", "gp"], 1.0))
        xgf = _get(["xGF", "xgf"], 0.0)
        xga = _get(["xGA", "xga"], 0.0)
        gf = _get(["GF", "gf"], 0.0)
        ga = _get(["GA", "ga"], 0.0)
        sf = _get(["SF", "sf"], 0.0)
        sa = _get(["SA", "sa"], 0.0)
        ff = _get(["FF", "ff"], 0.0)
        cf = _get(["CF", "cf"], 0.0)

        # Percentage-style fields may be 0-100 or 0-1; try both column names.
        xgf_pct = _get(["xGF%", "xgf_pct"], 50.0)
        if 0 < xgf_pct <= 1.0:
            xgf_pct *= 100.0

        scf_pct = _get(["SCF%", "scf_pct", "cf_pct"], 50.0)
        if 0 < scf_pct <= 1.0:
            scf_pct *= 100.0

        hdcf_pct = _get(["HDCF%", "hdcf_pct"], 50.0)
        if 0 < hdcf_pct <= 1.0:
            hdcf_pct *= 100.0

        pdo = _get(["PDO", "pdo"], 100.0)
        if pd.isna(pdo):
            pdo = 100.0

        sv_col = get_column_safe(all_df, {"sv_pct": ["Sv%", "sv_pct", "SV_PCT", "Save%"]}, "sv_pct")
        sv = defaults["SvPct"]
        if sv_col:
            try:
                raw_sv = r.get(sv_col)
                if raw_sv is not None and not (isinstance(raw_sv, float) and np.isnan(raw_sv)):
                    sv = float(raw_sv)
                    if 0 < sv <= 1.0:
                        sv *= 100.0
            except Exception:
                sv = defaults["SvPct"]

        ff_pct = _get(["FF%", "ff_pct"], 50.0)
        if 0 < ff_pct <= 1.0:
            ff_pct *= 100.0

        return {
            "xGFpg": safe_division(xgf, gp, defaults["xGFpg"]),
            "xGApG": safe_division(xga, gp, defaults["xGApG"]),
            "GFpg": safe_division(gf, gp, defaults["GFpg"]),
            "GApG": safe_division(ga, gp, defaults["GApG"]),
            "SFpg": safe_division(sf, gp, defaults["SFpg"]),
            "SApg": safe_division(sa, gp, defaults["SApg"]),
            "FFpg": safe_division(ff, gp, defaults["FFpg"]),
            "CFpg": safe_division(cf, gp, defaults["CFpg"]),
            "xGF%": xgf_pct,
            "SCF%": scf_pct,
            "CF%": scf_pct,
            "FF%": ff_pct,
            "HDCF%": hdcf_pct,
            "PDO": pdo,
            "SvPct": sv,
            "gsax_per_game": _get(["gsax_per_game"], 0.0),
        }
    except Exception as e:
        logger.error(f"Error deriving metrics for {abbr}: {e}")
        return defaults


def _team_id_from_abbr(abbr: str) -> Optional[int]:
    """Inverse of the small NHL PBP team_id table."""
    abbr_norm = abbr.upper()
    for tid, a in TEAM_ID_TO_ABBR.items():
        if a == abbr_norm:
            return tid
    return None


def team_last_n_metrics(team_abbr: str, date_str: str, n: int = 6) -> Tuple[float, float, float]:
    """
    Return actual last-N per-game averages for goals-for, goals-against, and shots-for.
    Uses the PBP shot store and game dates so the "last N" is chronological.
    Falls back to season-wide rates if PBP data is unavailable.
    """
    defaults = (
        LEAGUE_AVERAGES["goals_per_game"],
        LEAGUE_AVERAGES["goals_per_game"],
        LEAGUE_AVERAGES["shots_per_game"],
    )
    try:
        season = season_from_date(date_str)
        start_year = int(season[:4]) if len(str(season)) >= 4 else 2024
        shots = load_shot_store(start_year, stype=2)
        if shots.empty:
            raise ValueError("Shot store empty")

        team_id = _team_id_from_abbr(team_abbr)
        if team_id is None:
            raise ValueError(f"Unknown team {team_abbr}")

        team_shots = shots[shots["team_id"] == team_id]
        if team_shots.empty:
            raise ValueError(f"No shots for team {team_abbr}")

        # Build per-game totals
        game_ids = team_shots["game_id"].unique()
        dmap = game_date_map(start_year, stype=2)
        rows = []
        for gid in game_ids:
            gid_shots = shots[shots["game_id"] == gid]
            gf = int(gid_shots[(gid_shots["team_id"] == team_id) & (gid_shots["is_goal"] == 1)].shape[0])
            sf = int(gid_shots[(gid_shots["team_id"] == team_id) & (gid_shots["is_shot"] == 1)].shape[0])
            ga = int(gid_shots[(gid_shots["team_id"] != team_id) & (gid_shots["is_goal"] == 1)].shape[0])
            gd = dmap.get(int(gid)) if dmap else None
            rows.append({"game_id": gid, "gf": gf, "ga": ga, "sf": sf, "date": gd})

        df = pd.DataFrame(rows)
        df = df.dropna(subset=["date"]).sort_values("date").tail(n)
        if df.empty:
            raise ValueError("No dated games for last-N")

        gf_pg = float(df["gf"].mean()) if not df["gf"].empty else defaults[0]
        ga_pg = float(df["ga"].mean()) if not df["ga"].empty else defaults[1]
        sf_pg = float(df["sf"].mean()) if not df["sf"].empty else defaults[2]
        return (max(0.0, gf_pg), max(0.0, ga_pg), max(0.0, sf_pg))
    except Exception as e:
        logger.debug(f"True last-N metrics failed for {team_abbr}: {e}, falling back to season rates")
        try:
            season = season_from_date(date_str)
            df = get_team_rates_all(season, stype=2)
            if df is None or df.empty:
                return defaults
            row = _match_team(df, team_abbr, "team")
            if row.empty:
                return defaults
            r = row.iloc[0]
            gf_pg = float(r.get("goals_per_game", defaults[0])) or defaults[0]
            ga_pg = float(r.get("ga_per_game", defaults[1])) or defaults[1]
            sf_pg = float(r.get("shots_per_game", defaults[2])) or defaults[2]
            return (gf_pg, ga_pg, sf_pg)
        except Exception as e2:
            logger.error(f"Error getting season metrics for {team_abbr}: {e2}")
            return defaults


def team_last_n_goals_list(team_abbr: str, date_str: str, n: int = 8) -> List[int]:
    """Return the actual last-N goals scored by a team from the PBP shot store."""
    try:
        season = season_from_date(date_str)
        start_year = int(season[:4]) if len(str(season)) >= 4 else 2024
        shots = load_shot_store(start_year, stype=2)
        if shots.empty:
            raise ValueError("Shot store empty")

        team_id = _team_id_from_abbr(team_abbr)
        if team_id is None:
            raise ValueError(f"Unknown team {team_abbr}")

        team_shots = shots[shots["team_id"] == team_id]
        if team_shots.empty:
            raise ValueError(f"No shots for team {team_abbr}")

        dmap = game_date_map(start_year, stype=2)
        games = team_shots["game_id"].unique()
        goals_per_game = []
        dates = []
        for gid in games:
            gf = int(team_shots[(team_shots["game_id"] == gid) & (team_shots["is_goal"] == 1)].shape[0])
            goals_per_game.append(gf)
            dates.append(dmap.get(int(gid), ""))

        df = pd.DataFrame({"gf": goals_per_game, "date": dates}).dropna(subset=["date"]).sort_values("date").tail(n)
        if df.empty:
            raise ValueError("No dated games")
        return [int(g) for g in df["gf"].tolist()]
    except Exception as e:
        logger.debug(f"Actual last-N goals list failed for {team_abbr}: {e}, falling back to synthetic")
        try:
            gf_pg, _, _ = team_last_n_metrics(team_abbr, date_str, n=1)
            mean = max(0.5, gf_pg)
            return [int(max(0, round(mean + (np.random.random() - 0.5) * 1.5))) for _ in range(n)]
        except Exception as e2:
            logger.error(f"Error getting goals list for {team_abbr}: {e2}")
            return []

def team_recent_streak_factor(
    team_abbr: str,
    date_str: str,
    n: int = 10,
) -> float:
    """
    Compute a hot/cold momentum factor from the last N completed games.

    Uses the NHL schedule (cached) to determine wins, OT/SO losses, and goal
    differential. Points percentage (2 for a win, 1 for an OT/SO loss) is blended
    with per-game goal differential to produce a factor in [-0.05, +0.05].

    Positive = hot streak, negative = cold streak, 0 = neutral/insufficient data.
    """
    try:
        season = season_from_date(date_str)
        target_date = datetime.fromisoformat(date_str).date()
    except Exception as e:
        logger.debug(f"Could not parse date for streak factor: {e}")
        return 0.0

    try:
        sched = _cached_schedule_json(team_abbr.upper(), season)
        if not sched:
            return 0.0

        games = sched.get("games", [])
        records: List[Dict[str, Any]] = []
        for g in games:
            gd = _parse_game_date(g.get("gameDate") or g.get("startTimeUTC") or g.get("startTime"))
            if not gd or gd.date() >= target_date:
                continue
            state = str(g.get("gameState", "")).upper()
            if state not in ("OFF", "FINAL", "OVER"):
                continue

            home = (g.get("homeTeam") or {}).get("abbrev", "").upper()
            away = (g.get("awayTeam") or {}).get("abbrev", "").upper()
            hs = (g.get("homeTeam") or {}).get("score")
            aw = (g.get("awayTeam") or {}).get("score")
            if hs is None or aw is None:
                continue
            try:
                hs = int(hs)
                aw = int(aw)
            except Exception:
                continue

            pd = (g.get("periodDescriptor") or {})
            pt = str(pd.get("periodType", "")).upper()
            was_ot = pt in ("OT", "SO")

            if home == team_abbr.upper():
                gf, ga = hs, aw
            elif away == team_abbr.upper():
                gf, ga = aw, hs
            else:
                continue

            if gf > ga:
                points = 2
            elif gf < ga and was_ot:
                points = 1
            else:
                points = 0

            records.append({
                "date": gd.date(),
                "gf": gf,
                "ga": ga,
                "points": points,
                "goal_diff": gf - ga,
            })

        if not records:
            return 0.0

        records.sort(key=lambda x: x["date"], reverse=True)
        last_n = records[:n]
        sample_size = len(last_n)
        if sample_size < 3:
            return 0.0

        max_points = 2.0 * sample_size
        points_pct = sum(r["points"] for r in last_n) / max_points
        avg_goal_diff = safe_division(
            sum(r["goal_diff"] for r in last_n), float(sample_size), 0.0
        )

        # Normalize components to roughly [-1, 1]
        points_component = (points_pct - 0.5) * 2.0
        goal_component = max(-1.0, min(1.0, avg_goal_diff / 2.0))

        # 60% points, 40% goal differential, then scaled to a small ±5% factor
        momentum = 0.6 * points_component + 0.4 * goal_component
        momentum = max(-1.0, min(1.0, momentum))
        factor = round(momentum * 0.05, 4)

        logger.info(
            f"🔥 {team_abbr} last-{sample_size} momentum: "
            f"points%={points_pct:.1%}, GD/GM={avg_goal_diff:+.2f} → factor={factor:+.3f}"
        )
        return factor
    except Exception as e:
        logger.debug(f"Recent streak factor failed for {team_abbr}: {e}")
        return 0.0


def _get_loaded_calibrator() -> Optional[Calibrator]:
    """Return the fitted Calibrator from app state, or None if unavailable."""
    try:
        app_state = get_app_state()
        calib = app_state.get("calibrator")
        if isinstance(calib, Calibrator):
            return calib
    except Exception:
        pass
    try:
        calib = Calibrator.load("models/calibrator.pkl")
        return calib
    except Exception:
        return None


def _h2h_pct_from_schedules(
    home_abbr: str, away_abbr: str, game_date: _date, season: str
) -> float:
    """
    Compute home team's win percentage in recent head-to-head meetings.
    Uses cached schedules and fetches boxscores only for completed H2H games.
    """
    try:
        home_sched = _cached_schedule_json(home_abbr, season)
        away_sched = _cached_schedule_json(away_abbr, season)
        if not home_sched or not away_sched:
            return 0.5

        def _game_id(g: Dict[str, Any]) -> Optional[str]:
            return str(g.get("id") or g.get("gameId") or g.get("gamePk") or "")

        home_games = {gid: g for gid, g in [(_game_id(g), g) for g in home_sched.get("games", [])] if gid}
        away_games = {gid: g for gid, g in [(_game_id(g), g) for g in away_sched.get("games", [])] if gid}
        common_ids = sorted(set(home_games.keys()) & set(away_games.keys()))

        meetings = []
        for gid in common_ids:
            g = home_games[gid]
            gd = _parse_game_date(g.get("gameDate") or g.get("startTimeUTC") or g.get("startTime"))
            if not gd or gd.date() >= game_date:
                continue
            state = str(g.get("gameState", "")).upper()
            if state not in ("OFF", "FINAL", "OVER"):
                continue
            box = _safe_get_json(f"{NHL_API_BASE}/gamecenter/{gid}/boxscore")
            if not box:
                continue
            h = (box.get("homeTeam") or {}).get("abbrev", "").upper()
            hs = float((box.get("homeTeam") or {}).get("score", 0) or 0)
            as_ = float((box.get("awayTeam") or {}).get("score", 0) or 0)
            meetings.append({"home_team": h, "home_score": hs, "away_score": as_, "date": gd.date()})

        if not meetings:
            return 0.5

        # Last 5 meetings
        meetings.sort(key=lambda x: x["date"], reverse=True)
        meetings = meetings[:5]
        home_wins = 0
        for m in meetings:
            if m["home_team"] == home_abbr.upper():
                if m["home_score"] > m["away_score"]:
                    home_wins += 1
            else:
                if m["away_score"] > m["home_score"]:
                    home_wins += 1
        return home_wins / len(meetings)
    except Exception as e:
        logger.debug(f"H2H lookup failed for {home_abbr} v {away_abbr}: {e}")
        return 0.5


def _build_ml_features(
    home_abbr: str,
    away_abbr: str,
    game_date: _date,
    season: str,
    home_all: Dict[str, float],
    away_all: Dict[str, float],
    home_ev: Dict[str, float],
    away_ev: Dict[str, float],
    home_rec_gf: float,
    home_rec_ga: float,
    away_rec_gf: float,
    away_rec_ga: float,
    home_venue_pct: float,
    away_venue_pct: float,
    feat_home: Dict[str, Any],
    feat_away: Dict[str, Any],
    h2h_pct: float,
    feature_engine: Any,
    team_elo: Any,
) -> Dict[str, float]:
    """
    Build a leakage-free pre-game feature vector matching the training pipeline.
    Any missing fields are filled with neutral defaults so the model degrades
    gracefully when data is sparse.
    """
    try:
        elo_features = feature_engine.extract_team_features(home_abbr, away_abbr)
    except Exception as e:
        logger.debug(f"Feature engine failed: {e}")
        elo_features = {}

    home_elo = 1500.0
    away_elo = 1500.0
    try:
        home_elo = team_elo.get_team_rating(home_abbr)
        away_elo = team_elo.get_team_rating(away_abbr)
    except Exception:
        pass

    home_xgf_pg = home_all.get("xGFpg", LEAGUE_AVERAGES["goals_per_game"])
    away_xgf_pg = away_all.get("xGFpg", LEAGUE_AVERAGES["goals_per_game"])
    home_xga_pg = home_all.get("xGApG", LEAGUE_AVERAGES["goals_per_game"])
    away_xga_pg = away_all.get("xGApG", LEAGUE_AVERAGES["goals_per_game"])

    home_sf_pg = home_all.get("SFpg", LEAGUE_AVERAGES["shots_per_game"])
    away_sf_pg = away_all.get("SFpg", LEAGUE_AVERAGES["shots_per_game"])

    # Recent form (training normalizes by subtracting 3.0)
    home_recent_form_off = home_rec_gf - 3.0
    away_recent_form_off = away_rec_gf - 3.0

    features: Dict[str, float] = {
        **elo_features,
        "home_elo": float(home_elo),
        "away_elo": float(away_elo),

        # Season-average xG / xGA (replicate training scaling)
        "home_season_xgf_pg": float(home_xgf_pg / 0.06) if home_xgf_pg else 0.0,
        "away_season_xgf_pg": float(away_xgf_pg / 0.06) if away_xgf_pg else 0.0,
        "home_xgf_norm": float(home_xgf_pg / 6.0) if home_xgf_pg else 0.0,
        "away_xgf_norm": float(away_xgf_pg / 6.0) if away_xgf_pg else 0.0,
        "home_xgf_share": float(home_xgf_pg / max(home_xgf_pg + away_xgf_pg, 0.1)),

        # Recent form
        "home_recent_xgf_pg": float(home_xgf_pg),
        "home_recent_xga_pg": float(home_xga_pg),
        "away_recent_xgf_pg": float(away_xgf_pg),
        "away_recent_xga_pg": float(away_xga_pg),
        "home_recent_gf_pg": float(home_rec_gf),
        "away_recent_gf_pg": float(away_rec_gf),
        "home_recent_form_off": float(home_recent_form_off),
        "away_recent_form_off": float(away_recent_form_off),

        # Shots / pace
        "home_sf_pg": float(home_sf_pg / 0.06) if home_sf_pg else 0.0,
        "away_sf_pg": float(away_sf_pg / 0.06) if away_sf_pg else 0.0,
        "home_recent_sf_pg": float(home_sf_pg),
        "away_recent_sf_pg": float(away_sf_pg),

        # Venue
        "home_venue_pct": float(home_venue_pct),
        "away_venue_pct": float(away_venue_pct),
        "venue_diff": float(home_venue_pct - away_venue_pct),

        # Rest / travel
        "home_rest_days": float(feat_home.get("rest_days", 3.0)),
        "away_rest_days": float(feat_away.get("rest_days", 3.0)),
        "rest_diff": float(feat_away.get("rest_days", 3.0)) - float(feat_home.get("rest_days", 3.0)),
        "home_b2b": 1.0 if feat_home.get("is_b2b") else 0.0,
        "away_b2b": 1.0 if feat_away.get("is_b2b") else 0.0,
        "home_travel_km": float(feat_home.get("travel_km", 0.0)),
        "away_travel_km": float(feat_away.get("travel_km", 0.0)),
        "home_tz_diff": float(feat_home.get("tz_diff", 0.0)),
        "away_tz_diff": float(feat_away.get("tz_diff", 0.0)),

        # Head-to-head
        "h2h_home_pct": float(h2h_pct),

        # Differential features available in both training and inference
        "xgf_pct_diff": safe_division(home_all.get("xGF%", 50.0) - away_all.get("xGF%", 50.0), 50.0, 0.0),
        "gf_pg_diff": home_all.get("GFpg", 3.0) - away_all.get("GFpg", 3.0),
        "ga_pg_diff": away_all.get("GApG", 3.0) - home_all.get("GApG", 3.0),
        "sf_pg_diff": home_all.get("SFpg", 30.0) - away_all.get("SFpg", 30.0),
        "xga_pg_diff": home_all.get("xGApG", 3.0) - away_all.get("xGApG", 3.0),
    }

    return features


def estimate_gamma_shape_from_recent(goals: List[int]) -> Optional[float]:
    if not goals or len(goals) < 3:
        return None
    try:
        m = float(np.mean(goals))
        v = float(np.var(goals, ddof=1))
        if v <= m or m <= 0:
            return None
        k = safe_division(m * m, v - m, default=None)
        if k is None or k <= 0 or not np.isfinite(k):
            return None
        return max(0.5, min(10.0, k))
    except Exception as e:
        logger.debug(f"Could not estimate gamma shape: {e}")
        return None

def apply_per_sim_shock(mu_home: float, mu_away: float, sims: int, sigma: float = None) -> Tuple[np.ndarray, np.ndarray]:
    if sigma is None:
        sigma = SIMULATION_PARAMS["shock_sigma"]
    sims = int(max(0, sims or 0))
    m = -0.5 * (sigma ** 2)
    f_home = np.exp(np.random.normal(loc=m, scale=sigma, size=sims))
    f_away = np.exp(np.random.normal(loc=m, scale=sigma, size=sims))
    lam_h = np.clip(mu_home * f_home, SIMULATION_PARAMS["min_goals"], SIMULATION_PARAMS["max_goals"])
    lam_a = np.clip(mu_away * f_away, SIMULATION_PARAMS["min_goals"], SIMULATION_PARAMS["max_goals"])
    return lam_h, lam_a

def resolve_score_mode(final_home: np.ndarray, final_away: np.ndarray) -> Tuple[int, int]:
    """Return the most frequently occurring final score (home, away)."""
    pairs = list(zip(final_home.tolist(), final_away.tolist()))
    counts = Counter(pairs)
    if not counts:
        return 0, 0
    (h, a), _ = counts.most_common(1)[0]
    return int(h), int(a)

def apply_empty_net_adjustments(final_home: np.ndarray, final_away: np.ndarray, mu_home: float, mu_away: float, p_base: float = None) -> Tuple[np.ndarray, np.ndarray]:
    if p_base is None:
        p_base = SIMULATION_PARAMS.get("empty_net_probability", 0.28)
    fh = final_home.copy()
    fa = final_away.copy()
    diff = fh - fa
    one_goal = np.where(np.abs(diff) == 1)[0]
    if one_goal.size == 0:
        return fh, fa
    total_goals = fh + fa
    idxs = one_goal[total_goals[one_goal] >= 3]
    if idxs.size == 0:
        return fh, fa
    mu_gap = abs(mu_home - mu_away)
    # Lower base rate when the game is already high-scoring (more chances have
    # already resolved) and when teams are evenly matched (tighter checking).
    scale = max(0.35, 0.75 - 0.15 * mu_gap)
    # Pull the ceiling down for high-total games.
    high_total_discount = np.where(total_goals[idxs] >= 6, 0.7, 1.0)
    probs = np.clip(p_base * scale * high_total_discount, 0.0, 0.35)
    draws = np.random.random(size=idxs.size) < probs
    for idx, got_en in zip(idxs, draws):
        if not got_en:
            continue
        if fh[idx] > fa[idx]:
            fh[idx] += 1
        else:
            fa[idx] += 1
    return fh, fa

def opp_goalie_sv_dsv_from_df(
    goalie_df: pd.DataFrame,
    team_abbr: str,
    selected_goalie: Optional[str] = None
) -> Tuple[float, float, float, float]:
    sv_default = GOALIE_PARAMS["default_sv_pct"]
    dsv_default = GOALIE_PARAMS["default_dsv_pct"]
    gsaa_pg = 0.0
    gsax_per_60 = 0.0

    # FIX: Use passed-in argument goalie_df
    df = normalize_sv_column(goalie_df.copy())
    team_col = get_column_safe(df, COLUMN_VARIATIONS, "team")

    if team_col:
        # Use STRICT matching helper
        subset = _match_team(df, team_abbr, team_col)
        df = subset if not subset.empty else df

    if df.empty:
        return sv_default, dsv_default, gsaa_pg, gsax_per_60

    if selected_goalie:
        player_col = get_column_safe(df, COLUMN_VARIATIONS, "player")
        if player_col:
            df_sel = df.assign(
                _player=df[player_col].astype(str),
                _norm=lambda d: d["_player"].map(normalize_name_key),
                _last=lambda d: d["_player"].map(last_token_norm)
            )
            target_norm = normalize_name_key(selected_goalie)
            target_last = last_token_norm(selected_goalie)
            subset = df_sel[df_sel["_norm"] == target_norm]
            if subset.empty and target_last:
                subset = df_sel[df_sel["_last"] == target_last]
                if subset.empty:
                    subset = df_sel[df_sel["_player"].str.contains(target_last, case=False, na=False)]
            if not subset.empty:
                df = subset

    def tonum(s):
        return pd.to_numeric(s, errors="coerce").fillna(0.0)
    if "GS" not in df.columns:
        df.loc[:, "GS"] = 0
    if "GP" not in df.columns:
        df.loc[:, "GP"] = 0
    if "Sv%" not in df.columns:
        df.loc[:, "Sv%"] = np.nan
    df = df.assign(
        _gs=tonum(df["GS"]),
        _gp=tonum(df["GP"]),
        _sv=tonum(df["Sv%"]).fillna(0.0)
    ).sort_values(by=["_gs", "_gp", "_sv"], ascending=[False, False, False])
    r = df.iloc[0]
    sv = float(r["_sv"]) if pd.notna(r["_sv"]) else sv_default
    dsv = float(pd.to_numeric(r.get("dSV%", np.nan), errors="coerce")) if "dSV%" in df.columns else dsv_default
    if not np.isfinite(dsv):
        dsv = dsv_default

    gsaa_pg = 0.0
    if "GSAA" in df.columns and pd.notna(r.get("GSAA")):
        try:
            gsaa = float(pd.to_numeric(r.get("GSAA"), errors="coerce"))
            gp = float(r.get("GP", 0.0) or 0.0)
            if gp > 0:
                gsaa_pg = float(gsaa / gp)
        except Exception:
            gsaa_pg = 0.0

    # GSAX per 60 from the PBP/xG model (stronger future-goalie signal than raw Sv%).
    for gsax_col in ("gsax_per_60", "gsax_per60", "GSAX/60", "GSAX60"):
        if gsax_col in df.columns and pd.notna(r.get(gsax_col)):
            try:
                gsax_per_60 = float(pd.to_numeric(r.get(gsax_col), errors="coerce"))
                if np.isfinite(gsax_per_60):
                    break
            except Exception:
                continue
    if gsax_per_60 == 0.0 and "gsax" in df.columns and pd.notna(r.get("gsax")):
        try:
            gsax_total = float(pd.to_numeric(r.get("gsax"), errors="coerce"))
            gp = float(r.get("GP", 0.0) or 0.0)
            if gp > 0 and np.isfinite(gsax_total):
                gsax_per_60 = gsax_total / gp
        except Exception:
            gsax_per_60 = 0.0

    return sv, dsv, gsaa_pg, gsax_per_60

# TTL cache for schedule data (30 min)
_schedule_cache: Dict[str, Tuple[float, Optional[Dict[str, Any]]]] = {}
_SCHEDULE_CACHE_TTL = 1800  # seconds

def _cached_schedule_json(team_abbr: str, season: str) -> Optional[Dict[str, Any]]:
    """Cached NHL schedule fetch — caches for 30 min to avoid redundant API calls."""
    key = f"{team_abbr}_{season}"
    now = time.time()
    if key in _schedule_cache:
        ts, data = _schedule_cache[key]
        if now - ts < _SCHEDULE_CACHE_TTL:
            return data
    url = f"{NHL_API_BASE}/club-schedule-season/{team_abbr}/{season}"
    data = _safe_get_json(url)
    _schedule_cache[key] = (now, data)
    return data


def get_team_home_away_record(team_abbr: str, season: str, as_of: Optional[_date] = None) -> Dict[str, float]:
    """
    Compute home and road win percentages from the schedule, up to `as_of` (default: today).

    Returns neutral 50% if no data (let other factors decide), not artificial home bias.
    Uses cached schedule data to avoid redundant API calls per team per day.
    """
    try:
        if as_of is None:
            as_of = _date.today()

        sched = _cached_schedule_json(team_abbr.upper(), season)

        # Return neutral if data is missing - no artificial bias
        if not sched:
            logger.warning(f"⚠️ Schedule unavailable for {team_abbr}. Using neutral 50%.")
            return {'home_win_pct': 0.50, 'away_win_pct': 0.50}

        games = sched.get("games", [])
        if not games:
            logger.warning(f"⚠️ No games in schedule for {team_abbr}. Using neutral 50%.")
            return {'home_win_pct': 0.50, 'away_win_pct': 0.50}

        home_wins = home_otl = home_games = 0
        road_wins = road_otl = road_games = 0

        for g in games:
            gd = _parse_game_date(g.get("gameDate") or g.get("startTimeUTC") or g.get("startTime"))
            if not gd or gd.date() > as_of:
                continue

            # Only completed games
            state = str(g.get("gameState", "")).upper()
            if state not in ("OFF", "FINAL", "OVER"):
                continue

            home = (g.get("homeTeam") or {}).get("abbrev", "").upper()
            away = (g.get("awayTeam") or {}).get("abbrev", "").upper()
            hs = (g.get("homeTeam") or {}).get("score")
            as_ = (g.get("awayTeam") or {}).get("score")
            if hs is None or as_ is None:
                continue

            was_ot = False
            pd = (g.get("periodDescriptor") or {})
            pt = str(pd.get("periodType", "")).upper()
            if pt in ("OT", "SO"):
                was_ot = True

            if home == team_abbr.upper():
                home_games += 1
                if hs > as_:
                    home_wins += 1
                elif was_ot:
                    home_otl += 1
            elif away == team_abbr.upper():
                road_games += 1
                if as_ > hs:
                    road_wins += 1
                elif was_ot:
                    road_otl += 1

        def pct(w, otl, gp):
            # Small sample size: blend with neutral 50%
            if gp < 5:
                if gp == 0:
                    return 0.50
                actual = (w + 0.5 * otl) / gp
                # Blend: 50% actual, 50% neutral for small samples
                return 0.5 * actual + 0.5 * 0.50
            return (w + 0.5 * otl) / gp

        home_pct = pct(home_wins, home_otl, home_games)
        away_pct = pct(road_wins, road_otl, road_games)
        
        logger.debug(
            f"Home/Away record for {team_abbr}: "
            f"Home {home_pct:.3f} ({home_wins}W-{home_otl}OTL/{home_games}GP), "
            f"Away {away_pct:.3f} ({road_wins}W-{road_otl}OTL/{road_games}GP)"
        )
        
        return {
            'home_win_pct': home_pct,
            'away_win_pct': away_pct,
        }
    except Exception as e:
        logger.warning(f"Could not fetch home/away record for {team_abbr}: {e}")
        return {'home_win_pct': 0.50, 'away_win_pct': 0.50}

def calculate_venue_advantage(
    home_team: str,
    away_team: str,
    season: str,
) -> float:
    """
    Calculate venue advantage based on team-specific home/away records,
    calibrated to the observed league home-ice edge.

    2025-26 regular-season baseline:
      - Home teams won 54.5% of all games
      - Home moneyline favorites won 56.0% of their games

    The adjustment combines a league-average home-ice baseline with a
    team-specific over/under-performance term. A team that only wins 30% at
    home will have its home edge reduced (or flipped vs a strong road team),
    while a dominant home club keeps most/all of the baseline advantage.
    """
    home_record = get_team_home_away_record(home_team, season)
    away_record = get_team_home_away_record(away_team, season)

    home_pct = home_record['home_win_pct']
    away_pct = away_record['away_win_pct']

    params = VENUE_ADV_PARAMS

    # How much better/worse each team is in its venue vs the league average.
    home_over = home_pct - params["league_home_win_pct"]
    away_over = away_pct - params["league_away_win_pct"]
    net_overperformance = home_over - away_over

    # Baseline home-ice edge + team-specific deviation.
    # For an average home team vs an average road team this is exactly the
    # league baseline (~+0.30 goals ≈ 54.5% win prob for otherwise even teams).
    adjustment = (
        params["baseline_goals"]
        + params["overperformance_scale"] * net_overperformance
    )

    adjustment = max(
        -params["max_adjustment"],
        min(params["max_adjustment"], adjustment),
    )

    # Log the breakdown
    if adjustment > 0.02:
        logger.info(
            f"🏠 Venue advantage: {home_team} at home\n"
            f"   {home_team} home record: {home_pct:.1%} (Lg avg {params['league_home_win_pct']:.1%}) | "
            f"{away_team} away record: {away_pct:.1%} (Lg avg {params['league_away_win_pct']:.1%})\n"
            f"   Net overperformance: {net_overperformance:+.1%} → {adjustment:+.3f} goals to {home_team}"
        )
    elif adjustment < -0.02:
        logger.info(
            f"✈️  Road warrior advantage: {away_team} on the road\n"
            f"   {away_team} away record: {away_pct:.1%} (Lg avg {params['league_away_win_pct']:.1%}) | "
            f"{home_team} home record: {home_pct:.1%} (Lg avg {params['league_home_win_pct']:.1%})\n"
            f"   Net overperformance: {net_overperformance:+.1%} → {adjustment:+.3f} goals to {away_team}"
        )
    else:
        logger.info(
            f"⚖️  Neutral venue: {home_team} ({home_pct:.1%} home) vs "
            f"{away_team} ({away_pct:.1%} away) → {adjustment:+.3f} goals"
        )

    return adjustment

def calculate_automatic_injury_impact(
    team_abbr: str,
    season: str,
    player_stats_cache: Dict[str, Dict],
    team_stats_cache: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    Automatically calculate injury impact from player contribution %.

    get_team_injuries() can return multiple normalized name keys for the same
    player, so we deduplicate by the resolved player stats record.

    Returns:
        {
            'offense_impact': float,  # Negative = loss (capped)
            'n_injuries': int,        # Unique injured players with stats found
            'players': [              # Per-player details for UI display
                {
                    'name': str,
                    'points': int,
                    'team_points': float,
                    'contribution_pct': float,
                    'impact_pct': float,
                    'status': str,
                },
                ...
            ]
        }
    """
    try:
        from NHL.ApiScrape import get_team_injuries
        injuries = get_team_injuries(team_abbr)
    except Exception as e:
        logger.debug(f"Could not get injuries for {team_abbr}: {e}")
        injuries = {}

    if not injuries:
        return {'offense_impact': 0.0, 'n_injuries': 0, 'players': []}

    # Build a normalized-key -> display-name map from injuries.json so the UI
    # shows real player names instead of concatenated normalized keys.
    display_name_map: Dict[str, str] = {}
    try:
        import json, os
        if os.path.exists("injuries.json"):
            with open("injuries.json", "r") as f:
                data = json.load(f)
            items = data.get("injuries", []) if isinstance(data, dict) else data
            for item in items:
                if item.get("team") == team_abbr.upper() and item.get("injured", True):
                    display = item.get("player", "")
                    if display:
                        display_name_map[normalize_name_key(display)] = display
    except Exception:
        pass

    team_totals = team_stats_cache.get(team_abbr.upper(), {})
    team_points = team_totals.get('total_points', 700.0)  # League avg fallback

    total_offense_loss = 0.0
    seen_players: set[str] = set()
    player_details: List[Dict[str, Any]] = []

    def _display_from_key(k: str) -> str:
        """Convert a normalized name key (e.g. 'samreinhart') to a readable name."""
        # Some keys contain initials like 'l.brossoit'; normalize those first.
        k = str(k).replace(".", "").replace("_", " ")
        return re.sub(r"([a-z])([A-Z])", r"\1 \2", k).title()

    for player_name_key, injury_status in injuries.items():
        player_stats = player_stats_cache.get(player_name_key)
        if not player_stats:
            continue

        # get_team_injuries() stores several name variants per player.
        # Deduplicate by the underlying stats record (name + points + gp).
        dedupe_key = f"{player_stats.get('name', player_name_key)}|{player_stats.get('points', 0)}|{player_stats.get('gp', 0)}"
        if dedupe_key in seen_players:
            continue
        seen_players.add(dedupe_key)

        points = int(player_stats.get('points', 0))
        gp = int(player_stats.get('gp', 1))

        # Contribution is share of team *points*, not goals.
        contribution_pct = points / team_points if team_points > 0 else 0.0

        # Convert to negative impact; cap individual impact at -40%.
        offense_loss = max(-0.40, -contribution_pct)
        total_offense_loss += offense_loss

        raw_name = player_stats.get('name')
        name = (
            display_name_map.get(player_name_key)
            or (raw_name if raw_name and str(raw_name).strip() else None)
            or _display_from_key(player_name_key)
        )
        player_details.append({
            'name': name,
            'points': points,
            'gp': gp,
            'team_points': float(team_points),
            'contribution_pct': float(contribution_pct),
            'impact_pct': float(offense_loss),
            'status': str(injury_status),
        })

        logger.info(
            f"💥 {name}: "
            f"{points} pts / {team_points:.0f} team points = "
            f"{contribution_pct*100:.1f}% → {offense_loss*100:+.1f}% offense"
        )

    # Cap total at -80%
    total_offense_loss = max(-0.80, total_offense_loss)

    if total_offense_loss < -0.05:
        logger.warning(
            f"⚠️  {team_abbr} injury impact: {total_offense_loss*100:.1f}% offense loss "
            f"({len(player_details)} injuries)"
        )

    return {
        'offense_impact': total_offense_loss,
        'n_injuries': len(player_details),
        'players': sorted(player_details, key=lambda p: p['contribution_pct'], reverse=True),
    }


def _load_player_stats_for_injuries(season: str) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Per-player and per-team stats used by the injury-impact code path.
    Now backed by NHL API PBP (was NST HTML scrape). Cached per season.

    Returns:
        player_cache: {name_key: {name, team, gp, goals, assists, points}}
        team_cache:   {team_abbr: {total_goals, total_points}}
    """
    now = time.time()
    cached = _INJURY_CACHE.get(season)
    if cached and (now - cached[0]) < _INJURY_CACHE_TTL:
        return cached[1]

    player_cache: Dict[str, Dict] = {}
    team_cache: Dict[str, Dict] = {}
    try:
        start_year = int(str(season)[:4])
    except (ValueError, TypeError):
        logger.error(f"Invalid season format {season!r}")
        return player_cache, team_cache

    try:
        from NHL.StatsFromPBP import load_skater_rates_from_json, team_abbr_from_id
        from NHL.PlayByPlay import load_shot_store
        rates = load_skater_rates_from_json(start_year, 2)
        shots = load_shot_store(start_year, 2)
    except Exception as e:
        logger.error(f"Failed to load PBP stats for injuries ({season}): {e}")
        return player_cache, team_cache

    if not rates:
        logger.warning(f"No PBP skater rates for {season}")
        return player_cache, team_cache

    id_team: Dict[int, int] = {}
    if not shots.empty and {"shooter_id", "team_id"}.issubset(shots.columns):
        sub = shots.dropna(subset=["shooter_id", "team_id"])
        for sid, grp in sub.groupby("shooter_id"):
            try:
                mode = grp["team_id"].mode()
                id_team[int(sid)] = int(mode.iloc[0]) if not mode.empty else 0
            except Exception:
                continue

    per_player: Dict[int, Dict[str, int]] = {}
    if not shots.empty and "shooter_id" in shots.columns:
        agg = shots.dropna(subset=["shooter_id"]).groupby("shooter_id").agg(
            goals=("is_goal", "sum"),
            gp=("game_id", "nunique"),
        )
        for sid, row in agg.iterrows():
            try:
                per_player[int(sid)] = {"goals": int(row["goals"]), "gp": int(row["gp"])}
            except Exception:
                pass

    for name_key, d in rates.items():
        try:
            goals = int(d.get("goals", 0))
            assists = int(d.get("assists", 0))
            player_cache[name_key] = {
                "name": d.get("name", ""),
                "team": "",
                "gp": int(d.get("gp", 0)),
                "goals": goals,
                "assists": assists,
                "points": goals + assists,
            }
        except Exception as e:
            logger.debug(f"Injury-stats row error: {e}")

    for sid, stats in per_player.items():
        abbr = team_abbr_from_id(id_team.get(sid, 0))
        if not abbr:
            continue
        if abbr not in team_cache:
            team_cache[abbr] = {"total_goals": 0, "total_points": 0}
        # API player-summary only has goals; points are approximated as goals*2,
        # which is conservative but avoids KeyError on missing 'assists'.
        goals = int(stats.get("goals", 0))
        assists = int(stats.get("assists", goals))
        team_cache[abbr]["total_goals"] += goals
        team_cache[abbr]["total_points"] += goals + assists

    logger.info(
        f"Loaded injury stats: {len(player_cache)} players, {len(team_cache)} teams"
    )
    _INJURY_CACHE[season] = (time.time(), (dict(player_cache), dict(team_cache)))
    return player_cache, team_cache


def improved_winner_prediction(final_home, final_away, extra_features=None) -> float:
    """
    Calculate win probability from simulation results WITHOUT forcing extremes.
    Returns realistic probabilities based on actual simulation outcomes.
    """
    try:
        home_wins = np.sum(final_home > final_away)
        total_games = len(final_home)
        
        raw_prob = home_wins / total_games
        smoothed_prob = (home_wins + 1) / (total_games + 2)
        
        logger.debug(f"Win probability: {home_wins}/{total_games} = {smoothed_prob:.3f}")
        
        return float(smoothed_prob)
        
    except Exception as e:
        logger.error(f"Error calculating win probability: {e}")
        return 0.5

@log_performance
def simulate_matchup(
    home_abbr: str,
    away_abbr: str,
    game_date: _date,
    stype: int = 2,
    sims: int = None,
    trend_games: int = None,
    use_recent_window_days: Optional[int] = None,
    season_skill_for_lineups: Optional[Dict[str, Dict[str, float]]] = None,
    home_lineup_df: Optional[pd.DataFrame] = None,
    away_lineup_df: Optional[pd.DataFrame] = None,
    selected_home_goalie: Optional[str] = None,
    selected_away_goalie: Optional[str] = None,
    home_elo_override: Optional[float] = None,
    away_elo_override: Optional[float] = None,
    apply_injury_impact: bool = True,  # NEW PARAMETER
    home_b2b: bool = False,
    away_b2b: bool = False,
) -> Dict[str, Any]:
    if not home_abbr or not away_abbr:
        raise SimulationError("Team abbreviations cannot be empty")
    if not isinstance(game_date, _date):
        raise SimulationError(f"game_date must be date object, got {type(game_date)}")

    if sims is None or sims <= 0:
        try:
            from NHL.Config import DEFAULT_SIMULATIONS
            sims = int(max(1, DEFAULT_SIMULATIONS))
        except Exception:
            sims = 1000
    if trend_games is None or trend_games <= 0:
        try:
            from NHL.Config import DEFAULT_TREND_GAMES
            trend_games = int(max(1, DEFAULT_TREND_GAMES))
        except Exception:
            trend_games = 6

    lineup_shoot_home = lineup_shoot_away = 0.0
    try:
        import NHL.Prediction as pred_mod
    except Exception as e:
        logger.debug(f"NHLPrediction unavailable: {e}")
        pred_mod = None
    if pred_mod and season_skill_for_lineups is not None:
        if home_lineup_df is not None:
            try:
                lineup_shoot_home = pred_mod.lineup_shooting_factor(home_lineup_df, season_skill_for_lineups)
            except Exception as e:
                logger.debug(f"Could not compute home lineup factor: {e}")
        else:
            logger.info(f"No home lineup provided for {home_abbr}, assuming neutral lineup impact.")

        if away_lineup_df is not None:
            try:
                lineup_shoot_away = pred_mod.lineup_shooting_factor(away_lineup_df, season_skill_for_lineups)
            except Exception as e:
                logger.debug(f"Could not compute away lineup factor: {e}")
        else:
            logger.info(f"No away lineup provided for {away_abbr}, assuming neutral lineup impact.")


    season = season_from_date(game_date.isoformat())
    fd_str = td_str = ""
    if use_recent_window_days:
        td_day = game_date - timedelta(days=1)
        fd_day = td_day - timedelta(days=use_recent_window_days - 1)
        fd_str, td_str = fd_day.isoformat(), td_day.isoformat()
        logger.info(f"Using recent window: {fd_str} to {td_str}")

    all_df = get_team_rates_all(season, stype, fd=fd_str, td=td_str)
    ev_df = get_team_rates_ev(season, stype, fd=fd_str, td=td_str)
    _rate_limit_sleep()
    pp_df = get_team_rates_pp_per60(season, stype, fd=fd_str, td=td_str)
    _rate_limit_sleep()
    pk_df = get_team_rates_pk_per60(season, stype, fd=fd_str, td=td_str)
    _rate_limit_sleep()
    g_df = get_goalie_table(season, stype, fd=fd_str, td=td_str)

    # Rest/travel features are needed early for goalie prediction (b2b detection).
    feat_home = compute_rest_travel_features_fast(home_abbr, away_abbr, game_date, [])
    feat_away = compute_rest_travel_features_fast(away_abbr, home_abbr, game_date, [])

    # Allow the user/schedule to override the fatigue flags for back-to-back games.
    if home_b2b:
        feat_home["is_b2b"] = True
        feat_home["rest_days"] = min(feat_home.get("rest_days", 3.0), 0.0)
    if away_b2b:
        feat_away["is_b2b"] = True
        feat_away["rest_days"] = min(feat_away.get("rest_days", 3.0), 0.0)

    rest_mult_home = fatigue_multiplier(feat_home)
    rest_mult_away = fatigue_multiplier(feat_away)

    logger.info(
        f"User-selected goalies: home={selected_home_goalie or 'Auto'}, "
        f"away={selected_away_goalie or 'Auto'}"
    )

    # Auto-select the predicted starting goalie for each team when the caller
    # did not provide one. The predictor uses recent game TOI, rest, and
    # opponent strength so backups are chosen vs weak opponents and starters vs
    # strong ones.
    try:
        if not selected_home_goalie:
            selected_home_goalie = predict_starting_goalie(
                home_abbr, game_date.isoformat(),
                opponent_abbr=away_abbr,
                is_b2b=bool(feat_home.get("is_b2b")) if feat_home else False,
            )
            if selected_home_goalie:
                logger.info(f"Predicted home goalie for {home_abbr}: {selected_home_goalie}")
        if not selected_away_goalie:
            selected_away_goalie = predict_starting_goalie(
                away_abbr, game_date.isoformat(),
                opponent_abbr=home_abbr,
                is_b2b=bool(feat_away.get("is_b2b")) if feat_away else False,
            )
            if selected_away_goalie:
                logger.info(f"Predicted away goalie for {away_abbr}: {selected_away_goalie}")
    except Exception as e:
        logger.debug(f"Auto goalie selection failed: {e}")

    home_all = derive_all_pg_metrics(all_df, home_abbr)
    away_all = derive_all_pg_metrics(all_df, away_abbr)
    home_ev = derive_all_pg_metrics(ev_df, home_abbr)
    away_ev = derive_all_pg_metrics(ev_df, away_abbr)

    home_pp60, home_pk_ga60 = _extract_pp_pk_rates(pp_df, pk_df, home_abbr)
    away_pp60, away_pk_ga60 = _extract_pp_pk_rates(pp_df, pk_df, away_abbr)
    pen_diff_home = penalty_diff_per60(all_df, home_abbr)
    pen_diff_away = penalty_diff_per60(all_df, away_abbr)
    
    pp_impact_mult = 0.045
    max_pp_impact = SPECIAL_TEAMS_PARAMS.get("max_pp_goal_impact", 0.30)
    
    extra_pp_goals_home = pp_impact_mult * pen_diff_home
    extra_pp_goals_away = pp_impact_mult * pen_diff_away
    
    extra_pp_goals_home = max(-max_pp_impact, min(max_pp_impact, extra_pp_goals_home))
    extra_pp_goals_away = max(-max_pp_impact, min(max_pp_impact, extra_pp_goals_away))

    home_opp_sv, home_opp_dsv, home_opp_gsaa_pg, home_opp_gsax60 = opp_goalie_sv_dsv_from_df(
        g_df, away_abbr, selected_goalie=selected_away_goalie
    )
    away_opp_sv, away_opp_dsv, away_opp_gsaa_pg, away_opp_gsax60 = opp_goalie_sv_dsv_from_df(
        g_df, home_abbr, selected_goalie=selected_home_goalie
    )

    home_rec_gf, home_rec_ga, _ = team_last_n_metrics(home_abbr, game_date.isoformat(), n=max(10, trend_games))
    away_rec_gf, away_rec_ga, _ = team_last_n_metrics(away_abbr, game_date.isoformat(), n=max(10, trend_games))

    # Independent hot/cold streak signal based on last 5-10 games (points % + goal differential)
    home_momentum = team_recent_streak_factor(home_abbr, game_date.isoformat(), n=10)
    away_momentum = team_recent_streak_factor(away_abbr, game_date.isoformat(), n=10)

    if 'tz_diff' in feat_home:
        tz_pen_home = -0.01 * abs(feat_home.get('tz_diff', 0))
        rest_mult_home += tz_pen_home
        if tz_pen_home < 0:
            logger.info(f"✈️  {home_abbr} Timezone Penalty: {tz_pen_home:.3f} (diff: {feat_home['tz_diff']}h)")

    if 'tz_diff' in feat_away:
        tz_pen_away = -0.01 * abs(feat_away.get('tz_diff', 0))
        rest_mult_away += tz_pen_away
        if tz_pen_away < 0:
            logger.info(f"✈️  {away_abbr} Timezone Penalty: {tz_pen_away:.3f} (diff: {feat_away['tz_diff']}h)")

    # AUTOMATIC INJURY IMPACT CALCULATION
    if apply_injury_impact:
        player_stats_cache, team_stats_cache = _load_player_stats_for_injuries(season)
        
        home_injury_data = calculate_automatic_injury_impact(
            home_abbr, season, player_stats_cache, team_stats_cache
        )
        away_injury_data = calculate_automatic_injury_impact(
            away_abbr, season, player_stats_cache, team_stats_cache
        )
        
        injury_impact_home = home_injury_data['offense_impact']
        injury_impact_away = away_injury_data['offense_impact']
        
        if injury_impact_home < -0.05 or injury_impact_away < -0.05:
            logger.info(
                f"💉 Injury adjustments:\n"
                f"   {home_abbr}: {injury_impact_home*100:+.1f}% ({home_injury_data['n_injuries']} injuries)\n"
                f"   {away_abbr}: {injury_impact_away*100:+.1f}% ({away_injury_data['n_injuries']} injuries)"
            )
    else:
        home_injury_data = {'offense_impact': 0.0, 'n_injuries': 0, 'players': []}
        away_injury_data = {'offense_impact': 0.0, 'n_injuries': 0, 'players': []}
        injury_impact_home = 0.0
        injury_impact_away = 0.0
        logger.info("Injury impact disabled for this simulation")

    def _compute_expected_goals(
        offense_all: Dict[str, float],
        defense_all: Dict[str, float],
        pp_off_gf60: float,
        pk_def_ga60: float,
        opp_sv: float,
        opp_dsv: float,
        opp_gsaa_pg: float,
        opp_gsax_per60: float,
        rec_gfpg: float,
        rec_gapg_opp: float,
        lineup_delta: float,
        venue_adj: float,
        advanced_stats: Dict[str, float],
        extra_pp_goals: float,
        rest_mult: float,
        score_mult: float,
        injury_impact: float,
        momentum_factor: float = 0.0
    ) -> Tuple[float, Dict[str, float]]:
        league_sv = LEAGUE_AVERAGES["sv_pct"]
        base_xgf = (
            MODEL_WEIGHTS["xg_for_weight"] * offense_all["xGFpg"] +
            MODEL_WEIGHTS["gf_weight"] * offense_all["GFpg"]
        )
        xga_adj = MODEL_WEIGHTS["xga_weight"] * defense_all["xGApG"]
        pp_minutes = LEAGUE_AVERAGES["powerplay_minutes"]
        st_goals = MODEL_WEIGHTS["pp_pk_weight"] * (pp_off_gf60 + pk_def_ga60) * (pp_minutes / 60.0)
        st_goals += extra_pp_goals

        mu = max(0.3, base_xgf + xga_adj + st_goals)

        # Goalie impact: prefer GSAX/60 (Goals Saved Above Expected per 60) from the xG model,
        # fall back to dSV% / raw Sv% if GSAX is unavailable.
        goalie_adj_mult = 1.0
        if opp_gsax_per60 is not None and not math.isnan(opp_gsax_per60):
            # Positive GSAX = goalie outperforming expected → harder to score against.
            # Typical elite range ±0.5 GSAX/60; scale so 0.5 ≈ 5% goal impact.
            gsax_clip = max(-1.0, min(1.0, opp_gsax_per60))
            goalie_adj_mult *= (1.0 - 0.10 * gsax_clip)
        elif not math.isnan(opp_dsv) and abs(opp_dsv) < GOALIE_PARAMS["max_dsv_pct_impact"]:
            goalie_adj_mult *= (1.0 - MODEL_WEIGHTS["goalie_impact_weight"] * opp_dsv)
        else:
            goalie_adj_mult *= (1.0 + MODEL_WEIGHTS["goalie_impact_weight"] * (league_sv - opp_sv))
        # GSAA as a small secondary adjustment
        goalie_adj_mult *= (1.0 - 0.02 * max(-1.0, min(1.0, opp_gsaa_pg)))
        mu *= goalie_adj_mult

        # Advanced stat adjustments (percentages are 0-100 scale, rates are per-game)
        mu *= (1.0 + 0.10 * safe_division(advanced_stats.get("xGF%", 50.0) - 50.0, 50.0, 0.0))
        mu *= (1.0 + 0.06 * safe_division(advanced_stats.get("SCF%", 50.0) - 50.0, 50.0, 0.0))
        mu *= (1.0 + 0.05 * safe_division(advanced_stats.get("HDCF%", 50.0) - 50.0, 50.0, 0.0))
        # PDO is mostly luck: dampen it aggressively toward the mean. A high PDO
        # means the team has overperformed, so we slightly lower future expectation
        # (regression), and vice versa.
        pdo = advanced_stats.get("PDO", 100.0)
        if pdo is not None and not math.isnan(pdo):
            mu *= (1.0 - 0.012 * safe_division(pdo - 100.0, 100.0, 0.0))
        mu *= (1.0 + 0.09 * safe_division(
            advanced_stats.get("EVGFpg", offense_all["GFpg"]) - offense_all["GFpg"],
            max(1e-9, offense_all["GFpg"]),
            0.0
        ))
        mu *= (1.0 + 0.09 * safe_division(
            advanced_stats.get("EVxGFpg", offense_all["xGFpg"]) - offense_all["xGFpg"],
            max(1e-9, offense_all["xGFpg"]),
            0.0
        ))
        mu *= (1.0 + 0.04 * safe_division(
            advanced_stats.get("FFpg", LEAGUE_AVERAGES["shots_per_game"]) - LEAGUE_AVERAGES["shots_per_game"],
            max(1e-9, LEAGUE_AVERAGES["shots_per_game"]),
            0.0
        ))
        mu *= (1.0 + 0.02 * safe_division(
            advanced_stats.get("CFpg", LEAGUE_AVERAGES["shots_per_game"]) - LEAGUE_AVERAGES["shots_per_game"],
            max(1e-9, LEAGUE_AVERAGES["shots_per_game"]),
            0.0
        ))

        mu *= (1.0 + MODEL_WEIGHTS["recent_form_weight"] * (rec_gfpg - offense_all["GFpg"]))
        mu *= (1.0 + MODEL_WEIGHTS["opponent_defense_weight"] * (rec_gapg_opp - defense_all["GApG"]))
        mu *= (1.0 + MODEL_WEIGHTS["lineup_impact_weight"] * lineup_delta)
        mu *= (1.0 + MODEL_WEIGHTS["momentum_factor_weight"] * momentum_factor)
        mu += venue_adj

        mu *= rest_mult
        mu *= score_mult

        # Apply injury impact (multiplicative)
        mu *= (1.0 + injury_impact)

        mu = float(max(0.4, min(8.0, mu)))

        breakdown = component_breakdown(
            base_xgf=round(base_xgf, 3),
            xga_adj=round(xga_adj, 3),
            pp_pk=round(st_goals, 3),
            goalie_adj=round(goalie_adj_mult - 1.0, 4),
            recent_adj=round(MODEL_WEIGHTS["recent_form_weight"] * (rec_gfpg - offense_all["GFpg"]), 4),
            lineup_adj=round(MODEL_WEIGHTS["lineup_impact_weight"] * lineup_delta, 4),
            momentum_adj=round(MODEL_WEIGHTS["momentum_factor_weight"] * momentum_factor, 4),
            rest_adj=round(rest_mult - 1.0, 4),
            score_adj=round(score_mult - 1.0, 4),
            final_mu=round(mu, 3)
        )
        return mu, breakdown

    advanced_stats_home = {
        "xGF%": home_all.get("xGF%", 50.0),
        "SCF%": home_all.get("SCF%", 50.0),
        "CF%": home_all.get("CF%", 50.0),
        "FF%": home_all.get("FF%", 50.0),
        "HDCF%": home_all.get("HDCF%", 50.0),
        "PDO": home_all.get("PDO", 100.0),
        "FFpg": home_all.get("FFpg", 25.0),
        "CFpg": home_all.get("CFpg", 45.0),
        "EVGFpg": home_ev.get("GFpg", home_all.get("GFpg", 3.0)),
        "EVxGFpg": home_ev.get("xGFpg", home_all.get("xGFpg", 3.0)),
        "gsax_per_game": home_all.get("gsax_per_game", 0.0),
    }

    advanced_stats_away = {
        "xGF%": away_all.get("xGF%", 50.0),
        "SCF%": away_all.get("SCF%", 50.0),
        "CF%": away_all.get("CF%", 50.0),
        "FF%": away_all.get("FF%", 50.0),
        "HDCF%": away_all.get("HDCF%", 50.0),
        "PDO": away_all.get("PDO", 100.0),
        "FFpg": away_all.get("FFpg", 25.0),
        "CFpg": away_all.get("CFpg", 45.0),
        "EVGFpg": away_ev.get("GFpg", away_all.get("GFpg", 3.0)),
        "EVxGFpg": away_ev.get("xGFpg", away_all.get("xGFpg", 3.0)),
        "gsax_per_game": away_all.get("gsax_per_game", 0.0),
    }

    # ✅ PURE DATA-DRIVEN VENUE ADVANTAGE (reduced scaling)
    venue_advantage = calculate_venue_advantage(home_abbr, away_abbr, season)

    # Capture individual venue records for the ML feature vector (cached)
    home_venue_record = get_team_home_away_record(home_abbr, season, as_of=game_date)
    away_venue_record = get_team_home_away_record(away_abbr, season, as_of=game_date)
    home_venue_pct = home_venue_record['home_win_pct']
    away_venue_pct = away_venue_record['away_win_pct']

    # Head-to-head history for the ML feature vector and a small physics nudge.
    h2h_pct = _h2h_pct_from_schedules(home_abbr, away_abbr, game_date, season)

    # Small H2H momentum adjustment: a team that has dominated recent meetings
    # gets a slight psychological/matchup edge, capped to avoid overfitting.
    h2h_physics_adj = 0.0
    if h2h_pct != 0.5:
        # Convert H2H win% deviation into a tiny goal adjustment.
        # 80% H2H record ≈ +0.12 goals, 20% ≈ -0.12 goals.
        h2h_physics_adj = 0.30 * (h2h_pct - 0.5)
        h2h_physics_adj = max(-0.15, min(0.15, h2h_physics_adj))
        logger.info(f"📊 H2H physics nudge: {h2h_physics_adj:+.3f} goals to home ({h2h_pct:.1%} home H2H)")

    # Calculate baseline mus. Score effects are an in-game state effect and are
    # intentionally NOT applied to pre-simulation expectations.
    mu_home, br_home = _compute_expected_goals(
        home_all, away_all, home_pp60, away_pk_ga60,
        home_opp_sv, home_opp_dsv, home_opp_gsaa_pg, home_opp_gsax60,
        home_rec_gf, away_rec_ga,
        lineup_shoot_home - lineup_shoot_away,
        venue_advantage + h2h_physics_adj,
        advanced_stats_home, extra_pp_goals_home, rest_mult_home, 1.0,  # no pre-sim score effects
        injury_impact_home,
        momentum_factor=home_momentum
    )
    mu_away, br_away = _compute_expected_goals(
        away_all, home_all, away_pp60, home_pk_ga60,
        away_opp_sv, away_opp_dsv, away_opp_gsaa_pg, away_opp_gsax60,
        away_rec_gf, home_rec_ga,
        lineup_shoot_away - lineup_shoot_home,
        0.0,
        advanced_stats_away, extra_pp_goals_away, rest_mult_away, 1.0,  # no pre-sim score effects
        injury_impact_away,
        momentum_factor=away_momentum
    )
    
    logger.info(f"Expected goals (baseline) - Home: {mu_home:.2f}, Away: {mu_away:.2f}")
    logger.info(f"Using {season} season data (current season Elo only)")

    try:
        app_state = get_app_state()
        team_elo_system = app_state['team_elo']
        feature_engine = app_state.get('feature_engine')
        ml_model = app_state.get('ml_model')
        current_season = app_state.get('current_season', 'unknown')

        # Use recent-form-adjusted Elo for the ensemble so the Elo component
        # reflects the last ~5-7 games, not just long-term team strength.
        home_team_obj = team_elo_system.get_or_create_team(home_abbr)
        away_team_obj = team_elo_system.get_or_create_team(away_abbr)
        home_elo_raw = home_team_obj.rating
        away_elo_raw = away_team_obj.rating
        home_elo = home_team_obj.get_recent_form_rating(days=14)
        away_elo = away_team_obj.get_recent_form_rating(days=14)

        # Compute Elo win probability (standard logistic)
        elo_diff = home_elo - away_elo
        elo_win_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

        logger.info(
            f"Elo ratings: {home_abbr}={home_elo_raw:.0f}→{home_elo:.0f} (adj), "
            f"{away_abbr}={away_elo_raw:.0f}→{away_elo:.0f} (adj) "
            f"→ win prob={elo_win_prob:.3f}"
        )
    except Exception as e:
        logger.warning(f"Could not load Elo ratings for ensemble: {e}")
        elo_win_prob = None
        feature_engine = None
        ml_model = None

    # ── ML adjustment: blend the physics-based baseline with the trained ML view ──
    ml_prob: Optional[float] = None
    try:
        if (
            ml_model is not None
            and getattr(ml_model, "is_trained", False)
            and feature_engine is not None
        ):
            ml_features = _build_ml_features(
                home_abbr=home_abbr,
                away_abbr=away_abbr,
                game_date=game_date,
                season=season,
                home_all=home_all,
                away_all=away_all,
                home_ev=home_ev,
                away_ev=away_ev,
                home_rec_gf=home_rec_gf,
                home_rec_ga=home_rec_ga,
                away_rec_gf=away_rec_gf,
                away_rec_ga=away_rec_ga,
                home_venue_pct=home_venue_pct,
                away_venue_pct=away_venue_pct,
                feat_home=feat_home,
                feat_away=feat_away,
                h2h_pct=h2h_pct,
                feature_engine=feature_engine,
                team_elo=team_elo_system,
            )
            raw_ml_prob = ml_model.predict_proba(ml_features)

            # Calibrate the raw ML probability using the OOF-fitted calibrator.
            # The calibrator was trained on the same model's out-of-fold predictions,
            # so it is only safe to apply here, not to the final ensemble blend.
            try:
                calibrator = _get_loaded_calibrator()
                if calibrator is not None:
                    ml_prob = calibrator.predict_one(raw_ml_prob)
                else:
                    ml_prob = raw_ml_prob
            except Exception:
                ml_prob = raw_ml_prob

            # Convert the (calibrated) probability into an expected-goal delta.
            # Apply the same magnitude to both teams so the model does not inflate
            # totals whenever it has conviction.
            prob_delta = ml_prob - 0.5
            goal_delta = prob_delta * 1.2
            ml_mu_home = max(0.5, mu_home + goal_delta)
            ml_mu_away = max(0.5, mu_away - goal_delta)
            logger.info(
                f"ML adjustment: baseline {mu_home:.2f} v {mu_away:.2f} "
                f"→ ML {ml_mu_home:.2f} v {ml_mu_away:.2f} (raw p={raw_ml_prob:.3f}, cal p={ml_prob:.3f})"
            )
            mu_home, mu_away = ml_mu_home, ml_mu_away
    except Exception as e:
        logger.warning(f"ML adjustment failed, using physics baseline: {e}")
        ml_prob = None

    # Score-effects are an in-game state effect, not a pre-game expectation.
    # Do not apply them to pre-simulation mus; they would inflate totals based
    # on projected goal differential rather than actual score state.
    mu_home = max(0.4, min(8.0, mu_home))
    mu_away = max(0.4, min(8.0, mu_away))
    br_home["Score-effects adj (mult-1)"] = 0.0
    br_away["Score-effects adj (mult-1)"] = 0.0
    br_home["Final mu"] = round(mu_home, 3)
    br_away["Final mu"] = round(mu_away, 3)
    logger.info(
        f"Final pre-simulation mu: home {mu_home:.2f} | away {mu_away:.2f}"
    )

    k_home = estimate_gamma_shape_from_recent(team_last_n_goals_list(home_abbr, game_date.isoformat(), n=10))
    k_away = estimate_gamma_shape_from_recent(team_last_n_goals_list(away_abbr, game_date.isoformat(), n=10))

    lam_h_base, lam_a_base = apply_per_sim_shock(mu_home, mu_away, sims=sims)

    rho = SIMULATION_PARAMS.get("correlation_rho", 0.38)
    shared = shared_correlation_factor(sims, rho=rho)
    if shared.size == sims:
        lam_h_base *= shared
        lam_a_base *= shared

    blow = np.random.random(size=sims)
    boost_home = np.where(
        blow < SIMULATION_PARAMS["blowout_probability"] / 2,
        SIMULATION_PARAMS["blowout_boost"],
        1.0
    )
    boost_away = np.where(
        (blow >= SIMULATION_PARAMS["blowout_probability"] / 2) &
        (blow < SIMULATION_PARAMS["blowout_probability"]),
        SIMULATION_PARAMS["blowout_boost"],
        1.0
    )
    lam_h = lam_h_base * boost_home
    lam_a = lam_a_base * boost_away

    if k_home is not None:
        lam_h = np.random.gamma(shape=k_home, scale=lam_h / max(k_home, 1e-9), size=sims)
    if k_away is not None:
        lam_a = np.random.gamma(shape=k_away, scale=lam_a / max(k_away, 1e-9), size=sims)

    home_goals = np.random.poisson(lam_h)
    away_goals = np.random.poisson(lam_a)

    final_home = home_goals.copy()
    final_away = away_goals.copy()

    final_home, final_away = apply_empty_net_adjustments(final_home, final_away, mu_home, mu_away)

    mode_home_goals, mode_away_goals = resolve_score_mode(final_home, final_away)
    most_likely_total = int(mode_home_goals + mode_away_goals)

    raw_prob = improved_winner_prediction(final_home, final_away)
    sim_win_prob = max(0.05, min(0.95, raw_prob))

    # Ensemble: blend simulation, Elo, and ML win probabilities.
    # Each source captures a different time horizon / signal.
    try:
        w_elo = MODEL_WEIGHTS["elo_winprob_weight"]
        w_sim = MODEL_WEIGHTS["simulation_winprob_weight"]
        w_ml = MODEL_WEIGHTS.get("ml_winprob_weight", 0.0)

        components: List[Tuple[Optional[float], float]] = [
            (elo_win_prob, w_elo),
            (sim_win_prob, w_sim),
            (ml_prob, w_ml),
        ]
        available = [(p, w) for p, w in components if p is not None]
        total_weight = sum(w for _, w in available)

        if available and total_weight > 0:
            winner_prob = sum(p * w for p, w in available) / total_weight
            winner_prob = max(0.05, min(0.95, winner_prob))
            logger.info(
                f"Ensemble win prob: Sim={sim_win_prob:.3f}, Elo={elo_win_prob if elo_win_prob is not None else 'N/A'}, "
                f"ML={ml_prob if ml_prob is not None else 'N/A'} → Blend={winner_prob:.3f}"
            )
        else:
            winner_prob = sim_win_prob
    except Exception:
        winner_prob = sim_win_prob

    mean_home = float(np.mean(final_home))
    mean_away = float(np.mean(final_away))
    median_home = int(np.median(final_home)) if len(final_home) > 0 else int(mu_home)
    median_away = int(np.median(final_away)) if len(final_away) > 0 else int(mu_away)

    totals = final_home + final_away
    totals_hist = Counter(totals.tolist())
    totals_dist = {int(k): int(v) for k, v in sorted(totals_hist.items())}

    denom = float(max(1, sims))
    
    pre_final_home = home_goals.copy()
    pre_final_away = away_goals.copy()
    pre_ties = int(np.sum(pre_final_home == pre_final_away))
    reg_games = int(sims - pre_ties)
    ot_games = int(round(pre_ties * 0.6))
    so_games = int(pre_ties - ot_games)

    diff = final_home - final_away
    vol = float(np.std(diff))
    
    data_quality_penalty = 0.0
    if g_df.empty:
        data_quality_penalty += 0.10
    if all_df.empty:
        data_quality_penalty += 0.20
    
    conf = float(max(0.0, min(1.0, 1.0 - 0.08 * vol - data_quality_penalty)))

    breakdown = {"HOME": br_home, "AWAY": br_away}

    result = {
        "mu_home": float(mu_home),
        "mu_away": float(mu_away),
        "home_win_pct": round(100.0 * winner_prob, 1),
        "away_win_pct": round(100.0 * (1 - winner_prob), 1),
        "home_win_2plus_pct": round(100.0 * (int(np.sum((final_home - final_away) >= 2)) / denom), 1),
        "away_win_2plus_pct": round(100.0 * (int(np.sum((final_away - final_home) >= 2)) / denom), 1),
        "exp_home_goals": float(mean_home),
        "exp_away_goals": float(mean_away),
        "median_home_goals": median_home,
        "median_away_goals": median_away,
        "mode_home_goals": mode_home_goals,
        "mode_away_goals": mode_away_goals,
        "most_likely_total": most_likely_total,
        "dist_home": final_home,
        "dist_away": final_away,
        "mode_pair": (mode_home_goals, mode_away_goals),
        "display_pair": (mode_home_goals, mode_away_goals),
        "sims": int(sims),
        "exp_home_shots": float(0.6 * home_all["SFpg"] + 0.4 * away_all["SApg"]),
        "exp_away_shots": float(0.6 * away_all["SFpg"] + 0.4 * home_all["SApg"]),
        "totals_distribution": totals_dist,
        "regulation_games_pct": round(100.0 * reg_games / denom, 1),
        "ot_games_pct": round(100.0 * ot_games / denom, 1),
        "so_games_pct": round(100.0 * so_games / denom, 1),
        "confidence": round(conf, 3),
        "home_goalie": selected_home_goalie,
        "away_goalie": selected_away_goalie,
        "breakdown": breakdown,
        "home_injuries": home_injury_data,
        "away_injuries": away_injury_data,
    }

    logger.info(
        f"Simulation complete - Home: {result['exp_home_goals']:.2f}G "
        f"({result['home_win_pct']:.1f}%), Away: {result['exp_away_goals']:.2f}G "
        f"({result['away_win_pct']:.1f}%)"
    )
    return result

__all__ = [
    'simulate_matchup',
    'get_team_rates_all',
    'get_team_rates_ev',
    'get_team_rates_pp_per60',
    'get_team_rates_pk_per60',
    'get_goalie_table',
    'season_from_date',
    'prev_season_key',
    'team_last_n_metrics',
    'team_last_n_goals_list',
    'team_recent_streak_factor',
]