"""
Defensive impact estimation from public on-ice data.

Uses MoneyPuck's season-summary skaters.csv to compute a per-player
"defensive importance" score for injured-player adjustments.  The score
blends on-ice expected-goals-against suppression, shot-attempt suppression,
blocked shots, and takeaways, and is normalized so a top-pairing shutdown
defenseman missing a game has a meaningfully larger effect than a depth
forward missing one.

The output is keyed by normalized player name so it can be joined against the
existing injury-impact path in NHL.Simulation.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from NHL.MoneyPuck import load_season_summary, MP_CACHE_DIR
from NHL.Utils import normalize_name_key

logger = logging.getLogger(__name__)

# In-memory cache with short TTL to avoid repeated CSV reads.
_CACHE: Dict[str, Tuple[float, Any]] = {}
_CACHE_TTL = 600  # seconds

# Scaling: tuned so a top-pairing shutdown D (~2 std above mean) produces an
# individual defensive impact around -3% to -4% of opponent goals when injured.
# These constants are intentionally conservative; the final simulation caps
# total defensive impact to avoid overfitting.
DEFENSE_POSITIONS = {"D", "LD", "RD"}
LEAGUE_AVG_XGA60 = 2.85  # league-average all-situations xGA/60 (approximate)


def _per_60(value: float, icetime_seconds: float) -> float:
    """Convert a cumulative stat to a per-60-minutes rate."""
    if not icetime_seconds or icetime_seconds <= 0:
        return 0.0
    return (float(value) / float(icetime_seconds)) * 3600.0


def _icetime_for_team(df: pd.DataFrame, team: str, situation: str = "all") -> float:
    """
    Estimate team ice time for a given situation.

    MoneyPuck does not expose team TOI directly in the skaters summary.  For
    5-on-5 this is well approximated by the average of the five highest
    skater icetimes on the team (the regular lineup); for all-situations it
    understates special-teams but is consistent across teams.  We only use it
    to scale off-ice stats, and errors here are small relative to the delta
    we are measuring.
    """
    sub = df[(df["team"] == team) & (df["situation"] == situation)]
    if sub.empty:
        return 0.0
    top5 = sub["icetime"].dropna().sort_values(ascending=False).head(5)
    if top5.empty:
        return 0.0
    return float(top5.mean())


def _compute_player_defense_row(
    row: pd.Series,
    team_icetime: float,
    league_xga60: float,
) -> Optional[Dict[str, Any]]:
    """Compute defensive metrics for one skater/situation row."""
    icetime = float(row.get("icetime", 0) or 0)
    if icetime < 60:  # need at least a minute of ice time
        return None

    name = str(row.get("name", "")).strip()
    if not name:
        return None

    position = str(row.get("position", "") or "").upper().strip()
    team = str(row.get("team", "") or "").upper().strip()

    gp = int(row.get("games_played", 0) or 0)
    hours = icetime / 3600.0
    # Sensible sample-size floor. Defensemen who kill penalties or were
    # traded/injured can have lower 5-on-5 ice time and still matter.
    min_hours = 4.0 if position in DEFENSE_POSITIONS else 6.0
    if gp < 5 or hours < min_hours:
        return None

    # On-ice defensive rates (lower = better defense)
    onice_xga = float(row.get("OnIce_A_xGoals", 0) or 0)
    onice_ca = float(row.get("OnIce_A_shotAttempts", 0) or 0)
    onice_xga60 = _per_60(onice_xga, icetime)
    onice_ca60 = _per_60(onice_ca, icetime)

    # MoneyPuck already computes the team's share of xG with/without the player.
    # We use the difference (on - off) as a clean "net defensive impact" signal:
    # positive means the team out-chances opponents when this player is on.
    on_ice_xg_pct = float(row.get("onIce_xGoalsPercentage", 0.5) or 0.5)
    off_ice_xg_pct = float(row.get("offIce_xGoalsPercentage", 0.5) or 0.5)
    if on_ice_xg_pct <= 0.0 or off_ice_xg_pct <= 0.0:
        delta_xg_pct = 0.0
    else:
        # Clamp to sensible range
        on_ice_xg_pct = max(0.05, min(0.95, on_ice_xg_pct))
        off_ice_xg_pct = max(0.05, min(0.95, off_ice_xg_pct))
        delta_xg_pct = on_ice_xg_pct - off_ice_xg_pct

    # Individual defensive actions
    blocks60 = _per_60(float(row.get("shotsBlockedByPlayer", 0) or 0), icetime)
    takeaways60 = _per_60(float(row.get("I_F_takeaways", 0) or 0), icetime)
    giveaways60 = _per_60(float(row.get("I_F_giveaways", 0) or 0), icetime)
    hits60 = _per_60(float(row.get("I_F_hits", 0) or 0), icetime)

    # Games played / usage
    gp = int(row.get("games_played", 0) or 0)
    icetime_rank = int(row.get("iceTimeRank", 999) or 999)

    # Composite defensive score.
    pos_mult = 1.0 if position in DEFENSE_POSITIONS else 0.55

    # 1. Shot suppression relative to league average.
    #    Lower on-ice xGA/60 -> positive contribution.
    suppression = (league_xga60 - onice_xga60) * 0.8

    # 2. Relative impact: team is better with player on ice than off.
    #    A +0.05 delta is excellent; scale so it matters but doesn't dominate.
    relative_impact = delta_xg_pct * 10.0

    # 3. Individual defensive actions
    actions = blocks60 * 0.18 + takeaways60 * 0.25 - giveaways60 * 0.12 + hits60 * 0.05

    # 4. Usage premium: heavy-ice-time players are harder to replace.
    hours = icetime / 3600.0
    usage = min(1.0, hours / 25.0) * 0.4

    raw_score = pos_mult * (suppression + relative_impact + actions) + usage

    return {
        "name": name,
        "name_key": normalize_name_key(name),
        "team": team,
        "position": position,
        "situation": str(row.get("situation", "") or ""),
        "gp": gp,
        "icetime_seconds": icetime,
        "icetime_hours": hours,
        "onice_xga60": round(onice_xga60, 3),
        "onice_xg_pct": round(on_ice_xg_pct, 3),
        "office_xg_pct": round(off_ice_xg_pct, 3),
        "delta_xg_pct": round(delta_xg_pct, 3),
        "onice_ca60": round(onice_ca60, 2),
        "blocks60": round(blocks60, 2),
        "takeaways60": round(takeaways60, 2),
        "giveaways60": round(giveaways60, 2),
        "hits60": round(hits60, 2),
        "ice_time_rank": icetime_rank,
        "raw_defensive_score": round(raw_score, 3),
    }


def compute_defensive_impact_scores(
    season_year: int,
    situation: str = "5on5",
    force_refresh: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute per-player defensive importance scores from MoneyPuck data.

    Uses 5-on-5 by default because all-situations numbers are polluted by
    special-teams usage: a shutdown defenseman who kills penalties will look
    worse in "all" simply because the PK bleeds xG by design.

    Returns {name_key: {name, team, position, ..., defensive_score}}
    keyed by normalized player name for easy joining with injury data.
    """
    cache_key = f"defense_scores_{season_year}_{situation}"
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and (now - cached[0]) < _CACHE_TTL and not force_refresh:
        return cached[1]

    df = load_season_summary(season_year, "skaters", force=force_refresh)
    if df is None or df.empty:
        logger.warning(f"No MoneyPuck skater data for {season_year}")
        return {}

    # Fill missing numeric columns so downstream math doesn't NaN.
    numeric_cols = [
        "icetime", "games_played", "OnIce_A_xGoals", "OnIce_A_shotAttempts",
        "OffIce_A_xGoals", "OffIce_A_shotAttempts", "shotsBlockedByPlayer",
        "I_F_takeaways", "I_F_giveaways", "I_F_hits", "iceTimeRank",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Compute team icetime approximations once per team.
    team_icetimes: Dict[str, float] = {}
    teams = df["team"].dropna().unique()
    for team in teams:
        team_icetimes[str(team).upper()] = _icetime_for_team(df, str(team), situation)

    # Work on a clean copy for the requested situation.
    sub = df[df["situation"] == situation].copy()

    # Percentage columns are sometimes stored as 0-1, sometimes 0-100.
    # Normalize to 0-1 to be safe.
    pct_cols = ["onIce_xGoalsPercentage", "offIce_xGoalsPercentage"]
    for col in pct_cols:
        if col in sub.columns:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
            # If max > 1, assume percentage scale and divide by 100.
            if sub[col].max() > 1.5:
                sub[col] = sub[col] / 100.0

    # Compute a global league xGA/60 baseline from the requested situation.
    league_xga60 = LEAGUE_AVG_XGA60
    if not sub.empty:
        xga60_series = (sub["OnIce_A_xGoals"] / sub["icetime"].replace(0, np.nan)) * 3600
        league_xga60 = float(xga60_series.dropna().median()) or LEAGUE_AVG_XGA60

    rows: List[Dict[str, Any]] = []
    for _, row in df[df["situation"] == situation].iterrows():
        team = str(row.get("team", "") or "").upper()
        rec = _compute_player_defense_row(
            row,
            team_icetime=team_icetimes.get(team, 0.0),
            league_xga60=league_xga60,
        )
        if rec:
            rows.append(rec)

    if not rows:
        logger.warning(f"No defensive rows computed for {season_year} situation={situation}")
        return {}

    scores_df = pd.DataFrame(rows)

    # Robust z-score normalization across all skaters.
    mean_score = float(scores_df["raw_defensive_score"].mean())
    std_score = float(scores_df["raw_defensive_score"].std())
    if std_score and not math.isnan(std_score) and std_score > 0:
        scores_df["defensive_z"] = (scores_df["raw_defensive_score"] - mean_score) / std_score
    else:
        scores_df["defensive_z"] = 0.0

    # Final score: z-score, but clipped so extreme outliers don't dominate.
    scores_df["defensive_score"] = scores_df["defensive_z"].clip(-4.0, 4.0)

    # Percentile for quick UI interpretation.
    scores_df["defensive_percentile"] = scores_df["defensive_score"].rank(pct=True) * 100

    out: Dict[str, Dict[str, Any]] = {}
    for _, r in scores_df.iterrows():
        key = str(r["name_key"])
        if not key:
            continue
        # If a player appears more than once (shouldn't for one situation),
        # keep the row with the highest absolute defensive score.
        existing = out.get(key)
        if existing and abs(existing["defensive_score"]) >= abs(r["defensive_score"]):
            continue
        out[key] = {
            "name": str(r["name"]),
            "team": str(r["team"]),
            "position": str(r["position"]),
            "situation": str(r["situation"]),
            "gp": int(r["gp"]),
            "icetime_hours": round(float(r["icetime_hours"]), 1),
            "onice_xga60": float(r["onice_xga60"]),
            "delta_xg_pct": float(r["delta_xg_pct"]),
            "onice_xg_pct": float(r["onice_xg_pct"]),
            "onice_ca60": float(r["onice_ca60"]),
            "blocks60": float(r["blocks60"]),
            "takeaways60": float(r["takeaways60"]),
            "giveaways60": float(r["giveaways60"]),
            "hits60": float(r["hits60"]),
            "defensive_score": round(float(r["defensive_score"]), 3),
            "defensive_percentile": round(float(r["defensive_percentile"]), 1),
            "raw_defensive_score": round(float(r["raw_defensive_score"]), 3),
        }

    _CACHE[cache_key] = (now, dict(out))
    logger.info(
        f"Computed defensive impact scores for {len(out)} skaters "
        f"({season_year}, situation={situation}, league_xga60={league_xga60:.2f})"
    )
    return out


def get_player_defensive_score(
    name: str,
    season_year: int,
    situation: str = "all",
    force_refresh: bool = False,
) -> Optional[Dict[str, Any]]:
    """Look up a single player's defensive score by display name."""
    scores = compute_defensive_impact_scores(season_year, situation, force_refresh)
    return scores.get(normalize_name_key(name))


def defensively_important_injuries(
    injuries: List[Dict[str, Any]],
    season_year: int,
    top_n: int = 5,
    min_score: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Given a list of injured players, return the most defensively important ones.

    Only players with a positive defensive_score above `min_score` are returned,
    so the list highlights shutdown defensemen and strong two-way forwards whose
    absence actually weakens the team defensively.

    Each entry in `injuries` should have at least {"player": "..."}.
    """
    scores = compute_defensive_impact_scores(season_year, "5on5")
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for inj in injuries:
        name = str(inj.get("player", "")).strip()
        if not name:
            continue
        key = normalize_name_key(name)
        if key in seen:
            continue
        seen.add(key)
        score = scores.get(key)
        if score and score.get("defensive_score", 0) >= min_score:
            out.append({
                "name": name,
                "position": score["position"],
                "defensive_score": score["defensive_score"],
                "defensive_percentile": score["defensive_percentile"],
                "delta_xg_pct": score["delta_xg_pct"],
                "onice_xga60": score["onice_xga60"],
                "blocks60": score["blocks60"],
                "icetime_hours": score["icetime_hours"],
                "status": str(inj.get("status", "injured")),
            })

    out.sort(key=lambda x: x["defensive_score"], reverse=True)
    return out[:top_n]


__all__ = [
    "compute_defensive_impact_scores",
    "get_player_defensive_score",
    "defensively_important_injuries",
]
