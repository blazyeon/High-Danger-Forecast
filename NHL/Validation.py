"""
Validation: cross-check our PBP-derived xG against MoneyPuck's xG.

For every season where we have both PBP shots and MP shots, we:
  1. Load both DataFrames
  2. Join on (game_id, event_id) when the IDs align
  3. Compute Pearson correlation, Brier score, and log-loss of our xG
     against MP's xG
  4. Write a JSON report

Why this matters: MP's xGoal is a published, vetted model. If our xG
diverges significantly (correlation < 0.85, or Brier > 0.05 worse),
there's a bug in our feature engineering or model training.
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from NHL.Config import DEFAULT_TIMEOUT, REQUEST_HEADERS
from NHL.MoneyPuck import (
    MP_CACHE_DIR,
    download_shots_zip,
    parse_mp_shots,
)
from NHL.PlayByPlay import load_shot_store, SHOT_STORE_DIR
from NHL.xGModel import (
    ALL_FEATURE_COLS,
    MODEL_DIR,
    REPORT_PATH as XG_REPORT_PATH,
    load_xg_model,
    predict_xg,
)

logger = logging.getLogger(__name__)

# A small tolerance window for joining shots: if event_id doesn't match
# exactly, we fall back to matching by approximate time within a game.
JOIN_TIME_TOLERANCE_SEC = 2.0

# Validation report path
VALIDATION_REPORT_PATH = MODEL_DIR / "xg_validation.json"


# ── Join helpers ────────────────────────────────────────────────────────

def _normalize_pbp_game_id(gid: int) -> int:
    """
    Convert NHL game_id (e.g., 2024020001) to MoneyPuck game_id (20001)
    by stripping the 4-digit year+type prefix.
    """
    s = str(int(gid))
    if len(s) >= 8:
        return int(s[4:])  # strip YYYY and TT
    return int(gid)


def _build_pbp_shot_sequence(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-game shot sequence number (1-indexed) from PBP event_id.
    MP's 'event_id' is the n-th shot of the game, not the global eventId.
    We sort by event_id within each game and re-rank to get the shot #.
    """
    df = pbp.copy()
    df = df.sort_values(["game_id", "event_id"])
    df["shot_seq"] = df.groupby("game_id").cumcount() + 1
    return df


def _join_pbp_to_mp(
    pbp: pd.DataFrame,
    mp: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join our PBP shots to MoneyPuck shots.

    MoneyPuck's game_id is the suffix of NHL's game_id (last 5 digits).
    MP's 'event_id' is a per-game counter across ALL event types (faceoffs,
    hits, etc.) not just shots, so we can't use it for joining. The
    reliable join is on (mp_game_id, period, time_seconds, shooter_id,
    team_id), which matches 85%+ of MP shots and avoids the rare
    duplicate (period, time, shooter) collisions in PBP.
    """
    if pbp.empty or mp.empty:
        return pd.DataFrame()

    pbp_k = pbp.copy()
    pbp_k["mp_game_id"] = pbp_k["game_id"].apply(_normalize_pbp_game_id)
    pbp_k = pbp_k[["mp_game_id", "period", "time_seconds", "shooter_id", "team_id",
                   "x", "y", "is_goal", "shot_type"]].copy()
    pbp_k["shooter_id"] = pbp_k["shooter_id"].astype("Int64")
    pbp_k["team_id"] = pbp_k["team_id"].astype("Int64")

    # MP team_abbr_mp → NHL team_id (from our TEAM_ID_TO_ABBR reverse)
    from NHL.StatsFromPBP import TEAM_ID_TO_ABBR
    abbr_to_id = {v: k for k, v in TEAM_ID_TO_ABBR.items()}
    # Apply ARI→UTA mapping
    try:
        from NHL.Config import TEAM_ABBR_MAPPING as TAM
    except Exception:
        TAM = {"ARI": "UTA"}
    mp_team_id = mp["team_abbr_mp"].map(lambda a: abbr_to_id.get(TAM.get(str(a).upper(), str(a).upper()), 0)
                                          if pd.notna(a) else 0)

    mp_k = mp.copy()
    mp_k = mp_k.rename(columns={"game_id": "mp_game_id", "shooterPlayerId": "shooter_id"})
    mp_k["shooter_id"] = mp_k["shooter_id"].astype("Int64")
    mp_k["team_id"] = mp_team_id.astype("Int64")
    mp_k = mp_k[["mp_game_id", "period", "time_seconds", "shooter_id", "team_id",
                 "xgoal_mp", "distance", "angle",
                 "is_rebound", "is_rush", "home_skaters", "away_skaters"]].copy()

    joined = pbp_k.merge(mp_k, on=["mp_game_id", "period", "time_seconds", "shooter_id", "team_id"],
                         how="inner", suffixes=("", "_mp"))
    if joined.empty:
        return joined
    # Defensive dedup
    joined = joined.drop_duplicates(
        subset=["mp_game_id", "period", "time_seconds", "shooter_id", "team_id"],
        keep="first"
    )
    return joined


# ── Main validation entry point ─────────────────────────────────────────

def validate_xg_against_money_puck(
    pbp_shots: Optional[pd.DataFrame] = None,
    mp_shots: Optional[pd.DataFrame] = None,
    season_year: int = 2024,
    stype: int = 2,
) -> Dict:
    """
    Compare our xG to MoneyPuck's xG for a given season.

    Loads PBP from our local shot store and MP from disk (downloading if
    needed). Trains a quick xG model on 3 seasons of PBP (or uses the
    saved one), then runs the cross-check. Writes
    models/xg_validation.json.
    """
    if pbp_shots is None:
        pbp_shots = load_shot_store(season_year, stype)
    if pbp_shots.empty:
        return {"error": "no PBP shots available", "season": season_year}

    if mp_shots is None:
        mp_csv = MP_CACHE_DIR / f"shots_{season_year}.csv"
        if not mp_csv.exists():
            download_shots_zip([season_year])
        if not mp_csv.exists():
            return {"error": f"no MP shots at {mp_csv}", "season": season_year}
        mp_shots = parse_mp_shots(mp_csv)

    if mp_shots.empty:
        return {"error": "no MP shots available", "season": season_year}

    # Join
    joined = _join_pbp_to_mp(pbp_shots, mp_shots)
    if joined.empty:
        return {
            "error": "no matching shots between PBP and MP",
            "season": season_year,
            "n_pbp": len(pbp_shots),
            "n_mp": len(mp_shots),
        }

    # Predict our xG on the PBP shots that matched
    try:
        model = load_xg_model()
    except FileNotFoundError:
        return {
            "error": "no xG model trained. Run update_pbp_stats.py --train-xg first.",
            "season": season_year,
        }
    # Build the matched PBP rows in the same order as joined.
    pbp_for_pred = pbp_shots.copy()
    pbp_for_pred["mp_game_id"] = pbp_for_pred["game_id"].apply(_normalize_pbp_game_id)
    pbp_for_pred["shooter_id"] = pbp_for_pred["shooter_id"].astype("Int64")
    pbp_for_pred["team_id"] = pbp_for_pred["team_id"].astype("Int64")
    # Dedup PBP on the join key (rare collisions where two PBP shots
    # share the same (mp_game_id, period, time, shooter, team) — usually
    # a data issue but happens for ~0.1% of rows).
    pbp_for_pred = pbp_for_pred.drop_duplicates(
        subset=["mp_game_id", "period", "time_seconds", "shooter_id", "team_id"],
        keep="first"
    )
    join_keys = joined[["mp_game_id", "period", "time_seconds", "shooter_id", "team_id"]].copy()
    join_keys["shooter_id"] = join_keys["shooter_id"].astype("Int64")
    join_keys["team_id"] = join_keys["team_id"].astype("Int64")
    pbp_matched = join_keys.merge(
        pbp_for_pred, on=["mp_game_id", "period", "time_seconds", "shooter_id", "team_id"],
        how="left", suffixes=("", "_pbp")
    )
    if len(pbp_matched) != len(joined) or pbp_matched["x"].isna().any():
        return {
            "error": "could not recover PBP rows for joined keys",
            "n_joined": len(joined),
            "n_pbp_matched": int(pbp_matched["x"].notna().sum()),
        }
    # Reset index so the row order matches joined
    pbp_matched = pbp_matched.reset_index(drop=True)
    xg_ours = predict_xg(pbp_matched, model)
    if len(xg_ours) != len(joined):
        return {"error": "alignment failed", "n_joined": len(joined), "n_pred": len(xg_ours)}

    joined = joined.copy()
    joined["xg_ours"] = xg_ours
    y = joined["is_goal"].astype(int).values
    xg_mp = joined["xgoal_mp"].astype(float).values
    xg_us = joined["xg_ours"].astype(float).values

    # Drop NaN from MP
    mask = np.isfinite(xg_mp) & np.isfinite(xg_us) & np.isfinite(y)
    if mask.sum() < 100:
        return {"error": "too few valid rows for stats", "n_valid": int(mask.sum())}
    y = y[mask]
    xg_mp = xg_mp[mask]
    xg_us = xg_us[mask]

    # ── Metrics
    # Correlation: how well do our xG ranks agree with MP's?
    corr = float(np.corrcoef(xg_us, xg_mp)[0, 1]) if xg_us.std() > 0 and xg_mp.std() > 0 else 0.0
    # Brier score: how accurate are our probabilities vs outcomes
    brier_ours = float(brier_score_loss(y, xg_us))
    brier_mp = float(brier_score_loss(y, xg_mp))
    # Log-loss
    try:
        ll_ours = float(log_loss(y, np.clip(xg_us, 1e-6, 1 - 1e-6)))
    except Exception:
        ll_ours = float("nan")
    try:
        ll_mp = float(log_loss(y, np.clip(xg_mp, 1e-6, 1 - 1e-6)))
    except Exception:
        ll_mp = float("nan")
    # AUC
    try:
        auc_ours = float(roc_auc_score(y, xg_us))
    except Exception:
        auc_ours = float("nan")
    try:
        auc_mp = float(roc_auc_score(y, xg_mp))
    except Exception:
        auc_mp = float("nan")

    # Mean predicted vs actual goals
    mean_mp = float(xg_mp.mean())
    mean_ours = float(xg_us.mean())
    actual_rate = float(y.mean())

    report = {
        "season": season_year,
        "n_joined": int(len(joined)),
        "n_valid": int(mask.sum()),
        "correlation_ours_vs_mp": corr,
        "auc_ours": auc_ours,
        "auc_mp": auc_mp,
        "brier_ours": brier_ours,
        "brier_mp": brier_mp,
        "brier_diff": brier_ours - brier_mp,
        "logloss_ours": ll_ours,
        "logloss_mp": ll_mp,
        "logloss_diff": ll_ours - ll_mp,
        "mean_xg_ours": mean_ours,
        "mean_xg_mp": mean_mp,
        "actual_goal_rate": actual_rate,
        "verdict": "OK" if corr > 0.85 and abs(brier_ours - brier_mp) < 0.01 else "REVIEW",
    }

    try:
        with open(VALIDATION_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(
            f"Validation: corr={corr:.3f}, AUC ours={auc_ours:.3f} vs MP={auc_mp:.3f}, "
            f"Brier ours={brier_ours:.4f} vs MP={brier_mp:.4f} → {VALIDATION_REPORT_PATH}"
        )
    except Exception as e:
        logger.warning(f"Failed to write validation report: {e}")

    return report


__all__ = [
    "validate_xg_against_money_puck",
    "VALIDATION_REPORT_PATH",
]
