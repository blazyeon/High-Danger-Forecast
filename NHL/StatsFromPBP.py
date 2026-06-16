"""
Stats aggregator: PBP shots → team / skater / goalie rates.

Replaces the role of NST data in the simulation and prediction layers.
The output shapes are designed to be drop-in compatible with the existing
NST-driven functions, so callers don't have to change much.

Output contract (preserved from old NST versions):

  season_skater_rates(season, stype) → {name_key: {gpg, apg, sogpg, xgf_pg}}
    name_key is normalize_name_key(name). Consumer in Prediction.py
    iterates and reads gpg/apg/sogpg directly.

  compute_team_rates(season, stype, fd, td) → pd.DataFrame
    Per-team row with team, gp, xgf, xga, gf, ga, cf, ca, ff, fa, sf, sa,
    hdcf, hdca, goals_per_game, xgf_per_game, xgf_pct, cf_pct, ff_pct,
    sf_pct, hdcf_pct, sv_pct, pdo, sh_pct. The simulation in
    NHL/Simulation.py selects rows by team abbrev.

  compute_goalie_rates(season, stype) → pd.DataFrame
    Per-goalie row: name, gp, ga, gaa, sv, sa, sv_pct, xga, gsax, gsax_per_60.

Where stats come from:
- Shots: NHL API PBP (via NHL.PlayByPlay)
- Assists, primary/secondary attribution: PBP doesn't carry assists as
  cleanly as shots. For assists, we use the shot-store totals: a goal's
  assists are tracked in PBP via separate events (we approximate from
  the goal events). If precise assists are unavailable, we estimate
  from goal totals and team rate (NHL avg is ~1.6 assists per goal).
- HD shots: PBP doesn't tag high-danger directly. We use a proxy: shots
  from within the inner slot (distance < 25 ft, angle < 25°) are
  treated as high-danger. This is a standard proxy used in the
  MoneyPuck / NST public models.
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from NHL.Config import LEAGUE_AVERAGES
from NHL.Utils import normalize_name_key
from NHL.PlayByPlay import (
    load_shot_store,
    game_date_map,
    SHOT_STORE_DIR,
)

logger = logging.getLogger(__name__)

# Default date window for the in-progress season: from season start to today
DEFAULT_TODAY = date.today()

# High-danger proxy: shots from this distance/angle are "high danger"
HD_DISTANCE_FT = 25.0
HD_ANGLE_DEG = 25.0

# Team abbr normalization (Arizona/Utah mapping from Config is the source of
# truth; we just import it for consistency)
try:
    from NHL.Config import TEAM_ABBR_MAPPING, NST_ABBR_TO_FULL
    _TEAM_ABBR_MAPPING = TEAM_ABBR_MAPPING
except Exception:
    _TEAM_ABBR_MAPPING = {"ARI": "UTA"}


# ── Helpers ─────────────────────────────────────────────────────────────

def _filter_by_date(
    shots: pd.DataFrame,
    fd: str = "",
    td: str = "",
    season_year: Optional[int] = None,
    stype: int = 2,
) -> pd.DataFrame:
    """
    Filter shots to the [fd, td] date window. fd/td are YYYY-MM-DD strings.
    Empty strings mean "no bound on that side". season_year+stype scope the
    date lookup to one season's schedule (we keep the per-season shot store).
    """
    if not fd and not td:
        return shots
    if shots.empty:
        return shots
    if "game_id" not in shots.columns:
        return shots
    if season_year is None:
        return shots  # no way to map game_id → date

    dmap = game_date_map(season_year, stype=stype)
    game_dates = shots["game_id"].map(dmap)
    if fd:
        game_dates = game_dates[game_dates >= fd]
    if td:
        game_dates = game_dates[game_dates <= td]
    keep_ids = game_dates.dropna().index
    return shots.loc[shots.index.isin(keep_ids)]


def _normalize_team(abbr: Optional[str]) -> str:
    """Apply ARI→UTA and uppercase."""
    if abbr is None or (isinstance(abbr, float) and np.isnan(abbr)):
        return ""
    s = str(abbr).upper().strip()
    return _TEAM_ABBR_MAPPING.get(s, s)


def _is_high_danger(x: float, y: float) -> bool:
    """HD proxy: within HD_DISTANCE_FT of the net and HD_ANGLE_DEG degrees."""
    dx = 89.0 - abs(x) if x < 0 else 89.0 - x
    dx = max(0.0, dx)
    dy = y
    dist = math.sqrt(dx * dx + dy * dy)
    angle = abs(math.degrees(math.atan2(dy, dx))) if dx > 0 else 90.0
    return dist < HD_DISTANCE_FT and angle < HD_ANGLE_DEG


def _load_skater_rates_from_json(season_year: int, stype: int) -> Dict[str, Dict[str, float]]:
    """
    Fallback when the shot parquet is missing on the server.

    Loads the pre-computed full-season skater rates exported by
    update_pbp_stats.py. This avoids the live NHL API crawl that
    times out on Render. Date-window filtering is not available in
    this fallback; callers receive full-season rates.
    """
    json_path = Path("static/data/pbp_skater_stats.json")
    if not json_path.exists():
        return {}
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if payload.get("stype") != stype:
            return {}
        expected_season = f"{season_year}{season_year + 1}"
        if payload.get("season") != expected_season:
            return {}
        out: Dict[str, Dict[str, float]] = {}
        for rec in payload.get("data", []):
            name = rec.get("name", "")
            if not name:
                continue
            key = normalize_name_key(name)
            if not key:
                continue
            out[key] = {
                "gp": int(rec.get("gp", 0)),
                "goals": int(rec.get("goals", 0)),
                "assists": int(rec.get("assists", 0)),
                "shots": int(rec.get("shots", 0)),
                "gpg": float(rec.get("gpg", 0.0)),
                "apg": float(rec.get("apg", 0.0)),
                "sogpg": float(rec.get("sogpg", 0.0)),
                "xgf_pg": float(rec.get("xgf_pg", 0.0)),
            }
        logger.info(
            f"Loaded {len(out)} skater rates from fallback JSON for "
            f"{expected_season} stype={stype}"
        )
        return out
    except Exception as e:
        logger.warning(f"Failed to load skater rates fallback JSON: {e}")
        return {}


# ── Team rates ──────────────────────────────────────────────────────────

def compute_team_rates(
    season_year: int,
    stype: int = 2,
    fd: str = "",
    td: str = "",
) -> pd.DataFrame:
    """
    Per-team rates for a given season, optionally filtered to a date window.

    Returns a DataFrame with the columns the simulation expects.
    """
    shots = load_shot_store(season_year, stype)
    if shots.empty:
        return pd.DataFrame()
    shots = _filter_by_date(shots, fd, td, season_year, stype)

    # Pre-compute HD flag once and attach to the shots df
    hd = shots.apply(lambda r: _is_high_danger(float(r["x"]), float(r["y"])), axis=1)
    shots = shots.copy()
    shots["hd"] = hd

    out_rows: List[Dict] = []
    grouped = shots.groupby("team_id", dropna=False)
    # Resolve team_id → 3-letter abbrev for human-readable output. The
    # simulation's _match_team normalizes both sides through NST_ABBR_TO_FULL
    # so an abbrev here lines up with what NHL/Simulation.py and app.py
    # pass in. Team_id 0 / NaN fall through to "" which downstream code
    # will skip.
    for team_id, grp in grouped:
        gf = int(grp["is_goal"].sum())
        # All attempts = shots + missed + blocked
        attempts = len(grp)
        goals_against = 0  # filled below via cross-team
        # We can't know goals_against for a team without looking at the
        # opponent's shots. We compute it as total goals in games this
        # team played in minus goals the team itself scored.
        all_goals = int(grp["is_goal"].sum())
        # Simpler: ga = total goals in games - gf
        if "game_id" in grp.columns:
            game_ids = grp["game_id"].unique()
            other_shots = shots[shots["game_id"].isin(game_ids) & (shots["team_id"] != team_id)]
            goals_against = int(other_shots["is_goal"].sum())
        shots_for = int((grp["is_shot"] == 1).sum())
        shots_against = 0
        if "game_id" in grp.columns:
            game_ids = grp["game_id"].unique()
            other = shots[shots["game_id"].isin(game_ids) & (shots["team_id"] != team_id)]
            shots_against = int((other["is_shot"] == 1).sum())
        cf = shots_for + int((grp["event_type"] == "missed").sum()) + int((grp["event_type"] == "blocked").sum())
        ca = attempts - shots_for + shots_against  # rough
        # Fenwick = shots + missed (unblocked)
        ff = shots_for + int((grp["event_type"] == "missed").sum())
        # We don't have fa cleanly here; use the opponent fenwick proxy
        if "game_id" in grp.columns:
            other = shots[shots["game_id"].isin(game_ids) & (shots["team_id"] != team_id)]
            fa = int((other["is_shot"] == 1).sum()) + int((other["event_type"] == "missed").sum())
        else:
            fa = 0
        sf = shots_for
        sa = shots_against
        hdcf = int(grp["hd"].sum())
        hdca = 0
        if "game_id" in grp.columns:
            other = shots[shots["game_id"].isin(game_ids) & (shots["team_id"] != team_id)]
            hdca = int(other["hd"].sum())

        gp = int(grp["game_id"].nunique()) if "game_id" in grp.columns else 0
        gp = max(gp, 1)
        team_abbr = team_abbr_from_id(team_id)
        out_rows.append({
            "team": team_abbr or str(team_id),
            "team_id": int(team_id) if not pd.isna(team_id) else 0,
            "gp": gp,
            "gf": gf,
            "ga": goals_against,
            "cf": cf,
            "ca": ca,
            "ff": ff,
            "fa": fa,
            "sf": sf,
            "sa": sa,
            "hdcf": hdcf,
            "hdca": hdca,
            "goals_per_game": gf / gp,
            "shots_per_game": sf / gp,
            "xgf": float("nan"),  # filled by xG model if available
            "xga": float("nan"),
            "xgf_per_game": float("nan"),
            "xga_per_game": float("nan"),
            "cf_pct": cf / max(cf + ca, 1),
            "ff_pct": ff / max(ff + fa, 1),
            "sf_pct": sf / max(sf + sa, 1),
            "hdcf_pct": hdcf / max(hdcf + hdca, 1),
            "xgf_pct": float("nan"),
            "sv_pct": 1.0 - (goals_against / max(shots_against, 1)) if shots_against else LEAGUE_AVERAGES["sv_pct"],
            "sh_pct": gf / max(sf, 1),
            "pdo": float("nan"),
        })

    df = pd.DataFrame(out_rows)
    if df.empty:
        return df

    # Add xG from the model if it's available
    try:
        from NHL.xGModel import load_xg_model, predict_xg
        model = load_xg_model()
        xg_vals = predict_xg(shots, model)
        shots_with_xg = shots.copy()
        shots_with_xg["xg"] = xg_vals
        xgf_by_team = shots_with_xg.groupby("team_id")["xg"].sum()
        # Goals against: sum of xG from opponent shots in each game
        xga_by_team = {}
        for tid in df["team_id"]:
            game_ids = shots_with_xg[shots_with_xg["team_id"] == tid]["game_id"].unique()
            opp = shots_with_xg[
                (shots_with_xg["game_id"].isin(game_ids)) & (shots_with_xg["team_id"] != tid)
            ]
            xga_by_team[tid] = float(opp["xg"].sum())
        df["xgf"] = df["team_id"].map(xgf_by_team).fillna(0)
        df["xga"] = df["team_id"].map(xga_by_team).fillna(0)
        df["xgf_per_game"] = df["xgf"] / df["gp"]
        df["xga_per_game"] = df["xga"] / df["gp"]
        df["xgf_pct"] = df["xgf"] / (df["xgf"] + df["xga"]).replace(0, 1)
        df["gsax"] = df["ga"] - df["xga"]
    except FileNotFoundError:
        logger.debug("No xG model trained yet, leaving xg columns NaN")
    except Exception as e:
        logger.warning(f"xG enrichment failed: {e}")

    return df


# ── Skater rates ────────────────────────────────────────────────────────

def compute_skater_rates(
    season_year: int,
    stype: int = 2,
    fd: str = "",
    td: str = "",
) -> Dict[str, Dict[str, float]]:
    """
    Per-player rates for a given season, filtered to a date window.

    Returns {name_key: {gpg, apg, sogpg, xgf_pg}} where name_key is
    normalize_name_key(name). Drop-in compatible with the old
    season_skater_rates_from_nst return shape.
    """
    shots = load_shot_store(season_year, stype)
    if shots.empty:
        # On Render the PBP parquet may be missing on a cold start. Fall back
        # to the pre-computed full-season JSON so we don't crawl the NHL API
        # and hit the gunicorn timeout.
        if fd or td:
            logger.warning(
                f"Shot store missing for {season_year}-{season_year + 1} "
                f"and date window [{fd}, {td}] requested; using full-season "
                f"fallback JSON (window filter will be ignored)."
            )
        return _load_skater_rates_from_json(season_year, stype)
    shots = _filter_by_date(shots, fd, td, season_year, stype)

    # Group by shooter_id (stable) but expose by shooter_name (what callers
    # will look up). Both can fail — empty shooter_name means we skip the row.
    out: Dict[str, Dict[str, float]] = {}

    # Build assist estimates per player
    # PBP doesn't give us primary/secondary assists directly. We approximate:
    #   apg ≈ (gf * 1.6 / gp) * (player_shot_share)  (1.6 = NHL avg assists/goal)
    # But for simplicity here we compute apg from goal involvement: a
    # player on the ice for a goal gets ~0.7 of a goal (counted as either
    # the shooter or one of the assisters). Since we only have shooter
    # info from PBP, we default assists to (shooter_goals * 0.7) for the
    # goal events, then split. The NHL convention is the last 2 events
    # before a goal are assists; we approximate by counting all shots in
    # the prior 5 seconds as potential "pre-goal" touches and scaling.
    # This is rough — a real assist model would need the PBP play-by-play
    # to be processed in event order. Out of scope for this iteration.
    by_shooter = shots.groupby("shooter_id", dropna=True)
    games_per_shooter = shots.groupby("shooter_id")["game_id"].nunique()

    # Count actual assists: every row in `shots` with a non-null
    # `assist1_id` or `assist2_id` credits the corresponding player
    # with one assist. We do this by walking the goal rows once and
    # adding to per-shooter totals.
    assists_count: Dict[int, int] = {}
    if "assist1_id" in shots.columns:
        for pid in shots["assist1_id"].dropna().unique():
            try:
                pid_int = int(pid)
            except (TypeError, ValueError):
                continue
            n = int((shots["assist1_id"] == pid).sum())
            assists_count[pid_int] = assists_count.get(pid_int, 0) + n
    if "assist2_id" in shots.columns:
        for pid in shots["assist2_id"].dropna().unique():
            try:
                pid_int = int(pid)
            except (TypeError, ValueError):
                continue
            n = int((shots["assist2_id"] == pid).sum())
            assists_count[pid_int] = assists_count.get(pid_int, 0) + n

    for shooter_id, grp in by_shooter:
        try:
            shooter_id_int = int(shooter_id)
        except (TypeError, ValueError):
            continue
        name = grp["shooter_name"].iloc[0] if "shooter_name" in grp.columns else None
        if not name or str(name).strip() == "" or str(name).lower() == "nan":
            continue
        key = normalize_name_key(name)
        if not key:
            continue
        gp = max(1, int(games_per_shooter.get(shooter_id, 1)))
        goals = int(grp["is_goal"].sum())
        sog = int((grp["is_shot"] == 1).sum())
        # Real assists from the goal events (assist1 + assist2 columns).
        # The previous version used a goals*0.5 proxy; the PBP now
        # gives us actual primary/secondary assists.
        assists = assists_count.get(shooter_id_int, 0)
        # Add xG
        xgf = 0.0
        try:
            from NHL.xGModel import load_xg_model, predict_xg
            model = load_xg_model()
            xg_vals = predict_xg(grp, model)
            xgf = float(xg_vals.sum())
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug(f"xG for skater rates failed: {e}")

        out[key] = {
            "name": str(name),
            "gpg": goals / gp,
            "apg": assists / gp,
            "sogpg": sog / gp,
            "xgf_pg": xgf / gp,
            "gp": gp,
            "goals": goals,
            "shots": sog,
            "assists": assists,
        }

    return out


# ── Goalie rates ────────────────────────────────────────────────────────

def compute_goalie_rates(
    season_year: int,
    stype: int = 2,
    fd: str = "",
    td: str = "",
) -> pd.DataFrame:
    """
    Per-goalie rates. Returns a DataFrame with name, gp, ga, gaa, sv, sa,
    sv_pct, xga, gsax, gsax_per_60. The Flask app's goalie dropdown reads
    this.
    """
    shots = load_shot_store(season_year, stype)
    if shots.empty:
        return pd.DataFrame()
    shots = _filter_by_date(shots, fd, td, season_year, stype)

    by_goalie = shots.groupby("goalie_id", dropna=True)
    rows: List[Dict] = []
    for goalie_id, grp in by_goalie:
        name = grp["goalie_name"].iloc[0] if "goalie_name" in grp.columns else None
        if not name or str(name).strip() == "" or str(name).lower() == "nan":
            continue
        gp = int(grp["game_id"].nunique())
        sa = int((grp["is_shot"] == 1).sum())
        ga = int(grp["is_goal"].sum())
        sv = sa - ga
        # GAA: 60-min rate, assumes goalie played all 60 min in each game
        gaa = ga / max(gp, 1)
        sv_pct = sv / max(sa, 1) if sa else LEAGUE_AVERAGES["sv_pct"]
        # xGA from xG model
        xga = 0.0
        try:
            from NHL.xGModel import load_xg_model, predict_xg
            model = load_xg_model()
            xg_vals = predict_xg(grp, model)
            xga = float(xg_vals.sum())
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug(f"xG for goalie rates failed: {e}")
        gsax = ga - xga
        gsax_per_60 = gsax / max(gp, 1)
        rows.append({
            "name": str(name),
            "goalie_id": int(goalie_id) if not pd.isna(goalie_id) else 0,
            "gp": gp,
            "ga": ga,
            "gaa": gaa,
            "sa": sa,
            "sv": sv,
            "sv_pct": sv_pct,
            "xga": xga,
            "gsax": gsax,
            "gsax_per_60": gsax_per_60,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("gp", ascending=False).reset_index(drop=True)
    return df


# ── Team abbr from team_id (small static table) ────────────────────────

# A small lookup: NHL internal team_id → 3-letter abbrev. Used to populate
# the 'team' column in compute_team_rates for human-readable output.
# Source: NHL's public teams endpoint, hardcoded here for offline use.
# Note: 59 = Utah Hockey Club (the relocated Arizona franchise,
# renamed 2024-25). 30 is still ARI for the historical Arizona Coyotes
# games that predate the move.
TEAM_ID_TO_ABBR: Dict[int, str] = {
    # Mapping that matches the team_id values emitted by the NHL API
    # play-by-play feed we consume (this is NOT the global franchise id).
    1: "NJD", 2: "NYI", 3: "NYR", 4: "PHI", 5: "PIT", 6: "BOS",
    7: "BUF", 8: "MTL", 9: "OTT", 10: "TOR", 12: "CAR", 13: "FLA",
    14: "TBL", 15: "WSH", 16: "CHI", 17: "DET", 18: "NSH", 19: "STL",
    20: "CGY", 21: "COL", 22: "EDM", 23: "VAN", 24: "ANA", 25: "DAL",
    26: "LAK", 28: "SJS", 29: "CBJ", 30: "MIN",
    52: "WPG",  # PBP uses this id for the current Winnipeg Jets
    53: "ARI",  # legacy Arizona Coyotes (folded); remapped to UTA below
    54: "VGK", 55: "SEA", 59: "UTA", 68: "UTA",  # Utah Hockey Club
}


def team_abbr_from_id(team_id: Optional[int]) -> str:
    if team_id is None or (isinstance(team_id, float) and np.isnan(team_id)):
        return ""
    tid = int(team_id)
    abbr = TEAM_ID_TO_ABBR.get(tid, "")
    return _TEAM_ABBR_MAPPING.get(abbr, abbr)


__all__ = [
    "compute_team_rates",
    "compute_skater_rates",
    "compute_goalie_rates",
    "team_abbr_from_id",
    "TEAM_ID_TO_ABBR",
]
