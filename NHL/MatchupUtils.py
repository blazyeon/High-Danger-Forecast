"""
Pure utility functions for the matchup predictor and game frontend.
Extracted from GameFrontend.py and MatchupPredictor.py to remove Streamlit dependency.
No Streamlit imports — these are all pure Python computation functions.
"""
from __future__ import annotations
import json
import math
import logging
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

from NHL.Config import DIVISIONS, TEAM_ABBR_MAPPING, NST_ABBR_TO_FULL
from NHL.Utils import normalize_name_key

logger = logging.getLogger(__name__)


# ── Team picker helpers ────────────────────────────────────────────────

def build_team_options() -> Tuple[List[str], Dict[str, str]]:
    """
    Build team options for a team picker, grouped by division.

    Returns:
        (sorted_options, label_to_abbr) — display labels and their abbreviation map
    """
    options = []
    label_to_abbr = {}

    for division, teams in DIVISIONS.items():
        for abbr in teams:
            display_abbr = TEAM_ABBR_MAPPING.get(abbr, abbr)
            full_name = NST_ABBR_TO_FULL.get(abbr, NST_ABBR_TO_FULL.get(display_abbr, display_abbr))
            if full_name == display_abbr:
                full_name = abbr
            label = f"{division} — {display_abbr} ({full_name})"
            options.append(label)
            label_to_abbr[label] = display_abbr

    options.sort()
    return options, label_to_abbr


def build_teams_api_data() -> Dict[str, Any]:
    """
    Build team data suitable for JSON API responses.
    Returns dict with divisions, each containing team abbreviations, display names, and full names.
    Always provides a human-readable full name for every team.
    """
    divisions = {}
    for division, teams in DIVISIONS.items():
        division_teams = []
        for abbr in teams:
            display_abbr = TEAM_ABBR_MAPPING.get(abbr, abbr)
            # Try abbreviation first, then display abbreviation, then fall back to "City Name"
            full_name = NST_ABBR_TO_FULL.get(abbr, NST_ABBR_TO_FULL.get(display_abbr, ""))
            if not full_name or full_name == display_abbr:
                full_name = abbr
            division_teams.append({
                "abbr": display_abbr,
                "full_name": full_name,
            })
        divisions[division] = division_teams
    return divisions


# ── Game type detection ────────────────────────────────────────────────

def detect_game_type(game: Dict[str, Any]) -> int:
    """
    Infer game type (1=preseason, 2=regular, 3=postseason) from game ID encoding or metadata.
    """
    game_id = str(game.get("id", ""))
    if len(game_id) >= 6:
        try:
            type_code = game_id[4:6]
            if type_code == "01":
                return 1
            elif type_code == "02":
                return 2
            elif type_code == "03":
                return 3
        except Exception:
            pass
    game_type = game.get("gameType")
    if isinstance(game_type, int) and 1 <= game_type <= 3:
        return game_type
    season_stage = game.get("gameScheduleState") or game.get("seasonStage") or ""
    if isinstance(season_stage, str):
        s = season_stage.upper()
        if "PRE" in s:
            return 1
        if "POST" in s or "PLAYOFF" in s:
            return 3
    return 2


# ── Skater stats ──────────────────────────────────────────────────────

def build_skater_stats_df(
    df_alloc: pd.DataFrame,
    df_shots: pd.DataFrame,
    assists_map: Dict[str, int]
) -> pd.DataFrame:
    """Merge goal allocation, shot simulation, and assist data into a single DataFrame."""
    if df_alloc is None or df_alloc.empty:
        return pd.DataFrame(columns=["Name", "Position", "Goals", "Assists", "Points", "Shots"])
    rows = []
    for _, row in df_alloc.iterrows():
        name = row["Name"]
        position = row.get("Position", "NA")
        goals = int(round(row["Assigned Goals"]))
        shots = 0
        if df_shots is not None and not df_shots.empty and name in df_shots["Name"].values:
            try:
                shots = int(round(df_shots.loc[df_shots["Name"] == name, "SimShots"].values[0]))
            except Exception:
                shots = 0
        assists = assists_map.get(name, 0)
        points = goals + assists
        rows.append({
            "Name": name,
            "Position": position,
            "Goals": goals,
            "Assists": assists,
            "Points": points,
            "Shots": shots,
        })
    return pd.DataFrame(rows, columns=["Name", "Position", "Goals", "Assists", "Points", "Shots"])


# ── Goalie override helpers ───────────────────────────────────────────

def parse_goalie_override_json(b: bytes) -> Dict[str, str]:
    """Parse uploaded JSON bytes into a team→goalie name mapping dict."""
    mapping: Dict[str, str] = {}
    try:
        data = json.loads(b.decode("utf-8"))
        if not isinstance(data, list):
            return mapping
        for item in data:
            if not isinstance(item, dict):
                continue
            team = str(item.get("team", "")).strip().upper()
            goalie = str(item.get("goalie", "")).strip()
            if team and goalie:
                mapping[team] = goalie
    except Exception as e:
        logger.warning(f"Failed to parse goalie override JSON: {e}")
    return mapping


def merge_goalie_options(roster_options: List[str], mapped_name: Optional[str]) -> List[str]:
    """Merge API roster goalie names with uploaded JSON override, deduplicating by normalized key."""
    uniq: Dict[str, str] = {}
    for opt in roster_options:
        key = normalize_name_key(opt) if opt else ""
        if key and key not in uniq:
            uniq[key] = opt
    if mapped_name:
        key = normalize_name_key(mapped_name)
        if key and key not in uniq:
            uniq[key] = mapped_name
    return list(uniq.values())


def default_index_for_options(options: List[str], mapped_name: Optional[str]) -> int:
    """Return the default index for a goalie selector, offset by +1 for the 'Auto (model)' option."""
    if not mapped_name:
        return 0
    target_key = normalize_name_key(mapped_name)
    for i, opt in enumerate(options):
        try:
            if normalize_name_key(opt) == target_key:
                return i + 1  # +1 for "Auto (model)"
        except Exception:
            continue
    return 0


# ── Score extraction ──────────────────────────────────────────────────

def extract_scores_from_boxscore(box: Dict) -> Optional[Dict[str, int]]:
    """
    Robustly extract final home/away scores from an api-web boxscore or similar payload.
    Returns {'home': int, 'away': int} or None if not available.
    """
    if not box or not isinstance(box, dict):
        return None
    for hk, ak in [("homeTeam", "awayTeam"), ("home", "away")]:
        h = box.get(hk, {}) or {}
        a = box.get(ak, {}) or {}
        hscore = h.get("score") if "score" in h else h.get("goals")
        ascore = a.get("score") if "score" in a else a.get("goals")
        if hscore is not None and ascore is not None:
            try:
                return {"home": int(hscore), "away": int(ascore)}
            except Exception:
                continue
    try:
        live = box.get("liveData") or {}
        linescore = live.get("linescore") or {}
        teams = linescore.get("teams") or {}
        h = teams.get("home", {}) or {}
        a = teams.get("away", {}) or {}
        hscore = h.get("goals")
        ascore = a.get("goals")
        if hscore is not None and ascore is not None:
            return {"home": int(hscore), "away": int(ascore)}
    except Exception:
        pass
    return None


# ── Poisson / score distribution ──────────────────────────────────────

def poisson_pmf_log(k: int, lam: float) -> float:
    """Log Poisson PMF for numerical stability: log P = k*log(lam) - lam - lgamma(k+1)."""
    if lam <= 0:
        return 0.0 if k == 0 else float("-inf")
    return k * math.log(lam) - lam - math.lgamma(k + 1)


def score_combo_distribution(exp_home: float, exp_away: float, total: int) -> List[Tuple[int, int, float]]:
    """
    Return list of tuples (away_goals, home_goals, prob) for all splits summing to `total`.
    Prob computed via independent Poisson approximation and normalized across splits.
    """
    log_probs = []
    for away in range(0, total + 1):
        home = total - away
        logp_away = poisson_pmf_log(away, exp_away)
        logp_home = poisson_pmf_log(home, exp_home)
        logp = logp_away + logp_home
        log_probs.append(logp)
    max_log = max(log_probs) if log_probs else float("-inf")
    exp_probs = [math.exp(lp - max_log) for lp in log_probs]
    s = sum(exp_probs)
    if s <= 0:
        n = total + 1
        return [(i, total - i, 1.0 / n) for i in range(0, total + 1)]
    normalized = [p / s for p in exp_probs]
    result = []
    for away, p in enumerate(normalized):
        home = total - away
        result.append((away, home, float(p)))
    result.sort(key=lambda x: x[2], reverse=True)
    return result


def choose_non_tie_split(total: int, exp_home: float, exp_away: float, prefer_home_win: bool) -> Tuple[int, int]:
    """
    Given a total and expected goals, choose a split (away, home) that sums to total
    and yields a clear winner. Uses Poisson-proportional most likely split matching
    the preferred winner.
    """
    combos = score_combo_distribution(exp_home, exp_away, total)

    if not combos:
        if prefer_home_win:
            if total == 0:
                return (0, 1)
            home = max(1, total // 2 + 1)
            away = total - home
            return away, home
        else:
            if total == 0:
                return (1, 0)
            away = max(1, total // 2 + 1)
            home = total - away
            return away, home

    for away, home, p in combos:
        if prefer_home_win and home > away:
            return away, home
        if not prefer_home_win and away > home:
            return away, home

    for away, home, p in combos:
        if home != away:
            return away, home

    # All combos tie — force based on preference
    prefer_home = prefer_home_win

    if exp_home + exp_away <= 0:
        if prefer_home:
            home = (total // 2) + (1 if total % 2 == 0 else 1)
            away = total - home
            return away, home
        else:
            away = (total // 2) + (1 if total % 2 == 0 else 1)
            home = total - away
            return away, home

    r = exp_home / (exp_home + exp_away)
    home_est = int(round(r * total))
    home = min(total, max(0, home_est))
    away = total - home

    if prefer_home and home <= away:
        if home < total:
            home += 1
            away = total - home
        else:
            away = max(0, away - 1)
            home = total - away
    if not prefer_home and away <= home:
        if away < total:
            away += 1
            home = total - away
        else:
            home = max(0, home - 1)
            away = total - home

    return away, home


__all__ = [
    'build_team_options',
    'build_teams_api_data',
    'detect_game_type',
    'build_skater_stats_df',
    'parse_goalie_override_json',
    'merge_goalie_options',
    'default_index_for_options',
    'extract_scores_from_boxscore',
    'poisson_pmf_log',
    'score_combo_distribution',
    'choose_non_tie_split',
]