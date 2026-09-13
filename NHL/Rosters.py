"""
Roster + rookie projection utilities.

Fetches current NHL rosters (skaters + goalies, with player IDs) from the NHL
API and caches them to rosters.json. Also builds rookie point projections: for
skaters with no recent NHL games, we pull their most recent non-NHL
regular-season line (AHL / junior / college / Europe) from the player landing
endpoint and translate it to an NHL per-game pace using NHLe-style league
factors. The projection fills the "no NHL sample yet" gap for first-year
players during their first ~10 games.

Run via update_rosters.py (scheduled daily).
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from NHL.ApiScrape import _display_name, _position_code, _try_get_json
from NHL.Config import DIVISIONS, NHL_API_BASE
from NHL.StatsFromPBP import load_skater_rates_from_json
from NHL.Utils import normalize_name_key

logger = logging.getLogger(__name__)

ROSTERS_PATH = Path("rosters.json")
ROOKIE_PROJECTIONS_PATH = Path("rookie_projections.json")

ALL_TEAMS: List[str] = [abbr for div in DIVISIONS.values() for abbr in div]

# NHLe-style translation factors: estimated NHL points-per-game ≈
# (league points-per-game) * factor. Approximations, not authoritative.
_NHLE_FACTORS: Dict[str, float] = {
    "NHL": 1.0,
    "AHL": 0.44,
    "KHL": 0.74,
    "SHL": 0.58,
    "LIIGA": 0.54,
    "NL": 0.47,
    "DEL": 0.50,
    "CZECH": 0.51,
    "CZECHIA": 0.51,
    "SLOVAKIA": 0.46,
    "ALLSVENSKAN": 0.45,
    "HOCKEYALLSVENSKAN": 0.45,
    "VHL": 0.46,
    "OHL": 0.27,
    "WHL": 0.29,
    "QMJHL": 0.24,
    "NCAA": 0.41,
    "USHL": 0.23,
    "ECHL": 0.21,
    "BCHL": 0.15,
    "AJHL": 0.15,
    "OJHL": 0.13,
    "USPORTS": 0.18,
    "CIS": 0.18,
    "WHC": 0.55,
    "WC": 0.55,
    "OLYMPICS": 0.55,
}


def _league_factor(league: str) -> float:
    """Return the NHLe factor for a league abbreviation, with a sane fallback."""
    lg = (league or "").strip().upper()
    if lg in _NHLE_FACTORS:
        return _NHLE_FACTORS[lg]
    # Fallback: junior leagues translate lower than pro/European leagues.
    if any(t in lg for t in ("JR", "OHL", "WHL", "QMJHL", "USHL", "NAHL", "BCHL", "AJHL", "MHL")):
        return 0.25
    if "NCAA" in lg or "USPORT" in lg or "CIS" in lg:
        return 0.40
    return 0.45


# Junior / amateur leagues where age matters: a player putting up a given
# per-game line a year (or two) before their draft year is far more impressive
# than a draft-year peer, so these get an age adjustment. Pro leagues (KHL,
# Liiga, SHL, AHL, ...) are deliberately excluded.
_JUNIOR_LEAGUES = {
    "OHL", "WHL", "QMJHL", "USHL", "NCAA", "BCHL", "AJHL", "OJHL", "NAHL",
    "USPORTS", "CIS", "MHL",
}


def _draft_year(birth_date: Optional[str]) -> Optional[int]:
    """NHL draft year for a birth date (YYYY-MM-DD)."""
    if not birth_date:
        return None
    try:
        y, m, d = (int(x) for x in str(birth_date).split("-")[:3])
    except Exception:
        return None
    # A player is draft-eligible the year they turn 18, provided they are 18
    # by Sept 15 of that draft year.
    return y + 18 if (m < 9 or (m == 9 and d <= 15)) else y + 19


def _age_multiplier(birth_date: Optional[str], source_season: Any, league: str) -> float:
    """Age adjustment for junior-league production. Younger = more impressive."""
    lg = (league or "").strip().upper()
    if lg not in _JUNIOR_LEAGUES and "JR" not in lg:
        return 1.0
    draft_year = _draft_year(birth_date)
    season_start = int(str(source_season)[:4]) if source_season else None
    if not draft_year or not season_start:
        return 1.0
    draft_relative = season_start - (draft_year - 1)  # 0 = draft year, -1 = D-1
    if draft_relative <= -2:
        return 1.6
    if draft_relative == -1:
        return 1.3
    return 1.0  # draft year or over-ager: no boost


def fetch_team_roster(abbr: str) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch a team's current roster (skaters + goalies) with player IDs."""
    url = f"{NHL_API_BASE}/roster/{abbr}/current"
    data = _try_get_json(url)
    skaters: List[Dict[str, Any]] = []
    goalies: List[Dict[str, Any]] = []
    if not data:
        return {"skaters": skaters, "goalies": goalies}

    for group, bucket in (("forwards", skaters), ("defensemen", skaters), ("goalies", goalies)):
        for p in data.get(group, []) or []:
            name = _display_name(p)
            if not name:
                continue
            bucket.append({
                "name": name,
                "id": p.get("id"),
                "position": _position_code(p),
                "birthDate": p.get("birthDate"),
            })

    def _dedup(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out = []
        for r in rows:
            k = r["name"].lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
        return out

    return {"skaters": _dedup(skaters), "goalies": _dedup(goalies)}


def fetch_all_rosters(rate_limit: float = 0.35) -> Dict[str, Dict[str, Any]]:
    """Fetch current rosters for all teams."""
    teams: Dict[str, Dict[str, Any]] = {}
    for abbr in ALL_TEAMS:
        try:
            teams[abbr] = fetch_team_roster(abbr)
            logger.info(
                f"Roster {abbr}: {len(teams[abbr]['skaters'])} skaters, "
                f"{len(teams[abbr]['goalies'])} goalies"
            )
        except Exception as e:
            logger.warning(f"Failed to fetch roster for {abbr}: {e}")
            teams[abbr] = {"skaters": [], "goalies": []}
        time.sleep(rate_limit)
    return teams


def project_rookie_rates(player_id: Optional[int], birth_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Estimate an NHL per-game pace for a player from their most recent non-NHL
    regular-season lines. Returns None if no usable non-NHL data.
    """
    if not player_id:
        return None
    landing = _try_get_json(f"{NHL_API_BASE}/player/{player_id}/landing")
    if not landing:
        return None

    season_totals = landing.get("seasonTotals") or []
    reg = [e for e in season_totals if e.get("gameTypeId") == 2]
    # Most recent season first, then largest sample within it.
    reg.sort(key=lambda e: (int(e.get("season") or 0), int(e.get("gamesPlayed") or 0)), reverse=True)

    # Consider the two most recent distinct non-NHL seasons (draft year + the
    # year before). A single most-recent line can be a partial or mis-tagged
    # sample (short schedule, wrong league tag), so the prior season's
    # production is kept as a candidate too.
    window: List[Any] = []
    for e in reg:
        league = (e.get("leagueAbbrev") or "").strip().upper()
        if league in ("", "NHL"):
            continue
        season = e.get("season")
        if season not in window:
            window.append(season)
        if len(window) >= 2:
            break

    candidates = [
        e for e in reg
        if e.get("season") in window
        and (e.get("leagueAbbrev") or "").strip().upper() not in ("", "NHL")
    ]
    # Prefer a reliable sample (>=10 games); fall back to any sample so
    # short-season players still get a projection rather than nothing.
    reliable = [e for e in candidates if int(e.get("gamesPlayed") or 0) >= 10]
    if reliable:
        candidates = reliable

    best: Optional[Dict[str, Any]] = None
    for e in candidates:
        league = (e.get("leagueAbbrev") or "").strip().upper()
        gp = int(e.get("gamesPlayed") or 0)
        if gp <= 0:
            continue
        factor = _league_factor(league) * _age_multiplier(birth_date, e.get("season"), league)
        goals = float(e.get("goals") or 0)
        assists = float(e.get("assists") or 0)
        points = float(e.get("points") or 0)
        points_pg = (points / gp) * factor
        if best is not None and points_pg <= best["points_pg"]:
            continue
        goals_pg = (goals / gp) * factor
        assists_pg = (assists / gp) * factor
        shots = e.get("shots")
        if shots is not None:
            shots_pg = (float(shots) / gp) * factor
        else:
            # League lines often omit shots; back out an estimate from goals
            # using a typical NHL shooting percentage.
            shots_pg = (goals_pg / 0.09) if goals_pg > 0 else 0.0
        best = {
            "points_pg": round(points_pg, 4),
            "goals_pg": round(goals_pg, 4),
            "assists_pg": round(assists_pg, 4),
            "shots_pg": round(shots_pg, 4),
            "source_league": league,
            "source_season": e.get("season"),
            "source_games": gp,
        }
    return best


def _known_nhl_name_keys(season_start_year: int) -> set:
    """Normalized-name keys for players with recent NHL games (current + prior season)."""
    keys = set()
    for year in (season_start_year, season_start_year - 1):
        try:
            rates = load_skater_rates_from_json(year, 2)
        except Exception as e:
            logger.warning(f"Could not load skater rates for {year}: {e}")
            continue
        keys.update(rates.keys())
    return keys


def build_rookie_projections(
    rosters: Dict[str, Dict[str, Any]],
    season_start_year: int,
    rate_limit: float = 0.35,
) -> Dict[str, Dict[str, Any]]:
    """
    Build rookie projections for roster skaters with no recent NHL games.
    Returns a dict keyed by normalized name -> projection record.
    """
    known = _known_nhl_name_keys(season_start_year)
    projections: Dict[str, Dict[str, Any]] = {}

    for abbr, team in rosters.items():
        for skater in team.get("skaters", []):
            name = skater.get("name", "")
            key = normalize_name_key(name)
            if not key or key in known:
                continue  # established NHL player, or no usable name
            try:
                proj = project_rookie_rates(skater.get("id"), birth_date=skater.get("birthDate"))
            except Exception as e:
                logger.warning(f"Projection failed for {name} ({abbr}): {e}")
                proj = None
            if proj:
                proj["name"] = name
                proj["team"] = abbr
                projections[key] = proj
            time.sleep(rate_limit)
    return projections


def save_rosters(rosters: Dict[str, Dict[str, Any]], path: Path = ROSTERS_PATH) -> None:
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "teams": rosters}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_rosters(path: Path = ROSTERS_PATH) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return {}
    return data.get("teams", {})


def save_rookie_projections(
    projections: Dict[str, Dict[str, Any]], path: Path = ROOKIE_PROJECTIONS_PATH
) -> None:
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "players": projections}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_rookie_projections(path: Path = ROOKIE_PROJECTIONS_PATH) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return {}
    return data.get("players", {})


__all__ = [
    "ALL_TEAMS", "fetch_team_roster", "fetch_all_rosters",
    "project_rookie_rates", "build_rookie_projections",
    "save_rosters", "load_rosters",
    "save_rookie_projections", "load_rookie_projections",
]
