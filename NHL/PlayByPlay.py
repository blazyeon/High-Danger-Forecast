"""
NHL API Play-by-Play fetcher with on-disk caching.

Replaces the NST HTML scraper as the primary source of shot-level data.
The NHL API at api-web.nhle.com/v1 is the same source MoneyPuck and NST
build on top of — we hit it directly instead of scraping their renderings.

Verified PBP JSON shape (game 2023020001, 2023-10-10):
- Top-level keys: id, season, gameType, gameDate, plays[], rosterSpots[]
- Each play has: eventId, periodDescriptor.number, timeInPeriod (MM:SS),
  situationCode ("1551" = 5v5, "1541" = 5v4 PP, "1451" = 4v5 PK),
  typeCode (505=goal, 506=shot, 507=missed, 508=blocked, 502=faceoff,
  503=hit, 516=stoppage, 520=period start, 521=period end, 525=penalty)
- Shot details: xCoord (ft from center, ±89 = goal lines), yCoord (ft from
  center ice), zoneCode (O/D/N), shotType (wrist, slap, backhand, snap,
  tip-in, wrap-around, etc.), shootingPlayerId, goalieInNetId,
  eventOwnerTeamId, homeSOG, awaySOG, homeScore, awayScore.
  Note: PBP shot details carry player IDs only — names are looked up
  from the top-level rosterSpots[] array.

Schedule endpoint at /v1/schedule/{YYYY-MM-DD} returns a 7-day window with
game IDs. We walk all days in a season to enumerate game IDs, then fetch
each PBP. Season typically has 1,300+ games → ~90 seconds of fetches.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from NHL.Config import (
    NHL_API_BASE,
    REQUEST_HEADERS,
    DEFAULT_TIMEOUT,
    RATE_LIMIT_SLEEP_SECONDS,
    RATE_LIMIT_JITTER_SECONDS,
)

logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────

# Project root: this file is at NHL/PlayByPlay.py, root is one level up
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PBP_CACHE_DIR = Path(os.environ.get("PBP_CACHE_DIR", str(_PROJECT_ROOT / "pbp_cache")))
RAW_DIR = PBP_CACHE_DIR / "raw"
SHOT_STORE_DIR = PBP_CACHE_DIR / "shots"
SCHEDULE_DIR = PBP_CACHE_DIR / "schedule"

for _d in (RAW_DIR, SHOT_STORE_DIR, SCHEDULE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Rate-limited session ────────────────────────────────────────────────

_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """One shared session with retry + connection pooling."""
    global _session
    if _session is None:
        s = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=10,
            pool_maxsize=20,
        )
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers.update(REQUEST_HEADERS)
        _session = s
    return _session


def _sleep_rate_limit() -> None:
    import random
    jitter = random.uniform(0, RATE_LIMIT_JITTER_SECONDS)
    time.sleep(RATE_LIMIT_SLEEP_SECONDS + jitter)


# ── Game ID discovery (schedule walker) ─────────────────────────────────

def fetch_schedule_window(day: date) -> List[Dict]:
    """
    Fetch a 7-day schedule window from the NHL API.

    The endpoint /v1/schedule/{date} returns games for the 7 days starting
    at that date. We use this to enumerate game IDs for a season.

    Returns a list of game summaries: {id, date, season, gameType, ...}
    Empty list on failure.
    """
    url = f"{NHL_API_BASE}/schedule/{day.isoformat()}"
    cache_path = SCHEDULE_DIR / f"{day.isoformat()}.json"
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass  # fall through to fetch

    try:
        _sleep_rate_limit()
        resp = _get_session().get(url, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Schedule fetch failed for {day}: {e}")
        return []

    out: List[Dict] = []
    for week_day in data.get("gameWeek", []):
        for game in week_day.get("games", []):
            out.append({
                "id": game.get("id"),
                "date": week_day.get("date"),
                "season": game.get("season"),
                "gameType": game.get("gameType"),  # 1=pre, 2=reg, 3=playoff
                "homeTeam": game.get("homeTeam", {}).get("abbrev", ""),
                "awayTeam": game.get("awayTeam", {}).get("abbrev", ""),
                "gameState": game.get("gameState", ""),  # OFF/FUT/LIVE/CRIT/FINAL
            })

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    except Exception as e:
        logger.debug(f"Failed to write schedule cache {cache_path}: {e}")

    return out


def discover_season_games(
    season_start: date,
    season_end: date,
    stype: int = 2,
) -> List[Dict]:
    """
    Walk the schedule to collect all game IDs in [season_start, season_end].

    We step in 7-day windows (the schedule endpoint's natural stride). Stops
    as soon as a window's first day is past season_end. Dedupes by game id.
    """
    seen: Set[int] = set()
    games: List[Dict] = []
    cur = season_start
    while cur <= season_end:
        window = fetch_schedule_window(cur)
        for g in window:
            gid = g.get("id")
            if gid is None or gid in seen:
                continue
            try:
                gdate = date.fromisoformat(g["date"])
            except Exception:
                continue
            if gdate < season_start or gdate > season_end:
                continue
            if stype is not None and g.get("gameType") != stype:
                continue
            if g.get("gameState") not in ("OFF", "FINAL"):
                continue  # skip scheduled / live
            seen.add(gid)
            games.append(g)
        cur += timedelta(days=7)
    games.sort(key=lambda g: (g.get("date", ""), g.get("id", 0)))
    return games


# ── PBP fetch with on-disk cache ────────────────────────────────────────

def fetch_game_pbp(game_id: int, use_cache: bool = True) -> Optional[Dict]:
    """
    Fetch play-by-play JSON for a single game.

    Cached on disk at pbp_cache/raw/{game_id}.json. Set use_cache=False
    to force re-fetch.
    """
    cache_path = RAW_DIR / f"{game_id}.json"
    if use_cache and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Cache read failed for {game_id}: {e}, re-fetching")

    url = f"{NHL_API_BASE}/gamecenter/{game_id}/play-by-play"
    try:
        _sleep_rate_limit()
        resp = _get_session().get(url, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"PBP fetch failed for game {game_id}: {e}")
        return None

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.debug(f"Cache write failed for {game_id}: {e}")

    return data


# ── PBP → flat shot DataFrame ───────────────────────────────────────────

# PBP typeCode → event_type
_SHOT_TYPES = {505: "goal", 506: "shot", 507: "missed", 508: "blocked"}


def _parse_situation(sit_code: str) -> Tuple[int, int]:
    """
    Parse 4-digit situationCode (away goalie + away skaters + home skaters + home goalie)
    Returns (home_skaters, away_skaters).
    """
    if not sit_code or len(sit_code) != 4:
        return 5, 5
    try:
        away_g, away_s, home_s, home_g = (int(c) for c in sit_code)
        return home_s, away_s
    except Exception:
        return 5, 5


def count_pp_opportunities(game_id: Any, use_cache: bool = True) -> Tuple[int, int]:
    """
    Count actual power-play opportunities for each team from the play-by-play.

    The NHL boxscore endpoint no longer exposes a summary with teamGameStats,
    so we derive PP opportunities by tracking situationCode transitions.
    A new opportunity is counted when a team gains a skater advantage over
    the opponent while both goalies are in net.

    Returns (home_pp_opportunities, away_pp_opportunities).
    """
    pbp = fetch_game_pbp(game_id, use_cache=use_cache)
    if not pbp:
        return 0, 0

    home_pp = 0
    away_pp = 0
    home_pp_active = False
    away_pp_active = False

    for play in pbp.get("plays", []) or []:
        code = play.get("situationCode", "")
        if not code or len(code) != 4:
            continue
        try:
            away_goalie = int(code[0])
            away_skaters = int(code[1])
            home_skaters = int(code[2])
            home_goalie = int(code[3])
        except Exception:
            continue

        # Empty-net / pulled goalie situations reset standard PP tracking.
        if home_goalie == 0 or away_goalie == 0:
            home_pp_active = False
            away_pp_active = False
            continue

        home_advantage = home_skaters > away_skaters
        away_advantage = away_skaters > home_skaters

        if home_advantage and not home_pp_active:
            home_pp += 1
            home_pp_active = True
        if away_advantage and not away_pp_active:
            away_pp += 1
            away_pp_active = True
        if not home_advantage:
            home_pp_active = False
        if not away_advantage:
            away_pp_active = False

    return home_pp, away_pp


def _time_to_seconds(t: str, period: int) -> Optional[float]:
    """Convert MM:SS + period to seconds elapsed in the game (1-indexed period)."""
    if not t:
        return None
    try:
        mm, ss = t.split(":")
        return (period - 1) * 1200.0 + int(mm) * 60 + int(ss)
    except Exception:
        return None


def _roster_lookup(pbp: Dict) -> Dict[int, Dict[str, str]]:
    """
    Build a {player_id: {first, last, team_id, position}} lookup from
    the PBP's rosterSpots array. Used to enrich shot events with names
    (PBP details only carries player IDs, not names).
    """
    out: Dict[int, Dict[str, str]] = {}
    for spot in pbp.get("rosterSpots", []) or []:
        pid = spot.get("playerId")
        if pid is None:
            continue
        first = (spot.get("firstName") or {}).get("default", "")
        last = (spot.get("lastName") or {}).get("default", "")
        out[int(pid)] = {
            "first": str(first).strip(),
            "last": str(last).strip(),
            "team_id": spot.get("teamId"),
            "position": spot.get("positionCode", ""),
        }
    return out


def parse_pbp_events(pbp: Dict, game_id: Optional[int] = None) -> pd.DataFrame:
    """
    Flatten a PBP JSON into a shots DataFrame.

    Returns a DataFrame with one row per shot/goal/missed/blocked event.
    Empty DataFrame if no shot events. The columns are designed to be
    compatible with MoneyPuck's shot schema where possible.

    Player IDs come from different fields depending on event type:
      - shots (506): shootingPlayerId
      - goals (505): scoringPlayerId, assist1PlayerId, assist2PlayerId
      - missed (507) / blocked (508): shootingPlayerId
    We always populate `shooter_id` (the player who took the shot — for
    goals this is the scorer) and additionally emit `assist1_id` /
    `assist2_id` for goals. Names come from the top-level rosterSpots
    array; details never carry names.
    """
    rows: List[Dict] = []
    game_id = game_id or pbp.get("id")
    plays = pbp.get("plays", []) or []
    roster = _roster_lookup(pbp)

    def _name_for(pid) -> str:
        if pid is None:
            return ""
        try:
            entry = roster.get(int(pid), {})
        except (TypeError, ValueError):
            return ""
        if not entry:
            return ""
        return f"{entry.get('first', '')} {entry.get('last', '')}".strip()

    for play in plays:
        tcode = play.get("typeCode")
        if tcode not in _SHOT_TYPES:
            continue
        details = play.get("details") or {}
        x = details.get("xCoord")
        y = details.get("yCoord")
        if x is None or y is None:
            continue
        period = (play.get("periodDescriptor") or {}).get("number", 0)
        tip = play.get("timeInPeriod", "")
        sit = play.get("situationCode", "1551")
        home_s, away_s = _parse_situation(sit)
        # Shooter ID is the player who took the shot. For goals it's the
        # scoringPlayerId; for everything else it's shootingPlayerId.
        if tcode == 505:
            shooter_id = details.get("scoringPlayerId")
            assist1_id = details.get("assist1PlayerId")
            assist2_id = details.get("assist2PlayerId")
        else:
            shooter_id = details.get("shootingPlayerId")
            assist1_id = None
            assist2_id = None
        goalie_id = details.get("goalieInNetId")
        shooter_name = _name_for(shooter_id)
        goalie_name = _name_for(goalie_id)
        assist1_name = _name_for(assist1_id)
        assist2_name = _name_for(assist2_id)
        rows.append({
            "game_id": game_id,
            "event_id": play.get("eventId"),
            "period": period,
            "time_in_period": tip,
            "time_seconds": _time_to_seconds(tip, period),
            "situation_code": sit,
            "home_skaters": home_s,
            "away_skaters": away_s,
            "event_type": _SHOT_TYPES[tcode],
            "x": float(x),
            "y": float(y),
            "shot_type": (details.get("shotType") or "").lower().strip(),
            "shooter_id": shooter_id,
            "shooter_name": shooter_name,
            "goalie_id": goalie_id,
            "goalie_name": goalie_name,
            "assist1_id": assist1_id,
            "assist1_name": assist1_name,
            "assist2_id": assist2_id,
            "assist2_name": assist2_name,
            "team_id": details.get("eventOwnerTeamId"),
            "zone": details.get("zoneCode"),
            "is_goal": int(tcode == 505),
            "is_shot": int(tcode in (505, 506)),
            "home_score": details.get("homeScore"),
            "away_score": details.get("awayScore"),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ── Season-scale PBP ingestion ──────────────────────────────────────────

def iter_season_pbp(
    season_year: int,
    stype: int = 2,
    force_refresh: bool = False,
) -> Iterator[Tuple[Dict, pd.DataFrame]]:
    """
    Yield (game_meta, shots_df) for every game in the given season.

    season_year is the *start* year of the season (e.g., 2024 for 2024-25).
    Walks Oct→next-Apr to find regular season (stype=2) games. For
    playoffs, pass stype=3 and adjust dates externally.
    """
    season_start = date(season_year, 10, 1)
    season_end = date(season_year + 1, 6, 30)
    games = discover_season_games(season_start, season_end, stype=stype)
    logger.info(f"Discovered {len(games)} games for {season_year}-{season_year+1} stype={stype}")
    for g in games:
        gid = g["id"]
        pbp = fetch_game_pbp(gid, use_cache=not force_refresh)
        if pbp is None:
            continue
        shots = parse_pbp_events(pbp, game_id=gid)
        yield g, shots


def build_shot_store(
    season_year: int,
    stype: int = 2,
    out_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> Path:
    """
    Fetch all PBP for a season, parse to shots, write a single parquet file.

    Idempotent: skip games already in the on-disk cache unless force_refresh
    is True. The output parquet is at pbp_cache/shots/shots_{season}.parquet
    (or out_path if given).
    """
    if out_path is None:
        out_path = SHOT_STORE_DIR / f"shots_{season_year}_{stype}.parquet"

    all_shots: List[pd.DataFrame] = []
    meta_rows: List[Dict] = []
    n_fetched = 0
    n_cached = 0
    for g, shots in iter_season_pbp(season_year, stype, force_refresh=force_refresh):
        meta_rows.append(g)
        if not shots.empty:
            all_shots.append(shots)
        cache_path = RAW_DIR / f"{g['id']}.json"
        if cache_path.exists() and cache_path.stat().st_mtime < (time.time() - 3600):
            n_cached += 1
        else:
            n_fetched += 1

    if not all_shots:
        logger.warning(f"No shots found for {season_year} stype={stype}")
        return out_path

    df = pd.concat(all_shots, ignore_index=True)
    # Backfill is_home / team_abbr from meta if not in shots
    if meta_rows and "team_abbr" not in df.columns:
        meta_df = pd.DataFrame(meta_rows)
        meta_df = meta_df.rename(columns={"id": "game_id"})
        meta_df = meta_df[["game_id", "homeTeam", "awayTeam"]]
        df = df.merge(meta_df, on="game_id", how="left")
    # Pre-compute high-danger flag once at build time so every stats call
    # doesn't re-run the trigonometry on hundreds of thousands of rows.
    if not df.empty and "hd" not in df.columns and {"x", "y"}.issubset(df.columns):
        from NHL.StatsFromPBP import _vectorized_high_danger
        df["hd"] = _vectorized_high_danger(df["x"], df["y"]).astype(int)
    df.to_parquet(out_path, index=False)
    logger.info(
        f"Wrote {len(df):,} shots from {len(meta_rows)} games "
        f"(fetched {n_fetched}, cached {n_cached}) → {out_path}"
    )
    return out_path


def load_shot_store(season_year: int, stype: int = 2) -> pd.DataFrame:
    """
    Load a previously built shot store, or build it if missing.

    Returns an empty DataFrame if nothing exists and the network is down.
    """
    path = SHOT_STORE_DIR / f"shots_{season_year}_{stype}.parquet"
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}, rebuilding")
    try:
        build_shot_store(season_year, stype, out_path=path)
        if path.exists():
            return pd.read_parquet(path)
    except Exception as e:
        logger.error(f"Failed to build shot store for {season_year}: {e}")
    return pd.DataFrame()


# ── Schedule → game_id ↔ date mapping ───────────────────────────────────

def game_date_map(season_year: int, stype: int = 2) -> Dict[int, str]:
    """
    Return {game_id: 'YYYY-MM-DD'} for a season, useful for date-window
    filtering in the stats aggregator.
    """
    season_start = date(season_year, 10, 1)
    season_end = date(season_year + 1, 6, 30)
    games = discover_season_games(season_start, season_end, stype=stype)
    return {g["id"]: g["date"] for g in games if g.get("id") is not None}


__all__ = [
    "fetch_game_pbp",
    "fetch_schedule_window",
    "discover_season_games",
    "parse_pbp_events",
    "iter_season_pbp",
    "build_shot_store",
    "load_shot_store",
    "game_date_map",
    "PBP_CACHE_DIR",
    "RAW_DIR",
    "SHOT_STORE_DIR",
]
