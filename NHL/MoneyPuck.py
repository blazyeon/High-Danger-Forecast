"""
MoneyPuck CSV/zip loader.

MoneyPuck publishes pre-computed xG, GAR, WAR, and shot-level data as direct
file downloads at peter-tanner.com/moneypuck — no scraping, no rate limiting,
just HTTP GET on static files. Used as:

1. A historical backfill source (shots_2007-2024.zip, 1.96M shots)
2. A validation cross-check for our xG model (their xGoal column vs ours)
3. A source for season-level aggregates when we don't want to recompute

URL patterns confirmed via moneypuck.com/data.htm (and peter-tanner.com mirror):
- https://peter-tanner.com/moneypuck/downloads/shots_{YYYY}.zip
- https://peter-tanner.com/moneypuck/downloads/shots_2007-2024.zip
- https://peter-tanner.com/moneypuck/playerData/seasonSummary/{YYYY}/regular/{skaters,goalies,teams,lines}.csv
"""
from __future__ import annotations

import io
import logging
import os
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from NHL.Config import REQUEST_HEADERS, DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)

# MoneyPuck serves files from peter-tanner.com (mirror of moneypuck.com/data.htm)
MP_BASE = "https://peter-tanner.com/moneypuck"
MP_BASE_ALT = "https://moneypuck.com/moneypuck"  # for the all_teams.csv file

# On-disk cache
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MP_CACHE_DIR = Path(os.environ.get("MP_CACHE_DIR", str(_PROJECT_ROOT / "pbp_cache" / "mp")))
MP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Shot CSV columns we care about (full list is 60+)
# Normalize names so MP shots can be compared to our PBP shots.
# MoneyPuck uses 'xCord'/'yCord' (no 'd'); NHL API uses 'xCoord'/'yCoord'.
MP_SHOT_COLUMNS_TO_KEEP = [
    "shotID", "id", "game_id", "season", "isPlayoffGame",
    "homeTeamCode", "awayTeamCode", "teamCode", "isHomeTeam",
    "period", "time", "event", "goal", "shotWasOnGoal",
    "location", "xCord", "yCord", "xCordAdjusted", "yCordAdjusted",
    "shotAngle", "shotAngleAdjusted", "shotDistance", "shotType",
    "shotOnEmptyNet", "shotRebound", "shotRush",
    "homeSkatersOnIce", "awaySkatersOnIce",
    "homeEmptyNet", "awayEmptyNet",
    "shooterPlayerId", "shooterName",
    "goalieIdForShot", "goalieNameForShot",
    "xGoal",  # MoneyPuck's xG
]

# Mapping from MP columns to our normalized names (match PBP-derived names)
MP_TO_NORMAL = {
    "id": "event_id",
    "period": "period",
    "time": "time_seconds",
    "event": "event_type",  # SHOT/GOAL/MISS
    "goal": "is_goal",
    "xCord": "x",
    "yCord": "y",
    "xCordAdjusted": "x_adjusted",
    "yCordAdjusted": "y_adjusted",
    "shotType": "shot_type",
    "shotDistance": "distance",
    "shotAngle": "angle",
    "shotRebound": "is_rebound",
    "shotRush": "is_rush",
    "shotOnEmptyNet": "is_empty_net",
    "homeSkatersOnIce": "home_skaters",
    "awaySkatersOnIce": "away_skaters",
    "shooterName": "shooter_name",
    "goalieNameForShot": "goalie_name",
    "teamCode": "team_abbr_mp",
    "xGoal": "xgoal_mp",
}


# ── Session ─────────────────────────────────────────────────────────────

_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        retry = Retry(
            total=3, backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20))
        s.headers.update(REQUEST_HEADERS)
        _session = s
    return _session


# ── File downloads ──────────────────────────────────────────────────────

def _download(url: str, out_path: Path, force: bool = False) -> bool:
    """Download to out_path, skip if exists and not forcing."""
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        return True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        time.sleep(0.5)  # be polite
        resp = _get_session().get(url, timeout=120, stream=True)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
        logger.info(f"Downloaded {url} → {out_path} ({out_path.stat().st_size:,} bytes)")
        return True
    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        return False


def download_shots_zip(years: List[int], force: bool = False) -> List[Path]:
    """
    Download per-season shot ZIPs from MoneyPuck. Returns list of local
    .csv paths (extracted from the zips).
    """
    csvs: List[Path] = []
    for year in years:
        zip_path = MP_CACHE_DIR / f"shots_{year}.zip"
        if not _download(f"{MP_BASE}/downloads/shots_{year}.zip", zip_path, force=force):
            continue
        # Extract the CSV inside (typically a single shots_YYYY.csv)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
                if not names:
                    continue
                csv_name = next((n for n in names if n.endswith(".csv")), names[0])
                csv_path = MP_CACHE_DIR / f"shots_{year}.csv"
                if not csv_path.exists() or force:
                    with zf.open(csv_name) as src, open(csv_path, "wb") as dst:
                        dst.write(src.read())
                csvs.append(csv_path)
        except Exception as e:
            logger.error(f"Failed to extract {zip_path}: {e}")
    return csvs


def download_historical_shots_zip(force: bool = False) -> Optional[Path]:
    """Download the big multi-season bundle (2007-2024, 1.96M shots)."""
    zip_path = MP_CACHE_DIR / "shots_2007-2024.zip"
    if not _download(f"{MP_BASE}/downloads/shots_2007-2024.zip", zip_path, force=force):
        return None
    csv_path = MP_CACHE_DIR / "shots_2007-2024.csv"
    if not csv_path.exists() or force:
        try:
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
                csv_name = next((n for n in names if n.endswith(".csv")), names[0])
                with zf.open(csv_name) as src, open(csv_path, "wb") as dst:
                    dst.write(src.read())
        except Exception as e:
            logger.error(f"Failed to extract {zip_path}: {e}")
            return None
    return csv_path


def download_season_summary(year: int, kind: str, force: bool = False) -> Optional[Path]:
    """
    Download a season summary CSV. kind in {skaters, goalies, teams, lines}.
    """
    url = f"{MP_BASE}/playerData/seasonSummary/{year}/regular/{kind}.csv"
    out = MP_CACHE_DIR / f"mp_{kind}_{year}.csv"
    return out if _download(url, out, force=force) else None


# ── Parsing ─────────────────────────────────────────────────────────────

def parse_mp_shots(csv_path: Path, sample: bool = False) -> pd.DataFrame:
    """
    Load a MoneyPuck shot CSV, keep the columns we care about, and normalize
    names so it can be compared to our PBP-derived shots.

    If sample=True, reads the first 50k rows only (for quick testing).
    """
    try:
        nrows = 50_000 if sample else None
        df = pd.read_csv(csv_path, low_memory=False, nrows=nrows)
    except Exception as e:
        logger.error(f"Failed to read {csv_path}: {e}")
        return pd.DataFrame()

    keep = [c for c in MP_SHOT_COLUMNS_TO_KEEP if c in df.columns]
    df = df[keep].copy()
    df = df.rename(columns=MP_TO_NORMAL)

    # Normalize event_type to our PBP names: SHOT→shot, GOAL→goal, MISS→missed
    event_map = {"SHOT": "shot", "GOAL": "goal", "MISS": "missed"}
    if "event_type" in df.columns:
        df["event_type"] = df["event_type"].map(event_map).fillna(df["event_type"])

    # MoneyPuck season: 2009 = 2009-2010 → we store the start year
    if "season" in df.columns:
        df["season_start"] = df["season"].astype(int)

    # Drop rows with no coordinates
    if "x" in df.columns and "y" in df.columns:
        df = df.dropna(subset=["x", "y"])

    return df


def load_all_mp_shots(force: bool = False) -> pd.DataFrame:
    """
    Load the full historical MP shot set, downloading if needed.

    Strategy: download the big multi-season zip once, then load it. Faster
    than per-year downloads for the full backfill.
    """
    hist_path = MP_CACHE_DIR / "shots_2007-2024.csv"
    if not hist_path.exists():
        download_historical_shots_zip(force=force)
    if not hist_path.exists():
        logger.warning("No historical MP shot file available")
        return pd.DataFrame()

    return parse_mp_shots(hist_path)


__all__ = [
    "MP_BASE",
    "MP_CACHE_DIR",
    "download_shots_zip",
    "download_historical_shots_zip",
    "download_season_summary",
    "parse_mp_shots",
    "load_all_mp_shots",
]
