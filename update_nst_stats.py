#!/usr/bin/env python3
"""
NST Advanced Stats Scraper — exports JSON for the frontend.

Usage:
    python update_nst_stats.py [--season 20242025] [--sit s5v5]

Outputs (to static/data/):
    nst_team_stats.json
    nst_skater_stats.json
    nst_goalie_stats.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.naturalstattrick.com/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

# ── helpers ─────────────────────────────────────────────────────────────


def get_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3, backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.headers.update(HEADERS)
    return s


def fetch_table(url: str) -> Optional[pd.DataFrame]:
    """Fetch an HTML page and return the largest table as a DataFrame."""
    try:
        time.sleep(random.uniform(1.5, 3.5))  # human-like delay
        session = get_session()
        resp = session.get(url, timeout=40)
        resp.raise_for_status()
        html = StringIO(resp.text)
        # Try available parsers without hard-depending on lxml
        df = None
        for flavor in ("lxml", "html5lib", "bs4", None):
            try:
                tables = pd.read_html(html, flavor=flavor)
                if tables:
                    df = max(tables, key=lambda t: len(t) * len(t.columns))
                    break
            except Exception:
                html.seek(0)
        if df is None:
            return None
        df.columns = [str(c).strip() for c in df.columns]
        # drop empty rows/cols
        df = df.dropna(how="all").dropna(axis=1, how="all")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None


def _to_num(val: Any) -> Any:
    if pd.isna(val):
        return None
    s = str(val).replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return val


def _norm_key(name: str) -> str:
    """Normalise a column name to a snake-case key."""
    # strip non-alphanumeric, collapse spaces
    name = re.sub(r"[^A-Za-z0-9\s\%]", "", str(name))
    name = re.sub(r"\s+", " ", name).strip()
    u = name.upper().replace(" ", "").replace("%", "PCT")
    # common mappings
    MAP = {
        "TEAM": "team", "TM": "team",
        "PLAYER": "player", "NAME": "player", "SKATER": "player", "GOALIE": "player",
        "GP": "gp", "GAMESPLAYED": "gp",
        "G": "g", "GOALS": "g",
        "A": "a", "ASSISTS": "a",
        "P": "pts", "PTS": "pts", "POINTS": "pts",
        "CF": "cf", "CORSIFOR": "cf",
        "CA": "ca", "CORSIAGAINST": "ca",
        "CFPCT": "cf_pct", "CORSIFORPERCENTAGE": "cf_pct",
        "FF": "ff", "FENWICKFOR": "ff",
        "FA": "fa", "FENWICKAGAINST": "fa",
        "FFPCT": "ff_pct", "FENWICKFORPERCENTAGE": "ff_pct",
        "SF": "sf", "SHOTSFOR": "sf",
        "SA": "sa", "SHOTSAGAINST": "sa",
        "SFPCT": "sf_pct", "SHOTSFORPERCENTAGE": "sf_pct",
        "GF": "gf", "GOALSFOR": "gf",
        "GA": "ga", "GOALSAGAINST": "ga",
        "GFPCT": "gf_pct", "GOALSFORPERCENTAGE": "gf_pct",
        "XGF": "xgf", "EXPECTEDGOALSFOR": "xgf",
        "XGA": "xga", "EXPECTEDGOALSAGAINST": "xga",
        "XGFPCT": "xgf_pct", "EXPECTEDGOALSFORPERCENTAGE": "xgf_pct",
        "SCF": "scf", "SCORINGCHANCESFOR": "scf",
        "SCA": "sca", "SCORINGCHANCESAGAINST": "sca",
        "SCFPCT": "scf_pct", "SCORINGCHANCESFORPERCENTAGE": "scf_pct",
        "HDCF": "hdcf", "HIGHDANGERCORSIFOR": "hdcf",
        "HDCA": "hdca", "HIGHDANGERCORSIAGAINST": "hdca",
        "HDCFPC": "hdcf_pct", "HIGHDANGERCORSIFORPERCENTAGE": "hdcf_pct", "HDCF%": "hdcf_pct",
        "HDSF": "hdsf", "HIGHDANGERSHOTSFOR": "hdsf",
        "HDSA": "hdsa", "HIGHDANGERSHOTSAGAINST": "hdsa",
        "HDSFPCT": "hdsf_pct", "HIGHDANGERSHOTSFORPERCENTAGE": "hdsf_pct",
        "HDGF": "hdgf", "HIGHDANGERGOALSFOR": "hdgf",
        "HDGA": "hdga", "HIGHDANGERGOALSAGAINST": "hdga",
        "HDGFPCT": "hdgf_pct", "HIGHDANGERGOALSFORPERCENTAGE": "hdgf_pct",
        "SHPCT": "sh_pct", "SHOOTINGPERCENTAGE": "sh_pct", "SH%": "sh_pct",
        "SVPCT": "sv_pct", "SAVEPERCENTAGE": "sv_pct", "SV%": "sv_pct",
        "PDO": "pdo",
        "TOI": "toi", "TIMEONICE": "toi",
        "ATOI": "atoi", "AVERAGETIMEONICE": "atoi",
        "GS": "gs", "GAMESSTARTED": "gs",
        "W": "w", "WINS": "w",
        "L": "l", "LOSSES": "l",
        "OTL": "otl", "OTLOSSES": "otl",
        "GAA": "gaa", "GOALSAVERAGEAGAINST": "gaa",
        "GSAX": "gsax", "GOALSSAVEDABOVEEXPECTED": "gsax",
        "XGA": "xga",
        "MDGA": "mdga", "MEDIUMDANGERGOALSAGAINST": "mdga",
        "MDSA": "mdsa", "MEDIUMDANGERSHOTSAGAINST": "mdsa",
        "MDSVPCT": "mdsv_pct", "MDSV%": "mdsv_pct",
        "LDGA": "ldga", "LOWDANGERGOALSAGAINST": "ldga",
        "LDSA": "ldsa", "LOWDANGERSHOTSAGAINST": "ldsa",
        "LDSVPCT": "ldsv_pct", "LDSV%": "ldsv_pct",
    }
    return MAP.get(u, re.sub(r"[^a-z0-9]", "_", name.lower()).strip("_"))


def clean_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Rename columns to normalised keys and coerce numbers."""
    rename = {c: _norm_key(c) for c in df.columns}
    df = df.rename(columns=rename)
    records = df.to_dict(orient="records")
    for r in records:
        for k, v in r.items():
            r[k] = _to_num(v)
    return records


# ── URL builders ──────────────────────────────────────────────────────


def build_team_url(season: str, situation: str = "s5v5") -> str:
    base = "https://www.naturalstattrick.com/teamtable.php"
    params = {
        "fromseason": season, "thruseason": season, "stype": "2",
        "sit": situation, "score": "all", "rate": "n",
        "team": "all", "loc": "B", "gpf": "410", "fd": "", "td": "",
    }
    return f"{base}?{'&'.join(f'{k}={v}' for k, v in params.items())}"


def build_skater_url(season: str, situation: str = "s5v5") -> str:
    base = "https://www.naturalstattrick.com/playerteams.php"
    params = {
        "fromseason": season, "thruseason": season, "stype": "2",
        "sit": situation, "score": "all", "rate": "n",
        "team": "all", "pos": "S", "loc": "B", "toi": "0",
        "gpfilt": "none", "fd": "", "td": "",
    }
    return f"{base}?{'&'.join(f'{k}={v}' for k, v in params.items())}"


def build_goalie_url(season: str, situation: str = "s5v5") -> str:
    base = "https://www.naturalstattrick.com/playerteams.php"
    params = {
        "fromseason": season, "thruseason": season, "stype": "2",
        "sit": situation, "score": "all", "rate": "n",
        "team": "all", "pos": "G", "loc": "B", "toi": "0",
        "gpfilt": "none", "fd": "", "td": "",
    }
    return f"{base}?{'&'.join(f'{k}={v}' for k, v in params.items())}"


# ── writers ───────────────────────────────────────────────────────────


def save_json(name: str, data: List[Dict[str, Any]], season: str, situation: str, out_dir: Path):
    path = out_dir / f"nst_{name}_stats.json"
    payload = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "season": f"{season[:4]}-{season[4:]}",
        "situation": situation,
        "count": len(data),
        "data": data,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved {path} ({len(data)} records)")


# ── main ──────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Scrape NST and export JSON")
    parser.add_argument("--season", default=os.environ.get("NST_SEASON", "20242025"))
    parser.add_argument("--sit", default=os.environ.get("NST_SIT", "s5v5"), choices=["s5v5", "all"])
    parser.add_argument("--out", default="static/data", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scraping NST — season={args.season} situation={args.sit}")

    # ── teams ──
    team_df = fetch_table(build_team_url(args.season, args.sit))
    if team_df is not None and not team_df.empty:
        save_json("team", clean_df(team_df), args.season, args.sit, out_dir)
    else:
        logger.warning("Team table scrape failed — keeping old file if present")

    # ── skaters ──
    skater_df = fetch_table(build_skater_url(args.season, args.sit))
    if skater_df is not None and not skater_df.empty:
        # keep only players with ≥10 GP (avoids call-ups with tiny samples)
        rows = clean_df(skater_df)
        rows = [r for r in rows if (r.get("gp") or 0) >= 10]
        save_json("skater", rows, args.season, args.sit, out_dir)
    else:
        logger.warning("Skater table scrape failed — keeping old file if present")

    # ── goalies ──
    goalie_df = fetch_table(build_goalie_url(args.season, args.sit))
    if goalie_df is not None and not goalie_df.empty:
        rows = clean_df(goalie_df)
        rows = [r for r in rows if (r.get("gp") or 0) >= 5]
        save_json("goalie", rows, args.season, args.sit, out_dir)
    else:
        logger.warning("Goalie table scrape failed — keeping old file if present")


if __name__ == "__main__":
    main()
