"""
Scrape projected power-play units from Daily Faceoff line-combination pages.

Daily Faceoff embeds the full lineup as JSON in ``__NEXT_DATA__``, with each
player tagged by ``groupIdentifier`` (``pp1``, ``pp2``, ``f1``-``f4``,
``d1``-``d3``, ``g``, ``pk1``/``pk2``, ``ir``). We reuse the same team pages
the injury scraper already fetches, but read the JSON instead of slicing HTML,
so power-play membership is exact rather than guessed from last game's TOI.

Output shape (per team):
    {"pp1": ["Auston Matthews", ...], "pp2": [...],
     "updated_at": "...", "source": "Last Game (2026-10-06)"}
"""
from __future__ import annotations

import json
import logging
import time
from typing import Dict, List, Optional

from NHL.Config import REQUEST_HEADERS, DEFAULT_TIMEOUT
from NHL.InjuryScraper import DFO_SLUG_TO_ABBR, _fetch_page

logger = logging.getLogger(__name__)

BASE_URL = "https://www.dailyfaceoff.com/teams/{slug}/line-combinations"


def _slug_for_abbr(abbr: str) -> Optional[str]:
    """Reverse-lookup the Daily Faceoff slug for a team abbreviation."""
    target = (abbr or "").upper()
    for slug, a in DFO_SLUG_TO_ABBR.items():
        if a == target:
            return slug
    return None


def _parse_combinations(html: str) -> Optional[Dict]:
    """Extract and parse the ``combinations`` object from ``__NEXT_DATA__``."""
    start = html.find("__NEXT_DATA__")
    if start == -1:
        logger.warning("No __NEXT_DATA__ blob on page; Daily Faceoff markup changed?")
        return None
    gt = html.find(">", start)
    end = html.find("</script>", gt)
    if gt == -1 or end == -1:
        return None
    try:
        data = json.loads(html[gt + 1 : end])
        return data["props"]["pageProps"]["combinations"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Could not parse Daily Faceoff __NEXT_DATA__: {e}")
        return None


def _dedup_keep_order(names: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def scrape_team_pp_units(abbr: str) -> Optional[Dict]:
    """Scrape PP1/PP2 membership for a single team, or None on failure."""
    slug = _slug_for_abbr(abbr)
    if not slug:
        logger.warning(f"No Daily Faceoff slug for team {abbr}")
        return None

    html = _fetch_page(BASE_URL.format(slug=slug))
    if not html:
        return None

    combos = _parse_combinations(html)
    if not combos:
        return None

    pp1: List[str] = []
    pp2: List[str] = []
    for player in combos.get("players", []):
        name = player.get("name")
        if not name:
            continue
        group = (player.get("groupIdentifier") or "").lower()
        if group == "pp1":
            pp1.append(name)
        elif group == "pp2":
            pp2.append(name)

    return {
        "pp1": _dedup_keep_order(pp1),
        "pp2": _dedup_keep_order(pp2),
        "updated_at": combos.get("updatedAt"),
        "source": combos.get("sourceName"),
    }


def scrape_all_pp_units(rate_limit_seconds: float = 0.75) -> Dict:
    """Scrape PP units for every team and return them keyed by abbreviation."""
    teams: Dict[str, Dict] = {}
    for slug, abbr in DFO_SLUG_TO_ABBR.items():
        try:
            team = scrape_team_pp_units(abbr)
            if team:
                teams[abbr] = team
        except Exception as e:
            logger.warning(f"Failed to scrape PP units for {abbr}: {e}")
        time.sleep(rate_limit_seconds)

    updated_ats = [t.get("updated_at") for t in teams.values() if t.get("updated_at")]
    return {
        "source": "daily_faceoff",
        "updated_at": max(updated_ats) if updated_ats else None,
        "teams": teams,
    }


__all__ = ["scrape_team_pp_units", "scrape_all_pp_units"]
