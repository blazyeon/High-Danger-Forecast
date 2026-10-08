"""
Scrape shotpropz.com goals / shots-on-goal allowed by position, per team,
split home / away.

The site renders a plain WordPress table: each position (C / LW / RW / D) is
its own <table> preceded by an <h3> ("Goals Against to C" / "SOG Against to LW").
Every team row carries data-team="XXX" and a <td class="value"
data-sort-value="N"> holding the per-game rate. No JS or API auth is involved.

Output schema (what update_shotpropz.py writes to shotpropz.json):
    {
      "source": "shotpropz",
      "updated_at": "...",
      "goals_against": {"All": {POS: {ABBR: float}}, "Home": {...}, "Away": {...}},
      "sog_against":    {"All": {POS: {ABBR: float}}, "Home": {...}, "Away": {...}},
    }
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Dict

from bs4 import BeautifulSoup

from NHL.InjuryScraper import _fetch_page

logger = logging.getLogger(__name__)

BASE_URL = "https://shotpropz.com/nhl/{metric}-against-by-position/"

# group=CLR renders four position tables (C / LW / RW / D) on a single page.
GROUP = "CLR"
SPAN = "recent"
FILTER = "all"

METRICS = ("goals", "sog")
LOCATIONS = ("All", "Home", "Away")

# The <h3> above each table ends with the position ("... to C").
POSITIONS = {"C", "LW", "RW", "D"}


def _scrape_metric_location(metric: str, location: str) -> Dict[str, Dict[str, float]]:
    url = (
        f"{BASE_URL.format(metric=metric)}"
        f"?span={SPAN}&location={location}&group={GROUP}&filter={FILTER}"
    )
    html = _fetch_page(url)
    if not html:
        logger.warning("No HTML for %s/%s", metric, location)
        return {}

    soup = BeautifulSoup(html, "lxml")
    out: Dict[str, Dict[str, float]] = {}
    for table in soup.find_all("table"):
        heading = table.find_previous("h3")
        label = heading.get_text(strip=True) if heading else ""
        # "Goals Against to C" / "SOG Against to LW" -> trailing position token.
        pos = label.rsplit("to ", 1)[-1].strip().upper() if "to " in label else ""
        if pos not in POSITIONS:
            continue

        bucket: Dict[str, float] = {}
        for tr in table.find_all("tr", attrs={"data-team": True}):
            team = tr.get("data-team")
            value_cell = tr.find("td", class_="value")
            raw = value_cell.get("data-sort-value") if value_cell else None
            if not team or raw is None:
                continue
            try:
                bucket[str(team).upper()] = float(raw)
            except (TypeError, ValueError):
                continue
        if bucket:
            out[pos] = bucket
    return out


def scrape_all(rate_limit_seconds: float = 0.75) -> Dict:
    """Scrape every metric/location combination into one payload."""
    payload: Dict = {
        "source": "shotpropz",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "goals_against": {},
        "sog_against": {},
    }
    for metric in METRICS:
        for location in LOCATIONS:
            data = _scrape_metric_location(metric, location)
            payload[f"{metric}_against"][location] = data
            time.sleep(rate_limit_seconds)
    return payload
