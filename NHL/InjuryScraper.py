"""
Scrape current NHL injuries from Daily Faceoff line-combination pages.

Returns structured data matching the injuries.json format:
    {"team": "BUF", "player": "Jiri Kulich", "status": "ir", "injured": true}
"""
from __future__ import annotations

import logging
import re
import time
from typing import Dict, List, Optional

import requests

from NHL.Config import REQUEST_HEADERS, DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)

# Map Daily Faceoff URL slugs -> our 3-letter team abbreviations
DFO_SLUG_TO_ABBR: Dict[str, str] = {
    "anaheim-ducks": "ANA",
    "boston-bruins": "BOS",
    "buffalo-sabres": "BUF",
    "calgary-flames": "CGY",
    "carolina-hurricanes": "CAR",
    "chicago-blackhawks": "CHI",
    "columbus-blue-jackets": "CBJ",
    "colorado-avalanche": "COL",
    "dallas-stars": "DAL",
    "detroit-red-wings": "DET",
    "edmonton-oilers": "EDM",
    "florida-panthers": "FLA",
    "los-angeles-kings": "LAK",
    "minnesota-wild": "MIN",
    "montreal-canadiens": "MTL",
    "nashville-predators": "NSH",
    "new-jersey-devils": "NJD",
    "new-york-islanders": "NYI",
    "new-york-rangers": "NYR",
    "ottawa-senators": "OTT",
    "philadelphia-flyers": "PHI",
    "pittsburgh-penguins": "PIT",
    "san-jose-sharks": "SJS",
    "seattle-kraken": "SEA",
    "st-louis-blues": "STL",
    "tampa-bay-lightning": "TBL",
    "toronto-maple-leafs": "TOR",
    "utah-hockey-club": "UTA",
    "vancouver-canucks": "VAN",
    "vegas-golden-knights": "VGK",
    "washington-capitals": "WSH",
    "winnipeg-jets": "WPG",
}

BASE_URL = "https://www.dailyfaceoff.com/teams/{slug}/line-combinations"


def _fetch_page(url: str, retries: int = 3) -> Optional[str]:
    """Fetch page HTML with retries and rate-limit-friendly headers."""
    for attempt in range(retries):
        try:
            resp = requests.get(
                url,
                headers=REQUEST_HEADERS,
                timeout=DEFAULT_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 429:
                logger.warning(f"Rate limited by {url}; sleeping...")
                time.sleep(5 + attempt * 5)
            else:
                logger.warning(f"GET {url} returned {resp.status_code}")
        except Exception as e:
            logger.warning(f"Fetch {url} failed (attempt {attempt + 1}): {e}")
        time.sleep(0.5 + attempt)
    return None


def _parse_name_from_href(href: str) -> Optional[str]:
    """Extract a player name slug from a Daily Faceoff player link."""
    m = re.search(r"/players/news/([^/]+)/", href)
    if not m:
        return None
    slug = m.group(1).replace("-", " ").title()
    # Fix common name particles
    for old, new in [
        (" De ", " de "), (" Van ", " van "), (" Van Der ", " van der "),
        (" Di ", " di "), (" Del ", " del "), (" Du ", " du "),
        (" Ii", " II"), (" Iii", " III"), (" Jr", " Jr."),
    ]:
        slug = slug.replace(old, new)
    return slug


def scrape_team_injuries(abbr: str) -> List[Dict[str, str]]:
    """Scrape injuries for a single team abbreviation."""
    slug = None
    for s, a in DFO_SLUG_TO_ABBR.items():
        if a == abbr.upper():
            slug = s
            break
    if not slug:
        logger.warning(f"No Daily Faceoff slug for team {abbr}")
        return []

    url = BASE_URL.format(slug=slug)
    html = _fetch_page(url)
    if not html:
        return []

    out: List[Dict[str, str]] = []

    # Find the Injuries section in the raw HTML. The Next.js-rendered cards are
    # siblings of the heading, so a DOM walk misses them; slice the HTML and
    # split into per-player cards instead.
    start = html.lower().find(">injuries")
    if start == -1:
        start = html.lower().find("injuries</span>")
    if start == -1:
        logger.debug(f"No Injuries section found for {abbr}")
        return out

    end_terms = [
        "Click player jersey for news",
        'class="text-center text-xs text-gray-500',
        "Badges:",
        "Game-time decision",
    ]
    end = len(html)
    for term in end_terms:
        idx = html.find(term, start)
        if idx != -1:
            end = min(end, idx)
    chunk = html[start:end]

    # Each player card is wrapped in a `w-1/3 text-center` column.
    card_pattern = re.compile(r'<div class="w-1/3 text-center">(.*?)</div>\s*</div>\s*</div>', re.S | re.I)
    link_re = re.compile(r'<a href="/players/news/([^/]+)/(\d+)"', re.S | re.I)
    badge_re = re.compile(r'<span class="rounded-md bg-red-500[^>]*>([^<]+)</span>', re.S | re.I)
    img_alt_re = re.compile(r'<img alt="([^"]+)"', re.S | re.I)
    name_span_re = re.compile(r'<span class="text-xs font-bold uppercase[^>]*>([^<]+)</span>', re.S | re.I)

    seen = set()
    for card in card_pattern.findall(chunk):
        link_m = link_re.search(card)
        if not link_m:
            continue
        href_slug, player_id = link_m.group(1), link_m.group(2)

        badge_m = badge_re.search(card)
        if not badge_m:
            continue  # no red badge = not injured in this card
        status = badge_m.group(1).strip().lower()

        # Prefer the <img alt="..."> name; fall back to the visible name span.
        name = ""
        img_m = img_alt_re.search(card)
        if img_m:
            name = img_m.group(1).strip()
        if not name:
            span_m = name_span_re.search(card)
            if span_m:
                name = span_m.group(1).strip()
        if not name:
            name = _parse_name_from_href(href_slug) or ""
        if not name:
            continue

        norm = name.lower()
        if norm in seen:
            continue
        seen.add(norm)

        out.append({
            "team": abbr.upper(),
            "player": name,
            "status": status,
            "injured": True,
        })

    logger.info(f"Scraped {len(out)} injuries for {abbr} from Daily Faceoff")
    return out


def scrape_all_injuries(rate_limit_seconds: float = 0.75) -> List[Dict[str, str]]:
    """Scrape injuries for all teams and return a flat list."""
    all_injuries: List[Dict[str, str]] = []
    for slug, abbr in DFO_SLUG_TO_ABBR.items():
        try:
            team_injuries = scrape_team_injuries(abbr)
            all_injuries.extend(team_injuries)
        except Exception as e:
            logger.warning(f"Failed to scrape injuries for {abbr}: {e}")
        time.sleep(rate_limit_seconds)
    logger.info(f"Total scraped injuries: {len(all_injuries)}")
    return all_injuries


__all__ = ["scrape_team_injuries", "scrape_all_injuries", "DFO_SLUG_TO_ABBR"]
