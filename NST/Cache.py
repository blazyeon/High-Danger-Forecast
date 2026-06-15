"""
Natural Stat Trick scraper with connection pooling, robust HTML parsing,
in-memory caching with TTL, and URL builders.

.. deprecated::
    NST HTML scraping is fragile and has CORS issues. The data pipeline
    has been migrated to the NHL API play-by-play feed (see
    ``NHL.PlayByPlay``) plus MoneyPuck CSVs (see ``NHL.MoneyPuck``) for
    validation. Stats aggregators live in ``NHL.StatsFromPBP`` and the
    xG model is in ``NHL.xGModel``.

    This file remains importable for backwards compatibility (the app's
    startup and a few historical scripts may still reference it). The
    functions log a deprecation warning on first call. They will be
    removed in a future release.
"""
from __future__ import annotations

import logging
import re
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple
from io import StringIO
from urllib.parse import urlencode, urlparse

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Emit a deprecation warning the first time this module is imported.
warnings.warn(
    "NST.Cache is deprecated; use NHL.PlayByPlay + NHL.StatsFromPBP instead.",
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)

# Optional: import defaults from Config if available
try:
    from NHL.Config import REQUEST_HEADERS as _REQ_HEADERS
    from NHL.Config import DEFAULT_TIMEOUT as _DEFAULT_TIMEOUT
    from NHL.Config import RATE_LIMIT_SLEEP_SECONDS as _RATE_LIMIT
    from NHL.Config import MAX_RETRIES as _MAX_RETRIES
    from NHL.Config import RETRY_BACKOFF_BASE as _BACKOFF
    from NHL.Config import CONNECTION_POOL_SIZE as _POOL_SIZE
    from NHL.Config import CONNECTION_POOL_MAXSIZE as _POOL_MAX
except Exception:
    _REQ_HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; NHLGamePredictor/1.0; +https://github.com/)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    _DEFAULT_TIMEOUT = 20
    _RATE_LIMIT = 0.35
    _MAX_RETRIES = 3
    _BACKOFF = 0.75
    _POOL_SIZE = 10
    _POOL_MAX = 20

# ─── Connection-pooled session ─────────────────────────────────────────────

_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """Get or create a requests session with connection pooling and retry."""
    global _session
    if _session is None:
        _session = requests.Session()
        retry_strategy = Retry(
            total=_MAX_RETRIES,
            backoff_factor=_BACKOFF,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=_POOL_SIZE,
            pool_maxsize=_POOL_MAX,
        )
        _session.mount("http://", adapter)
        _session.mount("https://", adapter)
        _session.headers.update(_REQ_HEADERS)
    return _session


# ─── In-memory cache with TTL ──────────────────────────────────────────────

_CACHE: Dict[str, Tuple[float, Optional[pd.DataFrame]]] = {}
_CACHE_TTL_SECONDS = 6 * 60 * 60      # 6 hours for successful fetches
_NEG_CACHE_TTL_SECONDS = 5 * 60        # 5 minutes for failed fetches
_MAX_CACHE_ENTRIES = 500                # Prevent unbounded growth


def _now() -> float:
    return time.time()


def _cache_get(url: str) -> Optional[Optional[pd.DataFrame]]:
    entry = _CACHE.get(url)
    if not entry:
        return None
    ts, df = entry
    age = _now() - ts
    ttl = _NEG_CACHE_TTL_SECONDS if df is None else _CACHE_TTL_SECONDS
    if age <= ttl:
        return df
    _CACHE.pop(url, None)
    return None


def _cache_set(url: str, df: Optional[pd.DataFrame]) -> None:
    # Evict oldest entries if cache is full
    if len(_CACHE) >= _MAX_CACHE_ENTRIES:
        oldest_key = min(_CACHE, key=lambda k: _CACHE[k][0])
        _CACHE.pop(oldest_key, None)
    _CACHE[url] = (_now(), df)


# ─── HTML parsing helpers ──────────────────────────────────────────────────

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(c) for c in tup if str(c) != "nan"]).strip()
            for tup in df.columns.values
        ]
    else:
        df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"TEAM": "Team", "team": "Team", "Sv%": "Sv%", "SV%": "Sv%", "Saves%": "Sv%"})
    return df


def _best_table(tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if not tables:
        return None
    tables = [t for t in tables if isinstance(t, pd.DataFrame)]
    if not tables:
        return None
    tables.sort(key=lambda t: (len(t.index), len(t.columns)), reverse=True)
    return tables[0]


def _fetch_html(url: str) -> Optional[str]:
    try:
        time.sleep(_RATE_LIMIT)
        session = _get_session()
        resp = session.get(url, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.error(f"NST Cache: HTTP error fetching {url[:120]}: {e}")
        return None


def _parse_with_pandas(html: str) -> Optional[pd.DataFrame]:
    sio = StringIO(html)
    for flavor in ("lxml", "bs4", "html5lib"):
        try:
            tables = pd.read_html(sio, flavor=flavor)
            df = _best_table(tables)
            if df is not None:
                return _flatten_columns(df)
        except Exception:
            pass
        sio.seek(0)
    return None


def _parse_with_bs4_manual(html: str) -> Optional[pd.DataFrame]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("NST: BeautifulSoup not installed. Install with: pip install beautifulsoup4 html5lib")
        return None

    parser = "html5lib"
    try:
        import html5lib  # noqa: F401
    except ImportError:
        parser = "html.parser"

    try:
        soup = BeautifulSoup(html, parser)
        tables = soup.find_all("table")
        if not tables:
            return None

        tables.sort(key=lambda t: len(t.find_all("tr")), reverse=True)
        tb = tables[0]

        headers: List[str] = []
        thead = tb.find("thead")
        if thead:
            headers = [th.get_text(strip=True) for th in thead.find_all("th")]

        rows: List[List[str]] = []
        for tr in tb.find_all("tr"):
            tds = tr.find_all(["td", "th"])
            if not tds:
                continue
            row = [td.get_text(strip=True) for td in tds]
            rows.append(row)

        if not rows:
            return None
        if not headers:
            headers = [f"Col{i+1}" for i in range(len(rows[0]))]
            data_rows = rows
        else:
            data_rows = [r for r in rows if len(r) == len(headers)]

        if not data_rows:
            return None
        return _flatten_columns(pd.DataFrame(data_rows, columns=headers))
    except Exception as e:
        logger.error(f"NST Cache: Manual BeautifulSoup parse failed: {e}")
        return None


# ─── Main fetcher ──────────────────────────────────────────────────────────

def get_nst_table_from_url(url: str) -> Optional[pd.DataFrame]:
    """
    Fetch and parse a Natural Stat Trick table from a URL.
    Uses in-memory cache with TTL, multiple HTML parser fallbacks,
    and connection pooling.
    """
    cached = _cache_get(url)
    if cached is not None or url in _CACHE:
        return cached

    html = _fetch_html(url)
    if not html:
        _cache_set(url, None)
        return None

    df = _parse_with_pandas(html)
    if df is None or df.empty:
        df = _parse_with_bs4_manual(html)

    if df is None or df.empty:
        logger.warning(f"NST Cache: No usable table found at {url[:200]}")
        _cache_set(url, None)
        return None

    # Cleanup
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()

    _cache_set(url, df)
    return df


# ─── Alternative scraper (BeautifulSoup direct) ────────────────────────────

def scrape_nst_player_table(url: str, timeout: int = 30) -> Optional[pd.DataFrame]:
    """
    Scrape player stats table using BeautifulSoup directly.
    Use as a fallback if pandas.read_html fails.
    """
    try:
        session = _get_session()
        response = session.get(url, timeout=timeout)
        response.raise_for_status()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")

        table = soup.find("table", {"class": "sortable"})
        if not table:
            return get_nst_table_from_url(url)  # fallback to main fetcher

        headers_row = table.find("thead").find("tr") if table.find("thead") else table.find("tr")
        headers = [re.sub(r'\s+', ' ', th.get_text(strip=True)) for th in headers_row.find_all(["th", "td"])]

        rows_data = []
        tbody = table.find("tbody") if table.find("tbody") else table
        for tr in tbody.find_all("tr"):
            if tr.find("th") and not tr.find("td"):
                continue
            cells = [td.get_text(strip=True).replace(",", "") for td in tr.find_all(["td", "th"])]
            if len(cells) == len(headers):
                rows_data.append(cells)

        if not rows_data:
            return None

        return pd.DataFrame(rows_data, columns=headers)

    except Exception as e:
        logger.error(f"NST Scraper: Failed to scrape {url[:80]}: {e}")
        return None


def get_nst_player_stats(season: str, stype: int = 2) -> Optional[pd.DataFrame]:
    """
    Get player stats from NST for a given season.
    Uses build_nst_player_url to construct the URL.
    """
    url = build_nst_player_url(season, stype=stype)
    return get_nst_table_from_url(url)


# ─── URL builders ──────────────────────────────────────────────────────────

def build_nst_team_url(
    season: str,
    situation: str = "all",
    table: str = "teams",
    **kwargs
) -> str:
    """Build a Natural Stat Trick team stats URL."""
    base_url = "https://www.naturalstattrick.com/teamtable.php"
    params = {
        "fromseason": season,
        "thruseason": season,
        "stype": "2",
        "sit": situation,
        "score": "all",
        "rate": "n",
        "team": "all",
        "loc": "B",
        "gpf": "410",
        "fd": "",
        "td": "",
    }
    params.update(kwargs)
    return f"{base_url}?{urlencode(params)}"


def build_nst_player_url(
    season: str,
    situation: str = "all",
    position: str = "S",
    **kwargs
) -> str:
    """Build a Natural Stat Trick player stats URL."""
    base_url = "https://www.naturalstattrick.com/playerteams.php"
    params = {
        "fromseason": season,
        "thruseason": season,
        "stype": "2",
        "sit": situation,
        "score": "all",
        "rate": "n",
        "team": "all",
        "pos": position,
        "loc": "B",
        "toi": "0",
        "gpfilt": "none",
        "fd": "",
        "td": "",
    }
    params.update(kwargs)
    return f"{base_url}?{urlencode(params)}"


def validate_nst_url(url: str) -> bool:
    """Validate that a URL is a Natural Stat Trick URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc == "www.naturalstattrick.com"
    except Exception:
        return False


def test_nst_connection() -> bool:
    """Test if Natural Stat Trick is accessible."""
    test_url = build_nst_team_url("20232024", situation="all")
    try:
        df = get_nst_table_from_url(test_url)
        success = df is not None and not df.empty
        if success:
            logger.info("NST connection test successful")
        else:
            logger.warning("NST connection test failed: no data returned")
        return success
    except Exception as e:
        logger.error(f"NST connection test failed: {e}")
        return False


def clear_cache() -> int:
    """Clear the in-memory NST cache. Returns number of entries cleared."""
    count = len(_CACHE)
    _CACHE.clear()
    logger.info(f"Cleared NST cache ({count} entries)")
    return count


__all__ = [
    "get_nst_table_from_url",
    "scrape_nst_player_table",
    "get_nst_player_stats",
    "build_nst_team_url",
    "build_nst_player_url",
    "validate_nst_url",
    "test_nst_connection",
    "clear_cache",
]