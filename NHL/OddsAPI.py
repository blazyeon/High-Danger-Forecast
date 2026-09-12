"""
Thin client for The Odds API (v4) focused on NHL odds.
- API key from env ODDS_API_KEY or a .env file. No embedded key.
- Retries on 429 with exponential backoff
- Helpers:
    - fetch_nhl_odds_by_date: featured markets (h2h/spreads/totals) using /odds
    - fetch_nhl_events_by_date: list events on a specific date (free)
    - fetch_event_player_odds: player props for one event using /events/{id}/odds
    - fetch_nhl_player_props_by_date: aggregate player props for all events on a date
Docs: https://api.the-odds-api.com
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import date as _date, datetime, timedelta, timezone

import requests
import logging

from NHL.Utils import atomic_write_json, read_json_robust, LEAGUE_TZ

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.the-odds-api.com/v4"

# Where the last-seen Odds API quota is persisted for the low-quota UI banner.
QUOTA_CACHE_PATH = Path(__file__).resolve().parent.parent / "static" / "data" / "odds_quota.json"


class OddsAPIError(Exception):
    pass


def _get_api_key() -> Optional[str]:
    """Get API key from environment variable or a .env file."""
    # Try environment variable first
    key = os.getenv("ODDS_API_KEY")
    if key:
        return key
    # Try .env file
    try:
        from pathlib import Path
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("ODDS_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return None

def _headers() -> Dict[str, str]:
    return {
        "Accept": "application/json",
        "User-Agent": "NHLGamePredictor/1.0"
    }

def _iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def _log_quota_headers(hdrs: Dict[str, str]) -> None:
    """Log remaining Odds API quota and persist it for the low-quota banner."""
    remaining = hdrs.get("x-requests-remaining")
    used = hdrs.get("x-requests-used")
    if remaining is not None or used is not None:
        logger.info(f"Odds API quota — used={used}, remaining={remaining}")
        try:
            atomic_write_json(QUOTA_CACHE_PATH, {
                "remaining": _as_int(remaining),
                "used": _as_int(used),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            logger.debug(f"Could not persist Odds API quota: {e}")


def _as_int(value: Any) -> Optional[int]:
    """Coerce a header value to int, returning None when unparseable."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_odds_quota_status() -> Dict[str, Any]:
    """Return the last-known Odds API quota (remaining/used/updated_at)."""
    try:
        if QUOTA_CACHE_PATH.exists():
            return read_json_robust(QUOTA_CACHE_PATH)
    except Exception:
        pass
    return {"remaining": None, "used": None, "updated_at": None}


def _check_quota() -> None:
    """Raise early when the persisted quota is exhausted instead of hammering the API."""
    quota = get_odds_quota_status()
    remaining = quota.get("remaining")
    if remaining is not None and remaining <= 0:
        raise OddsAPIError("Odds API quota exhausted; try again later.")

def _retry_after_delay(resp: Any, attempt: int, backoff: float) -> float:
    """Return the sleep delay for a retryable response, honoring Retry-After."""
    retry_after = resp.headers.get("Retry-After")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            pass
    # Exponential backoff with jitter to avoid a thundering herd under quota pressure.
    return backoff ** attempt + random.uniform(0, 0.5)


def _request_with_retry(
    method: str,
    url: str,
    params: Dict[str, Any],
    max_retries: int = 4,
    timeout: int = 20
) -> Tuple[Any, Dict[str, str]]:
    backoff = 1.7
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.request(method, url, params=params, headers=_headers(), timeout=timeout)
        except Exception as e:
            last_err = e
            time.sleep(backoff ** attempt + random.uniform(0, 0.5))
            continue

        hdrs = {k.lower(): v for k, v in resp.headers.items()}
        if resp.status_code == 200:
            try:
                return resp.json(), hdrs
            except Exception as e:
                raise OddsAPIError(f"Failed to parse JSON: {e}")
        if resp.status_code in (429, 503):
            # Rate limited / temporarily unavailable
            time.sleep(_retry_after_delay(resp, attempt, backoff))
            continue
        # Other error
        detail = resp.text[:500] if hasattr(resp, "text") else f"status={resp.status_code}"
        raise OddsAPIError(f"Odds API {resp.status_code}: {detail}")
    raise OddsAPIError(f"Exceeded retries: {last_err}")

def _utc_day_window(day: _date) -> Tuple[str, str]:
    # The NHL schedules its game-day boundary in Eastern time. A game at 10 PM ET
    # is already past midnight UTC, so a UTC-midnight window would push it onto the
    # next day and return an off-by-one slate. Build the window from Eastern
    # midnight boundaries, then convert to UTC for the Odds API commenceTime params.
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(LEAGUE_TZ)
    except Exception:
        tz = timezone.utc
    start_local = datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=tz)
    end_local = start_local + timedelta(days=1) - timedelta(seconds=1)
    return _iso_utc(start_local), _iso_utc(end_local)

def fetch_nhl_odds_by_date(
    day: _date,
    regions: str,
    markets: List[str],
    bookmakers_csv: Optional[str] = None,
    odds_format: str = "american",
    base_url: str = DEFAULT_BASE_URL
) -> List[Dict[str, Any]]:
    """
    Featured markets (/v4/sports/icehockey_nhl/odds):
    Valid: h2h, spreads, totals (player_* NOT supported here).
    """
    api_key = _get_api_key()
    if not api_key:
        raise OddsAPIError("Missing API key.")
    _check_quota()

    commence_from, commence_to = _utc_day_window(day)
    params: Dict[str, Any] = {
        "apiKey": api_key,
        "regions": regions,
        "oddsFormat": odds_format,
        "dateFormat": "iso",
        "commenceTimeFrom": commence_from,
        "commenceTimeTo": commence_to,
    }
    if markets:
        params["markets"] = ",".join(markets)
    if bookmakers_csv:
        params["bookmakers"] = bookmakers_csv

    url = f"{base_url}/sports/icehockey_nhl/odds"
    data, hdrs = _request_with_retry("GET", url, params=params)
    _log_quota_headers(hdrs)

    if not isinstance(data, list):
        raise OddsAPIError(f"Unexpected response type: {type(data)}")
    return data

def fetch_nhl_events_by_date(
    day: _date,
    base_url: str = DEFAULT_BASE_URL
) -> List[Dict[str, Any]]:
    """
    Free endpoint: /v4/sports/icehockey_nhl/events
    Returns events without odds; used to enumerate event IDs for player props.
    """
    api_key = _get_api_key()
    if not api_key:
        raise OddsAPIError("Missing API key.")
    commence_from, commence_to = _utc_day_window(day)
    params = {
        "apiKey": api_key,
        "dateFormat": "iso",
        "commenceTimeFrom": commence_from,
        "commenceTimeTo": commence_to,
    }
    url = f"{base_url}/sports/icehockey_nhl/events"
    data, hdrs = _request_with_retry("GET", url, params=params)
    _log_quota_headers(hdrs)
    if not isinstance(data, list):
        raise OddsAPIError(f"Unexpected response type: {type(data)}")
    return data

def fetch_event_player_odds(
    event_id: str,
    regions: str,
    markets: List[str],
    bookmakers_csv: Optional[str] = None,
    odds_format: str = "american",
    base_url: str = DEFAULT_BASE_URL
) -> Optional[Dict[str, Any]]:
    """
    Player markets supported via /v4/sports/{sport}/events/{eventId}/odds
    Returns a single event object (or None if no markets available).
    """
    api_key = _get_api_key()
    if not api_key:
        raise OddsAPIError("Missing API key.")
    _check_quota()
    params: Dict[str, Any] = {
        "apiKey": api_key,
        "regions": regions,
        "oddsFormat": odds_format,
        "dateFormat": "iso",
    }
    if markets:
        params["markets"] = ",".join(markets)
    if bookmakers_csv:
        params["bookmakers"] = bookmakers_csv
    url = f"{base_url}/sports/icehockey_nhl/events/{event_id}/odds"
    data, hdrs = _request_with_retry("GET", url, params=params)
    _log_quota_headers(hdrs)
    if not isinstance(data, dict):
        return None
    # If no bookmakers/markets available, skip
    books = data.get("bookmakers", []) or []
    if not books:
        return None
    has_markets = any((bk.get("markets") or []) for bk in books)
    return data if has_markets else None

def fetch_nhl_player_props_by_date(
    day: _date,
    regions: str,
    markets: List[str],
    bookmakers_csv: Optional[str] = None,
    odds_format: str = "american",
    base_url: str = DEFAULT_BASE_URL
) -> List[Dict[str, Any]]:
    """
    Aggregate all player prop markets for all NHL events on a given date.
    Steps:
      1) /events (free) to list events for the day
      2) /events/{id}/odds for each event with player_* markets
    Costs: 1 per event per unique market group returned x regions.
    """
    events = fetch_nhl_events_by_date(day, base_url=base_url)
    out: List[Dict[str, Any]] = []
    for ev in events:
        ev_id = ev.get("id")
        if not ev_id:
            continue
        try:
            data = fetch_event_player_odds(
                event_id=ev_id,
                regions=regions,
                markets=markets,
                bookmakers_csv=bookmakers_csv,
                odds_format=odds_format,
                base_url=base_url
            )
            if data:
                out.append(data)
        except OddsAPIError as e:
            # Skip problematic event, continue others
            # Optionally you can log e
            continue
    return out