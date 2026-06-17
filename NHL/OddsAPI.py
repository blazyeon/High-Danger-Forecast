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
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import date as _date, datetime, timedelta, timezone

import requests
import logging

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.the-odds-api.com/v4"


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
            time.sleep(backoff ** attempt)
            continue

        hdrs = {k.lower(): v for k, v in resp.headers.items()}
        if resp.status_code == 200:
            try:
                return resp.json(), hdrs
            except Exception as e:
                raise OddsAPIError(f"Failed to parse JSON: {e}")
        if resp.status_code == 429:
            # Rate limited
            time.sleep(backoff ** attempt)
            continue
        # Other error
        detail = resp.text[:500] if hasattr(resp, "text") else f"status={resp.status_code}"
        raise OddsAPIError(f"Odds API {resp.status_code}: {detail}")
    raise OddsAPIError(f"Exceeded retries: {last_err}")

def _utc_day_window(day: _date) -> Tuple[str, str]:
    start_dt = datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=timezone.utc)
    end_dt = start_dt + timedelta(days=1) - timedelta(seconds=1)
    return _iso_utc(start_dt), _iso_utc(end_dt)

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
    data, _ = _request_with_retry("GET", url, params=params)
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
    data, _ = _request_with_retry("GET", url, params=params)
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