"""
Thin client for SharpAPI (https://docs.sharpapi.io) focused on NHL odds.

Replaces the previous The Odds API client. The public function signatures are
unchanged, so callers (NHL/BettingEdge.py, NHL/PlayerLinePredictor.py, app.py,
update_odds.py) do not need to know which provider is behind them.

SharpAPI returns one flat row per (event, book, market, outcome) rather than the
nested event/bookmaker/market tree The Odds API used. This module translates
those rows back into the tree shape the rest of the app expects:

    {"id", "home_team", "away_team", "commence_time",
     "bookmakers": [{"key", "title",
                     "markets": [{"key", "outcomes": [
                         {"name", "price", "point"[, "description"]}]}]}]}

Free tier limits handled here:
  - 12 requests per minute  -> `_pace()` throttles to 11 so a retry cannot
    tip the window over. There is no monthly quota, so no credit budgeting.
  - 60s data delay          -> irrelevant for pregame lines.
  - DraftKings + FanDuel    -> only books the key can see.

API key comes from env SHARPAPI_KEY or a .env file. No embedded key.
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time
from datetime import date as _date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from NHL.Utils import atomic_write_json, read_json_robust, LEAGUE_TZ

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.sharpapi.io/api/v1"

# Where the last-seen rate-limit state is persisted for the UI banner.
QUOTA_CACHE_PATH = Path(__file__).resolve().parent.parent / "static" / "data" / "odds_quota.json"

# SharpAPI's free tier allows 12 requests per rolling minute.
_RATE_LIMIT_PER_MIN = 12
# Hold one request back so a retry or a concurrent process cannot exhaust the window.
_RATE_LIMIT_BUDGET = _RATE_LIMIT_PER_MIN - 1

# SharpAPI market_type names differ from The Odds API's for a few markets.
# Map app-facing names -> SharpAPI names, and back for the response tree.
_TO_SHARPAPI_MARKET = {
    "h2h": "moneyline",
    "spreads": "puck_line",
    "totals": "total_goals",
    "player_total_saves": "player_saves",
}
_TO_INTERNAL_MARKET = {
    "moneyline": "h2h",
    "puck_line": "spreads",
    "total_goals": "totals",
    "player_saves": "player_total_saves",
}

# Request timestamps inside the current rolling minute, guarded by _rate_lock.
_rate_lock = threading.Lock()
_request_times: List[float] = []


class OddsAPIError(Exception):
    pass


class _CursorExpired(Exception):
    """
    SharpAPI rebuilt its odds store, invalidating our pagination cursor.

    Recoverable: the caller restarts the walk rather than failing the whole
    day's props. SharpAPI refreshes roughly every 15s, so a throttled multi-page
    walk will hit this regularly.
    """


def _get_api_key() -> Optional[str]:
    """Get the SharpAPI key from the environment or a .env file."""
    key = os.getenv("SHARPAPI_KEY")
    if key:
        return key
    try:
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("SHARPAPI_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return None


def _headers() -> Dict[str, str]:
    return {
        "Accept": "application/json",
        "User-Agent": "NHLGamePredictor/1.0",
        "X-API-Key": _get_api_key() or "",
    }


def _as_int(value: Any) -> Optional[int]:
    """Coerce a value to int, returning None when unparseable."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    """Coerce a value to float, returning None when unparseable."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_ts(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 UTC timestamp into an aware datetime."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _day_window(day: _date) -> Tuple[datetime, datetime]:
    """
    UTC window covering one NHL game day.

    The NHL schedules its game-day boundary in Eastern time, so a 10 PM ET game
    is already past midnight UTC. Build the window from Eastern midnight
    boundaries or late games land on the wrong date.
    """
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(LEAGUE_TZ)
    except Exception:
        tz = timezone.utc
    start_local = datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=tz)
    end_local = start_local + timedelta(days=1) - timedelta(seconds=1)
    return (
        start_local.astimezone(timezone.utc),
        end_local.astimezone(timezone.utc),
    )


def _pace() -> None:
    """
    Block until another request fits in SharpAPI's 12-per-minute window.

    Uses a rolling window of request timestamps rather than a fixed sleep so a
    long paging loop costs exactly as much wall-clock as the limit requires.
    """
    while True:
        with _rate_lock:
            now = time.monotonic()
            while _request_times and now - _request_times[0] >= 60.0:
                _request_times.pop(0)
            if len(_request_times) < _RATE_LIMIT_BUDGET:
                _request_times.append(now)
                return
            wait = 60.0 - (now - _request_times[0]) + 0.05
        logger.info(f"SharpAPI rate limit reached ({_RATE_LIMIT_BUDGET}/min); pacing for {wait:.1f}s")
        time.sleep(max(0.1, wait))


def _retry_after_seconds(value: Any) -> Optional[float]:
    """
    Interpret a retry_after value, whose units SharpAPI documents inconsistently
    (some examples are a duration in seconds, others an absolute epoch). Sniff
    by magnitude instead of trusting either.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    now = time.time()
    if v > 1e11:        # epoch milliseconds
        return max(1.0, v / 1000.0 - now)
    if v > 1e9:         # epoch seconds
        return max(1.0, v - now)
    return max(1.0, v)  # duration in seconds


def _note_rate_limit(hdrs: Dict[str, str]) -> None:
    """
    Log SharpAPI's per-minute rate state.

    The app's UI banner warns on a *low monthly quota*, which SharpAPI does not
    have, so `remaining` is deliberately reported as None here. The live
    per-minute figure is kept separately for diagnostics.
    """
    limit = hdrs.get("x-ratelimit-limit")
    remaining = hdrs.get("x-ratelimit-remaining")
    delay = hdrs.get("x-data-delay")
    if remaining is not None:
        logger.debug(f"SharpAPI rate limit — remaining={remaining}/{limit}, data_delay={delay}s")
    try:
        atomic_write_json(QUOTA_CACHE_PATH, {
            "provider": "sharpapi",
            "remaining": None,
            "used": None,
            "rate_limit_remaining": _as_int(remaining),
            "rate_limit_per_min": _as_int(limit),
            "data_delay_seconds": _as_int(delay),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        logger.debug(f"Could not persist SharpAPI rate state: {e}")


def get_odds_quota_status() -> Dict[str, Any]:
    """
    Return the last-known odds quota.

    `remaining` is always None for SharpAPI: it meters requests per minute and
    resets continuously, with no spendable balance. The UI banner keys off
    `remaining`, so leaving it None keeps it hidden instead of warning on every
    page load.
    """
    try:
        if QUOTA_CACHE_PATH.exists():
            cached = read_json_robust(QUOTA_CACHE_PATH)
            cached["remaining"] = None
            cached["used"] = None
            return cached
    except Exception:
        pass
    return {"provider": "sharpapi", "remaining": None, "used": None, "updated_at": None}


def _request_with_retry(
    method: str,
    url: str,
    params: Dict[str, Any],
    max_retries: int = 4,
    timeout: int = 30,
) -> Tuple[Any, Dict[str, str]]:
    backoff = 1.7
    last_err: Any = None
    for attempt in range(max_retries):
        _pace()
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

        if resp.status_code == 429:
            delay = None
            try:
                body = resp.json()
                delay = _retry_after_seconds((body.get("error") or {}).get("retry_after"))
            except Exception:
                pass
            if delay is None:
                delay = _retry_after_seconds(
                    hdrs.get("retry-after") or hdrs.get("x-ratelimit-reset")
                )
            if delay is None:
                delay = backoff ** attempt + random.uniform(0, 0.5)
            logger.warning(f"SharpAPI rate limited; sleeping {delay:.1f}s")
            time.sleep(delay)
            continue

        if resp.status_code >= 500:
            time.sleep(backoff ** attempt + random.uniform(0, 0.5))
            continue

        if resp.status_code == 400:
            try:
                err = (resp.json() or {}).get("error") or {}
            except Exception:
                err = {}
            if err.get("code") == "cursor_expired":
                raise _CursorExpired(str(err.get("message") or "pagination cursor expired"))
        if resp.status_code == 401:
            raise OddsAPIError("SharpAPI rejected the API key (401). Check SHARPAPI_KEY.")
        if resp.status_code == 403:
            detail = resp.text[:300]
            raise OddsAPIError(f"SharpAPI tier restriction (403): {detail}")

        detail = resp.text[:500] if hasattr(resp, "text") else f"status={resp.status_code}"
        raise OddsAPIError(f"SharpAPI {resp.status_code}: {detail}")

    raise OddsAPIError(f"Exceeded retries: {last_err}")


def _pagination(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Read the pagination block, which appears at the top level and/or under meta."""
    meta = payload.get("meta") or {}
    return payload.get("pagination") or meta.get("pagination") or {}


def _row_key(row: Dict[str, Any]) -> tuple:
    """Identity of one odds row, used to dedupe across restarted walks."""
    return (
        row.get("event_id"),
        row.get("sportsbook"),
        row.get("market_type"),
        row.get("selection"),
        row.get("player_name"),
        row.get("line"),
    )


def _fetch_odds_rows(
    market: Optional[str],
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
    event_ids: Optional[List[str]] = None,
    sportsbook: Optional[str] = None,
    allow_markets: Optional[set] = None,
    base_url: str = DEFAULT_BASE_URL,
    limit: int = 200,
    max_pages: int = 60,
    max_restarts: int = 3,
) -> List[Dict[str, Any]]:
    """
    Page /odds with the cursor and return the raw rows.

    When a day window is given, rows outside it are dropped and paging stops as
    soon as results pass the end of the window — /odds is ordered by event start
    time, so there is no reason to walk the rest of the upcoming board. This is
    what keeps a full day of props to a handful of requests instead of dozens.

    `allow_markets` filters rows to exact market types. SharpAPI's `market`
    parameter matches by prefix, so asking for `moneyline` also returns
    `moneyline_3-way`; the explicit filter keeps the tree to what was requested.

    Cursors expire whenever SharpAPI rebuilds its store, so a long walk restarts
    from the first page on that error. Rows are keyed by identity, so a restart
    cannot double-count what was already collected.
    """
    if not _get_api_key():
        raise OddsAPIError("Missing API key.")

    collected: Dict[tuple, Dict[str, Any]] = {}  # insertion-ordered

    for attempt in range(max_restarts):
        try:
            _walk_odds(
                market=market, start_dt=start_dt, end_dt=end_dt, event_ids=event_ids,
                sportsbook=sportsbook, allow_markets=allow_markets, base_url=base_url,
                limit=limit, max_pages=max_pages, collected=collected,
            )
            break
        except _CursorExpired:
            logger.warning(
                f"SharpAPI pagination cursor expired; restarting walk "
                f"({attempt + 1}/{max_restarts}, {len(collected)} rows kept so far)"
            )

    return list(collected.values())


def _walk_odds(
    market: Optional[str],
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
    event_ids: Optional[List[str]],
    sportsbook: Optional[str],
    allow_markets: Optional[set],
    base_url: str,
    limit: int,
    max_pages: int,
    collected: Dict[tuple, Dict[str, Any]],
) -> None:
    """One pass of the cursor walk, accumulating rows into `collected`."""
    url = f"{base_url}/odds"
    cursor: Optional[str] = None

    for _ in range(max_pages):
        params: Dict[str, Any] = {"league": "nhl", "limit": limit}
        if market:
            params["market"] = market
        if event_ids:
            params["event_id"] = ",".join(event_ids)
        if sportsbook:
            params["sportsbook"] = sportsbook
        if cursor:
            params["cursor"] = cursor
        # Exclude in-game prices: every caller wants pregame lines.
        params["is_live"] = "false"

        data, hdrs = _request_with_retry("GET", url, params=params)
        _note_rate_limit(hdrs)

        page_rows = data.get("data") or []

        # Decide the window stop on the raw rows: /odds is ordered by event start
        # time, so once a row passes the end of the window nothing after it can
        # belong to this day. Doing this before the market filter keeps that
        # signal even when a whole page is filtered away.
        past_window = False
        if start_dt is not None and end_dt is not None:
            windowed = []
            for row in page_rows:
                start = _parse_ts(row.get("event_start_time"))
                if start is None:
                    windowed.append(row)
                elif start > end_dt:
                    past_window = True
                    break
                elif start >= start_dt:
                    windowed.append(row)
            page_rows = windowed

        if allow_markets is not None:
            page_rows = [r for r in page_rows if str(r.get("market_type") or "") in allow_markets]

        for row in page_rows:
            collected.setdefault(_row_key(row), row)

        if past_window:
            return

        pag = _pagination(data)
        cursor = pag.get("next_cursor")
        if not pag.get("has_more") or not cursor:
            return


def _book_title(row: Dict[str, Any]) -> str:
    """Human-readable sportsbook name; SharpAPI nests it under sportsbook_ref."""
    ref = row.get("sportsbook_ref") or {}
    if isinstance(ref, dict) and ref.get("label"):
        return str(ref["label"])
    return str(row.get("sportsbook") or "").replace("_", " ").title()


def _internal_market(market_type: Any) -> str:
    """Map a SharpAPI market_type to the app-facing market key."""
    mk = str(market_type or "")
    return _TO_INTERNAL_MARKET.get(mk, mk)


def _rows_to_events(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fold flat SharpAPI rows into the nested event tree the app consumes.

    Outcome field names matter: `name` is the side ("Over"/"Under" for props,
    the team name for moneylines) and `description` carries the player, matching
    The Odds API's convention that NHL/PlayerLinePredictor.py reads.
    """
    events: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    for row in rows:
        # Suspended/closed markets keep a frozen price; do not show them as live.
        if row.get("is_active") is False:
            continue
        ev_id = row.get("event_id")
        home = row.get("home_team")
        away = row.get("away_team")
        if not ev_id or not home or not away:
            continue

        event = events.get(ev_id)
        if event is None:
            event = {
                "id": ev_id,
                "home_team": str(home),
                "away_team": str(away),
                "commence_time": row.get("event_start_time"),
                "bookmakers": [],
            }
            events[ev_id] = event
            order.append(ev_id)

        book_key = str(row.get("sportsbook") or "")
        book = next((b for b in event["bookmakers"] if b["key"] == book_key), None)
        if book is None:
            book = {"key": book_key, "title": _book_title(row), "markets": []}
            event["bookmakers"].append(book)

        market_key = _internal_market(row.get("market_type"))
        market = next((m for m in book["markets"] if m["key"] == market_key), None)
        if market is None:
            market = {"key": market_key, "outcomes": [], "last_update": row.get("timestamp")}
            book["markets"].append(market)

        selection = str(row.get("selection") or "").strip()
        point = _as_float(row.get("line"))

        # "Does X happen" markets (anytime goal scorer) carry the player's own
        # name as the selection and no line, and quote only the yes side. Present
        # them as an Over on a 0.5 line so the Over/Under machinery can price them.
        if market_key == "anytime_goal_scorer":
            selection = "Over"
            point = 0.5

        outcome: Dict[str, Any] = {
            "name": selection,
            "price": _as_int(row.get("odds_american")),
            "point": point,
        }
        # Books quote a two-sided main line plus an Over-only alternate ladder
        # ("2+ points", "3+ points"). Downstream code needs to tell them apart:
        # an alternate has no Under, so it can only ever be an Over bet.
        if row.get("is_main_line") is not None:
            outcome["is_main_line"] = bool(row.get("is_main_line"))
        if row.get("player_name"):
            outcome["description"] = str(row["player_name"])
        market["outcomes"].append(outcome)

    return [events[ev_id] for ev_id in order]


def _sharpapi_markets(markets: Optional[List[str]], default: List[str]) -> List[str]:
    """Translate app-facing market names to SharpAPI's market_type names."""
    names = markets or default
    return sorted({_TO_SHARPAPI_MARKET.get(m, m) for m in names})


def _fetch_markets(day: _date, markets: Optional[List[str]], default: List[str],
                   bookmakers_csv: Optional[str], base_url: str) -> List[Dict[str, Any]]:
    """Fetch one game day's rows for the given markets, filtered to exactly those."""
    wanted = _sharpapi_markets(markets, default)
    start_dt, end_dt = _day_window(day)
    rows = _fetch_odds_rows(
        market=",".join(wanted),
        start_dt=start_dt,
        end_dt=end_dt,
        sportsbook=bookmakers_csv,
        allow_markets=set(wanted),
        base_url=base_url,
    )
    return _rows_to_events(rows)


def fetch_nhl_odds_by_date(
    day: _date,
    regions: str,
    markets: List[str],
    bookmakers_csv: Optional[str] = None,
    odds_format: str = "american",
    base_url: str = DEFAULT_BASE_URL,
) -> List[Dict[str, Any]]:
    """
    Featured markets (moneyline / puck line / totals) for one game day.

    `regions` and `odds_format` are accepted for signature compatibility:
    SharpAPI has no regions concept (books are tier-determined) and returns
    American, decimal and implied probability on every row at once.
    """
    return _fetch_markets(day, markets, ["h2h", "spreads", "totals"], bookmakers_csv, base_url)


def fetch_nhl_events_by_date(
    day: _date,
    base_url: str = DEFAULT_BASE_URL,
) -> List[Dict[str, Any]]:
    """
    List the day's NHL games without odds.

    Derived from the moneyline board rather than /events, which is polluted with
    futures and outright markets that have no teams attached.
    """
    start_dt, end_dt = _day_window(day)
    rows = _fetch_odds_rows(
        market="moneyline",
        start_dt=start_dt,
        end_dt=end_dt,
        allow_markets={"moneyline"},
        base_url=base_url,
    )
    return [
        {
            "id": ev["id"],
            "home_team": ev["home_team"],
            "away_team": ev["away_team"],
            "commence_time": ev["commence_time"],
        }
        for ev in _rows_to_events(rows)
    ]


def fetch_event_player_odds(
    event_id: str,
    regions: str,
    markets: List[str],
    bookmakers_csv: Optional[str] = None,
    odds_format: str = "american",
    base_url: str = DEFAULT_BASE_URL,
) -> Optional[Dict[str, Any]]:
    """Player markets for a single event. Returns None when the event has none."""
    wanted = _sharpapi_markets(markets, [])
    rows = _fetch_odds_rows(
        market=",".join(wanted) if wanted else None,
        event_ids=[event_id],
        sportsbook=bookmakers_csv,
        allow_markets=set(wanted) if wanted else None,
        base_url=base_url,
    )
    events = _rows_to_events(rows)
    return events[0] if events else None


def fetch_nhl_player_props_by_date(
    day: _date,
    regions: str,
    markets: List[str],
    bookmakers_csv: Optional[str] = None,
    odds_format: str = "american",
    base_url: str = DEFAULT_BASE_URL,
) -> List[Dict[str, Any]]:
    """
    All player prop markets for every NHL event on a given date.

    One paged sweep of /odds, filtered to the day and stopped early once the
    results move past it, rather than a request per event as The Odds API
    required. Paging is throttled to the free tier's 12 requests per minute.
    """
    return _fetch_markets(day, markets, [], bookmakers_csv, base_url)
