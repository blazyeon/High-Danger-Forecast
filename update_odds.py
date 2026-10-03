"""
Daily NHL odds update script.

Fetches featured NHL odds (moneyline, puck line, totals) from SharpAPI
and writes them to static/data/odds_cache.json so the web app can serve
edges without hitting the API on every page load.

Run:
    python update_odds.py
    python update_odds.py --date 2025-11-15
    python update_odds.py --all
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date as _date, datetime, timedelta, timezone

from NHL.BettingEdge import (
    DEFAULT_CACHE_PATH,
    DEFAULT_MARKETS,
    DEFAULT_REGIONS,
    fetch_and_cache_odds,
    compute_and_cache_edges,
    OddsAPIError,
)
from NHL.OddsAPI import fetch_nhl_odds_window, league_date_of_event
from NHL.Utils import atomic_write_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# How far past today the sweep pre-computes odds and edges.
HORIZON_DAYS = 30


def _update_one(game_date: _date) -> int:
    """Fetch odds and compute edges for a single date. Returns 0 on success."""
    logger.info(f"Fetching NHL odds for {game_date}...")
    payload = None
    try:
        payload = fetch_and_cache_odds(game_date)
    except OddsAPIError as e:
        err_msg = str(e).lower()
        if "missing api key" in err_msg:
            # Leave the cache untouched so the app reports "no live odds"
            # rather than serving edges computed from nothing.
            logger.warning(f"No Odds API key configured; skipping edge computation for {game_date}.")
            return 0
        else:
            logger.error(f"Odds API error: {e}")
            return 1
    except Exception as e:
        logger.error(f"Unexpected error fetching odds: {e}")
        return 1

    if payload and payload.get("source") == "sharpapi":
        logger.info(f"Successfully cached {len(payload.get('events', []))} events.")

    if not payload or not payload.get("events"):
        logger.info(f"No odds events for {game_date}; skipping edge computation.")
        return 0

    logger.info(f"Computing and caching betting edges for {game_date}...")
    try:
        edge_payload = compute_and_cache_edges(game_date, odds_payload=payload)
        logger.info(f"Cached {len(edge_payload.get('games', []))} edge games.")
    except Exception as e:
        logger.error(f"Failed to compute betting edges: {e}")
        # Odds are already cached; do not fail the whole update.

    return 0


def _update_all(horizon_days: int = HORIZON_DAYS) -> int:
    """
    Refresh odds + edges for today and the next ``horizon_days``.

    One windowed /odds walk, then bucket the events by league day. The old
    per-date loop re-walked the entire board once per date (quadratic, since a
    far-off day must page past every earlier game), which ran past the daily
    job's 45-minute timeout and cancelled the run before Today's Picks was ever
    computed or committed.
    """
    today = _date.today()
    end = today + timedelta(days=horizon_days)
    logger.info(f"Fetching NHL odds for {today}..{end} in one sweep...")
    try:
        events = fetch_nhl_odds_window(today, end, DEFAULT_REGIONS, list(DEFAULT_MARKETS))
    except OddsAPIError as e:
        if "missing api key" in str(e).lower():
            logger.warning("No Odds API key configured; skipping edge computation.")
            return 0
        logger.error(f"Odds API error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error fetching odds: {e}")
        return 1

    if not events:
        # Leave the existing caches untouched rather than overwrite with an
        # empty board, same as the single-date path.
        logger.info(f"No odds events for {today}..{end}; skipping edge computation.")
        return 0

    fetched_at = datetime.now(timezone.utc).isoformat()
    by_date: dict = {}
    for event in events:
        game_day = league_date_of_event(event.get("commence_time"))
        if game_day is None or not (today <= game_day <= end):
            continue
        by_date.setdefault(game_day, []).append(event)

    # Keep the single-date odds cache pointed at today: it is what the app and
    # update_todays_picks.py read on their fast path.
    if by_date.get(today):
        atomic_write_json(
            DEFAULT_CACHE_PATH,
            {
                "date": today.isoformat(),
                "fetched_at": fetched_at,
                "source": "sharpapi",
                "events": by_date[today],
            },
            indent=2,
        )

    for game_day in sorted(by_date):
        logger.info(f"Computing and caching betting edges for {game_day}...")
        try:
            edge_payload = compute_and_cache_edges(
                game_day,
                odds_payload={
                    "events": by_date[game_day],
                    "source": "sharpapi",
                    "fetched_at": fetched_at,
                },
            )
            logger.info(
                f"Cached {len(edge_payload.get('games', []))} edge games for {game_day}."
            )
        except Exception as e:
            # One bad date must not abort the rest of the horizon.
            logger.error(f"Failed to compute betting edges for {game_day}: {e}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Update cached NHL odds")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date to fetch odds for (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch odds and compute edges for every date with odds over the next 30 days.",
    )
    args = parser.parse_args()

    if args.all:
        return _update_all()

    if args.date:
        try:
            game_date = _date.fromisoformat(args.date)
        except ValueError:
            logger.error(f"Invalid date format: {args.date}. Expected YYYY-MM-DD.")
            return 1
    else:
        game_date = _date.today()

    return _update_one(game_date)


if __name__ == "__main__":
    raise SystemExit(main())
