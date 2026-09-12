"""
Daily NHL odds update script.

Fetches featured NHL odds (moneyline, puck line, totals) from The Odds API
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
from datetime import date as _date, timedelta

from NHL.BettingEdge import (
    fetch_and_cache_odds,
    compute_and_cache_edges,
    OddsAPIError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _update_one(game_date: _date) -> int:
    """Fetch odds and compute edges for a single date. Returns 0 on success."""
    logger.info(f"Fetching NHL odds for {game_date}...")
    payload = None
    try:
        payload = fetch_and_cache_odds(game_date)
    except OddsAPIError as e:
        err_msg = str(e).lower()
        if "missing api key" in err_msg:
            # Do NOT fabricate edges from the demo fixture in the scheduled
            # production update — leave the cache untouched so the app reports
            # "no live odds" instead of demo value bets.
            logger.warning(f"No Odds API key configured; skipping edge computation for {game_date}.")
            return 0
        else:
            logger.error(f"Odds API error: {e}")
            return 1
    except Exception as e:
        logger.error(f"Unexpected error fetching odds: {e}")
        return 1

    if payload and payload.get("source") == "the-odds-api":
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
        today = _date.today()
        for offset in range(31):
            _update_one(today + timedelta(days=offset))
        return 0

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
