"""
Daily NHL odds update script.

Fetches featured NHL odds (moneyline, puck line, totals) from The Odds API
and writes them to static/data/odds_cache.json so the web app can serve
edges without hitting the API on every page load.

Run:
    python update_odds.py
    python update_odds.py --date 2025-11-15
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date as _date

from NHL.BettingEdge import (
    fetch_and_cache_odds,
    compute_and_cache_edges,
    load_demo_odds,
    OddsAPIError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Update cached NHL odds")
    parser.add_argument(
        "--date",
        type=str,
        default=_date.today().isoformat(),
        help="Date to fetch odds for (YYYY-MM-DD). Defaults to today.",
    )
    args = parser.parse_args()

    try:
        game_date = _date.fromisoformat(args.date)
    except ValueError:
        logger.error(f"Invalid date format: {args.date}. Expected YYYY-MM-DD.")
        return 1

    logger.info(f"Fetching NHL odds for {game_date}...")
    payload = None
    try:
        payload = fetch_and_cache_odds(game_date)
    except OddsAPIError as e:
        err_msg = str(e).lower()
        if "missing api key" in err_msg:
            logger.warning(f"No Odds API key configured; falling back to demo odds for {game_date}.")
            payload = load_demo_odds()
        else:
            logger.error(f"Odds API error: {e}")
            return 1
    except Exception as e:
        logger.error(f"Unexpected error fetching odds: {e}")
        return 1

    if payload and payload.get("source") == "the-odds-api":
        logger.info(f"Successfully cached {len(payload.get('events', []))} events.")

    logger.info(f"Computing and caching betting edges for {game_date}...")
    try:
        edge_payload = compute_and_cache_edges(game_date, odds_payload=payload)
        logger.info(f"Cached {len(edge_payload.get('games', []))} edge games.")
    except Exception as e:
        logger.error(f"Failed to compute betting edges: {e}")
        # Odds are already cached; do not fail the whole update.

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
