"""
Daily Today's Picks pre-computation script.

Runs the full slate of game simulations, betting edges, and player props for a
date and writes them to static/data/todays_picks_cache.json so the "Today's
Picks" tab opens instantly without re-running a simulation on every click.

Run:
    python update_todays_picks.py
    python update_todays_picks.py --date 2025-11-15
    python update_todays_picks.py --all
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date as _date, timedelta

from NHL.TodaysPicks import compute_and_cache_todays_picks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _update_one(game_date: _date) -> int:
    """Pre-compute picks for a single date. Returns 0 on success."""
    logger.info(f"Pre-computing Today's Picks for {game_date}...")
    try:
        payload = compute_and_cache_todays_picks(game_date)
        logger.info(
            f"Cached {len(payload.get('games', []))} games and "
            f"{len(payload.get('props', []))} props for {game_date}."
        )
    except Exception as e:
        logger.error(f"Failed to pre-compute Today's Picks for {game_date}: {e}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-compute Today's Picks cache")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date to pre-compute picks for (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Pre-compute picks for every date over the next 30 days.",
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
