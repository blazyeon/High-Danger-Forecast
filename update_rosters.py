#!/usr/bin/env python3
"""
Refresh rosters.json and rookie_projections.json.

Fetches current rosters (skaters + goalies, with player IDs) for all 32 teams
from the NHL API, and builds rookie point projections (non-NHL stats proxy) for
first-year players. Run daily; wired into daily_update.bat.

Usage:
    python update_rosters.py
    python update_rosters.py --dry-run
    python update_rosters.py --team MTL
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date

from NHL.Rosters import (
    build_rookie_projections,
    fetch_all_rosters,
    fetch_team_roster,
    save_rookie_projections,
    save_rosters,
)
from NHL.Utils import season_from_date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Update rosters + rookie projections")
    parser.add_argument("--team", help="Update a single team abbreviation only")
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing files")
    parser.add_argument("--rate-limit", type=float, default=0.35, help="Seconds between requests")
    args = parser.parse_args()

    season_start = int(season_from_date(date.today().isoformat())[:4])

    if args.team:
        rosters = {args.team.upper(): fetch_team_roster(args.team.upper())}
    else:
        rosters = fetch_all_rosters(rate_limit=args.rate_limit)

    projections = build_rookie_projections(rosters, season_start, rate_limit=args.rate_limit)

    if args.dry_run:
        print(json.dumps({"rosters": rosters, "rookie_projections": projections}, indent=2))
        return 0

    save_rosters(rosters)
    save_rookie_projections(projections)
    logger.info(
        f"Wrote rosters for {len(rosters)} teams and "
        f"{len(projections)} rookie projections."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
