#!/usr/bin/env python3
"""
Refresh lineups.json (projected power-play units) from Daily Faceoff.

Usage:
    python update_lineups.py
    python update_lineups.py --team TOR      # refresh a single team only
    python update_lineups.py --dry-run       # print, do not write
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from NHL.LineupScraper import scrape_all_pp_units, scrape_team_pp_units

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("lineups.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Update lineups.json from Daily Faceoff")
    parser.add_argument("--team", help="Update a single team abbreviation only")
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing file")
    parser.add_argument("--rate-limit", type=float, default=0.75, help="Seconds between team requests")
    args = parser.parse_args()

    if args.team:
        team = scrape_team_pp_units(args.team.upper())
        payload = {"source": "daily_faceoff", "updated_at": None, "teams": {}}
        if team:
            payload["teams"][args.team.upper()] = team
            payload["updated_at"] = team.get("updated_at")
    else:
        payload = scrape_all_pp_units(rate_limit_seconds=args.rate_limit)

    n_teams = len(payload.get("teams", {}))
    if not n_teams:
        logger.warning("No PP units scraped.")

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"Wrote PP units for {n_teams} teams to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
