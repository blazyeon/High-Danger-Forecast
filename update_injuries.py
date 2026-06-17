#!/usr/bin/env python3
"""
Refresh injuries.json from Daily Faceoff line-combination pages.

Usage:
    python update_injuries.py
    python update_injuries.py --team BUF      # refresh a single team only
    python update_injuries.py --dry-run       # print, do not write
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from NHL.InjuryScraper import scrape_all_injuries, scrape_team_injuries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("injuries.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Update injuries.json from Daily Faceoff")
    parser.add_argument("--team", help="Update a single team abbreviation only")
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing file")
    parser.add_argument("--rate-limit", type=float, default=0.75, help="Seconds between team requests")
    args = parser.parse_args()

    if args.team:
        injuries = scrape_team_injuries(args.team.upper())
    else:
        injuries = scrape_all_injuries(rate_limit_seconds=args.rate_limit)

    if not injuries:
        logger.warning("No injuries found.")

    if args.dry_run:
        print(json.dumps(injuries, indent=2))
        return 0

    OUTPUT_PATH.write_text(json.dumps(injuries, indent=2), encoding="utf-8")
    logger.info(f"Wrote {len(injuries)} injuries to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
