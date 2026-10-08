#!/usr/bin/env python3
"""
Refresh shotpropz.json (goals / SOG allowed by position, per team, home/away).

Usage:
    python update_shotpropz.py
    python update_shotpropz.py --dry-run       # print, do not write
    python update_shotpropz.py --rate-limit 1.0
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from NHL.ShotPropzScraper import scrape_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("shotpropz.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Update shotpropz.json from shotpropz.com")
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing file")
    parser.add_argument("--rate-limit", type=float, default=0.75, help="Seconds between requests")
    args = parser.parse_args()

    payload = scrape_all(rate_limit_seconds=args.rate_limit)

    rows = 0
    for metric in ("goals_against", "sog_against"):
        for positions in (payload.get(metric) or {}).values():
            for bucket in (positions or {}).values():
                rows += len(bucket)

    if not rows:
        logger.warning("No shotpropz data scraped.")

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"Wrote {rows} team-position rows to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
