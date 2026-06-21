
#!/usr/bin/env python3
"""
NHL API + MoneyPuck stats refresher — exports JSON for the frontend.

This replaces `update_nst_stats.py`. The pipeline is:

  NHL API PBP (PlayByPlay.py)  →  raw JSON cache + shot parquet
  MoneyPuck CSVs (MoneyPuck.py) →  validation cross-check
  xG model (xGModel.py)        →  trained on the shot parquet
  Stats aggregator (StatsFromPBP.py)  →  team / skater / goalie rates
                                          written to static/data/

Usage:
    python update_pbp_stats.py --season 2024 --stype 2
    python update_pbp_stats.py --season 2024 --train-xg
    python update_pbp_stats.py --seasons 2023 2024 --out static/data

Outputs (to --out, default static/data/):
    pbp_team_stats.json
    pbp_skater_stats.json
    pbp_goalie_stats.json
    pbp_shot_store_2024_2.parquet  (written by PlayByPlay.build_shot_store)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Make sure project root is on the import path when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from NHL.PlayByPlay import build_shot_store, load_shot_store
from NHL.StatsFromPBP import (
    compute_team_rates,
    compute_skater_rates,
    compute_goalie_rates,
    compute_season_skater_stats,
    compute_season_goalie_stats,
    merge_xg_into_skater_stats,
    merge_xg_into_goalie_stats,
)
from NHL.MoneyPuck import download_shots_zip, MP_CACHE_DIR, parse_mp_shots
from NHL.Validation import validate_xg_against_money_puck

logger = logging.getLogger("update_pbp_stats")


def _df_to_records(df: pd.DataFrame) -> List[Dict]:
    """Convert a DataFrame to JSON-safe records (NaN → None)."""
    if df is None or df.empty:
        return []
    # Replace NaN/Inf with None for JSON compatibility
    return json.loads(df.where(pd.notna(df), None).to_json(orient="records"))


def _write_json(out_path: Path, payload: Dict) -> None:
    """Write a JSON payload with a stable key order and a freshness timestamp."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {**payload, "updated_at": ts}
    out_path.write_text(json.dumps(payload, indent=2))


def _current_season_start_year() -> int:
    """Return the start year of the current NHL season (Oct cutoff)."""
    today = pd.Timestamp.utcnow()
    return today.year if today.month >= 10 else today.year - 1


def write_season_outputs(
    season_year: int,
    stype: int,
    out_dir: Path,
) -> Dict[str, int]:
    """
    Compute team/skater/goalie rates for one season and write JSON.

    Per-season files are always written. Generic files (pbp_*_stats.json) are
    only overwritten when we are processing the current in-progress season,
    so a bulk historical refresh does not clobber the prediction cache.

    Returns counts (teams, skaters, goalies) for the run log.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    season_str = f"{season_year}{season_year + 1}"
    meta = {"season": season_str, "stype": stype}
    is_current = season_year == _current_season_start_year()

    logger.info(f"Computing team rates for {season_str} (stype={stype})")
    team_df = compute_team_rates(season_year, stype)
    team_records = _df_to_records(team_df)
    team_payload = {**meta, "data": team_records}
    if is_current:
        _write_json(out_dir / "pbp_team_stats.json", team_payload)
    _write_json(out_dir / f"pbp_team_stats_{season_str}.json", team_payload)

    logger.info(f"Computing skater stats for {season_str} (stype={stype})")
    skater_records = compute_season_skater_stats(season_year, stype)
    skater_records = merge_xg_into_skater_stats(skater_records, season_year, stype)
    skater_payload = {**meta, "data": skater_records}
    if is_current:
        # Prediction path prefers the PBP-derived rates for the live season.
        try:
            pbp_skater_rates = compute_skater_rates(season_year, stype)
            current_skater_records = [
                {
                    "name": d.get("name", ""),
                    "gp": d.get("gp", 0),
                    "goals": d.get("goals", 0),
                    "assists": d.get("assists", 0),
                    "points": d.get("goals", 0) + d.get("assists", 0),
                    "shots": d.get("shots", 0),
                    "gpg": d.get("gpg", 0.0),
                    "apg": d.get("apg", 0.0),
                    "sogpg": d.get("sogpg", 0.0),
                    "xgf_pg": d.get("xgf_pg", 0.0),
                }
                for d in pbp_skater_rates.values()
            ]
            _write_json(out_dir / "pbp_skater_stats.json", {**meta, "data": current_skater_records})
        except Exception as e:
            logger.warning(f"Could not build PBP skater rates for current season: {e}")
            _write_json(out_dir / "pbp_skater_stats.json", skater_payload)
    _write_json(out_dir / f"pbp_skater_stats_{season_str}.json", skater_payload)

    logger.info(f"Computing goalie stats for {season_str} (stype={stype})")
    goalie_records = compute_season_goalie_stats(season_year, stype)
    goalie_records = merge_xg_into_goalie_stats(goalie_records, season_year, stype)
    goalie_payload = {**meta, "data": goalie_records}
    if is_current:
        try:
            pbp_goalie_df = compute_goalie_rates(season_year, stype)
            _write_json(out_dir / "pbp_goalie_stats.json", {**meta, "data": _df_to_records(pbp_goalie_df)})
        except Exception as e:
            logger.warning(f"Could not build PBP goalie rates for current season: {e}")
            _write_json(out_dir / "pbp_goalie_stats.json", goalie_payload)
    _write_json(out_dir / f"pbp_goalie_stats_{season_str}.json", goalie_payload)

    return {
        "teams": len(team_records),
        "skaters": len(skater_records),
        "goalies": len(goalie_records),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Single start year to refresh (e.g., 2024 for the 2024-25 season)",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=None,
        help="Multiple start years to refresh",
    )
    parser.add_argument(
        "--stype",
        type=int,
        default=2,
        help="Season type: 2=regular, 3=playoffs (default 2)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="static/data",
        help="Output directory (default static/data)",
    )
    parser.add_argument(
        "--train-xg",
        action="store_true",
        help="Re-train the xG model on the latest season's shots",
    )
    parser.add_argument(
        "--skip-mp",
        action="store_true",
        help="Skip MoneyPuck download and validation",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-fetch PBP JSONs even if cached (slow)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging (DEBUG)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Default season: current season (Oct → next June → use that)
    today = pd.Timestamp.utcnow()
    if today.month >= 10:
        default_year = today.year
    else:
        default_year = today.year - 1
    seasons = args.seasons or ([args.season] if args.season else [default_year])
    out_dir = Path(args.out)

    t0 = time.time()
    for season_year in seasons:
        logger.info(f"=== Season {season_year}-{season_year + 1} (stype={args.stype}) ===")
        # 1. PBP fetch + shot store build
        try:
            build_shot_store(
                season_year,
                args.stype,
                force_refresh=args.force_refresh,
            )
        except Exception as e:
            logger.error(f"PBP build failed for {season_year}: {e}")
            continue

        # 2. Write frontend JSON
        try:
            counts = write_season_outputs(season_year, args.stype, out_dir)
            logger.info(
                f"Wrote {counts['teams']} teams, {counts['skaters']} skaters, "
                f"{counts['goalies']} goalies → {out_dir}"
            )
        except Exception as e:
            logger.error(f"Output write failed for {season_year}: {e}")
            continue

    # 3. Optional: retrain xG model on the most recent season
    if args.train_xg:
        from NHL.xGModel import train_xg_model, load_xg_model, REPORT_PATH
        latest = seasons[-1]
        logger.info(f"Training xG model on {latest}-{latest + 1} shots...")
        shots = load_shot_store(latest, args.stype)
        if shots.empty:
            logger.warning("No shots available, skipping xG training")
        else:
            try:
                # Filter to regular season rows for the target year only
                report = train_xg_model(shots)
                logger.info(
                    f"xG model: val AUC={report['val']['auc']:.3f} "
                    f"brier={report['val']['brier']:.4f} → {REPORT_PATH}"
                )
            except Exception as e:
                logger.error(f"xG training failed: {e}")

    # 4. MoneyPuck validation cross-check (unless skipped)
    if not args.skip_mp:
        latest = seasons[-1]
        logger.info(f"Validating xG against MoneyPuck for {latest}-{latest + 1}...")
        try:
            download_shots_zip([latest])
            report = validate_xg_against_money_puck(season_year=latest, stype=args.stype)
            if "error" in report:
                logger.warning(f"Validation: {report['error']}")
            else:
                logger.info(
                    f"Validation: corr={report.get('correlation_ours_vs_mp', 0):.3f}, "
                    f"AUC ours={report.get('auc_ours', 0):.3f} "
                    f"vs MP={report.get('auc_mp', 0):.3f} → "
                    f"{report.get('verdict', '?')}"
                )
        except Exception as e:
            logger.warning(f"Validation step failed (non-fatal): {e}")

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
