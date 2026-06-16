"""
Tune team Elo parameters using walk-forward validation.

Searches over K-factor, xGF%/GF%/SF weights, and regression factor.
Objective: minimize log-loss of pre-game Elo win probabilities.

Run:
    python tune_elo.py
    python tune_elo.py --seasons 20242025 --n-samples 40
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import sys

sys.path.insert(0, str(Path(__file__).parent))

from EloMl.Database import EloDatabase
from EloMl.Ratings import EloConfig, TeamEloSystem
from NHL.Config import CURRENT_SEASON_YEAR, EARLIEST_SEASON_YEAR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _available_databases(root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for y in range(EARLIEST_SEASON_YEAR, CURRENT_SEASON_YEAR + 1):
        key = f"{y}{y + 1}"
        path = root / f"training_data_{key}.db"
        if path.exists():
            out[key] = path
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Elo parameters")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=str,
        default=None,
        help="Seasons to tune on (default: all available)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Number of random config samples to try",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="elo_tuning_results.json",
        help="Where to write tuning results",
    )
    return parser.parse_args()


def load_games(db_path: Path, season: str) -> List[Dict[str, Any]]:
    db = EloDatabase(str(db_path))
    cursor = db.conn.cursor()
    cursor.execute(
        """
        SELECT
            game_id,
            game_date,
            home_team,
            away_team,
            home_score,
            away_score,
            home_xgf,
            away_xgf,
            home_xga,
            away_xga,
            home_sf,
            away_sf,
            is_ot_so
        FROM game_results
        WHERE season = ?
        ORDER BY game_date ASC, game_id ASC
        """,
        (season,),
    )
    games = []
    for row in cursor.fetchall():
        games.append(
            {
                "game_id": row[0],
                "game_date": pd.to_datetime(row[1]).date(),
                "home_team": row[2],
                "away_team": row[3],
                "home_score": row[4],
                "away_score": row[5],
                "home_xgf": float(row[6] or 0.0),
                "away_xgf": float(row[7] or 0.0),
                "home_xga": float(row[8] or 0.0),
                "away_xga": float(row[9] or 0.0),
                "home_sf": int(row[10] or 30),
                "away_sf": int(row[11] or 30),
                "is_ot_so": bool(row[12]) if row[12] is not None else False,
            }
        )
    db.close()
    return games


def evaluate_config(games: List[Dict[str, Any]], config: EloConfig) -> float:
    """Walk forward, return log-loss of Elo predictions."""
    team_elo = TeamEloSystem(config)
    probs = []
    outcomes = []

    # Skip first 10 games of each team to avoid cold-start noise
    team_games_seen: Dict[str, int] = {}

    for game in games:
        home_team = game["home_team"]
        away_team = game["away_team"]

        home_elo = team_elo.get_team_rating(home_team)
        away_elo = team_elo.get_team_rating(away_team)
        prob = float(np.clip(1.0 / (1.0 + 10.0 ** (-(home_elo - away_elo) / 400.0)), 1e-6, 1 - 1e-6))

        if team_games_seen.get(home_team, 0) >= 5 and team_games_seen.get(away_team, 0) >= 5:
            probs.append(prob)
            outcomes.append(1.0 if game["home_score"] > game["away_score"] else 0.0)

        # Update Elo
        home_obj = team_elo.get_or_create_team(home_team)
        away_obj = team_elo.get_or_create_team(away_team)
        is_ot_so = game.get("is_ot_so", False)
        if game["home_score"] > game["away_score"]:
            home_result, away_result = 1.0, 0.25 if is_ot_so else 0.0
        elif game["away_score"] > game["home_score"]:
            home_result, away_result = 0.25 if is_ot_so else 0.0, 1.0
        else:
            home_result, away_result = 0.5, 0.5

        home_obj.update(
            opponent_rating=away_obj.rating,
            team_gf=game["home_score"],
            team_ga=game["away_score"],
            team_xgf=game["home_xgf"],
            team_xga=game["away_xgf"],
            team_sf=game["home_sf"],
            team_sa=game["away_sf"],
            result=home_result,
            config=config,
        )
        away_obj.update(
            opponent_rating=home_obj.rating,
            team_gf=game["away_score"],
            team_ga=game["home_score"],
            team_xgf=game["away_xgf"],
            team_xga=game["home_xgf"],
            team_sf=game["away_sf"],
            team_sa=game["home_sf"],
            result=away_result,
            config=config,
        )

        team_games_seen[home_team] = team_games_seen.get(home_team, 0) + 1
        team_games_seen[away_team] = team_games_seen.get(away_team, 0) + 1

    if not probs:
        return float("inf")
    return float(log_loss(np.array(outcomes), np.array(probs)))


def sample_config(rng: np.random.Generator) -> EloConfig:
    """Sample a random Elo configuration."""
    base = EloConfig()
    return EloConfig(
        k_factor_team=float(rng.uniform(8.0, 40.0)),
        k_factor_player=base.k_factor_player,
        k_factor_goalie=base.k_factor_goalie,
        regression_factor=float(rng.uniform(0.0, 0.08)),
        xgf_pct_weight=float(rng.uniform(0.0, 2.0)),
        gf_weight=float(rng.uniform(0.0, 2.0)),
        shot_attempt_weight=float(rng.uniform(0.0, 1.0)),
        pp_pk_weight=base.pp_pk_weight,
        min_rating=base.min_rating,
        max_rating=base.max_rating,
    )


def config_to_dict(config: EloConfig) -> Dict[str, Any]:
    return {
        "k_factor_team": config.k_factor_team,
        "regression_factor": config.regression_factor,
        "xgf_pct_weight": config.xgf_pct_weight,
        "gf_weight": config.gf_weight,
        "shot_attempt_weight": config.shot_attempt_weight,
    }


def run_tuning(args: argparse.Namespace) -> int:
    root = Path(__file__).parent
    db_map = _available_databases(root)

    if args.seasons:
        db_map = {k: v for k, v in db_map.items() if k in args.seasons}

    if not db_map:
        logger.error("No training databases found.")
        return 1

    all_games: List[Dict[str, Any]] = []
    for season, db_path in sorted(db_map.items()):
        games = load_games(db_path, season)
        logger.info(f"Loaded {len(games)} games from {db_path.name}")
        all_games.extend(games)

    all_games.sort(key=lambda g: (g["game_date"], g["game_id"]))
    logger.info(f"Total games for tuning: {len(all_games)}")

    rng = np.random.default_rng(42)
    results = []
    best_ll = float("inf")
    best_cfg = EloConfig()

    # Always evaluate the default config first as a baseline
    baseline_ll = evaluate_config(all_games, EloConfig())
    logger.info(f"Baseline Elo logloss: {baseline_ll:.4f}")
    results.append(
        {
            "config": config_to_dict(EloConfig()),
            "logloss": baseline_ll,
            "is_default": True,
        }
    )

    for i in range(args.n_samples):
        cfg = sample_config(rng)
        ll = evaluate_config(all_games, cfg)
        logger.info(
            f"[{i + 1}/{args.n_samples}] logloss={ll:.4f}  "
            f"K={cfg.k_factor_team:.1f} reg={cfg.regression_factor:.3f} "
            f"xgf={cfg.xgf_pct_weight:.2f} gf={cfg.gf_weight:.2f} sf={cfg.shot_attempt_weight:.2f}"
        )
        results.append({"config": config_to_dict(cfg), "logloss": ll})
        if ll < best_ll:
            best_ll = ll
            best_cfg = cfg

    results.sort(key=lambda x: x["logloss"])

    output = {
        "n_games": len(all_games),
        "seasons": sorted(db_map.keys()),
        "baseline_logloss": baseline_ll,
        "best_logloss": best_ll,
        "improvement": round(baseline_ll - best_ll, 5),
        "best_config": config_to_dict(best_cfg),
        "all_results": results[:20],  # top 20
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info("=" * 60)
    logger.info("TUNING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Baseline logloss: {baseline_ll:.4f}")
    logger.info(f"Best logloss:     {best_ll:.4f}  ({baseline_ll - best_ll:+.4f})")
    logger.info(f"Best config: {output['best_config']}")
    logger.info(f"Results written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(run_tuning(parse_args()))
