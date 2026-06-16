"""
Backtest the prediction pipeline on historical games.

Evaluates three signals:
1. Elo-only win probability
2. ML-only win probability
3. Simple ensemble blend

Run:
    python backtest.py
    python backtest.py --seasons 20232024 20242025 --model models/main_model.pkl
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

import sys

sys.path.insert(0, str(Path(__file__).parent))

from EloMl.Database import EloDatabase
from EloMl.Features import EloFeatureEngine
from EloMl.MLModel import EloMLPredictor
from EloMl.Ratings import EloConfig, PlayerEloSystem, TeamEloSystem
from NHL.Config import CURRENT_SEASON_YEAR, EARLIEST_SEASON_YEAR, MODEL_WEIGHTS
from NHL.Features import compute_rest_travel_features_fast
from NHL.Utils import season_from_date

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
    parser = argparse.ArgumentParser(description="Backtest NHL prediction signals")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=str,
        default=None,
        help="Seasons to backtest (default: all available)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/main_model.pkl",
        help="Path to trained ML model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backtest_results.json",
        help="Where to write results",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=50,
        help="Minimum games required to run backtest",
    )
    parser.add_argument(
        "--use-oof",
        action="store_true",
        default=True,
        help="Use walk-forward out-of-fold ML probabilities from models/calibration_data.json (fair backtest)",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default="models/calibration_data.json",
        help="Path to calibration/OOF predictions JSON",
    )
    parser.add_argument(
        "--no-oof",
        action="store_true",
        help="Disable OOF and use the final trained model directly (may look ahead)",
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
            season,
            home_team,
            away_team,
            home_score,
            away_score,
            home_xgf,
            away_xgf,
            home_xga,
            away_xga,
            is_ot_so,
            home_sf,
            away_sf
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
                "season": row[2],
                "home_team": row[3],
                "away_team": row[4],
                "home_score": row[5],
                "away_score": row[6],
                "home_xgf": float(row[7] or 0.0),
                "away_xgf": float(row[8] or 0.0),
                "home_xga": float(row[9] or 0.0),
                "away_xga": float(row[10] or 0.0),
                "is_ot_so": bool(row[11]) if row[11] is not None else False,
                "home_sf": int(row[12] or 30),
                "away_sf": int(row[13] or 30),
            }
        )
    db.close()
    return games


def _venue_record(
    home_team: str,
    away_team: str,
    completed_games: List[Dict[str, Any]],
) -> Tuple[float, float]:
    home_games = [g for g in completed_games if g["home_team"] == home_team]
    away_games = [g for g in completed_games if g["away_team"] == away_team]

    def _pct(team: str, games_subset: List[Dict[str, Any]], is_home: bool) -> float:
        wins = ot = gp = 0
        for g in games_subset:
            gp += 1
            if is_home:
                if g["home_score"] > g["away_score"]:
                    wins += 1
                elif g["is_ot_so"] and g["home_score"] < g["away_score"]:
                    ot += 1
            else:
                if g["away_score"] > g["home_score"]:
                    wins += 1
                elif g["is_ot_so"] and g["away_score"] < g["home_score"]:
                    ot += 1
        if gp == 0:
            return 0.50
        raw = (wins + 0.5 * ot) / gp
        return 0.5 * raw + 0.25

    return _pct(home_team, home_games, True), _pct(away_team, away_games, False)


def _recent_stats(
    team: str, is_home: bool, n: int, completed_games: List[Dict[str, Any]]
) -> Dict[str, float]:
    relevant = [
        g
        for g in completed_games
        if (is_home and g["home_team"] == team) or (not is_home and g["away_team"] == team)
    ]
    relevant = relevant[-n:]
    if not relevant:
        return {
            "gf_pg": 3.0,
            "ga_pg": 3.0,
            "xgf_pg": 3.0,
            "xga_pg": 3.0,
            "sf_pg": 30.0,
            "sa_pg": 30.0,
        }
    gp = len(relevant)
    gf = sum(g["home_score"] if is_home else g["away_score"] for g in relevant)
    ga = sum(g["away_score"] if is_home else g["home_score"] for g in relevant)
    xgf = sum(g["home_xgf"] if is_home else g["away_xgf"] for g in relevant)
    xga = sum(g["home_xga"] if is_home else g["away_xga"] for g in relevant)
    sf = sum(g["home_sf"] if is_home else g["away_sf"] for g in relevant)
    sa = sum(g["away_sf"] if is_home else g["home_sf"] for g in relevant)
    return {
        "gf_pg": gf / gp,
        "ga_pg": ga / gp,
        "xgf_pg": xgf / gp,
        "xga_pg": xga / gp,
        "sf_pg": sf / gp,
        "sa_pg": sa / gp,
    }


def _load_oof_predictions(path: str) -> Dict[str, float]:
    """Load walk-forward out-of-fold ML probabilities by game_id."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        return {str(r["game_id"]): float(r["pred_home_win"]) for r in records if "game_id" in r and "pred_home_win" in r}
    except Exception as e:
        logger.warning(f"Could not load OOF predictions from {path}: {e}")
        return {}


def _h2h_pct(home_team: str, away_team: str, completed_games: List[Dict[str, Any]]) -> float:
    h2h = [
        g
        for g in completed_games
        if (g["home_team"] == home_team and g["away_team"] == away_team)
        or (g["home_team"] == away_team and g["away_team"] == home_team)
    ]
    wins = gp = 0
    for g in h2h[-5:]:
        gp += 1
        if g["home_team"] == home_team and g["home_score"] > g["away_score"]:
            wins += 1
        elif g["home_team"] == away_team and g["away_score"] > g["home_score"]:
            wins += 1
    return wins / gp if gp else 0.5


def _season_to_date_stats(
    team: str, is_home: bool, completed_games: List[Dict[str, Any]]
) -> Dict[str, float]:
    relevant = [
        g
        for g in completed_games
        if (is_home and g["home_team"] == team) or (not is_home and g["away_team"] == team)
    ]
    if not relevant:
        return {"gf_pg": 3.0, "ga_pg": 3.0, "xgf_pg": 3.0, "xga_pg": 3.0, "sf_pg": 30.0, "sa_pg": 30.0}
    gp = len(relevant)
    gf = sum(g["home_score"] if is_home else g["away_score"] for g in relevant)
    ga = sum(g["away_score"] if is_home else g["home_score"] for g in relevant)
    xgf = sum(g["home_xgf"] if is_home else g["away_xgf"] for g in relevant)
    xga = sum(g["home_xga"] if is_home else g["away_xga"] for g in relevant)
    sf = sum(g["home_sf"] if is_home else g["away_sf"] for g in relevant)
    return {
        "gf_pg": gf / gp,
        "ga_pg": ga / gp,
        "xgf_pg": xgf / gp,
        "xga_pg": xga / gp,
        "sf_pg": sf / gp,
        "sa_pg": 30.0,
    }


def _build_ml_features(
    game: Dict[str, Any],
    team_elo: TeamEloSystem,
    player_elo: PlayerEloSystem,
    feature_engine: EloFeatureEngine,
    completed: List[Dict[str, Any]],
) -> Dict[str, float]:
    home_team = game["home_team"]
    away_team = game["away_team"]
    game_date = game["game_date"]

    elo_features = feature_engine.extract_team_features(home_team, away_team)
    home_recent = _recent_stats(home_team, True, 10, completed)
    away_recent = _recent_stats(away_team, False, 10, completed)
    home_std = _season_to_date_stats(home_team, True, completed)
    away_std = _season_to_date_stats(away_team, False, completed)
    home_pct, away_pct = _venue_record(home_team, away_team, completed)

    try:
        home_rest, away_rest = compute_rest_travel_features_fast(
            home_team, away_team, game_date, completed
        ), compute_rest_travel_features_fast(
            away_team, home_team, game_date, completed
        )
    except Exception:
        home_rest = away_rest = {
            "is_b2b": False,
            "rest_days": 3.0,
            "opp_rest_days": 3.0,
            "travel_km": 0.0,
            "tz_diff": 0.0,
        }

    h2h_pct = _h2h_pct(home_team, away_team, completed)

    home_xgf_pg = game["home_xgf"] / max(game["home_sf"], 1) * 30 if game["home_sf"] else 0.0
    away_xgf_pg = game["away_xgf"] / max(game["away_sf"], 1) * 30 if game["away_sf"] else 0.0
    home_xga_pg = game["home_xga"] / max(game["home_sf"], 1) * 30 if game["home_sf"] else 0.0
    away_xga_pg = game["away_xga"] / max(game["away_sf"], 1) * 30 if game["away_sf"] else 0.0

    return {
        **elo_features,
        "home_elo": team_elo.get_team_rating(home_team),
        "away_elo": team_elo.get_team_rating(away_team),
        "home_season_xgf_pg": home_xgf_pg / 0.06 if home_xgf_pg else 0.0,
        "away_season_xgf_pg": away_xgf_pg / 0.06 if away_xgf_pg else 0.0,
        "home_xgf_norm": home_xgf_pg / 6.0 if home_xgf_pg else 0.0,
        "away_xgf_norm": away_xgf_pg / 6.0 if away_xgf_pg else 0.0,
        "home_xgf_share": home_xgf_pg / max(home_xgf_pg + away_xgf_pg, 0.1),
        "home_recent_xgf_pg": home_recent["xgf_pg"],
        "home_recent_xga_pg": home_recent["xga_pg"],
        "away_recent_xgf_pg": away_recent["xgf_pg"],
        "away_recent_xga_pg": away_recent["xga_pg"],
        "home_recent_gf_pg": home_recent["gf_pg"],
        "away_recent_gf_pg": away_recent["gf_pg"],
        "home_recent_form_off": home_recent["xgf_pg"] - 3.0,
        "away_recent_form_off": away_recent["xgf_pg"] - 3.0,
        "home_sf_pg": game["home_sf"] / 0.06 if game["home_sf"] else 0.0,
        "away_sf_pg": game["away_sf"] / 0.06 if game["away_sf"] else 0.0,
        "home_recent_sf_pg": home_recent["sf_pg"],
        "away_recent_sf_pg": away_recent["sf_pg"],
        "home_venue_pct": home_pct,
        "away_venue_pct": away_pct,
        "venue_diff": home_pct - away_pct,
        "home_rest_days": float(home_rest.get("rest_days", 3.0)),
        "away_rest_days": float(away_rest.get("rest_days", 3.0)),
        "rest_diff": float(away_rest.get("rest_days", 3.0)) - float(home_rest.get("rest_days", 3.0)),
        "home_b2b": 1.0 if home_rest.get("is_b2b") else 0.0,
        "away_b2b": 1.0 if away_rest.get("is_b2b") else 0.0,
        "home_travel_km": float(home_rest.get("travel_km", 0.0)),
        "away_travel_km": float(away_rest.get("travel_km", 0.0)),
        "home_tz_diff": float(home_rest.get("tz_diff", 0.0)),
        "away_tz_diff": float(away_rest.get("tz_diff", 0.0)),
        "h2h_home_pct": h2h_pct,

        "xgf_pct_diff": (
            (home_std["xgf_pg"] / max(home_std["xgf_pg"] + home_std["xga_pg"], 0.1) -
             away_std["xgf_pg"] / max(away_std["xgf_pg"] + away_std["xga_pg"], 0.1)) * 100.0
        ) / 50.0,
        "gf_pg_diff": home_std["gf_pg"] - away_std["gf_pg"],
        "ga_pg_diff": away_std["ga_pg"] - home_std["ga_pg"],
        "sf_pg_diff": home_std["sf_pg"] - away_std["sf_pg"],
        "xga_pg_diff": home_std["xga_pg"] - away_std["xga_pg"],
    }


def _clip(p: float) -> float:
    return float(np.clip(p, 1e-6, 1 - 1e-6))


def _report(name: str, y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    preds = (probs >= 0.5).astype(int)
    return {
        "name": name,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "logloss": float(log_loss(y_true, probs)),
        "brier": float(brier_score_loss(y_true, probs)),
        "auc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.5,
    }


def _calibration_buckets(y_true: np.ndarray, probs: np.ndarray) -> List[Dict[str, Any]]:
    probs = np.clip(probs, 0.0, 1.0)
    buckets = []
    for lo in np.arange(0.0, 1.0, 0.1):
        hi = lo + 0.1
        mask = (probs >= lo) & (probs < hi)
        n = int(np.sum(mask))
        if n:
            actual = float(np.mean(y_true[mask]))
            mean_pred = float(np.mean(probs[mask]))
            buckets.append(
                {
                    "range": f"{lo:.1f}-{hi:.1f}",
                    "n": n,
                    "actual": round(actual, 3),
                    "predicted": round(mean_pred, 3),
                    "error": round(mean_pred - actual, 3),
                }
            )
    return buckets


def _ensemble_prob(elo_p: float, sim_p: float, ml_p: Optional[float], weights: Dict[str, float]) -> float:
    """Blend Elo, simulation, and ML probabilities."""
    if ml_p is None:
        return _clip(elo_p)
    elo_w = weights.get("elo_winprob_weight", 0.30)
    sim_w = weights.get("simulation_winprob_weight", 0.45)
    ml_w = weights.get("ml_winprob_weight", 0.25)
    total = elo_w + sim_w + ml_w
    if total <= 0:
        return _clip(elo_p)
    return _clip((elo_w * elo_p + sim_w * sim_p + ml_w * ml_p) / total)


def _grid_search_weights(
    records: List[Dict[str, Any]],
    y_true: np.ndarray,
    steps: int = 21,
) -> Dict[str, Any]:
    """Grid-search ensemble weights (Elo/Sim/ML) using sim=Elo proxy in backtest.

    Only games with an ML probability participate in the search so the result is
    not swamped by Elo-only early-season games.
    """
    best_ll = float("inf")
    best = dict(MODEL_WEIGHTS)
    ml_mask = np.array([r["ml_prob"] is not None for r in records])
    if np.sum(ml_mask) == 0:
        return {"best": best, "best_logloss": best_ll, "history": []}
    filtered_records = [r for r in records if r["ml_prob"] is not None]
    y_true_f = y_true[ml_mask]
    history: List[Dict[str, float]] = []

    # Search 2-D simplex: vary Elo+Sim combined weight vs ML weight.
    # Simulation is proxied by Elo here because running 10k sims per game is too
    # expensive for a full-season backtest. The resulting "Elo+Sim" weight is
    # split back into the two Config weights in a fixed ratio.
    sim_share = 0.60  # of the combined Elo+Sim budget, simulation gets this share
    for ml_w_int in range(steps):
        ml_w = ml_w_int / (steps - 1)
        combined = 1.0 - ml_w
        elo_w = combined * (1.0 - sim_share)
        sim_w = combined * sim_share
        probs = np.array(
            [
                _ensemble_prob(
                    r["elo_prob"],
                    r["sim_prob"],
                    r["ml_prob"],
                    {
                        "elo_winprob_weight": elo_w,
                        "simulation_winprob_weight": sim_w,
                        "ml_winprob_weight": ml_w,
                    },
                )
                for r in filtered_records
            ],
            dtype=float,
        )
        ll = log_loss(y_true_f, np.clip(probs, 1e-6, 1 - 1e-6))
        history.append({"elo": round(elo_w, 3), "sim": round(sim_w, 3), "ml": round(ml_w, 3), "logloss": round(ll, 5)})
        if ll < best_ll:
            best_ll = ll
            best = {"elo_winprob_weight": elo_w, "simulation_winprob_weight": sim_w, "ml_winprob_weight": ml_w}

    return {"best": best, "best_logloss": best_ll, "history": history}


def run_backtest(args: argparse.Namespace) -> int:
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

    if len(all_games) < args.min_games:
        logger.error(f"Only {len(all_games)} games, need at least {args.min_games}")
        return 1

    config = EloConfig()
    player_elo = PlayerEloSystem(config)
    team_elo = TeamEloSystem(config)
    feature_engine = EloFeatureEngine(player_elo, team_elo, config)

    use_oof = args.use_oof and not args.no_oof
    oof_probs: Dict[str, float] = {}
    if use_oof:
        oof_probs = _load_oof_predictions(args.calibration)
        if oof_probs:
            logger.info(f"Loaded {len(oof_probs)} OOF ML probabilities from {args.calibration}")

    ml_model: Optional[EloMLPredictor] = None
    if not use_oof:
        model_path = Path(args.model)
        if model_path.exists():
            try:
                ml_model = EloMLPredictor(model_id="main", config=config)
                ml_model.load(str(model_path))
                logger.info(f"Loaded ML model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load ML model: {e}")

    records = []
    completed: List[Dict[str, Any]] = []

    for i, game in enumerate(all_games):
        if (i + 1) % 500 == 0:
            logger.info(f"Backtesting game {i + 1}/{len(all_games)}...")

        home_team = game["home_team"]
        away_team = game["away_team"]
        home_elo = team_elo.get_team_rating(home_team)
        away_elo = team_elo.get_team_rating(away_team)
        elo_prob = _clip(1.0 / (1.0 + 10.0 ** (-(home_elo - away_elo) / 400.0)))

        ml_prob = None
        if use_oof:
            ml_prob = oof_probs.get(str(game["game_id"]))
            if ml_prob is not None:
                ml_prob = _clip(ml_prob)
        elif ml_model is not None and ml_model.is_trained:
            try:
                features = _build_ml_features(
                    game, team_elo, player_elo, feature_engine, completed
                )
                ml_prob = _clip(ml_model.predict_proba(features))
            except Exception as e:
                logger.debug(f"ML prediction failed for {game['game_id']}: {e}")

        actual = 1.0 if game["home_score"] > game["away_score"] else 0.0

        records.append(
            {
                "game_id": game["game_id"],
                "game_date": game["game_date"].isoformat(),
                "season": game["season"],
                "home_team": home_team,
                "away_team": away_team,
                "home_score": game["home_score"],
                "away_score": game["away_score"],
                "actual_home_win": int(actual),
                "elo_prob": round(elo_prob, 4),
                "sim_prob": round(elo_prob, 4),  # backtest proxy; simulation too slow for full history
                "ml_prob": round(ml_prob, 4) if ml_prob is not None else None,
                "home_elo": round(home_elo, 1),
                "away_elo": round(away_elo, 1),
            }
        )

        # Update Elo AFTER recording the prediction
        try:
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
        except Exception as e:
            logger.warning(f"Elo update failed for {game['game_id']}: {e}")

        completed.append(game)

    y_true = np.array([r["actual_home_win"] for r in records], dtype=float)
    elo_probs = np.array([r["elo_prob"] for r in records], dtype=float)

    # Baseline ensemble using current Config weights (sim proxied by Elo)
    baseline_probs = np.array(
        [_ensemble_prob(r["elo_prob"], r["sim_prob"], r["ml_prob"], MODEL_WEIGHTS) for r in records],
        dtype=float,
    )

    results = {
        "n_games": len(records),
        "elo": _report("Elo", y_true, elo_probs),
        "ensemble_baseline": _report("Elo+Sim+ML (Config)", y_true, baseline_probs),
        "elo_calibration": _calibration_buckets(y_true, elo_probs),
        "ensemble_baseline_calibration": _calibration_buckets(y_true, baseline_probs),
        "seasons": sorted(db_map.keys()),
        "baseline_weights": {
            "elo_winprob_weight": MODEL_WEIGHTS["elo_winprob_weight"],
            "simulation_winprob_weight": MODEL_WEIGHTS["simulation_winprob_weight"],
            "ml_winprob_weight": MODEL_WEIGHTS["ml_winprob_weight"],
        },
    }

    ml_mask = np.array([r["ml_prob"] is not None for r in records])
    if np.sum(ml_mask) > 0:
        ml_probs = np.array(
            [r["ml_prob"] for r in records if r["ml_prob"] is not None],
            dtype=float,
        )
        results["ml"] = _report("ML", y_true[ml_mask], ml_probs)
        results["ml_calibration"] = _calibration_buckets(y_true[ml_mask], ml_probs)

    # Optimize ensemble weights
    logger.info("\n🔎 Optimizing ensemble weights...")
    grid = _grid_search_weights(records, y_true)
    opt_weights = grid["best"]
    opt_probs = np.array(
        [_ensemble_prob(r["elo_prob"], r["sim_prob"], r["ml_prob"], opt_weights) for r in records],
        dtype=float,
    )
    results["ensemble_optimized"] = _report("Elo+Sim+ML (Optimized)", y_true, opt_probs)
    results["ensemble_optimized_calibration"] = _calibration_buckets(y_true, opt_probs)
    results["optimized_weights"] = {
        "elo_winprob_weight": round(opt_weights["elo_winprob_weight"], 3),
        "simulation_winprob_weight": round(opt_weights["simulation_winprob_weight"], 3),
        "ml_winprob_weight": round(opt_weights["ml_winprob_weight"], 3),
    }
    results["weight_search_history"] = grid["history"]

    # Top teams by final Elo
    results["final_elo_rankings"] = [
        {"team": t, "rating": round(r, 1)} for t, r in team_elo.get_team_rankings()[:10]
    ]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    for key in ("elo", "ml", "ensemble_baseline", "ensemble_optimized"):
        if key in results:
            r = results[key]
            logger.info(
                f"{r['name']:20s}  Acc={r['accuracy']:.3f}  "
                f"LogLoss={r['logloss']:.4f}  Brier={r['brier']:.4f}  AUC={r['auc']:.3f}"
            )
    logger.info(f"\nBaseline weights:  {results['baseline_weights']}")
    logger.info(f"Optimized weights: {results['optimized_weights']}")
    logger.info(f"\nDetailed results written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(run_backtest(parse_args()))
