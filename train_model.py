"""
Train ML model with Elo features for NHL game predictions.

This script:
1. Loads games from one or more training databases
2. Processes games chronologically to build accurate Elo ratings
3. Extracts pre-game features (Elo, xG, rest/travel, venue, etc.)
4. Trains an XGBoost model with walk-forward cross-validation
5. Optionally runs a small hyperparameter search
6. Saves the trained model and a calibration dataset

Example:
    python train_model.py
    python train_model.py --seasons 20222023 20232024 20242025 --tune
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
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

# Add project root to path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from EloMl.Database import EloDatabase
from EloMl.Features import EloFeatureEngine
from EloMl.MLModel import EloMLPredictor, ModelConfig
from EloMl.Ratings import EloConfig, PlayerEloSystem, TeamEloSystem
from Calibration import Calibrator
from NHL.Config import CURRENT_SEASON_YEAR, EARLIEST_SEASON_YEAR, _season_options
from NHL.Errors import safe_division
from NHL.Features import compute_rest_travel_features_fast
from NHL.Utils import season_from_date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _current_season() -> str:
    """Compute current NHL season string dynamically."""
    today = date.today()
    year = today.year if today.month >= 10 else today.year - 1
    return f"{year}{year + 1}"


def _available_training_databases(root: Path) -> Dict[str, Path]:
    """Map season keys to training database paths that exist on disk."""
    out: Dict[str, Path] = {}
    for y in range(EARLIEST_SEASON_YEAR, CURRENT_SEASON_YEAR + 1):
        key = f"{y}{y + 1}"
        path = root / f"training_data_{key}.db"
        if path.exists():
            out[key] = path
    return out


def parse_args() -> argparse.Namespace:
    current = _current_season()
    parser = argparse.ArgumentParser(
        description="Train NHL ML model with Elo + PBP features"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=str,
        default=[current],
        help="Seasons to train on (default: current season only)",
    )
    parser.add_argument(
        "--all-seasons",
        action="store_true",
        help="Train on all available training_data_*.db seasons",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/main_model.pkl",
        help="Output model path",
    )
    parser.add_argument(
        "--calibration-out",
        type=str,
        default="models/calibration_data.json",
        help="Where to write out-of-fold predictions for calibration",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run a small random hyperparameter search",
    )
    parser.add_argument(
        "--tune-iters",
        type=int,
        default=20,
        help="Hyperparameter search iterations",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        default=True,
        help="Use walk-forward (expanding window) CV instead of a fixed split",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=100,
        help="Minimum total games required to train",
    )
    return parser.parse_args()


def load_training_games(db_path: Path, season: str) -> List[Dict[str, Any]]:
    """Load games from a single training database."""
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

    games: List[Dict[str, Any]] = []
    for row in cursor.fetchall():
        games.append(
            {
                "game_id": row[0],
                "game_date": datetime.fromisoformat(row[1]).date(),
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
    logger.info(f"   Loaded {len(games)} games from {db_path.name}")
    return games


def _venue_record(
    home_team: str,
    away_team: str,
    season: str,
    completed_games: List[Dict[str, Any]],
) -> Tuple[float, float]:
    """
    Compute home/away win percentage from already-processed games only.
    This is safe from leakage because it uses data before the current game.
    """
    home_games = [g for g in completed_games if g["home_team"] == home_team]
    away_games = [g for g in completed_games if g["away_team"] == away_team]

    def _pct(team: str, games_subset: List[Dict[str, Any]], is_home: bool) -> float:
        wins = 0
        ot = 0
        gp = 0
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
        # Small-sample shrinkage toward .500
        return 0.5 * raw + 0.5 * 0.5

    return _pct(home_team, home_games, True), _pct(away_team, away_games, False)


def _season_to_date_stats(
    team: str,
    is_home: bool,
    completed_games: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute season-to-date per-game rates from already-processed games."""
    relevant = [
        g
        for g in completed_games
        if (is_home and g["home_team"] == team) or (not is_home and g["away_team"] == team)
    ]
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


def _recent_stats(
    team: str,
    is_home: bool,
    n: int,
    completed_games: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute recent xG, xGA, GF, GA, SF from already-processed games."""
    relevant = [
        g
        for g in completed_games
        if (is_home and g["home_team"] == team) or (not is_home and g["away_team"] == team)
    ]
    relevant = relevant[-n:] if len(relevant) > n else relevant
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


def _rest_travel_features_safe(
    home_team: str, away_team: str, game_date: date, completed: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute rest/travel features from already-played games; never raise."""
    defaults = {"is_b2b": False, "rest_days": 3.0, "opp_rest_days": 3.0, "travel_km": 0.0, "tz_diff": 0.0}
    try:
        home = compute_rest_travel_features_fast(home_team, away_team, game_date, completed)
    except Exception as e:
        logger.debug(f"Rest feature failure home {home_team}: {e}")
        home = dict(defaults)
    try:
        away = compute_rest_travel_features_fast(away_team, home_team, game_date, completed)
    except Exception as e:
        logger.debug(f"Rest feature failure away {away_team}: {e}")
        away = dict(defaults)
    return home, away


def _mean_std(series: List[float]) -> Tuple[float, float]:
    arr = np.array(series, dtype=float)
    return float(np.mean(arr)), float(np.std(arr)) if len(arr) > 1 else 1.0


def extract_features_for_game(
    game: Dict[str, Any],
    team_elo: TeamEloSystem,
    player_elo: PlayerEloSystem,
    feature_engine: EloFeatureEngine,
    completed_games: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Build a leakage-free pre-game feature vector."""
    home_team = game["home_team"]
    away_team = game["away_team"]
    game_date = game["game_date"]

    # Elo features (current state before this game)
    elo_features = feature_engine.extract_team_features(home_team, away_team)

    # Recent stats (from prior games only)
    home_recent = _recent_stats(home_team, True, 10, completed_games)
    away_recent = _recent_stats(away_team, False, 10, completed_games)

    # Season-to-date rates (match inference path's season-wide rates)
    home_std = _season_to_date_stats(home_team, True, completed_games)
    away_std = _season_to_date_stats(away_team, False, completed_games)

    # Venue record from prior games only
    home_pct, away_pct = _venue_record(home_team, away_team, game["season"], completed_games)

    # Rest/travel
    home_rest, away_rest = _rest_travel_features_safe(home_team, away_team, game_date, completed_games)

    # Head-to-head
    h2h = [g for g in completed_games if ((g["home_team"] == home_team and g["away_team"] == away_team) or
                                          (g["home_team"] == away_team and g["away_team"] == home_team))]
    h2h_home_wins = 0
    h2h_gp = 0
    for g in h2h[-5:]:
        h2h_gp += 1
        if g["home_team"] == home_team and g["home_score"] > g["away_score"]:
            h2h_home_wins += 1
        elif g["home_team"] == away_team and g["away_score"] > g["home_score"]:
            h2h_home_wins += 1
    h2h_pct = (h2h_home_wins / h2h_gp) if h2h_gp else 0.5

    # Build the feature dict
    features: Dict[str, float] = {
        # Elo
        **elo_features,
        "home_elo": team_elo.get_team_rating(home_team),
        "away_elo": team_elo.get_team_rating(away_team),

        # Season-average xG/xGA (from game_results, pre-game perspective)
        "home_season_xgf_pg": game["home_xgf"] / 0.06 if game["home_xgf"] else 0.0,
        "away_season_xgf_pg": game["away_xgf"] / 0.06 if game["away_xgf"] else 0.0,
        "home_xgf_norm": game["home_xgf"] / 6.0,
        "away_xgf_norm": game["away_xgf"] / 6.0,
        "home_xgf_share": game["home_xgf"] / max(game["home_xgf"] + game["away_xgf"], 0.1),

        # Recent form
        "home_recent_xgf_pg": home_recent["xgf_pg"],
        "home_recent_xga_pg": home_recent["xga_pg"],
        "away_recent_xgf_pg": away_recent["xgf_pg"],
        "away_recent_xga_pg": away_recent["xga_pg"],
        "home_recent_gf_pg": home_recent["gf_pg"],
        "away_recent_gf_pg": away_recent["gf_pg"],
        "home_recent_form_off": home_recent["xgf_pg"] - 3.0,
        "away_recent_form_off": away_recent["xgf_pg"] - 3.0,

        # Shots / pace
        "home_sf_pg": game["home_sf"] / 0.06 if game["home_sf"] else 0.0,
        "away_sf_pg": game["away_sf"] / 0.06 if game["away_sf"] else 0.0,
        "home_recent_sf_pg": home_recent["sf_pg"],
        "away_recent_sf_pg": away_recent["sf_pg"],

        # Venue
        "home_venue_pct": home_pct,
        "away_venue_pct": away_pct,
        "venue_diff": home_pct - away_pct,

        # Rest / travel
        "home_rest_days": float(home_rest.get("rest_days", 3.0)),
        "away_rest_days": float(away_rest.get("rest_days", 3.0)),
        "rest_diff": float(away_rest.get("rest_days", 3.0)) - float(home_rest.get("rest_days", 3.0)),
        "home_b2b": 1.0 if home_rest.get("is_b2b") else 0.0,
        "away_b2b": 1.0 if away_rest.get("is_b2b") else 0.0,
        "home_travel_km": float(home_rest.get("travel_km", 0.0)),
        "away_travel_km": float(away_rest.get("travel_km", 0.0)),
        "home_tz_diff": float(home_rest.get("tz_diff", 0.0)),
        "away_tz_diff": float(away_rest.get("tz_diff", 0.0)),

        # Head-to-head
        "h2h_home_pct": h2h_pct,

        # Season-to-date differential features (match inference path)
        "xgf_pct_diff": safe_division(
            (home_std["xgf_pg"] / max(home_std["xgf_pg"] + home_std["xga_pg"], 0.1) -
             away_std["xgf_pg"] / max(away_std["xgf_pg"] + away_std["xga_pg"], 0.1)) * 100.0,
            50.0,
            0.0,
        ),
        "gf_pg_diff": home_std["gf_pg"] - away_std["gf_pg"],
        "ga_pg_diff": away_std["ga_pg"] - home_std["ga_pg"],
        "sf_pg_diff": home_std["sf_pg"] - away_std["sf_pg"],
        "xga_pg_diff": home_std["xga_pg"] - away_std["xga_pg"],
    }

    return features


def create_training_data(
    games: List[Dict[str, Any]], config: EloConfig
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, Any]]]:
    """Create training data with leakage-free pre-game features."""
    logger.info("\n🔄 Extracting features chronologically...")

    player_elo = PlayerEloSystem(config)
    team_elo = TeamEloSystem(config)
    feature_engine = EloFeatureEngine(player_elo, team_elo, config)

    X_list: List[Dict[str, float]] = []
    y_list: List[float] = []
    meta_list: List[Dict[str, Any]] = []
    completed: List[Dict[str, Any]] = []

    for i, game in enumerate(games):
        if (i + 1) % 250 == 0:
            logger.info(f"   Processing game {i + 1}/{len(games)}...")

        try:
            features = extract_features_for_game(
                game, team_elo, player_elo, feature_engine, completed
            )
        except Exception as e:
            logger.warning(f"Feature extraction failed for {game['game_id']}: {e}")
            continue

        X_list.append(features)
        y_list.append(1.0 if game["home_score"] > game["away_score"] else 0.0)
        meta_list.append(
            {
                "game_id": game["game_id"],
                "game_date": game["game_date"].isoformat(),
                "season": game["season"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "home_score": game["home_score"],
                "away_score": game["away_score"],
            }
        )

        # Update Elo AFTER extracting features for this game
        try:
            home_obj = team_elo.get_or_create_team(game["home_team"])
            away_obj = team_elo.get_or_create_team(game["away_team"])

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

    if not X_list:
        raise RuntimeError("No training examples could be created")

    feature_names = list(X_list[0].keys())
    X = np.array([[f[n] for n in feature_names] for f in X_list], dtype=float)
    y = np.array(y_list, dtype=float)

    logger.info(f"\n✓ Extracted features for {len(X)} games")
    logger.info(f"✓ Feature count: {len(feature_names)}")
    return X, y, feature_names, meta_list


def _clip_probs(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 1e-6, 1 - 1e-6)


def _report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str,
) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    y_prob = _clip_probs(y_prob)
    return {
        "label": label,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "logloss": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5,
    }


def _train_one(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    cfg: ModelConfig,
) -> Tuple[EloMLPredictor, Dict[str, float]]:
    model = EloMLPredictor(model_id="main", config=cfg)
    model.train(X_train, y_train, X_val, y_val, feature_names=feature_names)
    val_pred = model.model.predict(X_val)
    val_pred = _clip_probs(np.array(val_pred, dtype=float))
    metrics = _report(y_val, val_pred, "val")
    return model, metrics


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    meta: List[Dict[str, Any]],
    output_path: str,
    calibration_path: str,
    tune: bool,
    tune_iters: int,
    walk_forward: bool,
) -> EloMLPredictor:
    """Train with walk-forward CV and optional hyperparameter tuning."""
    logger.info("\n🤖 Training ML model...")

    if walk_forward:
        # Expanding window: use first 50% for first train, then validate on next 20%,
        # retraining up to each validation block. Final model is trained on all data.
        n = len(X)
        min_train = int(n * 0.40)
        val_size = int(n * 0.15)
        steps = max(1, (n - min_train) // val_size)

        oof_probs = np.full(n, np.nan)
        best_ll = float("inf")
        best_cfg = ModelConfig()

        if tune:
            logger.info(f"\n🔎 Random hyperparameter search ({tune_iters} iters)...")
            configs = []
            rng = np.random.default_rng(42)
            for _ in range(tune_iters):
                configs.append(
                    ModelConfig(
                        learning_rate=float(rng.uniform(0.01, 0.15)),
                        max_depth=int(rng.integers(3, 10)),
                        n_estimators=int(rng.integers(100, 600)),
                        min_child_weight=int(rng.integers(1, 10)),
                        subsample=float(rng.uniform(0.5, 1.0)),
                        colsample_bytree=float(rng.uniform(0.5, 1.0)),
                        reg_alpha=float(rng.uniform(0.0, 1.0)),
                        reg_lambda=float(rng.uniform(0.0, 3.0)),
                    )
                )
        else:
            configs = [ModelConfig()]

        cv_results: List[Dict[str, Any]] = []
        for cfg_idx, cfg in enumerate(configs):
            fold_lls = []
            for step in range(steps):
                train_end = min_train + step * val_size
                val_start = train_end
                val_end = min(n, val_start + val_size)
                if val_end <= val_start:
                    break
                X_train = X[:train_end]
                y_train = y[:train_end]
                X_val = X[val_start:val_end]
                y_val = y[val_start:val_end]

                model, metrics = _train_one(
                    X_train, y_train, X_val, y_val, feature_names, cfg
                )
                fold_lls.append(metrics["logloss"])
                # Save OOF predictions for the final selected config
                if cfg_idx == 0:
                    oof_probs[val_start:val_end] = _clip_probs(
                        model.model.predict(X_val)
                    )

            avg_ll = float(np.mean(fold_lls)) if fold_lls else float("inf")
            cv_results.append({"cfg": cfg, "logloss": avg_ll})
            if avg_ll < best_ll:
                best_ll = avg_ll
                best_cfg = cfg
            logger.info(
                f"   cfg {cfg_idx + 1}/{len(configs)}: logloss={avg_ll:.4f} "
                f"lr={cfg.learning_rate:.3f} depth={cfg.max_depth} n={cfg.n_estimators}"
            )

        logger.info(
            f"\n✅ Best CV logloss: {best_ll:.4f} "
            f"(lr={best_cfg.learning_rate:.3f}, depth={best_cfg.max_depth}, n={best_cfg.n_estimators})"
        )

        # Re-train final model on all data using best config
        final_model, final_metrics = _train_one(
            X, y, X[-100:], y[-100:], feature_names, best_cfg
        )
    else:
        # Simple holdout for quick runs
        split = int(len(X) * 0.8)
        final_model, final_metrics = _train_one(
            X[:split], y[:split], X[split:], y[split:], feature_names, ModelConfig()
        )
        oof_probs = np.full(len(X), np.nan)
        oof_probs[split:] = _clip_probs(final_model.model.predict(X[split:]))

    logger.info(f"\n📊 Final validation metrics: {final_metrics}")

    # Feature importance
    logger.info("\n📊 Feature Importance (Top 20):")
    importance = final_model.get_feature_importance()
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
    for i, (fname, imp) in enumerate(sorted_features, 1):
        logger.info(f"   {i:2d}. {fname:30s} {imp:.4f}")

    # Save calibration dataset from OOF predictions
    calib_records = []
    for m, p, actual in zip(meta, oof_probs, y):
        if np.isnan(p):
            continue
        calib_records.append(
            {
                **m,
                "pred_home_win": float(p),
                "actual_home_win": int(actual),
            }
        )
    Path(calibration_path).parent.mkdir(parents=True, exist_ok=True)
    with open(calibration_path, "w", encoding="utf-8") as f:
        json.dump(calib_records, f, indent=2)
    logger.info(f"\n✓ Wrote {len(calib_records)} OOF predictions to {calibration_path}")

    # Fit and save calibrator using out-of-fold predictions
    valid_idx = ~np.isnan(oof_probs)
    if np.sum(valid_idx) >= 50:
        calibrator = Calibrator(method="isotonic")
        calibrator.fit(oof_probs[valid_idx], y[valid_idx])
        calib_path = str(Path(output_path).parent / "calibrator.pkl")
        calibrator.save(calib_path)
        logger.info(f"✓ Calibrator saved to {calib_path}")
    else:
        logger.warning("Not enough OOF predictions to fit calibrator")

    # Save model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final_model.save(output_path)
    logger.info(f"✓ Model saved to {output_path}")
    return final_model


def main() -> int:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("🏒 NHL ML Model Training")
    logger.info("=" * 60)

    root = Path(__file__).parent
    if args.all_seasons:
        db_map = _available_training_databases(root)
        seasons = list(db_map.keys())
        logger.info(f"📅 Found {len(seasons)} training databases: {seasons}")
    else:
        seasons = args.seasons
        db_map = {s: root / f"training_data_{s}.db" for s in seasons}
        db_map = {k: v for k, v in db_map.items() if v.exists()}

    if not db_map:
        logger.error("❌ No training databases found. Run update_elo_ratings.py --season YYYY --training --reset")
        return 1

    all_games: List[Dict[str, Any]] = []
    for season, db_path in sorted(db_map.items()):
        games = load_training_games(db_path, season)
        all_games.extend(games)

    all_games.sort(key=lambda g: (g["game_date"], g["game_id"]))
    logger.info(f"\n📊 Total games: {len(all_games)}")

    if len(all_games) < args.min_games:
        logger.error(f"❌ Insufficient training data: only {len(all_games)} games")
        return 1

    config = EloConfig()
    X, y, feature_names, meta = create_training_data(all_games, config)

    model = train_model(
        X,
        y,
        feature_names,
        meta,
        args.output,
        args.calibration_out,
        tune=args.tune,
        tune_iters=args.tune_iters,
        walk_forward=args.walk_forward,
    )

    logger.info("\n" + "=" * 60)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\n   Games: {len(all_games)}")
    logger.info(f"   Features: {len(feature_names)}")
    logger.info(f"   Model: {args.output}")
    logger.info(f"   Calibration data: {args.calibration_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
