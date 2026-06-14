"""
Auto-improvement engine: Tests model variants and promotes the best.
Periodically retrains test variants on recent game data so they can
compete fairly with the main model.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from datetime import date, datetime, timedelta
import numpy as np
import logging
from pathlib import Path

from .MLModel import EloMLPredictor, ModelConfig
from .Database import EloDatabase
from .Features import EloFeatureEngine
from .Ratings import PlayerEloSystem, TeamEloSystem, EloConfig

logger = logging.getLogger(__name__)


class AutoImprovementEngine:
    """
    Runs multiple test models in shadow mode, retrains them on recent data,
    and promotes the best performer to production.
    """

    def __init__(
        self,
        database: EloDatabase,
        models_dir: str = "models",
        evaluation_window: int = 50  # Games before evaluation
    ):
        self.db = database
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.evaluation_window = evaluation_window

        # Main production model
        self.main_model: Optional[EloMLPredictor] = None

        # Test models
        self.test_models: List[EloMLPredictor] = []

        # Performance tracking
        self.games_since_eval = 0
        self.games_since_retrain = 0
        self._retrain_interval = 200  # Retrain test models every 200 games

        self._initialize_models()

    def _initialize_models(self):
        """Initialize main and test models"""

        # Try to load existing main model
        main_path = self.models_dir / "main_model.pkl"
        if main_path.exists():
            try:
                self.main_model = EloMLPredictor(model_id="main")
                self.main_model.load(str(main_path))
                logger.info("Loaded existing main model")
            except Exception as e:
                logger.error(f"Failed to load main model: {e}")
                self.main_model = None

        # Create main model if none exists
        if self.main_model is None:
            self.main_model = EloMLPredictor(
                model_id="main",
                config=ModelConfig()
            )
            logger.info("Created new main model")

        # Create test model variants
        self._create_test_models()

    def _create_test_models(self):
        """Create test model variants with different configurations"""

        test_configs = [
            # Variant 1: Higher Elo weight
            ModelConfig(
                elo_feature_weight=0.4,
                learning_rate=0.05,
                max_depth=6,
                n_estimators=200
            ),

            # Variant 2: Lower Elo weight, more trees
            ModelConfig(
                elo_feature_weight=0.2,
                learning_rate=0.03,
                max_depth=5,
                n_estimators=300
            ),

            # Variant 3: Deeper trees, regularization
            ModelConfig(
                elo_feature_weight=0.3,
                learning_rate=0.05,
                max_depth=8,
                n_estimators=150,
                reg_alpha=0.2,
                reg_lambda=1.5
            ),

            # Variant 4: Lighter model
            ModelConfig(
                elo_feature_weight=0.35,
                learning_rate=0.07,
                max_depth=4,
                n_estimators=250,
                min_child_weight=5
            ),

            # Variant 5: Conservative
            ModelConfig(
                elo_feature_weight=0.25,
                learning_rate=0.04,
                max_depth=5,
                n_estimators=200,
                subsample=0.7,
                colsample_bytree=0.7
            )
        ]

        self.test_models = []
        for i, config in enumerate(test_configs):
            model = EloMLPredictor(
                model_id=f"test_{i+1}",
                config=config
            )
            self.test_models.append(model)

        logger.info(f"Created {len(self.test_models)} test model variants")

    def _load_recent_games(self, n_days: int = 90) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
        """
        Load recent game data from the database for retraining test models.
        Returns (X, y, feature_names) or None if insufficient data.
        """
        try:
            cursor = self.db.conn.cursor()

            # Get recent games with Elo features
            cutoff = (date.today() - timedelta(days=n_days)).isoformat()
            cursor.execute("""
                SELECT home_team, away_team, home_score, away_score,
                       home_xgf, away_xgf, is_ot_so, home_sf, away_sf
                FROM game_results
                WHERE game_date >= ? AND home_xgf IS NOT NULL
                ORDER BY game_date ASC
            """, (cutoff,))

            rows = cursor.fetchall()
            if len(rows) < 100:
                logger.warning(f"Only {len(rows)} recent games available for retraining (need 100+)")
                return None

            # Build training data using Elo ratings at each game time
            config = EloConfig()
            team_elo = TeamEloSystem(config)

            X_list = []
            y_list = []

            for row in rows:
                home_team, away_team = row[0], row[1]
                home_score, away_score = row[2], row[3]
                home_xgf = row[4] or 0.0
                away_xgf = row[5] or 0.0
                is_ot_so = bool(row[6]) if row[6] is not None else False

                # Get Elo ratings
                home_elo = self.db.get_latest_team_elo(home_team) or 1500.0
                away_elo = self.db.get_latest_team_elo(away_team) or 1500.0

                total_xgf = max(home_xgf + away_xgf, 0.1)

                features = {
                    'home_xgf_norm': home_xgf / 6.0,
                    'away_xgf_norm': away_xgf / 6.0,
                    'home_xgf_pct': home_xgf / total_xgf,
                    'away_xgf_pct': away_xgf / total_xgf,
                    'home_shot_quality': home_xgf / 30.0 if home_xgf > 0 else 0.0,
                    'away_shot_quality': away_xgf / 30.0 if away_xgf > 0 else 0.0,
                    'home_sf_norm': 0.0,
                    'away_sf_norm': 0.0,
                    'elo_diff': (home_elo - away_elo) / 200.0,
                    'home_elo_norm': (home_elo - 1500.0) / 200.0,
                    'away_elo_norm': (away_elo - 1500.0) / 200.0,
                    'home_elo_level': home_elo / 2000.0,
                    'away_elo_level': away_elo / 2000.0,
                    'home_elo_momentum': 0.0,
                    'away_elo_momentum': 0.0,
                    'elo_strength_product': (home_elo * away_elo) / (1500.0 ** 2),
                    'elo_momentum_diff': 0.0,
                    'home_elo_experience': 0.0,
                    'away_elo_experience': 0.0,
                    'home_forward_elo_avg': 0.0,
                    'away_forward_elo_avg': 0.0,
                    'home_forward_elo_max': 0.0,
                    'away_forward_elo_max': 0.0,
                    'home_defense_elo_avg': 0.0,
                    'away_defense_elo_avg': 0.0,
                    'home_goalie_elo': 0.0,
                    'away_goalie_elo': 0.0,
                    'home_forward_depth': 0.0,
                    'away_forward_depth': 0.0,
                    'forward_elo_diff': 0.0,
                    'defense_elo_diff': 0.0,
                    'goalie_elo_diff': 0.0,
                }

                X_list.append(features)
                y_list.append(1.0 if home_score > away_score else 0.0)

                # Update in-memory Elo for subsequent games
                home_team_obj = team_elo.get_or_create_team(home_team)
                away_team_obj = team_elo.get_or_create_team(away_team)

                score_diff = abs(home_score - away_score)
                is_ot = is_ot_so or (score_diff == 1)

                if home_score > away_score:
                    home_result = 1.0
                    away_result = 0.25 if is_ot else 0.0
                elif away_score > home_score:
                    home_result = 0.25 if is_ot else 0.0
                    away_result = 1.0
                else:
                    home_result = 0.5
                    away_result = 0.5

                home_team_obj.update(
                    opponent_rating=away_team_obj.rating,
                    team_gf=home_score, team_ga=away_score,
                    team_xgf=home_xgf, team_xga=away_xgf,
                    team_sf=30, team_sa=30,
                    result=home_result, config=config
                )
                away_team_obj.update(
                    opponent_rating=home_team_obj.rating,
                    team_gf=away_score, team_ga=home_score,
                    team_xgf=away_xgf, team_xga=home_xgf,
                    team_sf=30, team_sa=30,
                    result=away_result, config=config
                )

            feature_names = list(X_list[0].keys())
            X = np.array([[f[name] for name in feature_names] for f in X_list])
            y = np.array(y_list)

            logger.info(f"Loaded {len(X)} recent games for retraining")
            return X, y, feature_names

        except Exception as e:
            logger.error(f"Error loading recent games for retraining: {e}")
            return None

    def retrain_test_models(self):
        """
        Retrain all test model variants on recent game data.
        This ensures test models actually have trained weights to compare.
        """
        logger.info("Retraining test model variants on recent data...")

        result = self._load_recent_games(n_days=90)
        if result is None:
            logger.warning("Insufficient data for retraining test models")
            return

        X, y, feature_names = result

        # Use 80/20 split
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        for model in self.test_models:
            try:
                model.train(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    feature_names=feature_names
                )
                logger.info(f"  Retrained {model.model_id}")
            except Exception as e:
                logger.warning(f"  Failed to retrain {model.model_id}: {e}")

        self.games_since_retrain = 0
        logger.info("Test model retraining complete")

    def predict_with_all_models(
        self,
        features: Dict[str, float],
        baseline_mu_home: float,
        baseline_mu_away: float
    ) -> Dict[str, Tuple[float, float]]:
        """
        Get predictions from all models.

        Returns:
            Dict mapping model_id to (adjusted_mu_home, adjusted_mu_away)
        """
        predictions = {}

        # Main model prediction
        if self.main_model and self.main_model.is_trained:
            predictions['main'] = self.main_model.predict_adjustment(
                features, baseline_mu_home, baseline_mu_away
            )
        else:
            predictions['main'] = (baseline_mu_home, baseline_mu_away)

        # Test model predictions
        for model in self.test_models:
            if model.is_trained:
                predictions[model.model_id] = model.predict_adjustment(
                    features, baseline_mu_home, baseline_mu_away
                )

        return predictions

    def record_game_result(
        self,
        game_date: date,
        features: Dict[str, float],
        baseline_mu_home: float,
        baseline_mu_away: float,
        actual_home_win: int  # 1 if home won, 0 otherwise
    ):
        """Record game result for all models"""

        # Get predictions from all models
        all_predictions = self.predict_with_all_models(
            features, baseline_mu_home, baseline_mu_away
        )

        # Convert expected goals to win probability using logistic function
        def goals_to_prob(mu_home: float, mu_away: float) -> float:
            diff = mu_home - mu_away
            return 1.0 / (1.0 + np.exp(-0.5 * diff))

        # Record performance for each model
        for model_id, (adj_home, adj_away) in all_predictions.items():
            prob_home_win = goals_to_prob(adj_home, adj_away)

            # Calculate metrics
            brier = (prob_home_win - actual_home_win) ** 2
            logloss = -(
                actual_home_win * np.log(np.clip(prob_home_win, 1e-9, 1 - 1e-9)) +
                (1 - actual_home_win) * np.log(np.clip(1 - prob_home_win, 1e-9, 1e-9))
            )

            # Save to database
            self.db.save_model_performance(
                model_id=model_id,
                model_version="1.0",
                game_date=game_date,
                prediction=prob_home_win,
                actual=actual_home_win,
                brier=brier,
                logloss=logloss,
                features=features
            )

            # Update model's internal tracking
            if model_id == 'main' and self.main_model:
                self.main_model.record_performance(
                    prob_home_win, actual_home_win, brier, logloss
                )
            else:
                for model in self.test_models:
                    if model.model_id == model_id:
                        model.record_performance(
                            prob_home_win, actual_home_win, brier, logloss
                        )

        self.games_since_eval += 1
        self.games_since_retrain += 1

        # Check if it's time to retrain test models
        if self.games_since_retrain >= self._retrain_interval:
            self.retrain_test_models()

        # Check if it's time to evaluate and potentially promote
        if self.games_since_eval >= self.evaluation_window:
            self._evaluate_and_promote()
            self.games_since_eval = 0

    def _evaluate_and_promote(self):
        """Evaluate all models and promote the best to main"""

        logger.info("Evaluating model performance...")

        # Get performance for all models
        performances = {}

        # Main model
        main_perf = self.db.get_model_performance_summary(
            'main', days=30
        )
        if main_perf and main_perf.get('n_predictions', 0) > 20:
            performances['main'] = main_perf

        # Test models
        for model in self.test_models:
            perf = self.db.get_model_performance_summary(
                model.model_id, days=30
            )
            if perf and perf.get('n_predictions', 0) > 20:
                performances[model.model_id] = perf

        if len(performances) < 2:
            logger.info("Not enough model data for evaluation")
            return

        # Find best model by combined score (lower is better)
        best_model_id = None
        best_score = float('inf')

        for model_id, perf in performances.items():
            score = 0.6 * perf['avg_brier'] + 0.4 * perf['avg_logloss']

            logger.info(
                f"{model_id}: Brier={perf['avg_brier']:.4f}, "
                f"LogLoss={perf['avg_logloss']:.4f}, "
                f"Acc={perf['accuracy']:.3f}, Score={score:.4f}"
            )

            if score < best_score:
                best_score = score
                best_model_id = model_id

        # Promote if test model is better than main
        if best_model_id and best_model_id != 'main':
            logger.info(f"🏆 Promoting {best_model_id} to main model!")

            # Find the test model
            for model in self.test_models:
                if model.model_id == best_model_id:
                    # Save old main model as backup
                    if self.main_model and self.main_model.is_trained:
                        backup_path = self.models_dir / f"main_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                        self.main_model.save(str(backup_path))

                    # Promote test model to main
                    old_config = model.config
                    model.model_id = 'main'
                    self.main_model = model
                    self.main_model.save(str(self.models_dir / "main_model.pkl"))

                    # Create new test model with same config
                    new_test = EloMLPredictor(
                        model_id=best_model_id,
                        config=old_config
                    )
                    self.test_models[self.test_models.index(model)] = new_test

                    break
        else:
            logger.info("✓ Main model is still the best")

        # Save current main model
        if self.main_model and self.main_model.is_trained:
            self.main_model.save(str(self.models_dir / "main_model.pkl"))

    def get_best_prediction(
        self,
        features: Dict[str, float],
        baseline_mu_home: float,
        baseline_mu_away: float
    ) -> Tuple[float, float]:
        """Get prediction from the main (best) model"""

        if self.main_model and self.main_model.is_trained:
            return self.main_model.predict_adjustment(
                features, baseline_mu_home, baseline_mu_away
            )
        else:
            return baseline_mu_home, baseline_mu_away


__all__ = ['AutoImprovementEngine']