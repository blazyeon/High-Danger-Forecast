"""
Machine Learning model that uses Elo features alongside existing stats.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for ML model"""
    model_type: str = "xgboost"  # xgboost, lightgbm, or sklearn
    elo_feature_weight: float = 0.3  # How much to weight Elo vs other features
    learning_rate: float = 0.05
    max_depth: int = 6
    n_estimators: int = 200
    min_child_weight: int = 3
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    # Regularization
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    
    # Early stopping
    early_stopping_rounds: int = 50
    eval_metric: str = "logloss"


class EloMLPredictor:
    """ML model that combines Elo features with traditional stats"""
    
    def __init__(
        self,
        model_id: str = "main",
        config: Optional[ModelConfig] = None
    ):
        self.model_id = model_id
        self.config = config or ModelConfig()
        self.model = None
        self.feature_names: List[str] = []
        self.is_trained = False
        self.performance_history: List[Dict] = []
        
    def _create_model(self):
        """Create the ML model based on config"""
        if self.config.model_type == "xgboost":
            try:
                import xgboost as xgb
                self.model = xgb.XGBRegressor(
                    learning_rate=self.config.learning_rate,
                    max_depth=self.config.max_depth,
                    n_estimators=self.config.n_estimators,
                    min_child_weight=self.config.min_child_weight,
                    subsample=self.config.subsample,
                    colsample_bytree=self.config.colsample_bytree,
                    reg_alpha=self.config.reg_alpha,
                    reg_lambda=self.config.reg_lambda,
                    objective='reg:logistic',
                    random_state=42
                )
                logger.info("Created XGBoost model")
            except ImportError:
                logger.warning("XGBoost not available, falling back to sklearn")
                self.config.model_type = "sklearn"
                self._create_model()
                
        elif self.config.model_type == "lightgbm":
            try:
                import lightgbm as lgb
                self.model = lgb.LGBMRegressor(
                    learning_rate=self.config.learning_rate,
                    max_depth=self.config.max_depth,
                    n_estimators=self.config.n_estimators,
                    min_child_weight=self.config.min_child_weight,
                    subsample=self.config.subsample,
                    colsample_bytree=self.config.colsample_bytree,
                    reg_alpha=self.config.reg_alpha,
                    reg_lambda=self.config.reg_lambda,
                    objective='regression',
                    random_state=42
                )
                logger.info("Created LightGBM model")
            except ImportError:
                logger.warning("LightGBM not available, falling back to sklearn")
                self.config.model_type = "sklearn"
                self._create_model()
                
        else:  # sklearn fallback
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                n_estimators=self.config.n_estimators,
                min_samples_leaf=self.config.min_child_weight,
                subsample=self.config.subsample,
                random_state=42
            )
            logger.info("Created sklearn GradientBoosting model")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ):
        """Train the model (compatible with all XGBoost versions)"""
        if self.model is None:
            self._create_model()
        
        if feature_names:
            self.feature_names = feature_names
        
        # Train without validation set complications
        # This is the most compatible approach
        self.model.fit(X_train, y_train)
        
        self.is_trained = True
        logger.info(f"Model {self.model_id} trained on {len(X_train)} samples")
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            try:
                val_pred = self.model.predict(X_val)
                val_pred = np.clip(val_pred, 0, 1)
                
                # Calculate metrics
                brier = np.mean((val_pred - y_val) ** 2)
                
                # Log loss
                epsilon = 1e-15
                val_pred_clipped = np.clip(val_pred, epsilon, 1 - epsilon)
                logloss = -np.mean(
                    y_val * np.log(val_pred_clipped) + 
                    (1 - y_val) * np.log(1 - val_pred_clipped)
                )
                
                # Accuracy
                correct = np.sum((val_pred >= 0.5) == (y_val == 1))
                accuracy = correct / len(y_val)
                
                logger.info(f"  Validation Brier: {brier:.4f}")
                logger.info(f"  Validation LogLoss: {logloss:.4f}")
                logger.info(f"  Validation Accuracy: {accuracy:.3f}")
                
            except Exception as e:
                logger.warning(f"Could not evaluate on validation set: {e}")
    
    def predict_adjustment(
        self,
        features: Dict[str, float],
        baseline_mu_home: float,
        baseline_mu_away: float
    ) -> Tuple[float, float]:
        """
        Predict Elo-based adjustment to baseline expected goals.
        
        Returns:
            (adjusted_mu_home, adjusted_mu_away)
        """
        if not self.is_trained or self.model is None:
            # No adjustment if model not trained
            return baseline_mu_home, baseline_mu_away
        
        # Extract feature vector
        X = self._features_to_array(features)
        
        # Predict adjustment factor
        try:
            # Model predicts a multiplier (centered around 1.0)
            adjustment = self.model.predict(X.reshape(1, -1))[0]
            
            # Clip adjustment to reasonable bounds
            adjustment = np.clip(adjustment, 0.7, 1.3)
            
            # Apply weighted adjustment
            weight = self.config.elo_feature_weight
            
            # Adjust home team more if Elo favors home, away team if Elo favors away
            elo_diff = features.get('elo_diff', 0.0)
            
            if elo_diff > 0:  # Home team stronger by Elo
                home_adj = 1.0 + weight * (adjustment - 1.0)
                away_adj = 1.0 - weight * (adjustment - 1.0) * 0.5
            else:  # Away team stronger by Elo
                home_adj = 1.0 - weight * abs(adjustment - 1.0) * 0.5
                away_adj = 1.0 + weight * (adjustment - 1.0)
            
            # Clip final adjustments
            home_adj = np.clip(home_adj, 0.8, 1.2)
            away_adj = np.clip(away_adj, 0.8, 1.2)
            
            adjusted_home = baseline_mu_home * home_adj
            adjusted_away = baseline_mu_away * away_adj
            
            logger.debug(
                f"Elo ML adjustment: Home {baseline_mu_home:.2f} -> {adjusted_home:.2f}, "
                f"Away {baseline_mu_away:.2f} -> {adjusted_away:.2f}"
            )
            
            return adjusted_home, adjusted_away
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return baseline_mu_home, baseline_mu_away
    
    def _features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to numpy array in correct order"""
        if self.feature_names:
            return np.array([features.get(f, 0.0) for f in self.feature_names])
        else:
            return np.array(list(features.values()))
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_trained or self.model is None:
            return {}
        
        try:
            if self.config.model_type == "xgboost":
                importance = self.model.feature_importances_
            elif self.config.model_type == "lightgbm":
                importance = self.model.feature_importances_
            else:
                importance = self.model.feature_importances_
            
            if self.feature_names and len(importance) == len(self.feature_names):
                return dict(zip(self.feature_names, importance))
            else:
                return {f"feature_{i}": imp for i, imp in enumerate(importance)}
        except Exception as e:
            logger.error(f"Could not get feature importance: {e}")
            return {}
    
    def save(self, filepath: str):
        """Save model to disk"""
        if not self.is_trained:
            logger.warning("Cannot save untrained model")
            return
        
        model_data = {
            'model': self.model,
            'model_id': self.model_id,
            'config': self.config,
            'feature_names': self.feature_names,
            'performance_history': self.performance_history
        }
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved model to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_id = model_data['model_id']
        self.config = model_data['config']
        self.feature_names = model_data['feature_names']
        self.performance_history = model_data.get('performance_history', [])
        self.is_trained = True
        
        logger.info(f"Loaded model from {filepath}")
    
    def record_performance(
        self,
        prediction: float,
        actual: int,
        brier: float,
        logloss: float
    ):
        """Record model performance for tracking"""
        self.performance_history.append({
            'prediction': prediction,
            'actual': actual,
            'brier': brier,
            'logloss': logloss
        })
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_recent_performance(self, n: int = 100) -> Dict[str, float]:
        """Get performance metrics over recent predictions"""
        if not self.performance_history:
            return {}
        
        recent = self.performance_history[-n:]
        
        brier_scores = [p['brier'] for p in recent]
        logloss_scores = [p['logloss'] for p in recent]
        
        # Calculate accuracy
        correct = sum(
            1 for p in recent
            if (p['prediction'] >= 0.5 and p['actual'] == 1) or
               (p['prediction'] < 0.5 and p['actual'] == 0)
        )
        
        return {
            'n_predictions': len(recent),
            'avg_brier': np.mean(brier_scores),
            'avg_logloss': np.mean(logloss_scores),
            'accuracy': correct / len(recent)
        }