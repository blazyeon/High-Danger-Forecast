"""
Probability calibration and online update scaffolding.
- Safe no-op when model or history not available.
"""
from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np

import logging
logger = logging.getLogger(__name__)

class Calibrator:
    """
    Probability calibrator for home-win probabilities.
    Supports isotonic regression and Platt (sigmoid) scaling.
    Falls back to identity if no fit data or scikit-learn unavailable.
    """
    def __init__(self, method: str = "auto"):
        self.method = method
        self._fitted = False
        self._model = None

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> None:
        if len(probs) < 50:
            return
        probs_arr = np.asarray(probs, dtype=float)
        outcomes_arr = np.asarray(outcomes, dtype=float)

        methods_to_try = []
        if self.method == "auto":
            methods_to_try = ["isotonic", "sigmoid"]
        else:
            methods_to_try = [self.method]

        best_model = None
        best_method = "identity"
        best_score = float("inf")
        # Use last 20% as a tiny holdout for method selection; the full fit is used
        # for the chosen model because the OOF preds are already out-of-fold.
        n = len(probs_arr)
        holdout = max(20, int(0.2 * n))
        train_idx = np.arange(n - holdout)
        val_idx = np.arange(n - holdout, n)

        for method in methods_to_try:
            try:
                if method == "isotonic":
                    from sklearn.isotonic import IsotonicRegression
                    model = IsotonicRegression(out_of_bounds="clip")
                    model.fit(probs_arr[train_idx], outcomes_arr[train_idx])
                    preds = model.predict(probs_arr[val_idx])
                elif method == "sigmoid":
                    from sklearn.linear_model import LogisticRegression
                    # Add small noise to avoid singular features
                    X = probs_arr[train_idx].reshape(-1, 1)
                    model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=200)
                    model.fit(X, outcomes_arr[train_idx])
                    preds = model.predict_proba(probs_arr[val_idx].reshape(-1, 1))[:, 1]
                else:
                    continue

                preds = np.clip(preds, 1e-6, 1 - 1e-6)
                score = -np.mean(
                    outcomes_arr[val_idx] * np.log(preds) +
                    (1 - outcomes_arr[val_idx]) * np.log(1 - preds)
                )
                if score < best_score:
                    best_score = score
                    best_model = model
                    best_method = method
            except Exception as e:
                logger.debug(f"{method} calibrator fit failed: {e}")
                continue

        if best_model is not None:
            # Refit on full data for the chosen method.
            try:
                if best_method == "isotonic":
                    best_model.fit(probs_arr, outcomes_arr)
                elif best_method == "sigmoid":
                    best_model.fit(probs_arr.reshape(-1, 1), outcomes_arr)
                self._model = best_model
                self._fitted = True
                self.method = best_method
                logger.info(f"Fitted {best_method} calibrator (holdout logloss={best_score:.4f})")
            except Exception as e:
                logger.debug(f"Final {best_method} refit failed: {e}")
                self._fitted = False
                self._model = None

    def predict(self, probs: np.ndarray) -> np.ndarray:
        if self._fitted and self._model is not None:
            try:
                return self._model.predict(probs.astype(float))
            except Exception:
                pass
        return probs

    def predict_one(self, p: float) -> float:
        try:
            arr = np.array([p], dtype=float)
            return float(self.predict(arr)[0])
        except Exception:
            return float(p)

    def save(self, filepath: str) -> None:
        """Persist fitted calibrator to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "fitted": self._fitted, "method": self.method}, f)
        logger.info(f"Saved calibrator to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "Calibrator":
        """Load a fitted calibrator from disk."""
        cal = cls(method="identity")
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            cal._model = data.get("model")
            cal._fitted = bool(data.get("fitted", False))
            cal.method = data.get("method", "isotonic")
            logger.info(f"Loaded calibrator from {filepath} (fitted={cal._fitted})")
        except Exception as e:
            logger.warning(f"Could not load calibrator from {filepath}: {e}")
        return cal


_GLOBAL_CALIBRATOR = Calibrator()

def calibrate_prob(p_home: float) -> float:
    """Calibrate a single probability (safe no-op if calibrator not fitted)."""
    return _GLOBAL_CALIBRATOR.predict_one(p_home)
