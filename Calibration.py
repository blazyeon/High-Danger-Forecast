"""
Probability calibration and online update scaffolding.
- Safe no-op when model or history not available.
"""
from __future__ import annotations

from typing import List
import numpy as np

import logging
logger = logging.getLogger(__name__)

class Calibrator:
    """
    Placeholder calibrator. If scikit-learn/fit history available, uses isotonic regression.
    Otherwise, returns input probability unchanged.
    """
    def __init__(self, method: str = "isotonic"):
        self.method = method
        self._fitted = False
        self._model = None

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> None:
        try:
            from sklearn.isotonic import IsotonicRegression
            if len(probs) < 50:
                return
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(probs.astype(float), outcomes.astype(float))
            self._model = ir
            self._fitted = True
            self.method = "isotonic"
            logger.info("Fitted isotonic calibrator on rolling dataset")
        except Exception as e:
            logger.debug(f"Calibrator fit failed, staying as identity: {e}")
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

_GLOBAL_CALIBRATOR = Calibrator()

def calibrate_prob(p_home: float) -> float:
    """Calibrate a single probability (safe no-op if calibrator not fitted)."""
    return _GLOBAL_CALIBRATOR.predict_one(p_home)

def update_calibration(rolling_probs: List[float], rolling_outcomes: List[int]) -> None:
    """Fit/update the calibrator with recent outcomes (e.g., daily cron/backtest)."""
    try:
        p = np.asarray(rolling_probs, dtype=float)
        y = np.asarray(rolling_outcomes, dtype=int)
        if len(p) >= 50:
            _GLOBAL_CALIBRATOR.fit(p, y)
    except Exception as e:
        logger.debug(f"Calibration update failed: {e}")