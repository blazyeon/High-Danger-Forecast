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
    Isotonic regression calibrator for home-win probabilities.
    Falls back to identity if no fit data or scikit-learn unavailable.
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
