"""
xG (Expected Goals) model.

A simple but well-engineered logistic regression on shot features. The NHL
API gives us x/y coordinates, shot type, and game situation; from those we
engineer distance, angle, rebound/rush flags, and a man-advantage indicator.

Target: P(is_goal) per shot. Trained on 2-3 historical seasons of PBP
shots, validated against MoneyPuck's published xGoal column.

Why logistic regression and not xgboost?
- Logistic regression on shot features typically gets AUC 0.74-0.78
- xgboost gets to ~0.78-0.80, but adds complexity, no calibration, harder
  to debug. We can swap in a richer model later. The plan's threshold
  (AUC ≥ 0.74) is realistic for LR.
- A well-calibrated LR is what most production xG models started as.
"""
from __future__ import annotations

import json
import logging
import math
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────

# NHL rink: net at x=±89 ft, center ice at x=0
# We flip coordinates so all shots attack the +x direction (toward the
# right net at x=89, y=0). This matches MoneyPuck's xCordAdjusted scheme.
NET_X = 89.0
NET_Y = 0.0

# Standard NHL shot types seen in PBP data
KNOWN_SHOT_TYPES = [
    "wrist", "slap", "snap", "backhand", "tip-in", "deflect",
    "wrap-around", "poke", "bat",
]

# Feature columns produced by build_features
FEATURE_COLS_NUMERIC = [
    "distance", "angle", "distance_x_angle",
    "is_rebound", "is_rush",
    "home_skaters", "away_skaters", "skaters_diff",
    "is_empty_net", "is_home", "period", "time_seconds",
]
FEATURE_COLS_SHOT_TYPE = [f"shot_type_{t}" for t in KNOWN_SHOT_TYPES]

ALL_FEATURE_COLS = FEATURE_COLS_NUMERIC + FEATURE_COLS_SHOT_TYPE

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("XGMODEL_DIR", str(_PROJECT_ROOT / "models")))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "xg_model.pkl"
REPORT_PATH = MODEL_DIR / "xg_model_report.json"


# ── Feature engineering ─────────────────────────────────────────────────

def _flip_to_attacking(x: float, y: float, is_home: Optional[int] = None) -> Tuple[float, float]:
    """
    Normalize shot coordinates so all shots attack the +x net at (89, 0).

    The PBP coordinate system is from the *defending* team's perspective:
    - Home team defending left: x in [-89, 89], positive x is the away goal
    - Visiting team defending right: x in [-89, 89], positive x is the home goal

    We don't have is_home in the raw PBP shot (the plan said is_home; we
    backfill it later from meta). When is_home is None, we assume the
    default convention (positive x = attacking right).

    Per MoneyPuck: xCordAdjusted is always positive for shots at the
    right net. We use a simple rule: if x < 0, flip sign of both x and y.
    """
    if x < 0:
        return -x, -y
    return x, y


def _shot_distance_angle(x: float, y: float) -> Tuple[float, float]:
    """Euclidean distance and shot angle (degrees) from net at (89, 0)."""
    dx = NET_X - x
    dy = NET_Y - y
    dist = math.sqrt(dx * dx + dy * dy)
    angle = abs(math.degrees(math.atan2(dy, dx)))
    return dist, angle


def _parse_situation(sit: str) -> Tuple[int, int]:
    """
    Parse 4-digit situationCode: away_g, away_s, home_s, home_g.
    Returns (home_skaters, away_skaters).
    """
    if not sit or len(str(sit)) != 4:
        return 5, 5
    try:
        away_g, away_s, home_s, home_g = (int(c) for c in str(sit))
        return home_s, away_s
    except Exception:
        return 5, 5


def build_features(
    shots: pd.DataFrame,
    game_meta: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build feature matrix from a shots DataFrame.

    Required columns: x, y, period, home_skaters, away_skaters, is_goal
    (or event_type). Optional: shot_type, time_seconds, is_rebound, is_rush.

    If game_meta is provided (with columns game_id, homeTeam, awayTeam), we
    backfill is_home by joining on game_id and comparing team_id to the
    home team abbrev. But PBP doesn't give us the team abbrev on the shot,
    only team_id. For the current xG model we default is_home=0 (no home
    advantage baked into xG) — venue effects are handled separately in
    the simulation layer.

    Returns a DataFrame with the engineered columns.
    """
    df = shots.copy()
    if df.empty:
        return pd.DataFrame(columns=ALL_FEATURE_COLS)

    # ── Coordinates → distance, angle, flipped coords
    coords = df[["x", "y"]].astype(float).values
    flipped = np.array([_flip_to_attacking(x, y) for x, y in coords])
    df["x_adj"] = flipped[:, 0]
    df["y_adj"] = flipped[:, 1]
    da = np.array([_shot_distance_angle(x, y) for x, y in flipped])
    df["distance"] = da[:, 0]
    df["angle"] = da[:, 1]
    df["distance_x_angle"] = df["distance"] * df["angle"]

    # ── Categorical → one-hot
    shot_type = df.get("shot_type", pd.Series([""] * len(df))).fillna("").astype(str)
    shot_type_norm = shot_type.str.lower().str.strip()
    for st in KNOWN_SHOT_TYPES:
        df[f"shot_type_{st}"] = (shot_type_norm == st).astype(int)
    # Catch-all: unknown shot types get a fallback "other" via the residual
    known_mask = shot_type_norm.isin(KNOWN_SHOT_TYPES)
    df["shot_type_other"] = (~known_mask).astype(int)

    # ── Situation
    if "home_skaters" not in df.columns or "away_skaters" not in df.columns:
        sit = df.get("situation_code", pd.Series(["1551"] * len(df))).astype(str)
        parsed = np.array([_parse_situation(s) for s in sit])
        df["home_skaters"] = parsed[:, 0]
        df["away_skaters"] = parsed[:, 1]
    df["home_skaters"] = df["home_skaters"].clip(0, 6)
    df["away_skaters"] = df["away_skaters"].clip(0, 6)
    df["skaters_diff"] = df["home_skaters"] - df["away_skaters"]

    # ── Flags
    df["is_rebound"] = df.get("is_rebound", pd.Series([0] * len(df))).fillna(0).astype(int)
    df["is_rush"] = df.get("is_rush", pd.Series([0] * len(df))).fillna(0).astype(int)
    df["is_empty_net"] = df.get("is_empty_net", pd.Series([0] * len(df))).fillna(0).astype(int)
    df["is_home"] = df.get("is_home", pd.Series([0] * len(df))).fillna(0).astype(int)
    df["period"] = df.get("period", pd.Series([1] * len(df))).fillna(1).clip(1, 5).astype(int)
    df["time_seconds"] = df.get("time_seconds", pd.Series([0.0] * len(df))).fillna(0.0).astype(float)

    return df[ALL_FEATURE_COLS + ["shot_type_other"]]


# ── Training ────────────────────────────────────────────────────────────

def _target(shots: pd.DataFrame) -> np.ndarray:
    """Extract y target (1 if goal, 0 otherwise)."""
    if "is_goal" in shots.columns:
        return shots["is_goal"].astype(int).values
    if "event_type" in shots.columns:
        return (shots["event_type"] == "goal").astype(int).values
    raise ValueError("shots DataFrame must have 'is_goal' or 'event_type' column")


def train_xg_model(
    shots: pd.DataFrame,
    out_path: Optional[Path] = None,
    holdout_season: Optional[int] = None,
    random_state: int = 42,
) -> Dict:
    """
    Train a logistic regression xG model on the given shots.

    The holdout is by season (not random): if holdout_season is given, the
    most-recent full season is held out for evaluation. If not given, we
    use a random 80/20 split.

    Returns a report dict with metrics.
    """
    if out_path is None:
        out_path = MODEL_PATH

    if shots.empty or "x" not in shots.columns or "y" not in shots.columns:
        raise ValueError("shots must be non-empty with x/y columns")

    # Filter to valid rows: have x/y, finite, not too far from the net
    df = shots.dropna(subset=["x", "y"]).copy()
    df = df[np.isfinite(df["x"]) & np.isfinite(df["y"])]
    # Drop impossible coordinates
    df = df[(df["x"].abs() <= 100) & (df["y"].abs() <= 50)]

    if df.empty:
        raise ValueError("No valid shots after cleaning")

    X_df = build_features(df)
    y = _target(df)

    if holdout_season is not None and "season_start" in df.columns:
        train_mask = df["season_start"] != holdout_season
        X_train, y_train = X_df[train_mask], y[train_mask]
        X_val, y_val = X_df[~train_mask], y[~train_mask]
        if len(X_val) == 0:
            logger.warning(f"No shots for holdout season {holdout_season}, using random split")
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_df, y, test_size=0.2, random_state=random_state, stratify=y
            )
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_df, y, test_size=0.2, random_state=random_state, stratify=y
        )

    feature_cols = list(X_df.columns)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = LogisticRegression(max_iter=1000, C=1.0, random_state=random_state)
    model.fit(X_train_s, y_train)

    # ── Metrics
    train_proba = model.predict_proba(X_train_s)[:, 1]
    val_proba = model.predict_proba(X_val_s)[:, 1]
    train_pred = (train_proba >= 0.5).astype(int)
    val_pred = (val_proba >= 0.5).astype(int)

    report = {
        "model_type": "LogisticRegression",
        "feature_cols": feature_cols,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "train": {
            "auc": float(roc_auc_score(y_train, train_proba)),
            "brier": float(brier_score_loss(y_train, train_proba)),
            "logloss": float(log_loss(y_train, train_proba)),
            "accuracy": float(accuracy_score(y_train, train_pred)),
            "goal_rate": float(y_train.mean()),
        },
        "val": {
            "auc": float(roc_auc_score(y_val, val_proba)),
            "brier": float(brier_score_loss(y_val, val_proba)),
            "logloss": float(log_loss(y_val, val_proba)),
            "accuracy": float(accuracy_score(y_val, val_pred)),
            "goal_rate": float(y_val.mean()),
        },
        "mean_predicted_xg": float(val_proba.mean()),
        "total_predicted_goals": float(val_proba.sum()),
        "total_actual_goals": int(y_val.sum()),
    }

    # ── Per-feature coefficient (for debugging)
    coefs = list(zip(feature_cols, model.coef_[0]))
    coefs.sort(key=lambda c: abs(c[1]), reverse=True)
    report["top_coefficients"] = [
        {"feature": f, "coef": float(c)} for f, c in coefs[:15]
    ]

    # ── Save model + scaler
    artifact = {"model": model, "scaler": scaler, "feature_cols": feature_cols}
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(
        f"xG model trained. val AUC={report['val']['auc']:.4f} "
        f"brier={report['val']['brier']:.4f} logloss={report['val']['logloss']:.4f} "
        f"→ {out_path}"
    )

    return report


# ── Inference ───────────────────────────────────────────────────────────

def load_xg_model(path: Optional[Path] = None):
    """Load a trained xG model artifact. Returns dict {model, scaler, feature_cols}."""
    if path is None:
        path = MODEL_PATH
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_xg(shots: pd.DataFrame, model_artifact: Optional[dict] = None) -> np.ndarray:
    """
    Predict xG for a shots DataFrame. Returns array of P(is_goal), one per row.

    If model_artifact is None, loads from disk. If the model file doesn't
    exist, returns an array of league-average goals-per-shot (~0.092) as
    a fallback so the simulation can keep running.
    """
    if model_artifact is None:
        try:
            model_artifact = load_xg_model()
        except FileNotFoundError:
            logger.warning("No trained xG model found, returning league-average 0.092")
            return np.full(len(shots), 0.092, dtype=float)

    if shots.empty:
        return np.zeros(0, dtype=float)

    # Drop rows with NaN x/y or impossible coordinates — the model
    # cannot handle NaN inputs. The remaining rows keep their original
    # positional index so callers can map back.
    valid_mask = (
        shots["x"].notna() & shots["y"].notna()
        & np.isfinite(shots["x"]) & np.isfinite(shots["y"])
        & (shots["x"].abs() <= 100) & (shots["y"].abs() <= 50)
    )
    if not valid_mask.any():
        return np.full(len(shots), 0.092, dtype=float)

    valid_shots = shots[valid_mask]
    X = build_features(valid_shots)
    if X.empty:
        return np.full(len(shots), 0.092, dtype=float)
    # Drop any rows where features are still NaN/inf — build_features
    # uses fillna(0) for most columns, but if a new column is added
    # without a fill, NaN can sneak in. We re-mask before scaling.
    feat_finite = np.isfinite(X.values).all(axis=1)
    if not feat_finite.all():
        valid_mask = valid_mask.copy()
        valid_mask[valid_mask.values] = feat_finite
        valid_shots = valid_shots[feat_finite]
        X = X[feat_finite]
    if X.empty:
        return np.full(len(shots), 0.092, dtype=float)
    scaler = model_artifact["scaler"]
    feature_cols = model_artifact["feature_cols"]
    # Ensure all expected cols are present (in case of empty feature build)
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
    Xs = scaler.transform(X[feature_cols])
    out_proba = model_artifact["model"].predict_proba(Xs)[:, 1]

    # Re-assemble in original positions: invalid rows get the league
    # average so the caller sees a numeric value per row, not a sparse array.
    out = np.full(len(shots), 0.092, dtype=float)
    out[valid_mask.values] = out_proba
    return out


__all__ = [
    "build_features",
    "train_xg_model",
    "load_xg_model",
    "predict_xg",
    "MODEL_PATH",
    "REPORT_PATH",
    "ALL_FEATURE_COLS",
]
