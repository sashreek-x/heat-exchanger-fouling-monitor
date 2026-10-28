"""Regression metrics with explicit handling of near-zero targets."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if not len(y_true):
        raise ValueError("No finite target/prediction pairs are available.")
    denom = np.abs(y_true) > 1e-8
    mape = float(np.mean(np.abs((y_true[denom] - y_pred[denom]) / y_true[denom])) * 100) if denom.any() else float("nan")
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "MAPE_pct_nonzero_targets": mape,
        "n_evaluated": int(len(y_true)),
    }
