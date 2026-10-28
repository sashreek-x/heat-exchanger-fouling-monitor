"""Train and compare regression models from a CSV file."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from src.features import engineer_features
from src.metrics import regression_metrics
from src.data.validation import validate_frame

def build_models(random_state=42):
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=random_state, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state, n_estimators=200, learning_rate=0.04, max_depth=2, loss="huber"),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=random_state, max_iter=250, learning_rate=0.06, l2_regularization=1.0),
        "mlp": TransformedTargetRegressor(regressor=MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", alpha=0.001, max_iter=1500, early_stopping=True, random_state=random_state), transformer=StandardScaler()),
    }

def train_compare(df: pd.DataFrame, target: str, output_dir="artifacts", test_size=0.2, random_state=42, use_time_split=False):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found.")
    warnings = validate_frame(df, target)
    work = df.copy()
    y = pd.to_numeric(work[target], errors="coerce")
    valid = y.notna() & np.isfinite(y)
    work, y = work.loc[valid].copy(), y.loc[valid].astype(float)
    if len(work) < 30:
        raise ValueError("At least 30 labelled rows are recommended for a meaningful train/holdout workflow.")
    X = engineer_features(work)
    if X.shape[1] == 0:
        raise ValueError("No usable predictor columns remain after feature engineering.")
    # Prefer a chronological holdout if a usable timestamp exists.
    if use_time_split and "timestamp" in work.columns:
        times = pd.to_datetime(work["timestamp"], errors="coerce")
        order = times.sort_values(kind="stable").index
        split_at = int(len(order) * (1-test_size))
        train_idx, test_idx = order[:split_at], order[split_at:]
        X_train, X_test, y_train, y_test = X.loc[train_idx], X.loc[test_idx], y.loc[train_idx], y.loc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    numeric_features = list(X_train.columns)
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                              ("scaler", StandardScaler())]), numeric_features)
    ], remainder="drop")
    rows, fitted, estimators = [], {}, {}
    cv = KFold(n_splits=min(5, max(2, len(X_train)//10)), shuffle=True, random_state=random_state)
    for name, estimator in build_models(random_state).items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        try:
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            metrics = regression_metrics(y_test, pred)
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=None)
            row = {"model": name, **metrics, "CV_RMSE_mean": float(-cv_scores.mean()), "CV_RMSE_std": float(cv_scores.std())}
            rows.append(row); fitted[name] = pipe
            estimators[name] = {"test_predictions": pred.tolist()}
        except Exception as exc:
            rows.append({"model": name, "error": f"{type(exc).__name__}: {exc}"})
    results = pd.DataFrame(rows)
    if "RMSE" not in results:
        raise RuntimeError("All model training attempts failed.")
    successful = results.loc[results["RMSE"].notna()].sort_values("RMSE")
    if successful.empty:
        raise RuntimeError("No model produced valid holdout metrics.")
    best_name = str(successful.iloc[0]["model"])
    best = fitted[best_name]
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "model_comparison.csv", index=False)
    predictions = pd.DataFrame({"y_true": y_test, "y_pred": best.predict(X_test)}, index=y_test.index).sort_index()
    predictions.to_csv(out / "best_model_holdout_predictions.csv", index_label="row_index")
    joblib.dump(best, out / "best_model.joblib")
    metadata = {
        "target": target, "best_model": best_name, "features": numeric_features,
        "n_rows_labelled": int(len(work)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "split": "chronological" if use_time_split and "timestamp" in work.columns else "random holdout",
        "random_state": random_state, "warnings": warnings,
        "data_note": "Metrics describe only the supplied dataset and split; synthetic demo data is not real-world validation."
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # Permutation importance computed on untouched holdout set.
    try:
        pi = permutation_importance(best, X_test, y_test, n_repeats=8, random_state=random_state,
                                    scoring="neg_root_mean_squared_error")
        imp = pd.DataFrame({"feature": X_test.columns, "importance_mean": pi.importances_mean,
                            "importance_std": pi.importances_std}).sort_values("importance_mean", ascending=False)
        imp.to_csv(out / "permutation_importance.csv", index=False)
    except Exception as exc:
        (out / "importance_warning.txt").write_text(str(exc), encoding="utf-8")
    return results, best_name, metadata

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="CSV with one row per observation")
    p.add_argument("--target", default="fouling_resistance_m2K_W")
    p.add_argument("--output-dir", default="artifacts")
    p.add_argument("--time-split", action="store_true", help="Use chronological split if timestamp is available")
    p.add_argument("--test-size", type=float, default=0.2)
    args = p.parse_args()
    df = pd.read_csv(args.data)
    results, best, meta = train_compare(df, args.target, args.output_dir, args.test_size, use_time_split=args.time_split)
    print("\nModel comparison (lower RMSE/MAE is better; higher R² is better):")
    print(results.to_string(index=False))
    print(f"\nSelected model: {best}\nSaved artifacts to: {args.output_dir}")
    print("NOTE:", meta["data_note"])
if __name__ == "__main__":
    main()
