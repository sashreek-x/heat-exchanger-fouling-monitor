"""Domain logic for a heat-exchanger fouling prediction prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


TARGET = "fouling_resistance_m2k_w"
FEATURES = [
    "hot_in_c", "hot_out_c", "cold_in_c", "cold_out_c",
    "flow_rate_kg_s", "pressure_drop_kpa", "runtime_hours",
    "hot_delta_c", "cold_delta_c", "mean_approach_c",
]
REQUIRED_INPUTS = [
    "hot_in_c", "hot_out_c", "cold_in_c", "cold_out_c",
    "flow_rate_kg_s", "pressure_drop_kpa", "runtime_hours",
]


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    mae: float
    rmse: float
    r2: float


def synthetic_data(rows: int = 730, seed: int = 19) -> pd.DataFrame:
    """Create demonstration data with operationally plausible fouling behavior."""
    rng = np.random.default_rng(seed)
    runtime = np.arange(rows) * 8.0
    cleaning_cycle = np.mod(runtime, 1_200)
    hot_in = 160 + rng.normal(0, 3.5, rows)
    cold_in = 28 + rng.normal(0, 2.0, rows)
    flow = np.clip(14 + rng.normal(0, 1.3, rows), 8, None)
    base_fouling = 0.000025 + cleaning_cycle * 0.000000055
    pressure_drop = 42 + cleaning_cycle * 0.010 + rng.normal(0, 1.8, rows)
    heat_penalty = base_fouling * 25_000 + rng.normal(0, 0.5, rows)
    hot_out = hot_in - (35 - heat_penalty + rng.normal(0, 1.1, rows))
    cold_out = cold_in + (28 - heat_penalty + rng.normal(0, 1.1, rows))
    target = np.clip(
        base_fouling + (15 - flow) * 0.000006 + rng.normal(0, 0.000009, rows),
        0.000005,
        None,
    )
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=rows, freq="8h"),
        "hot_in_c": hot_in, "hot_out_c": hot_out,
        "cold_in_c": cold_in, "cold_out_c": cold_out,
        "flow_rate_kg_s": flow, "pressure_drop_kpa": pressure_drop,
        "runtime_hours": runtime, TARGET: target,
    })


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate raw inputs and add temperature-derived predictors."""
    missing = [column for column in REQUIRED_INPUTS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    data = frame.copy()
    for column in REQUIRED_INPUTS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["hot_delta_c"] = data["hot_in_c"] - data["hot_out_c"]
    data["cold_delta_c"] = data["cold_out_c"] - data["cold_in_c"]
    data["mean_approach_c"] = ((data["hot_in_c"] - data["cold_out_c"]) +
                               (data["hot_out_c"] - data["cold_in_c"])) / 2
    return data


def train_models(frame: pd.DataFrame) -> tuple[list[ModelResult], pd.DataFrame]:
    data = engineer_features(frame)
    if TARGET not in data.columns:
        raise ValueError(f"Training data needs a '{TARGET}' column.")
    y = pd.to_numeric(data[TARGET], errors="coerce")
    usable = data.loc[y.notna()].copy()
    y = y.loc[y.notna()]
    if len(usable) < 30:
        raise ValueError("At least 30 labelled rows are needed to train a model.")
    x_train, x_test, y_train, y_test = train_test_split(
        usable[FEATURES], y, test_size=0.25, random_state=42
    )
    candidates: Iterable[tuple[str, Pipeline]] = [
        ("Ridge regression", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])),
        ("Random forest", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1
            )),
        ])),
    ]
    results = []
    predictions = pd.DataFrame({"actual": y_test}, index=y_test.index)
    for name, pipeline in candidates:
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        predictions[name] = predicted
        results.append(ModelResult(
            name=name, pipeline=pipeline,
            mae=mean_absolute_error(y_test, predicted),
            rmse=mean_squared_error(y_test, predicted) ** 0.5,
            r2=r2_score(y_test, predicted),
        ))
    return sorted(results, key=lambda item: item.rmse), predictions


def score(frame: pd.DataFrame, model: Pipeline, threshold: float) -> pd.DataFrame:
    data = engineer_features(frame)
    data["predicted_fouling_resistance_m2k_w"] = model.predict(data[FEATURES])
    data["alert"] = np.where(
        data["predicted_fouling_resistance_m2k_w"] >= threshold,
        "Inspect / clean", "Normal"
    )
    return data
