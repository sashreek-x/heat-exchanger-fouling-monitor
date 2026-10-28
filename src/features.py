"""Leakage-conscious feature engineering for heat-exchanger observations."""
from __future__ import annotations
import numpy as np
import pandas as pd

ENGINEERING_COLUMNS = [
    "hot_inlet_temp_C", "hot_outlet_temp_C", "cold_inlet_temp_C",
    "cold_outlet_temp_C", "hot_mass_flow_kg_s", "cold_mass_flow_kg_s",
    "hot_cp_kJ_kgK", "cold_cp_kJ_kgK", "pressure_drop_kPa", "operating_hours",
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create physically motivated features from available measured inputs.

    Target columns are deliberately not read here. Derived values are NaN when
    the required measurements are absent or physically invalid.
    """
    X = df.copy()
    for col in ENGINEERING_COLUMNS:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    if {"hot_inlet_temp_C", "hot_outlet_temp_C"} <= set(X.columns):
        X["hot_delta_T_K"] = X["hot_inlet_temp_C"] - X["hot_outlet_temp_C"]
    if {"cold_outlet_temp_C", "cold_inlet_temp_C"} <= set(X.columns):
        X["cold_delta_T_K"] = X["cold_outlet_temp_C"] - X["cold_inlet_temp_C"]
    if {"hot_mass_flow_kg_s", "hot_cp_kJ_kgK", "hot_delta_T_K"} <= set(X.columns):
        X["hot_duty_kW"] = X["hot_mass_flow_kg_s"] * X["hot_cp_kJ_kgK"] * X["hot_delta_T_K"]
    if {"cold_mass_flow_kg_s", "cold_cp_kJ_kgK", "cold_delta_T_K"} <= set(X.columns):
        X["cold_duty_kW"] = X["cold_mass_flow_kg_s"] * X["cold_cp_kJ_kgK"] * X["cold_delta_T_K"]

    # Remove raw timestamp and known target-like columns from predictors.
    drop_cols = [c for c in X.columns if c.lower() in {
        "timestamp", "fouling_resistance_m2k_w", "fouling_resistance",
        "target", "label", "fouling_label"
    }]
    X = X.drop(columns=drop_cols, errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan)
    # Drop all-empty and constant columns to reduce avoidable pipeline errors.
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=True) > 1] if len(X) > 1 else X
    return X
