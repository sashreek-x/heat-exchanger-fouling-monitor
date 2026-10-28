"""Input validation helpers."""
from __future__ import annotations
import pandas as pd

REQUIRED_NUMERIC = [
    "hot_inlet_temp_C", "hot_outlet_temp_C", "cold_inlet_temp_C",
    "cold_outlet_temp_C", "hot_mass_flow_kg_s", "cold_mass_flow_kg_s",
    "hot_cp_kJ_kgK", "cold_cp_kJ_kgK",
]

def validate_frame(df: pd.DataFrame, target: str | None = None) -> list[str]:
    """Return actionable validation warnings; do not silently rewrite the input."""
    warnings = []
    if df.empty:
        warnings.append("The uploaded dataset has no rows.")
        return warnings
    if target and target not in df.columns:
        warnings.append(f"Target column '{target}' is missing.")
    missing = [c for c in REQUIRED_NUMERIC if c not in df.columns]
    if missing:
        warnings.append("Missing recommended engineering columns: " + ", ".join(missing))
    for col in REQUIRED_NUMERIC:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.isna().mean() > 0.25:
                warnings.append(f"'{col}' has more than 25% non-numeric/missing values.")
    if "timestamp" in df.columns:
        parsed = pd.to_datetime(df["timestamp"], errors="coerce")
        if parsed.notna().sum() == 0:
            warnings.append("Column 'timestamp' exists but no timestamps could be parsed.")
    return warnings
