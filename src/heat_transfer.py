"""Transparent heat-transfer calculations with basic physical validity checks."""
from __future__ import annotations
import math
import numpy as np

def heat_duty_kW(mass_flow_kg_s: float, cp_kJ_kgK: float, inlet_C: float, outlet_C: float) -> float:
    if mass_flow_kg_s < 0 or cp_kJ_kgK <= 0:
        raise ValueError("Mass flow must be non-negative and specific heat must be positive.")
    return float(mass_flow_kg_s * cp_kJ_kgK * (inlet_C - outlet_C))

def lmtd_counterflow_K(hot_in_C: float, hot_out_C: float, cold_in_C: float, cold_out_C: float) -> float:
    d1 = hot_in_C - cold_out_C
    d2 = hot_out_C - cold_in_C
    if d1 <= 0 or d2 <= 0:
        raise ValueError("Invalid terminal temperature differences for this counter-current formula.")
    if math.isclose(d1, d2, rel_tol=1e-8, abs_tol=1e-8):
        return float((d1 + d2) / 2.0)
    return float((d1 - d2) / math.log(d1 / d2))

def overall_U_W_m2K(duty_kW: float, area_m2: float, lmtd_K: float) -> float:
    if duty_kW < 0 or area_m2 <= 0 or lmtd_K <= 0:
        raise ValueError("Duty must be non-negative; area and LMTD must be positive.")
    return float(duty_kW * 1000.0 / (area_m2 * lmtd_K))

def added_fouling_resistance_m2K_W(U_fouled: float, U_clean: float) -> float:
    if U_fouled <= 0 or U_clean <= 0:
        raise ValueError("Both overall heat-transfer coefficients must be positive.")
    return float(1.0 / U_fouled - 1.0 / U_clean)

def calculate_row(row: dict, area_m2: float, clean_U_W_m2K: float | None = None) -> dict:
    qh = heat_duty_kW(row["hot_mass_flow_kg_s"], row["hot_cp_kJ_kgK"],
                      row["hot_inlet_temp_C"], row["hot_outlet_temp_C"])
    qc = heat_duty_kW(row["cold_mass_flow_kg_s"], row["cold_cp_kJ_kgK"],
                      row["cold_outlet_temp_C"], row["cold_inlet_temp_C"])
    lmtd = lmtd_counterflow_K(row["hot_inlet_temp_C"], row["hot_outlet_temp_C"],
                              row["cold_inlet_temp_C"], row["cold_outlet_temp_C"])
    # Hot-side duty is reported; duty imbalance is retained as a diagnostic.
    U = overall_U_W_m2K(qh, area_m2, lmtd)
    result = {"hot_duty_kW": qh, "cold_duty_kW": qc, "duty_imbalance_pct": (
        abs(qh-qc) / max(abs(qh), abs(qc), 1e-9) * 100.0
    ), "lmtd_K": lmtd, "U_W_m2K": U}
    if clean_U_W_m2K is not None:
        result["added_fouling_resistance_m2K_W"] = added_fouling_resistance_m2K_W(U, clean_U_W_m2K)
    return result
