"""Generate explicitly synthetic demo data to smoke-test the application."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def make_demo_data(rows=1500, seed=42):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-01-01", periods=rows, freq="h")
    hours = np.arange(rows, dtype=float)
    hot_in = rng.normal(155, 7, rows)
    cold_in = rng.normal(28, 2, rows)
    hot_out = hot_in - rng.normal(35, 4, rows)
    cold_out = cold_in + rng.normal(23, 3, rows)
    hot_flow = rng.normal(8.5, 0.8, rows).clip(4, 12)
    cold_flow = rng.normal(10.0, 1.0, rows).clip(5, 15)
    hot_cp = rng.normal(2.1, 0.08, rows).clip(1.6, 2.6)
    cold_cp = rng.normal(4.18, 0.04, rows).clip(3.8, 4.4)
    pressure_drop = (20 + 0.3*cold_flow**2 + 0.004*hours + rng.normal(0, 1.5, rows)).clip(5, None)
    # Illustrative hidden synthetic relationship only; not a physical plant model.
    target = (0.00008 + 0.000000035*hours + 0.0000009*np.maximum(pressure_drop-25, 0)
              + 0.0000005*np.maximum(100-hot_out, 0) + rng.normal(0, 0.000004, rows)).clip(0.00001, 0.001)
    return pd.DataFrame({
        "timestamp": t.astype(str), "hot_inlet_temp_C": hot_in, "hot_outlet_temp_C": hot_out,
        "cold_inlet_temp_C": cold_in, "cold_outlet_temp_C": cold_out,
        "hot_mass_flow_kg_s": hot_flow, "cold_mass_flow_kg_s": cold_flow,
        "hot_cp_kJ_kgK": hot_cp, "cold_cp_kJ_kgK": cold_cp,
        "pressure_drop_kPa": pressure_drop, "operating_hours": hours,
        "fouling_resistance_m2K_W": target,
    })

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/demo_synthetic.csv")
    p.add_argument("--rows", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.rows < 30: p.error("--rows must be at least 30")
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    make_demo_data(a.rows, a.seed).to_csv(out, index=False)
    print(f"Wrote {a.rows} SYNTHETIC rows to {out}. Not real or experimentally validated data.")
if __name__ == "__main__": main()
