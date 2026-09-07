# Heat Exchanger Fouling Monitor

A local prototype for predicting heat-exchanger fouling resistance from operating and thermal data. It supports a CSV upload or a realistic synthetic dataset, compares regression models, and surfaces maintenance alerts. It also converts predicted resistance to an equivalent fouling thickness and estimates time to a user-selected thickness limit.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

## Expected data

Upload a CSV with these columns (units may be adapted in the app):

| Column | Meaning |
| --- | --- |
| `timestamp` | Measurement time (optional) |
| `hot_in_c`, `hot_out_c` | Hot-side inlet/outlet temperature, °C |
| `cold_in_c`, `cold_out_c` | Cold-side inlet/outlet temperature, °C |
| `flow_rate_kg_s` | Process flow rate, kg/s |
| `pressure_drop_kpa` | Exchanger pressure drop, kPa |
| `runtime_hours` | Hours since cleaning or inspection |
| `fouling_resistance_m2k_w` | Target fouling resistance, m²·K/W |

## Thickness forecast

The monitoring view derives equivalent thickness using `Rf = thickness / k`. Set the deposit's effective thermal conductivity from laboratory data or defensible literature, then choose the maintenance thickness limit. The app fits a linear trend to the latest 60 measurements and forecasts the operating hours remaining until that limit.

This is a planning signal, not a guarantee: validate it against known cleaning events before using it for operational decisions.
