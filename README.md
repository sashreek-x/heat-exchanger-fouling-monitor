# Heat Exchanger Fouling Monitor

A local prototype for predicting heat-exchanger fouling resistance from operating and thermal data. It supports a CSV upload or a realistic synthetic dataset, compares regression models, and surfaces maintenance alerts.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
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

The target column is only required for model training. For production, train on labelled history, persist the approved model in a governed artifact store, and score live process data through a protected service rather than exposing credentials in this app.
