# Heat Exchanger Fouling Monitor — ML + Heat Transfer

A college-level software project for **monitoring heat-exchanger thermal performance and estimating fouling-related degradation** using machine learning and heat-transfer calculations.

> **Important data-integrity note:** The included demo-data generator creates *synthetic illustrative data*. It is for checking the pipeline and UI only. It is not industrial data, and demo metrics must not be presented as experimentally validated results. For a defensible study, add a documented public dataset or measured/simulated data with a clearly defined target.

## What is included

- Data validation, unit-aware feature engineering, missing-value handling and a reproducible train/test workflow.
- Baseline models: Linear Regression, Random Forest, Gradient Boosting, and optional HistGradientBoosting.
- Optional MLP neural-network model.
- Metrics: MAE, RMSE, R² and MAPE (with safeguards around near-zero targets).
- Holdout evaluation and cross-validation; model selection based on validation RMSE.
- Feature importance for tree models; permutation importance for general models.
- Heat-transfer calculations for heat duty, LMTD, overall heat-transfer coefficient and fouling-resistance estimate when required measurements/design inputs are available.
- Streamlit dashboard for CSV upload, model training, prediction, trend visualization and CSV export.
- Tests, sample schema, configuration, and a technical report outline.

## Quick start

Python 3.10–3.12 recommended.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 1. Run a pipeline smoke test with synthetic demo data

```bash
python -m src.data.make_demo_data --output data/demo_synthetic.csv --rows 1500
python -m src.train --data data/demo_synthetic.csv --target fouling_resistance_m2K_W --output-dir artifacts
```

This confirms that the pipeline runs. The generated data encodes a known synthetic relationship, so the resulting metrics are **not evidence of real-world performance**.

### 2. Launch the dashboard

```bash
streamlit run app.py
```

The dashboard lets you upload a CSV and select a target column. Use data with one row per timestamp/operating observation and stable units.

## Dataset contract

Recommended input columns (use these exact names where possible):

| Column | Unit | Meaning |
|---|---|---|
| `timestamp` | ISO datetime | Optional observation time |
| `hot_inlet_temp_C` | °C | Hot/process-side inlet temperature |
| `hot_outlet_temp_C` | °C | Hot/process-side outlet temperature |
| `cold_inlet_temp_C` | °C | Cold/utility-side inlet temperature |
| `cold_outlet_temp_C` | °C | Cold/utility-side outlet temperature |
| `hot_mass_flow_kg_s` | kg/s | Hot-side mass flow |
| `cold_mass_flow_kg_s` | kg/s | Cold-side mass flow |
| `hot_cp_kJ_kgK` | kJ/(kg·K) | Hot-side specific heat capacity |
| `cold_cp_kJ_kgK` | kJ/(kg·K) | Cold-side specific heat capacity |
| `pressure_drop_kPa` | kPa | Optional pressure drop |
| `operating_hours` | h | Optional elapsed operating time |
| `fouling_resistance_m2K_W` | m²·K/W | Target label, only if known/derived with documented assumptions |

At minimum, choose a target that exists in your real dataset. Do not train on a target column calculated using the same target's future information or leak target-derived features into the inputs.

## Scientific framing and target choice

Two distinct tasks are supported conceptually and should not be conflated:

1. **Supervised fouling-resistance prediction**: requires trustworthy labels for fouling resistance, from a documented experiment, validated simulation, or a justified calculation.
2. **Performance-deviation monitoring**: estimates expected clean-condition performance from operating conditions and flags sustained deviation. This can be more realistic when fouling itself is not directly measured, but requires a defensible clean baseline and operational context.

A model cannot identify fouling uniquely from temperatures alone in every plant: flow changes, product composition, sensor drift, control-valve behavior and changing duty can produce similar signals. Validate by time-based split and, where possible, by holding out a whole exchanger or operating campaign.

## Heat-transfer equations used

- Heat duty (single side): `Q = m_dot * cp * (T_in - T_out)`; with cp in kJ/(kg·K), Q is kW.
- Counter-current LMTD: `LMTD = (ΔT1 - ΔT2) / ln(ΔT1 / ΔT2)`, where `ΔT1 = Th,in - Tc,out` and `ΔT2 = Th,out - Tc,in`. Valid only if both terminal differences are positive; near-equal differences use the limiting value.
- Overall coefficient: `U = Q / (A * LMTD)`; if Q is kW and A is m², convert Q to W before computing U in W/(m²·K).
- Approximate added fouling resistance: `R_f ≈ 1/U_fouled - 1/U_clean`, only when comparable area, flow regime, properties, and correction factors are used. Negative values may indicate noise or an invalid baseline; they should be investigated rather than silently clipped.

These are simplified calculations. For real shell-and-tube exchangers, consider correction factors, phase change, heat losses, property variation, flow arrangement and uncertainty.

## Project structure

```text
.
├── app.py
├── config.yaml
├── requirements.txt
├── data/
│   ├── README.md
│   └── sample_schema.csv
├── docs/
│   ├── methodology.md
│   └── report_template.md
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── make_demo_data.py
│   │   └── validation.py
│   ├── features.py
│   ├── heat_transfer.py
│   ├── metrics.py
│   └── train.py
└── tests/
    ├── test_features.py
    └── test_heat_transfer.py
```

## Recommended final evaluation protocol

1. Document dataset source, license, process, sensor tags, units, cleaning steps and target derivation.
2. Sort chronologically where timestamps exist; use a time-based holdout rather than a random split for time-series process data.
3. Fit preprocessing **only on training data** using a scikit-learn Pipeline.
4. Compare models on the same holdout and report MAE, RMSE, R² and (if appropriate) MAPE.
5. Include baseline comparisons, residual plots, feature-importance caveats and error by operating regime.
6. Report actual sample count, train/test period, data limitations, and no claims unsupported by evidence.
7. Save model and metadata; do not deploy for safety-critical decisions without independent engineering validation.

## Limitations

- Synthetic demo data is not a public or industrial dataset.
- Feature importance is associational, not causal.
- R² can be negative on a holdout set and MAPE is unstable when targets approach zero.
- A trained model is not proof of a fouling mechanism; model outputs are decision support, not a replacement for engineering inspection.
