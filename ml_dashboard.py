"""Streamlit interface for heat-exchanger fouling ML."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from src.data.validation import validate_frame
from src.features import engineer_features
from src.train import train_compare
from src.heat_transfer import calculate_row

st.set_page_config(page_title="Heat Exchanger Fouling Monitor", page_icon="🌡️", layout="wide")
st.title("🌡️ Heat Exchanger Fouling Monitor")
st.caption("Heat-transfer engineering + machine-learning decision support. Not a safety instrument.")

with st.sidebar:
    st.header("Data & model")
    uploaded = st.file_uploader("Upload observation CSV", type=["csv"])
    target = st.text_input("Target column", "fouling_resistance_m2K_W")
    time_split = st.checkbox("Chronological holdout (recommended for time-series)", value=True)
    st.markdown("---")
    st.write("Demo data is synthetic. Upload documented data for meaningful evaluation.")

if uploaded is None:
    st.info("Upload a CSV to begin, or generate a local demo dataset using the README instructions.")
    st.markdown("**Recommended columns:** temperatures, mass flows, heat capacities, pressure drop, timestamp, and a documented target label.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

st.subheader("Dataset preview")
st.write(f"{len(df):,} rows × {len(df.columns)} columns")
st.dataframe(df.head(30), use_container_width=True)
warnings = validate_frame(df, target)
for warning in warnings:
    st.warning(warning)

if target not in df.columns:
    st.error(f"Target '{target}' is not in this file. Available columns: {', '.join(df.columns)}")
    st.stop()

if st.button("Train and compare models", type="primary"):
    with st.spinner("Training models and evaluating the holdout set…"):
        try:
            outdir = Path("artifacts")
            results, best, meta = train_compare(df, target, outdir, use_time_split=time_split)
            st.session_state["results"] = results
            st.session_state["best"] = best
            st.session_state["meta"] = meta
            st.session_state["outdir"] = str(outdir)
        except Exception as exc:
            st.error(f"Training failed: {type(exc).__name__}: {exc}")

if "results" in st.session_state:
    results = st.session_state["results"]
    best = st.session_state["best"]
    meta = st.session_state["meta"]
    st.subheader("Model comparison")
    st.dataframe(results, use_container_width=True)
    successful = results[results["RMSE"].notna()].sort_values("RMSE")
    if not successful.empty:
        bestrow = successful.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Selected model", best)
        c2.metric("Holdout MAE", f"{bestrow['MAE']:.6g}")
        c3.metric("Holdout RMSE", f"{bestrow['RMSE']:.6g}")
        c4.metric("Holdout R²", f"{bestrow['R2']:.4f}")
        st.caption(f"Split: {meta['split']} · labelled rows: {meta['n_rows_labelled']} · test rows: {meta['n_test']}")
        st.warning(meta["data_note"])
        pred_path = Path(st.session_state["outdir"]) / "best_model_holdout_predictions.csv"
        if pred_path.exists():
            pred = pd.read_csv(pred_path)
            fig, ax = plt.subplots()
            ax.scatter(pred["y_true"], pred["y_pred"], alpha=0.65)
            lo = min(pred["y_true"].min(), pred["y_pred"].min())
            hi = max(pred["y_true"].max(), pred["y_pred"].max())
            ax.plot([lo, hi], [lo, hi], linestyle="--")
            ax.set_xlabel("Measured / labelled target")
            ax.set_ylabel("Predicted target")
            ax.set_title("Holdout predictions")
            st.pyplot(fig)
            plt.close(fig)
        imp_path = Path(st.session_state["outdir"]) / "permutation_importance.csv"
        if imp_path.exists():
            st.subheader("Permutation importance")
            imp = pd.read_csv(imp_path)
            st.dataframe(imp, use_container_width=True)
            fig, ax = plt.subplots()
            top = imp.head(12).sort_values("importance_mean")
            ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
            ax.set_xlabel("Increase in RMSE when shuffled (higher = more useful)")
            st.pyplot(fig)
            plt.close(fig)
        st.download_button("Download model comparison CSV", results.to_csv(index=False), "model_comparison.csv", "text/csv")
        st.download_button("Download metadata JSON", json.dumps(meta, indent=2), "metadata.json", "application/json")

st.markdown("---")
st.subheader("Optional physics calculation")
st.caption("This calculation uses a counter-current LMTD and requires comparable operating conditions and design area.")
with st.expander("Calculate heat duty, LMTD and U from one observation"):
    a, b, c = st.columns(3)
    with a:
        hi = st.number_input("Hot inlet (°C)", value=150.0)
        ho = st.number_input("Hot outlet (°C)", value=110.0)
        mh = st.number_input("Hot mass flow (kg/s)", min_value=0.01, value=8.0)
        cph = st.number_input("Hot cp (kJ/kg·K)", min_value=0.01, value=2.1)
    with b:
        ci = st.number_input("Cold inlet (°C)", value=25.0)
        co = st.number_input("Cold outlet (°C)", value=65.0)
        mc = st.number_input("Cold mass flow (kg/s)", min_value=0.01, value=10.0)
        cpc = st.number_input("Cold cp (kJ/kg·K)", min_value=0.01, value=4.18)
    with c:
        area = st.number_input("Heat-transfer area (m²)", min_value=0.01, value=25.0)
        uclean = st.number_input("Clean U (W/m²·K; 0 = skip)", min_value=0.0, value=0.0)
    if st.button("Calculate thermal indicators"):
        try:
            result = calculate_row({
                "hot_mass_flow_kg_s": mh, "hot_cp_kJ_kgK": cph, "hot_inlet_temp_C": hi, "hot_outlet_temp_C": ho,
                "cold_mass_flow_kg_s": mc, "cold_cp_kJ_kgK": cpc, "cold_inlet_temp_C": ci, "cold_outlet_temp_C": co,
            }, area, uclean if uclean > 0 else None)
            st.json(result)
            if "added_fouling_resistance_m2K_W" in result and result["added_fouling_resistance_m2K_W"] < 0:
                st.warning("Estimated added fouling resistance is negative. Check measurement quality, baseline comparability and assumptions.")
        except Exception as exc:
            st.error(str(exc))
