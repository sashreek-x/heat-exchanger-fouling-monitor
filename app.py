import pandas as pd
import plotly.express as px
import streamlit as st

from fouling_monitor import TARGET, engineer_features, score, synthetic_data, train_models

st.set_page_config(page_title="Fouling Monitor", page_icon="🌡️", layout="wide")
st.title("Heat Exchanger Fouling Monitor")
st.caption("Train, compare, and monitor fouling-resistance predictions from process data.")

with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Upload labelled process CSV", type="csv")
    threshold = st.number_input(
        "Maintenance threshold (m²·K/W)", min_value=0.00001,
        max_value=0.001, value=0.00008, step=0.000005, format="%.6f"
    )
    st.caption("Demo data is used until a CSV is supplied.")

try:
    raw = pd.read_csv(upload) if upload else synthetic_data()
    data = engineer_features(raw)
    results, validation = train_models(data)
except (ValueError, pd.errors.ParserError) as error:
    st.error(str(error))
    st.stop()

best = results[0]
scored = score(data, best.pipeline, threshold)
latest = scored.iloc[-1]

metric_a, metric_b, metric_c, metric_d = st.columns(4)
metric_a.metric("Selected model", best.name)
metric_b.metric("Validation MAE", f"{best.mae:.6f}")
metric_c.metric("Latest predicted Rf", f"{latest['predicted_fouling_resistance_m2k_w']:.6f}")
metric_d.metric("Status", latest["alert"], delta=None)

tab_monitor, tab_models, tab_data = st.tabs(["Monitor", "Model comparison", "Data quality"])
with tab_monitor:
    chart_data = scored.copy()
    chart_data["row"] = range(1, len(chart_data) + 1)
    x_axis = "timestamp" if "timestamp" in chart_data.columns else "row"
    fig = px.line(chart_data, x=x_axis,
                  y=["predicted_fouling_resistance_m2k_w", TARGET],
                  labels={"value": "Fouling resistance (m²·K/W)", "variable": "Series"},
                  title="Fouling resistance: measured vs predicted")
    fig.add_hline(y=threshold, line_dash="dash", line_color="#ef7d32",
                  annotation_text="maintenance threshold")
    st.plotly_chart(fig, use_container_width=True)
    alerts = scored.loc[scored["alert"] != "Normal", [
        col for col in ["timestamp", "runtime_hours", "pressure_drop_kpa",
                         "predicted_fouling_resistance_m2k_w", "alert"] if col in scored
    ]]
    st.subheader("Active maintenance candidates")
    st.dataframe(alerts.tail(20), use_container_width=True, hide_index=True)

with tab_models:
    comparison = pd.DataFrame([{
        "Model": result.name, "MAE": result.mae, "RMSE": result.rmse, "R²": result.r2
    } for result in results])
    st.dataframe(comparison.style.format({"MAE": "{:.6f}", "RMSE": "{:.6f}", "R²": "{:.3f}"}),
                 use_container_width=True, hide_index=True)
    long_validation = validation.reset_index(drop=True).melt(
        id_vars="actual", var_name="model", value_name="prediction"
    )
    st.plotly_chart(px.scatter(long_validation, x="actual", y="prediction", color="model",
                               title="Holdout validation: prediction vs actual"),
                    use_container_width=True)

with tab_data:
    st.write(f"**{len(data):,}** records loaded. Missing values are median-imputed during modelling.")
    st.dataframe(data.head(25), use_container_width=True, hide_index=True)
    st.download_button("Download scored data", scored.to_csv(index=False),
                       file_name="fouling_predictions.csv", mime="text/csv")
