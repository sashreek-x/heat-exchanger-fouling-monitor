import pandas as pd
import plotly.express as px
import streamlit as st

from fouling_monitor import (
    TARGET, add_thickness_estimate, engineer_features, forecast_hours_to_thickness,
    score, synthetic_data, train_models,
)

st.set_page_config(page_title="Fouling Monitor", page_icon="🌡️", layout="wide")
st.title("Heat Exchanger Fouling Monitor")
st.caption("Train, compare, and monitor fouling-resistance predictions from process data.")

with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Upload labelled process CSV", type="csv")
    resistance_threshold = st.number_input(
        "Fouling-resistance threshold (m²·K/W)", min_value=0.00001,
        max_value=0.001, value=0.00008, step=0.000005, format="%.6f"
    )
    deposit_k = st.number_input(
        "Deposit thermal conductivity (W/m·K)", min_value=0.01,
        max_value=10.0, value=0.70, step=0.05,
        help="Use a lab- or literature-backed effective conductivity for the expected deposit."
    )
    thickness_limit_mm = st.number_input(
        "Maintenance thickness limit (mm)", min_value=0.001,
        max_value=10.0, value=0.060, step=0.005
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
scored = score(data, best.pipeline, resistance_threshold)
scored = add_thickness_estimate(scored, deposit_k)
hours_to_limit = forecast_hours_to_thickness(scored, thickness_limit_mm)
latest = scored.iloc[-1]

metric_a, metric_b, metric_c, metric_d = st.columns(4)
metric_a.metric("Selected model", best.name)
metric_b.metric("Validation MAE", f"{best.mae:.6f}")
metric_c.metric("Latest predicted Rf", f"{latest['predicted_fouling_resistance_m2k_w']:.6f}")
if hours_to_limit is None:
    forecast_label = "Trend unavailable"
elif hours_to_limit == 0:
    forecast_label = "Limit reached"
else:
    forecast_label = f"{hours_to_limit / 24:.1f} days"
metric_d.metric("Thickness limit forecast", forecast_label)

tab_monitor, tab_models, tab_data = st.tabs(["Monitor", "Model comparison", "Data quality"])
with tab_monitor:
    chart_data = scored.copy()
    chart_data["row"] = range(1, len(chart_data) + 1)
    x_axis = "timestamp" if "timestamp" in chart_data.columns else "row"
    fig = px.line(chart_data, x=x_axis,
                  y=["predicted_fouling_resistance_m2k_w", TARGET],
                  labels={"value": "Fouling resistance (m²·K/W)", "variable": "Series"},
                  title="Fouling resistance: measured vs predicted")
    fig.add_hline(y=resistance_threshold, line_dash="dash", line_color="#ef7d32",
                  annotation_text="maintenance threshold")
    st.plotly_chart(fig, use_container_width=True)
    alerts = scored.loc[scored["alert"] != "Normal", [
        col for col in ["timestamp", "runtime_hours", "pressure_drop_kpa",
                         "predicted_fouling_resistance_m2k_w",
                         "predicted_fouling_thickness_mm", "alert"] if col in scored
    ]]
    st.subheader("Active maintenance candidates")
    st.dataframe(alerts.tail(20), use_container_width=True, hide_index=True)
    thickness_chart = px.line(
        chart_data, x=x_axis, y="predicted_fouling_thickness_mm",
        title="Equivalent fouling thickness forecast",
        labels={"predicted_fouling_thickness_mm": "Thickness (mm)"},
    )
    thickness_chart.add_hline(y=thickness_limit_mm, line_dash="dash", line_color="#ef7d32",
                             annotation_text="thickness limit")
    st.plotly_chart(thickness_chart, use_container_width=True)
    if hours_to_limit is not None:
        message = ("The limit is already reached; schedule inspection/cleaning now."
                   if hours_to_limit == 0 else
                   f"Based on the latest trend, the {thickness_limit_mm:.3f} mm limit is expected in "
                   f"about {hours_to_limit:.0f} operating hours ({hours_to_limit / 24:.1f} days).")
        st.info(message)
    else:
        st.caption("A time-to-limit estimate needs at least eight recent measurements with a rising thickness trend.")

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
