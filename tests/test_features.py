import pandas as pd
from src.features import engineer_features
from src.data.make_demo_data import make_demo_data

def test_engineering_features_and_no_target_leakage():
    df = make_demo_data(40)
    X = engineer_features(df)
    assert "hot_duty_kW" in X.columns
    assert "cold_duty_kW" in X.columns
    assert "fouling_resistance_m2K_W" not in X.columns
    assert "timestamp" not in X.columns
