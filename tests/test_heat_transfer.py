import math
import pytest
from src.heat_transfer import heat_duty_kW, lmtd_counterflow_K, overall_U_W_m2K, added_fouling_resistance_m2K_W

def test_heat_duty():
    assert heat_duty_kW(2, 4, 100, 80) == pytest.approx(160)

def test_lmtd_equal_differences():
    assert lmtd_counterflow_K(100, 80, 40, 60) == pytest.approx(40)

def test_lmtd_invalid():
    with pytest.raises(ValueError):
        lmtd_counterflow_K(100, 50, 70, 60)

def test_overall_u_units():
    assert overall_U_W_m2K(10, 2, 25) == pytest.approx(200)

def test_added_fouling_resistance():
    assert added_fouling_resistance_m2K_W(100, 200) == pytest.approx(0.005)
