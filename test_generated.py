from discount import calculate_discount

import pytest
from your_module import calculate_discount  # replace 'your_module' with the actual module name

def test_valid_price_and_discount():
    assert calculate_discount(price=100, discount_pct=20) == 80.0

def test_non_numeric_price():
    with pytest.raises(TypeError):
        calculate_discount(price="string", discount_pct=20)

def test_discount_percentage_outside_valid_range():
    with pytest.raises(ValueError):
        calculate_discount(price=100, discount_pct=-1)
    with pytest.raises(ValueError):
        calculate_discount(price=100, discount_pct=101)

def test_zero_discount_percentage():
    assert calculate_discount(price=100, discount_pct=0) == 100.0

def test_100_percent_discount():
    assert calculate_discount(price=100, discount_pct=100) == 0.0

def test_negative_price():
    assert calculate_discount(price=-100, discount_pct=20) == -80.0

def test_floating_point_price_and_discount():
    assert calculate_discount(price=100.5, discount_pct=20.5) == 80.025

def test_edge_cases():
    assert calculate_discount(price=0, discount_pct=0) == 0.0
    assert calculate_discount(price=0, discount_pct=100) == 0.0

def test_max_and_min_float_values():
    with pytest.raises(OverflowError):
        calculate_discount(price=float('inf'), discount_pct=20)
    with pytest.raises(OverflowError):
        calculate_discount(price=float('-inf'), discount_pct=20)