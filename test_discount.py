import pytest
from discount import calculate_discount

def test_valid_numeric_price_with_valid_discount_percentage():
    assert calculate_discount(100.0, 10.0) == 90.0

def test_non_numeric_price_type():
    with pytest.raises(TypeError):
        calculate_discount("string", 10.0)
    with pytest.raises(TypeError):
        calculate_discount([1, 2, 3], 10.0)

def test_price_as_negative_number():
    assert calculate_discount(-100.0, 0.0) == -100.0
    assert calculate_discount(-100.0, 10.0) == -90.0

def test_price_as_zero():
    assert calculate_discount(0.0, 0.0) == 0.0
    assert calculate_discount(0.0, 10.0) == 0.0

def test_discount_percentage_as_zero():
    assert calculate_discount(100.0, 0.0) == 100.0

def test_discount_percentage_as_100():
    assert calculate_discount(100.0, 100.0) == 0.0

def test_discount_percentage_as_negative_number():
    with pytest.raises(ValueError):
        calculate_discount(100.0, -10.0)

def test_discount_percentage_as_number_greater_than_100():
    with pytest.raises(ValueError):
        calculate_discount(100.0, 110.0)

def test_discount_percentage_as_non_numeric_type():
    with pytest.raises(TypeError):
        calculate_discount(100.0, "string")
    with pytest.raises(TypeError):
        calculate_discount(100.0, [1, 2, 3])

def test_boundary_values_for_discount_percentage():
    assert calculate_discount(100.0, 0.01) == 99.99
    assert calculate_discount(100.0, 99.99) == 0.01

def test_rounding_of_result_to_two_decimal_places():
    assert calculate_discount(100.0, 10.005) == 89.99

def test_result_with_large_values():
    assert calculate_discount(1000000.0, 10.0) == 900000.0
    assert calculate_discount(100.0, 99.99) == 0.01
