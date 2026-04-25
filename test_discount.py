import pytest
from discount import *

class TestCalculateDiscount:
    def test_calculate_discount_with_valid_inputs(self):
        assert calculate_discount(100, 10) == 90.0

    def test_calculate_discount_with_non_numeric_price(self):
        with pytest.raises(TypeError):
            calculate_discount('100', 10)

    def test_calculate_discount_with_non_numeric_discount_pct(self):
        with pytest.raises(TypeError):
            calculate_discount(100, '10')

    def test_calculate_discount_with_discount_pct_less_than_zero(self):
        with pytest.raises(ValueError):
            calculate_discount(100, -10)

    def test_calculate_discount_with_discount_pct_greater_than_100(self):
        with pytest.raises(ValueError):
            calculate_discount(100, 110)

    def test_calculate_discount_with_discount_pct_equal_to_zero(self):
        assert calculate_discount(100, 0) == 100.0

    def test_calculate_discount_with_discount_pct_equal_to_100(self):
        assert calculate_discount(100, 100) == 0.0

    def test_calculate_discount_with_negative_price(self):
        assert calculate_discount(-100, 10) == -90.0

    def test_calculate_discount_with_zero_price(self):
        assert calculate_discount(0, 10) == 0.0

    def test_calculate_discount_with_very_large_price(self):
        assert calculate_discount(1e10, 10) == 9e9

    def test_calculate_discount_with_very_large_discount_pct(self):
        with pytest.raises(ValueError):
            calculate_discount(100, 1e10)

    def test_calculate_discount_with_edge_cases_for_rounding(self):
        assert calculate_discount(0.005, 10) == 0.0

    def test_calculate_discount_with_identical_inputs(self):
        assert calculate_discount(100, 10) == calculate_discount(100, 10)

    def test_calculate_discount_with_multiple_valid_inputs(self):
        assert calculate_discount(100, 10) == 90.0
        assert calculate_discount(200, 20) == 160.0
        assert calculate_discount(300, 30) == 210.0

