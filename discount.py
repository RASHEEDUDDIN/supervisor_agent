def calculate_discount(price: float, discount_pct: float) -> float:
    """Apply a percentage discount to a price and return the final amount."""
    if not isinstance(price, (int, float)):
        raise TypeError("price must be numeric")
    if discount_pct < 0 or discount_pct > 100:
        raise ValueError(f"discount_pct must be 0-100, got {discount_pct}")
    return round(price * (1 - discount_pct / 100), 2)
