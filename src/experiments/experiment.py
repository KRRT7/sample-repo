def gcd_recursive(a: int, b: int) -> int:
    """Calculate greatest common divisor using Euclidean algorithm with iteration."""
    while b != 0:
        a, b = b, a % b
    return a
