from typing import Dict, Tuple, Callable, Any, List, TypeVar
import time


def time_based_cache(expiry_seconds: int) -> Callable:
    """Manual implementation of a time-based cache decorator."""

    def decorator(func: Callable) -> Callable:
        cache: Dict[str, Tuple[Any, float]] = {}

        def wrapper(*args, **kwargs) -> Any:
            key_parts = [repr(arg) for arg in args]
            key_parts.extend(f"{k}:{repr(v)}" for k, v in sorted(kwargs.items()))
            key = ":".join(key_parts)

            current_time = time.time()

            if key in cache:
                result, timestamp = cache[key]
                if current_time - timestamp < expiry_seconds:
                    return result

            result = func(*args, **kwargs)

            cache[key] = (result, current_time)

            return result

        return wrapper

    return decorator


def fibonacci(n: int) -> int:
    """Calculate Fibonacci numbers without memoization."""

    if n <= 1:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


def matrix_chain_order(matrices: List[Tuple[int, int]]) -> int:
    """
    Find the minimum number of operations needed to multiply a chain of matrices.

    Args:
        matrices: A list of matrix dimensions as tuples (rows, cols)

    Returns:
        Minimum number of operations
    """
    n = len(matrices)
    memo = {}

    def dp(i: int, j: int) -> int:
        if i == j:
            return 0
        if (i, j) in memo:
            return memo[(i, j)]

        min_ops = float("inf")

        for k in range(i, j):
            # matrices[i][0] corresponds to the number of rows in the ith matrix
            # matrices[k][1] corresponds to the number of columns in the kth matrix
            # matrices[j][1] corresponds to the number of columns in the jth matrix
            cost = (
                dp(i, k)
                + dp(k + 1, j)
                + matrices[i][0] * matrices[k][1] * matrices[j][1]
            )
            min_ops = min(min_ops, cost)

        memo[(i, j)] = min_ops
        return min_ops

    return dp(0, n - 1)
