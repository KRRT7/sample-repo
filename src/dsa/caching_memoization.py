from typing import Callable, Any
import time
from functools import lru_cache


# def time_based_cache(expiry_seconds: int) -> Callable:
#     """Manual implementation of a time-based cache decorator."""

#     def decorator(func: Callable) -> Callable:
#         cache: dict[str, tuple[Any, float]] = {}

#         def wrapper(*args, **kwargs) -> Any:
#             key_parts = [repr(arg) for arg in args]
#             key_parts.extend(f"{k}:{repr(v)}" for k, v in sorted(kwargs.items()))
#             key = ":".join(key_parts)

#             current_time = time.time()

#             if key in cache:
#                 result, timestamp = cache[key]
#                 if current_time - timestamp < expiry_seconds:
#                     return result

#             result = func(*args, **kwargs)

#             cache[key] = (result, current_time)

#             return result

#         return wrapper

#     return decorator


# def matrix_chain_order(matrices: list[tuple[int, int]]) -> int:
#     """
#     Find the minimum number of operations needed to multiply a chain of matrices.

#     Args:
#         matrices: A list of matrix dimensions as tuples (rows, cols)

#     Returns:
#         Minimum number of operations
#     """
#     n = len(matrices)

#     def dp(i: int, j: int) -> int:
#         if i == j:
#             return 0

#         min_ops = float("inf")

#         for k in range(i, j):
#             cost = (
#                 dp(i, k)
#                 + dp(k + 1, j)
#                 + matrices[i][0] * matrices[k][1] * matrices[j][1]
#             )
#             min_ops = min(min_ops, cost)

#         return min_ops

#     return dp(0, n - 1)


@lru_cache(None)
def binomial_coefficient(n: int, k: int) -> int:
    if k == 0 or k == n:
        return 1
    return binomial_coefficient(n - 1, k - 1) + binomial_coefficient(n - 1, k)


def edit_distance(str1: str, str2: str, m: int, n: int) -> int:
    if m == 0:
        return n
    if n == 0:
        return m

    if str1[m - 1] == str2[n - 1]:
        return edit_distance(str1, str2, m - 1, n - 1)

    return 1 + min(
        edit_distance(str1, str2, m, n - 1),  # Insert
        edit_distance(str1, str2, m - 1, n),  # Remove
        edit_distance(str1, str2, m - 1, n - 1),  # Replace
    )


# def coin_change(coins: list[int], amount: int, index: int) -> int:
#     if amount == 0:
#         return 1
#     if amount < 0 or index >= len(coins):
#         return 0

#     return coin_change(coins, amount - coins[index], index) + coin_change(
#         coins, amount, index + 1
#     )


def knapsack(weights: list[int], values: list[int], capacity: int, n: int) -> int:
    if n == 0 or capacity == 0:
        return 0

    if weights[n - 1] > capacity:
        return knapsack(weights, values, capacity, n - 1)

    return max(
        values[n - 1] + knapsack(weights, values, capacity - weights[n - 1], n - 1),
        knapsack(weights, values, capacity, n - 1),
    )


def word_break(s: str, word_dict: list[str], start: int) -> bool:
    if start == len(s):
        return True

    for end in range(start + 1, len(s) + 1):
        if s[start:end] in word_dict and word_break(s, word_dict, end):
            return True

    return False


def catalan(n: int) -> int:
    if n <= 1:
        return 1

    result = 0
    for i in range(n):
        result += catalan(i) * catalan(n - i - 1)

    return result


# Time-based cache decorator
def time_based_cache(expiry_seconds: int) -> Callable:
    """Manual implementation of a time-based cache decorator."""

    def decorator(func: Callable) -> Callable:
        cache: dict[str, tuple[Any, float]] = {}

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


# Optimized matrix_chain_order function with memoization
def matrix_chain_order(matrices: list[tuple[int, int]]) -> int:
    """
    Find the minimum number of operations needed to multiply a chain of matrices.

    Args:
        matrices: A list of matrix dimensions as tuples (rows, cols)

    Returns:
        Minimum number of operations
    """
    n = len(matrices)

    dp_cache = {}

    def dp(i: int, j: int) -> int:
        if i == j:
            return 0

        if (i, j) in dp_cache:
            return dp_cache[(i, j)]

        min_ops = float("inf")

        for k in range(i, j):
            cost = (
                dp(i, k)
                + dp(k + 1, j)
                + matrices[i][0] * matrices[k][1] * matrices[j][1]
            )
            min_ops = min(min_ops, cost)

        dp_cache[(i, j)] = min_ops
        return min_ops

    return dp(0, n - 1)
