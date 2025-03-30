from typing import Callable, Any
import time


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


def knapsack(weights: list[int], values: list[int], capacity: int, n: int) -> int:
    # Create a memoization table to store results of subproblems
    memo = [[-1 for _ in range(capacity + 1)] for _ in range(n + 1)]

    def knapsack_recursive(n: int, capacity: int) -> int:
        # Base case: If no items left or capacity is 0
        if n == 0 or capacity == 0:
            return 0

        # If the result is already in the memo table, return it
        if memo[n][capacity] != -1:
            return memo[n][capacity]

        if weights[n - 1] > capacity:
            memo[n][capacity] = knapsack_recursive(n - 1, capacity)
        else:
            memo[n][capacity] = max(
                values[n - 1] + knapsack_recursive(n - 1, capacity - weights[n - 1]),
                knapsack_recursive(n - 1, capacity),
            )

        return memo[n][capacity]

    return knapsack_recursive(n, capacity)


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
