def matrix_sum(matrix: list[list[int]]) -> list[int]:
    return [row_sum for row in matrix if (row_sum := sum(row)) > 0]
