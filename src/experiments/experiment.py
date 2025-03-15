from typing import List
import math

import numpy as np

from typing import List, Tuple, Union
import pandas as pd
from typing import Any
from typing import Dict
from datetime import datetime
from typing import Callable

import math
import random


Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]
Vector = Union[List[float], np.ndarray]


def matrix_transpose(matrix: np.ndarray) -> np.ndarray:
    rows, cols = matrix.shape
    result = np.zeros((cols, rows))
    for i in range(rows):
        for j in range(cols):
            result[j, i] = matrix[i, j]
    return result


def numpy_mean(arr: np.ndarray) -> float:
    total = 0
    for i in range(len(arr)):
        total += arr[i]
    return total / len(arr)


def element_wise_log(arr: np.ndarray) -> np.ndarray:
    # Use boolean indexing to apply log only to the positive elements
    result = np.zeros_like(arr)
    positive_mask = arr > 0
    result[positive_mask] = np.log(arr[positive_mask])
    return result
