from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import math
from typing import Callable, Any
import random


Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]
Vector = Union[List[float], np.ndarray]


def numpy_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape
    if cols_A != rows_B:
        raise ValueError("Incompatible matrices")
    result = np.zeros((rows_A, cols_B))
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i, j] += A[i, k] * B[k, j]
    return result


def numpy_filter(arr: np.ndarray) -> np.ndarray:
    result = []
    for i in arr:
        if i % 2 == 0:
            result.append(i)
    return np.array(result)


def elementwise_multiply_by_2(arr: np.ndarray) -> np.ndarray:
    result = np.zeros_like(arr)
    for i in range(len(arr)):
        result[i] = arr[i] * 2
    return result


# derived from https://github.com/langchain-ai/langchain/pull/8151
def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(X) == 0 or len(Y) == 0:
        return np.array([])
    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}."
        )

    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
    similarity = np.dot(X / X_norm, (Y / Y_norm).T)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    return similarity


def cosine_similarity_top_k(
    X: Matrix,
    Y: Matrix,
    top_k: Optional[int] = 5,
    score_threshold: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    """Row-wise cosine similarity with optional top-k and score threshold filtering.

    Args:
        X: Matrix.
        Y: Matrix, same width as X.
        top_k: Max number of results to return.
        score_threshold: Minimum cosine similarity of results.

    Returns:
        Tuple of two lists. First contains two-tuples of indices (X_idx, Y_idx),
            second contains corresponding cosine similarities.
    """
    if len(X) == 0 or len(Y) == 0:
        return [], []
    score_array = cosine_similarity(X, Y)
    sorted_idxs = score_array.flatten().argsort()[::-1]
    top_k = top_k or len(sorted_idxs)
    top_idxs = sorted_idxs[:top_k]
    score_threshold = score_threshold or -1.0
    top_idxs = top_idxs[score_array.flatten()[top_idxs] > score_threshold]
    ret_idxs = [(x // score_array.shape[1], x % score_array.shape[1]) for x in top_idxs]
    scores = score_array.flatten()[top_idxs].tolist()
    return ret_idxs, scores


def pca(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    # this function is used to perform PCA on a dataset
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Calculate covariance matrix
    cov_matrix = np.zeros((X.shape[1], X.shape[1]))
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            for k in range(X.shape[0]):
                cov_matrix[i, j] += X_centered[k, i] * X_centered[k, j]
            cov_matrix[i, j] /= X.shape[0] - 1

    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort by eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select top n_components
    components = eigenvectors[:, :n_components]

    # Project data onto components
    X_transformed = np.zeros((X.shape[0], n_components))
    for i in range(X.shape[0]):
        for j in range(n_components):
            for k in range(X.shape[1]):
                X_transformed[i, j] += X_centered[i, k] * components[k, j]

    return X_transformed, components


def kmeans_clustering(
    X: np.ndarray, k: int, max_iter: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = X.shape[0]

    # Randomly initialize centroids
    centroid_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[centroid_indices]

    for _ in range(max_iter):
        # Assign samples to closest centroids
        labels = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            min_dist = float("inf")
            for j in range(k):
                dist = 0
                for feat in range(X.shape[1]):
                    dist += (X[i, feat] - centroids[j, feat]) ** 2
                dist = np.sqrt(dist)
                if dist < min_dist:
                    min_dist = dist
                    labels[i] = j

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k)

        for i in range(n_samples):
            cluster = labels[i]
            counts[cluster] += 1
            for feat in range(X.shape[1]):
                new_centroids[cluster, feat] += X[i, feat]

        for j in range(k):
            if counts[j] > 0:
                for feat in range(X.shape[1]):
                    new_centroids[j, feat] /= counts[j]

        # Check for convergence
        if np.array_equal(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids, labels


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")

    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


def matrix_inverse(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")

    n = matrix.shape[0]
    identity = np.eye(n)
    augmented = np.hstack((matrix, identity))

    for i in range(n):
        pivot = augmented[i, i]
        augmented[i] = augmented[i] / pivot

        for j in range(n):
            if i != j:
                factor = augmented[j, i]
                augmented[j] = augmented[j] - factor * augmented[i]

    return augmented[:, n:]


def manual_convolution_1d(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    signal_len = len(signal)
    kernel_len = len(kernel)
    result_len = signal_len - kernel_len + 1
    result = np.zeros(result_len)

    for i in range(result_len):
        for j in range(kernel_len):
            result[i] += signal[i + j] * kernel[j]

    return result


def numpy_sort(arr: np.ndarray) -> np.ndarray:
    result = arr.copy()
    n = len(result)

    for i in range(n):
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]

    return result


def euclidean_distance_matrix(X: Matrix, Y: Matrix) -> np.ndarray:
    """Calculate pairwise Euclidean distances between rows of X and Y."""
    X = np.array(X)
    Y = np.array(Y)

    result = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            squared_diff = 0
            for k in range(X.shape[1]):
                squared_diff += (X[i, k] - Y[j, k]) ** 2
            result[i, j] = np.sqrt(squared_diff)

    return result


def softmax(x: np.ndarray) -> np.ndarray:
    result = np.zeros_like(x)
    for i in range(len(x)):
        exp_x = np.exp(x[i] - np.max(x[i]))
        result[i] = exp_x / np.sum(exp_x)
    return result


def FFT(x: np.ndarray) -> np.ndarray:
    """implementation of Fast Fourier Transform."""
    n = len(x)
    if n == 1:
        return x

    # Split into even and odd indices
    even = FFT(x[0::2])
    odd = FFT(x[1::2])

    factor = np.exp(-2j * np.pi * np.arange(n) / n)

    result = np.zeros(n, dtype=complex)
    half_n = n // 2
    for k in range(half_n):
        result[k] = even[k] + factor[k] * odd[k]
        result[k + half_n] = even[k] - factor[k] * odd[k]

    return result


# still to verify


def singular_value_decomposition(
    A: np.ndarray, k: int = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Manual implementation of truncated SVD."""
    # Get covariance matrices
    AAT = np.dot(A, A.T)
    ATA = np.dot(A.T, A)

    # Get eigenvalues and eigenvectors
    U_vals, U_vecs = np.linalg.eigh(AAT)
    V_vals, V_vecs = np.linalg.eigh(ATA)

    # Sort in descending order
    U_idx = U_vals.argsort()[::-1]
    U_vals, U_vecs = U_vals[U_idx], U_vecs[:, U_idx]

    V_idx = V_vals.argsort()[::-1]
    V_vals, V_vecs = V_vals[V_idx], V_vecs[:, V_idx]

    # Calculate singular values
    S = np.sqrt(U_vals)

    # Truncate if k is specified
    if k is not None:
        U_vecs = U_vecs[:, :k]
        S = S[:k]
        V_vecs = V_vecs[:, :k]

    # Return U, S, V^T
    return U_vecs, np.diag(S), V_vecs.T


def image_rotation(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    angle_radians = np.radians(angle_degrees)
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)

    height, width = image.shape[:2]
    center_y, center_x = height // 2, width // 2

    # Calculate new dimensions
    new_height = int(abs(height * cos_theta) + abs(width * sin_theta))
    new_width = int(abs(width * cos_theta) + abs(height * sin_theta))

    # Create output image
    rotated = np.zeros(
        (new_height, new_width, image.shape[2])
        if len(image.shape) > 2
        else (new_height, new_width)
    )

    # Calculate new center
    new_center_y, new_center_x = new_height // 2, new_width // 2

    # Rotation matrix
    for y in range(new_height):
        for x in range(new_width):
            # Translate to origin
            offset_y = y - new_center_y
            offset_x = x - new_center_x

            # Apply rotation
            original_y = int(offset_y * cos_theta - offset_x * sin_theta + center_y)
            original_x = int(offset_y * sin_theta + offset_x * cos_theta + center_x)

            # Check if original pixel is inside the image
            if 0 <= original_y < height and 0 <= original_x < width:
                rotated[y, x] = image[original_y, original_x]

    return rotated


def gaussian_blur(
    image: np.ndarray, kernel_size: int = 3, sigma: float = 1.0
) -> np.ndarray:
    """Apply Gaussian blur to an image manually."""
    # Create Gaussian kernel
    k = kernel_size // 2
    y, x = np.ogrid[-k : k + 1, -k : k + 1]
    kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()

    # Get image dimensions
    height, width = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else image.shape[2]

    # Create output image
    output = np.zeros_like(image)

    # Apply convolution
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                weighted_sum = 0
                weight_sum = 0

                for ky in range(-k, k + 1):
                    for kx in range(-k, k + 1):
                        ny, nx = y + ky, x + kx

                        if 0 <= ny < height and 0 <= nx < width:
                            if channels == 1:
                                pixel_value = image[ny, nx]
                            else:
                                pixel_value = image[ny, nx, c]

                            weight = kernel[ky + k, kx + k]
                            weighted_sum += pixel_value * weight
                            weight_sum += weight

                if weight_sum > 0:
                    if channels == 1:
                        output[y, x] = weighted_sum / weight_sum
                    else:
                        output[y, x, c] = weighted_sum / weight_sum

    return output


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Apply histogram equalization to an image manually."""
    # Get image dimensions
    height, width = image.shape
    total_pixels = height * width

    # Calculate histogram
    histogram = np.zeros(256, dtype=int)
    for y in range(height):
        for x in range(width):
            histogram[image[y, x]] += 1

    # Calculate cumulative distribution function
    cdf = np.zeros(256, dtype=float)
    cdf[0] = histogram[0] / total_pixels
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + histogram[i] / total_pixels

    # Apply histogram equalization
    equalized = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            equalized[y, x] = np.round(cdf[image[y, x]] * 255)

    return equalized


def matrix_decomposition_LU(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """LU decomposition."""
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        # Upper triangular matrix
        for k in range(i, n):
            sum_val = 0
            for j in range(i):
                sum_val += L[i, j] * U[j, k]
            U[i, k] = A[i, k] - sum_val

        # Lower triangular matrix
        L[i, i] = 1  # Diagonal elements of L are 1
        for k in range(i + 1, n):
            sum_val = 0
            for j in range(i):
                sum_val += L[k, j] * U[j, i]
            if U[i, i] == 0:
                raise ValueError("Cannot perform LU decomposition")
            L[k, i] = (A[k, i] - sum_val) / U[i, i]

    return L, U


def gradient_descent(
    X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, iterations: int = 1000
) -> np.ndarray:
    """Implement gradient descent for linear regression"""
    m, n = X.shape
    weights = np.zeros(n)

    for _ in range(iterations):
        # Make predictions
        predictions = np.zeros(m)
        for i in range(m):
            for j in range(n):
                predictions[i] += X[i, j] * weights[j]

        # Calculate error
        errors = predictions - y

        # Update weights
        gradient = np.zeros(n)
        for j in range(n):
            for i in range(m):
                gradient[j] += errors[i] * X[i, j]
            gradient[j] /= m

        # Apply gradient descent
        for j in range(n):
            weights[j] -= learning_rate * gradient[j]

    return weights


def dataframe_filter(df: pd.DataFrame, column: str, value: Any) -> pd.DataFrame:
    """Filter a DataFrame manually based on a column value."""
    indices = []
    for i in range(len(df)):
        if df.iloc[i][column] == value:
            indices.append(i)
    return df.iloc[indices].reset_index(drop=True)


def groupby_mean(df: pd.DataFrame, group_col: str, value_col: str) -> dict[Any, float]:
    """Group by a column and compute mean of values in another column."""
    sums = {}
    counts = {}

    for i in range(len(df)):
        group = df.iloc[i][group_col]
        value = df.iloc[i][value_col]

        if group in sums:
            sums[group] += value
            counts[group] += 1
        else:
            sums[group] = value
            counts[group] = 1

    result = {}
    for group in sums:
        result[group] = sums[group] / counts[group]

    return result


def dataframe_merge(
    left: pd.DataFrame, right: pd.DataFrame, left_on: str, right_on: str
) -> pd.DataFrame:
    """Manually merge two DataFrames (inner join)."""
    result_data = []
    left_cols = list(left.columns)
    right_cols = [col for col in right.columns if col != right_on]

    # Create lookup dictionary for right DataFrame
    right_dict = {}
    for i in range(len(right)):
        key = right.iloc[i][right_on]
        if key not in right_dict:
            right_dict[key] = []
        right_dict[key].append(i)

    # Perform merge
    for i in range(len(left)):
        left_row = left.iloc[i]
        key = left_row[left_on]

        if key in right_dict:
            for right_idx in right_dict[key]:
                right_row = right.iloc[right_idx]
                new_row = {}

                # Add columns from left DataFrame
                for col in left_cols:
                    new_row[col] = left_row[col]

                # Add columns from right DataFrame
                for col in right_cols:
                    new_row[col] = right_row[col]

                result_data.append(new_row)

    return pd.DataFrame(result_data)


def pivot_table(
    df: pd.DataFrame, index: str, columns: str, values: str, aggfunc: str = "mean"
) -> dict[Any, dict[Any, float]]:
    """Manually create a pivot table."""
    result = {}

    # Define aggregation function
    if aggfunc == "mean":

        def agg_func(values):
            return sum(values) / len(values)

    elif aggfunc == "sum":

        def agg_func(values):
            return sum(values)

    elif aggfunc == "count":

        def agg_func(values):
            return len(values)

    else:
        raise ValueError(f"Unsupported aggregation function: {aggfunc}")

    # Group data
    grouped_data = {}
    for i in range(len(df)):
        row = df.iloc[i]
        index_val = row[index]
        column_val = row[columns]
        value = row[values]

        if index_val not in grouped_data:
            grouped_data[index_val] = {}

        if column_val not in grouped_data[index_val]:
            grouped_data[index_val][column_val] = []

        grouped_data[index_val][column_val].append(value)

    # Apply aggregation
    for index_val in grouped_data:
        result[index_val] = {}
        for column_val in grouped_data[index_val]:
            result[index_val][column_val] = agg_func(
                grouped_data[index_val][column_val]
            )

    return result


def apply_function(df: pd.DataFrame, column: str, func: Callable) -> List[Any]:
    result = []
    for i in range(len(df)):
        value = df.iloc[i][column]
        result.append(func(value))
    return result


def fillna(df: pd.DataFrame, column: str, value: Any) -> pd.DataFrame:
    result = df.copy()
    for i in range(len(df)):
        if pd.isna(df.iloc[i][column]):
            result.iloc[i, df.columns.get_loc(column)] = value
    return result


def drop_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    if subset is None:
        subset = df.columns.tolist()

    seen = set()
    keep_indices = []

    for i in range(len(df)):
        # Create a tuple of values for comparison
        values = tuple(df.iloc[i][col] for col in subset)

        if values not in seen:
            seen.add(values)
            keep_indices.append(i)

    return df.iloc[keep_indices].reset_index(drop=True)


def sort_values(df: pd.DataFrame, by: str, ascending: bool = True) -> pd.DataFrame:
    indices = list(range(len(df)))

    # Simple bubble sort
    for i in range(len(df)):
        for j in range(0, len(df) - i - 1):
            if ascending:
                condition = df.iloc[indices[j]][by] > df.iloc[indices[j + 1]][by]
            else:
                condition = df.iloc[indices[j]][by] < df.iloc[indices[j + 1]][by]

            if condition:
                indices[j], indices[j + 1] = indices[j + 1], indices[j]

    return df.iloc[indices].reset_index(drop=True)


def rolling_mean(series: pd.Series, window: int) -> List[float]:
    """Calculate rolling mean (moving average)."""
    result = []
    values = series.tolist()

    for i in range(len(values)):
        if i < window - 1:
            result.append(np.nan)
        else:
            window_sum = 0
            for j in range(window):
                window_sum += values[i - j]
            result.append(window_sum / window)

    return result


def describe(series: pd.Series) -> dict[str, float]:
    values = [v for v in series if not pd.isna(v)]
    n = len(values)

    if n == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "25%": np.nan,
            "50%": np.nan,
            "75%": np.nan,
            "max": np.nan,
        }

    # Sort values for percentiles
    sorted_values = sorted(values)

    # Calculate statistics
    mean = sum(values) / n

    # Standard deviation
    variance = sum((x - mean) ** 2 for x in values) / n
    std = variance**0.5

    # Percentiles
    def percentile(p):
        idx = int(p * n / 100)
        if idx >= n:
            idx = n - 1
        return sorted_values[idx]

    return {
        "count": n,
        "mean": mean,
        "std": std,
        "min": sorted_values[0],
        "25%": percentile(25),
        "50%": percentile(50),
        "75%": percentile(75),
        "max": sorted_values[-1],
    }


def correlation(df: pd.DataFrame) -> dict[Tuple[str, str], float]:
    """Calculate correlation matrix."""
    numeric_columns = [
        col for col in df.columns if np.issubdtype(df[col].dtype, np.number)
    ]
    n_cols = len(numeric_columns)
    result = {}

    for i in range(n_cols):
        col_i = numeric_columns[i]
        for j in range(n_cols):
            col_j = numeric_columns[j]

            # Extract non-NA values
            values_i = []
            values_j = []

            for k in range(len(df)):
                if not pd.isna(df.iloc[k][col_i]) and not pd.isna(df.iloc[k][col_j]):
                    values_i.append(df.iloc[k][col_i])
                    values_j.append(df.iloc[k][col_j])

            # Calculate correlation
            n = len(values_i)
            if n == 0:
                result[(col_i, col_j)] = np.nan
                continue

            mean_i = sum(values_i) / n
            mean_j = sum(values_j) / n

            var_i = sum((x - mean_i) ** 2 for x in values_i) / n
            var_j = sum((x - mean_j) ** 2 for x in values_j) / n

            std_i = var_i**0.5
            std_j = var_j**0.5

            if std_i == 0 or std_j == 0:
                result[(col_i, col_j)] = np.nan
                continue

            cov = (
                sum((values_i[k] - mean_i) * (values_j[k] - mean_j) for k in range(n))
                / n
            )
            corr = cov / (std_i * std_j)

            result[(col_i, col_j)] = corr

    return result


def melt(df: pd.DataFrame, id_vars: List[str], value_vars: List[str]) -> pd.DataFrame:
    result_data = []

    for i in range(len(df)):
        # Get ID values
        id_values = {id_var: df.iloc[i][id_var] for id_var in id_vars}

        # Create a new row for each value variable
        for value_var in value_vars:
            new_row = {
                **id_values,
                "variable": value_var,
                "value": df.iloc[i][value_var],
            }
            result_data.append(new_row)

    return pd.DataFrame(result_data)


def reindex(df: pd.DataFrame, new_index: List[Any]) -> pd.DataFrame:
    # Create a mapping of old indices to data
    index_map = {df.index[i]: i for i in range(len(df))}

    # Create new DataFrame with new index
    new_data = []
    for idx in new_index:
        if idx in index_map:
            new_data.append(df.iloc[index_map[idx]])
        else:
            # Insert NaN for missing indices
            new_row = pd.Series({col: np.nan for col in df.columns})
            new_data.append(new_row)

    result = pd.DataFrame(new_data)
    result.index = new_index

    return result


def qr_algorithm(
    A: List[List[float]], max_iter: int = 100, epsilon: float = 1e-10
) -> List[float]:
    """Find eigenvalues of a matrix using QR algorithm."""
    n = len(A)

    # Copy A to avoid modifying the original
    matrix = [row[:] for row in A]

    for _ in range(max_iter):
        # Compute QR decomposition using Gram-Schmidt
        Q = [[0.0 for _ in range(n)] for _ in range(n)]
        R = [[0.0 for _ in range(n)] for _ in range(n)]

        for j in range(n):
            # Copy the j-th column of matrix to v
            v = [matrix[i][j] for i in range(n)]

            for k in range(j):
                # Calculate dot product of j-th column with k-th orthogonal vector
                dot_product = sum(Q[i][k] * matrix[i][j] for i in range(n))
                R[k][j] = dot_product

                # Subtract projection
                for i in range(n):
                    v[i] -= dot_product * Q[i][k]

            # Calculate norm of the vector
            norm = math.sqrt(sum(v[i] ** 2 for i in range(n)))

            if norm > epsilon:
                for i in range(n):
                    Q[i][j] = v[i] / norm
                R[j][j] = norm
            else:
                for i in range(n):
                    Q[i][j] = 0.0

        # Compute new matrix A = R * Q
        new_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    new_matrix[i][j] += R[i][k] * Q[k][j]

        # Check for convergence
        converged = True
        for i in range(n):
            for j in range(n):
                if i > j and abs(new_matrix[i][j]) > epsilon:
                    converged = False
                    break

        matrix = new_matrix

        if converged:
            break

    # Return diagonal elements as eigenvalues
    return [matrix[i][i] for i in range(n)]


def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Multiply two matrices using the standard algorithm."""
    if not A or not B or len(A[0]) != len(B):
        raise ValueError("Invalid matrix dimensions for multiplication")

    rows_A = len(A)
    cols_A = len(A[0])
    cols_B = len(B[0])

    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]

    return result


def power_by_squaring(x: float, n: int) -> float:
    """Compute x^n using naive repeated multiplication."""
    result = 1.0
    for _ in range(n):
        result *= x
    return result


def numerical_integration_rectangle(
    f: Callable[[float], float], a: float, b: float, n: int
) -> float:
    """Integrate a function using the rectangle method with n subdivisions."""
    if a > b:
        a, b = b, a

    h = (b - a) / n
    result = 0.0

    for i in range(n):
        x = a + i * h
        result += f(x)

    return result * h


def binomial_coefficient_recursive(n: int, k: int) -> int:
    """Calculate binomial coefficient using recursive formula."""
    if k == 0 or k == n:
        return 1
    return binomial_coefficient_recursive(
        n - 1, k - 1
    ) + binomial_coefficient_recursive(n - 1, k)


def naive_matrix_determinant(matrix: List[List[float]]) -> float:
    """Calculate determinant using cofactor expansion."""
    n = len(matrix)

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    determinant = 0
    for j in range(n):
        # Create submatrix by removing first row and column j
        submatrix = []
        for i in range(1, n):
            row = []
            for k in range(n):
                if k != j:
                    row.append(matrix[i][k])
            submatrix.append(row)

        sign = (-1) ** j
        determinant += sign * matrix[0][j] * naive_matrix_determinant(submatrix)

    return determinant


def slow_matrix_inverse(matrix: List[List[float]]) -> List[List[float]]:
    """Calculate matrix inverse using cofactor method."""
    n = len(matrix)
    determinant = naive_matrix_determinant(matrix)

    if abs(determinant) < 1e-10:
        raise ValueError("Matrix is singular, cannot be inverted")

    # Calculate cofactor matrix
    cofactors = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            # Create submatrix by removing row i and column j
            submatrix = []
            for r in range(n):
                if r != i:
                    row = []
                    for c in range(n):
                        if c != j:
                            row.append(matrix[r][c])
                    submatrix.append(row)

            sign = (-1) ** (i + j)
            cofactors[i][j] = sign * naive_matrix_determinant(submatrix)

    # Transpose cofactor matrix
    adjoint = [[cofactors[j][i] for j in range(n)] for i in range(n)]

    # Divide by determinant
    inverse = [[adjoint[i][j] / determinant for j in range(n)] for i in range(n)]

    return inverse


def monte_carlo_pi(num_samples: int) -> float:
    """Estimate π using Monte Carlo method."""
    inside_circle = 0

    for _ in range(num_samples):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        if x**2 + y**2 <= 1:
            inside_circle += 1

    return 4 * inside_circle / num_samples


def sieve_of_eratosthenes(n: int) -> List[int]:
    """Find all primes up to n using sieve of Eratosthenes."""
    if n <= 1:
        return []

    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False

    return [i for i in range(2, n + 1) if is_prime[i]]


def linear_equation_solver(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve system of linear equations Ax = b using Gaussian elimination."""
    n = len(A)

    # Create augmented matrix [A|b]
    augmented = [row[:] + [b[i]] for i, row in enumerate(A)]

    # Forward elimination
    for i in range(n):
        # Find pivot
        max_idx = i
        for j in range(i + 1, n):
            if abs(augmented[j][i]) > abs(augmented[max_idx][i]):
                max_idx = j

        # Swap rows
        augmented[i], augmented[max_idx] = augmented[max_idx], augmented[i]

        # Eliminate below
        for j in range(i + 1, n):
            factor = augmented[j][i] / augmented[i][i]
            for k in range(i, n + 1):
                augmented[j][k] -= factor * augmented[i][k]

    # Back substitution
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = augmented[i][n]
        for j in range(i + 1, n):
            x[i] -= augmented[i][j] * x[j]
        x[i] /= augmented[i][i]

    return x


def newton_raphson_sqrt(x: float, epsilon: float = 1e-10, max_iter: int = 100) -> float:
    """Calculate square root using Newton-Raphson method."""
    if x < 0:
        raise ValueError("Cannot compute square root of negative number")

    if x == 0:
        return 0

    guess = x / 2  # Initial guess

    for _ in range(max_iter):
        next_guess = 0.5 * (guess + x / guess)
        if abs(next_guess - guess) < epsilon:
            return next_guess
        guess = next_guess

    return guess


def lagrange_interpolation(points: List[Tuple[float, float]], x: float) -> float:
    """Interpolate a function value using Lagrange polynomials."""
    result = 0.0
    n = len(points)

    for i in range(n):
        term = points[i][1]
        for j in range(n):
            if i != j:
                term *= (x - points[j][0]) / (points[i][0] - points[j][0])
        result += term

    return result


def bisection_method(
    f: Callable[[float], float],
    a: float,
    b: float,
    epsilon: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Find a root of function f using bisection method."""
    if f(a) * f(b) > 0:
        raise ValueError("Function must have opposite signs at endpoints")

    for _ in range(max_iter):
        c = (a + b) / 2
        fc = f(c)

        if abs(fc) < epsilon:
            return c

        if f(a) * fc < 0:
            b = c
        else:
            a = c

    return (a + b) / 2
