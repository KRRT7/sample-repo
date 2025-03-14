from typing import Dict, Tuple, Callable, Any
import time
import functools


def time_based_cache(expiry_seconds: int) -> Callable:
    """Manual implementation of a time-based cache decorator."""

    def decorator(func: Callable) -> Callable:
        cache: Dict[Tuple, Tuple[Any, float]] = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create a key from the function arguments
            key = (args, frozenset(kwargs.items()))

            current_time = time.time()

            # Check if result is in cache and not expired
            if key in cache:
                result, timestamp = cache[key]
                if current_time - timestamp < expiry_seconds:
                    return result

            # Compute the result
            result = func(*args, **kwargs)

            # Update cache
            cache[key] = (result, current_time)

            return result

        return wrapper

    return decorator


def segmented_cache() -> Callable:
    """Cache segmented by argument types to reduce lookup time."""

    def decorator(func: Callable) -> Callable:
        caches: Dict[str, Dict[str, Any]] = {}

        def wrapper(*args, **kwargs) -> Any:
            # Create a segment key based on argument types
            segment_parts = [type(arg).__name__ for arg in args]
            segment_parts.extend(
                f"{k}:{type(v).__name__}" for k, v in sorted(kwargs.items())
            )
            segment_key = ":".join(segment_parts)

            # Ensure segment exists
            if segment_key not in caches:
                caches[segment_key] = {}

            # Create value key based on argument values
            value_parts = [repr(arg) for arg in args]
            value_parts.extend(f"{k}:{repr(v)}" for k, v in sorted(kwargs.items()))
            value_key = ":".join(value_parts)

            # Check if result is in cache segment
            segment = caches[segment_key]
            if value_key in segment:
                return segment[value_key]

            # Calculate the result
            result = func(*args, **kwargs)

            # Add to cache
            segment[value_key] = result

            return result

        return wrapper

    return decorator


def memoize_method() -> Callable:
    """Memoization decorator designed specifically for class methods."""

    def decorator(method: Callable) -> Callable:
        # Use a separate cache for each instance
        def wrapper(self, *args, **kwargs) -> Any:
            # Initialize cache if it doesn't exist
            if not hasattr(self, "_method_cache"):
                self._method_cache = {}
            if method.__name__ not in self._method_cache:
                self._method_cache[method.__name__] = {}

            cache = self._method_cache[method.__name__]

            # Create a key from the method arguments
            key_parts = [repr(arg) for arg in args]
            key_parts.extend(f"{k}:{repr(v)}" for k, v in sorted(kwargs.items()))
            key = ":".join(key_parts)

            # Check if result is in cache
            if key in cache:
                return cache[key]

            # Calculate the result
            result = method(self, *args, **kwargs)

            # Add to cache
            cache[key] = result

            return result

        return wrapper

    return decorator
