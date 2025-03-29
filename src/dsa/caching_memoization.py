from typing import Dict, Tuple, Callable, Any
import time


def time_based_cache(expiry_seconds: int) -> Callable:
    """Manual implementation of a time-based cache decorator."""

    def decorator(func: Callable) -> Callable:
        cache: Dict[str, Tuple[Any, float]] = {}

        def generate_key(args: Tuple[Any], kwargs: Dict[str, Any]) -> str:
            """Generates a cache key from function arguments and keyword arguments."""
            key_parts = [repr(arg) for arg in args]
            key_parts.extend(f"{k}:{repr(v)}" for k, v in sorted(kwargs.items()))
            return ":".join(key_parts)

        def wrapper(*args, **kwargs) -> Any:
            # Generate key from function arguments
            key = generate_key(args, kwargs)
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
