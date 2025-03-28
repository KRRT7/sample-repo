from typing import Dict, Tuple, Callable, Any
import time



def time_based_cache(expiry_seconds: int) -> Callable:
    """Manual implementation of a time-based cache decorator."""
    def decorator(func: Callable) -> Callable:
        cache: Dict[str, Tuple[Any, float]] = {}
        
        def wrapper(*args, **kwargs) -> Any:
            # Create a key from the function arguments
            key_parts = [repr(arg) for arg in args]
            key_parts.extend(f"{k}:{repr(v)}" for k, v in sorted(kwargs.items()))
            key = ":".join(key_parts)
            
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