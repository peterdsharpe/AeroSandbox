import time
from typing import Tuple, Any, Callable


def time_function(
        func: Callable
) -> Tuple[float, Any]:
    """
    Runs a given callable and tells you how long it took to run, in seconds. Also returns the result of the function
        (if any), for good measure. Args: func:

    Returns: A Tuple of (time_taken, result).
        - time_taken is a float of the time taken to run the function, in seconds.
        - result is the result of the function, if any.

    """
    start_ns = time.time_ns()
    result = func()
    return (
        (time.time_ns() - start_ns) / 1e9,
        result
    )
