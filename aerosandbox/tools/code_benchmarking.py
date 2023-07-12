import time
from typing import Tuple, Any, Callable
import aerosandbox.numpy as np


def time_function(
        func: Callable,
        repeats: int = None,
        desired_runtime: float = None,
        runtime_reduction = np.min,
) -> Tuple[float, Any]:
    """
    Runs a given callable and tells you how long it took to run, in seconds. Also returns the result of the function
        (if any), for good measure. Args: func:

    Returns: A Tuple of (time_taken, result).
        - time_taken is a float of the time taken to run the function, in seconds.
        - result is the result of the function, if any.

    """
    if (repeats is not None) and (desired_runtime is not None):
        raise ValueError("You can't specify both repeats and desired_runtime!")

    def time_function_once():
        start_ns = time.time_ns()
        result = func()
        return (
            (time.time_ns() - start_ns) / 1e9,
            result
        )

    runtimes = []

    t, result = time_function_once()
    runtimes.append(t)

    if (desired_runtime is not None) and (repeats is None):
        repeats = int(desired_runtime // t) - 1
        # print(f"Running {func.__name__} {repeats} times to get a desired runtime of {desired_runtime} seconds.")

    if repeats is None:
        repeats = 0

    for _ in range(repeats):
        t, _ = time_function_once()
        runtimes.append(t)


    return (
        runtime_reduction(runtimes),
        result
    )

if __name__ == '__main__':

    def f():
        time.sleep(0.1)

    print(time_function(f, desired_runtime=1))