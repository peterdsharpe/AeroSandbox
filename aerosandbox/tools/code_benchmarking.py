import time
from typing import Tuple, Any, Callable
import aerosandbox.numpy as np


class Timer(object):
    """
    A context manager for timing things. Use it like this:

    with Timer("My timer"):  # You can optionally give it a name
        # Do stuff

    Results are printed to stdout. You can access the runtime (in seconds) directly with:

    with Timer("My timer") as t:
        # Do stuff
    print(t.runtime)
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.t_start = time.perf_counter_ns()

    def __exit__(self, type, value, traceback):
        self.runtime = (time.perf_counter_ns() - self.t_start) / 1e9
        if self.name:
            print(f'[{self.name}]')
        print(f'Elapsed: {runtime} sec')


def time_function(
        func: Callable,
        repeats: int = None,
        desired_runtime: float = None,
        runtime_reduction=np.min,
) -> Tuple[float, Any]:
    """
    Runs a given callable and tells you how long it took to run, in seconds. Also returns the result of the function
        (if any), for good measure.

    Args:

        func: The function to run. Should take no arguments; use a lambda function or functools.partial if you need
            to pass arguments.

        repeats: The number of times to run the function. If None, runs until desired_runtime is met.

        desired_runtime: The desired runtime of the function, in seconds. If None, runs until repeats is met.

        runtime_reduction: A function that takes in a list of runtimes and returns a reduced value. For example,
            np.min will return the minimum runtime, np.mean will return the mean runtime, etc. Default is np.min.

    Returns: A Tuple of (time_taken, result).
        - time_taken is a float of the time taken to run the function, in seconds.
        - result is the result of the function, if any.

    """
    if (repeats is not None) and (desired_runtime is not None):
        raise ValueError("You can't specify both repeats and desired_runtime!")

    def time_function_once():
        start_ns = time.perf_counter_ns()
        result = func()
        return (
            (time.perf_counter_ns() - start_ns) / 1e9,
            result
        )

    runtimes = []

    t, result = time_function_once()

    if t == 0:
        t = 1e-2
    else:
        runtimes.append(t)

    if (desired_runtime is not None) and (repeats is None):
        repeats = int(desired_runtime // t) - 1
        # print(f"Running {func.__name__} {repeats} times to get a desired runtime of {desired_runtime} seconds.")

    if repeats is None:
        repeats = 0

    for _ in range(repeats):
        t, _ = time_function_once()
        if t != 0:
            runtimes.append(t)

    if len(runtimes) == 0:
        runtimes = [0.]

    return (
        runtime_reduction(runtimes),
        result
    )


if __name__ == '__main__':

    def f():
        time.sleep(0.1)


    print(time_function(f, desired_runtime=1))
