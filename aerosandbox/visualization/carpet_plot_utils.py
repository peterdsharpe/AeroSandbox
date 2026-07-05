"""
Provide utilities for making better carpet plots.
"""

import aerosandbox.numpy as np

import signal
from contextlib import contextmanager


@contextmanager
def time_limit(seconds):
    """
    Run a block of code with a timeout.

    This way, you can sweep through points to make a carpet plot without getting stuck on a
    particular point that may not terminate in a reasonable amount of time.

    Only runs on Linux!

    Parameters
    ----------
    seconds : float
        Duration of timeout [seconds]. May be a float.

    Examples
    --------
    Attempt to set x equal to the value of a complicated function. If it takes longer than 5
    seconds, skip it.

    >>> try:
    >>>     with time_limit(5):
    >>>         x = complicated_function()
    >>> except TimeoutError:
    >>>     x = np.nan
    """

    def signal_handler(signum, frame):
        raise TimeoutError()

    try:
        old_handler = signal.signal(signal.SIGALRM, signal_handler)
    except AttributeError:
        raise OSError(
            "signal.SIGALRM could not be found. This is probably because you're not using Linux."
        )
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if old_handler is not None:
            signal.signal(signal.SIGALRM, old_handler)


def remove_nans(array):
    """
    Remove NaN values from a 1D array.

    Parameters
    ----------
    array
        A 1D array of data.

    Returns
    -------
    ndarray
        The array with all NaN values stripped.
    """
    return array[~np.isnan(array)]


def patch_nans(array, verbose: bool = True):
    """
    Patch NaN values in a 2D array.

    Can patch holes or entire regions. Uses Laplacian smoothing.

    The input array is not modified; a patched copy is returned.

    Parameters
    ----------
    array
        The 2D array to patch.
    verbose : bool
        Should we print the patching progress to the console?

    Returns
    -------
    ndarray
        A copy of the array with all NaN values patched.
    """
    array = np.copy(array)
    original_nans = np.isnan(array)

    def nanfrac(array):
        return np.sum(np.isnan(array)) / len(array.flatten())

    def item(i, j):
        if i < 0 or j < 0:  # don't allow wrapping other than what's controlled here
            return np.nan
        try:
            return array[i, j % array.shape[1]]  # allow wrapping around day of year
        except IndexError:
            return np.nan

    def print_title(name):
        if verbose:
            print(f"{name}\nIter | NaN Fraction")

    def print_progress(iter):
        if verbose:
            print(f"{iter:4} | {nanfrac(array):.6f}")

    # Bridging
    print_title("Bridging")
    print_progress(0)
    iter = 1
    last_nanfrac = nanfrac(array)
    making_progress = True
    while making_progress:
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if not np.isnan(array[i, j]):
                    continue

                pairs = [
                    [item(i, j - 1), item(i, j + 1)],
                    [item(i - 1, j), item(i + 1, j)],
                    [item(i - 1, j + 1), item(i + 1, j - 1)],
                    [item(i - 1, j - 1), item(i + 1, j + 1)],
                ]

                for pair in pairs:
                    a = pair[0]
                    b = pair[1]

                    if not (np.isnan(a) or np.isnan(b)):
                        array[i, j] = (a + b) / 2
                        break  # Bridge with the first valid pair (pairs are in priority order)
        print_progress(iter)
        making_progress = nanfrac(array) != last_nanfrac
        last_nanfrac = nanfrac(array)
        iter += 1

    # Spreading
    for neighbors_to_spread in [4, 3, 2, 1]:
        print_title(f"Spreading with {neighbors_to_spread} neighbors")
        print_progress(0)
        iter = 1
        last_nanfrac = nanfrac(array)
        making_progress = True
        while making_progress:
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    if not np.isnan(array[i, j]):
                        continue

                    neighbors = np.array(
                        [
                            item(i, j - 1),
                            item(i, j + 1),
                            item(i - 1, j),
                            item(i + 1, j),
                            item(i - 1, j + 1),
                            item(i + 1, j - 1),
                            item(i - 1, j - 1),
                            item(i + 1, j + 1),
                        ]
                    )

                    valid_neighbors = neighbors[np.logical_not(np.isnan(neighbors))]

                    if len(valid_neighbors) > neighbors_to_spread:
                        array[i, j] = np.mean(valid_neighbors)
            print_progress(iter)
            making_progress = nanfrac(array) != last_nanfrac
            last_nanfrac = nanfrac(array)
            iter += 1
        if last_nanfrac == 0:
            break

    assert last_nanfrac == 0, "Could not patch all NaNs!"

    # Diffusing
    print_title(
        "Diffusing"
    )  # TODO Perhaps use skimage gaussian blur kernel or similar instead of "+" stencil?
    for iter in range(50):
        if verbose:
            print(f"{iter + 1:4}")
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if original_nans[i, j]:
                    neighbors = np.array(
                        [
                            item(i, j - 1),
                            item(i, j + 1),
                            item(i - 1, j),
                            item(i + 1, j),
                        ]
                    )

                    valid_neighbors = neighbors[np.logical_not(np.isnan(neighbors))]

                    array[i, j] = np.mean(valid_neighbors)

    return array


if __name__ == "__main__":
    import numpy as np
    from numpy import linalg

    def complicated_function():
        print("Starting...")
        n = 10000
        linalg.solve(np.random.randn(n, n), np.random.randn(n))
        print("Finished")
        return True

    try:
        with time_limit(1):
            complicated_function()
    except TimeoutError:
        print("Timed out.")
