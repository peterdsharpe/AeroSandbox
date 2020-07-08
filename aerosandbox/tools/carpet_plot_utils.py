"""
Utilities for making better carpet plots
"""

import numpy as np

def patch_nans(array):
    """
    Patches NaN values in a 2D array. Can patch holes or entire regions. Uses Laplacian smoothing.
    :param array:
    :return:
    """
    original_nans = np.isnan(array)

    nanfrac = lambda array: np.sum(np.isnan(array)) / len(array.flatten())

    def item(i, j):
        if i < 0 or j < 0:  # don't allow wrapping other than what's controlled here
            return np.nan
        try:
            return array[i, j % array.shape[1]]  # allow wrapping around day of year
        except IndexError:
            return np.nan

    print_title = lambda name: print(f"{name}\nIter | NaN Fraction")
    print_progress = lambda iter: print(f"{iter:4} | {nanfrac(array):.6f}")

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
                        continue
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

                    neighbors = np.array([
                        item(i, j - 1), item(i, j + 1),
                        item(i - 1, j), item(i + 1, j),
                        item(i - 1, j + 1), item(i + 1, j - 1),
                        item(i - 1, j - 1), item(i + 1, j + 1),
                    ])

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
    print_title("Diffusing")
    for iter in range(50):
        print(f"{iter + 1:4}")
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if original_nans[i, j]:
                    neighbors = np.array([
                        item(i, j - 1),
                        item(i, j + 1),
                        item(i - 1, j),
                        item(i + 1, j),
                    ])

                    valid_neighbors = neighbors[np.logical_not(np.isnan(neighbors))]

                    array[i, j] = np.mean(valid_neighbors)

    return array
