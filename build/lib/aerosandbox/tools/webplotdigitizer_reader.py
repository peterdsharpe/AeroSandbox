"""
A series of utilities for working with CSV data extracted from WebPlotDigitizer.

https://automeris.io/WebPlotDigitizer/
https://github.com/ankitrohatgi/WebPlotDigitizer

"""
import numpy as np
from typing import Dict
from pathlib import Path
from typing import Union


def string_to_float(s: str) -> float:
    """Converts a string input to a float. If not possible, returns NaN."""
    try:
        return float(s)
    except ValueError:
        return np.nan


def remove_nan_rows(a: np.ndarray) -> np.ndarray:
    """Removes any rows in a 2D ndarray where any of the entries are NaN."""
    nan_rows = np.any(np.isnan(a), axis=1)
    return a[~nan_rows, :]


def read_webplotdigitizer_csv(
        filename: Union[Path, str],
) -> Dict[str, np.ndarray]:
    """
    Reads a CSV file produced by WebPlotDigitizer (https://automeris.io/WebPlotDigitizer/).

    If there's only one data series, produces a Dict with key "data" and value 2D ndarray.

    If there are multiple data series, produces a Dict with keys of the names and values of 2D ndarrays.

    2D ndarrays are sorted by their X-values before being returned.

    Args:
        filename: Filename, as a string or pathlib Path, or equivalent.

    Returns: A dictionary where keys are series names and values are data points.

    """
    delimiter = ","
    with open(filename, "r") as f:
        lines = f.readlines()

    has_titles = np.any([
        np.isnan(string_to_float(s))
        for s in lines[0].split(delimiter)
    ])

    if has_titles:
        titles = lines[0].split(delimiter)[::2]
        first_data_row = 2
    else:
        titles = ["data"]
        first_data_row = 0

    all_data = np.array([
        [string_to_float(item) for item in line.split(delimiter)]
        for line in lines[first_data_row:]
    ], dtype=float)

    output = {}

    for i, title in enumerate(titles):

        series = all_data[:, 2 * i: 2 * i + 2]
        all_nan_rows = np.all(np.isnan(series), axis=1)
        series = series[~all_nan_rows, :]

        sort_order = np.argsort(series[:, 0])

        output[title] = series[sort_order, :]

    return output
