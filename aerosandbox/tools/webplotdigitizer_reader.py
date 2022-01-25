"""
A series of utilities for working with CSV data extracted from WebPlotDigitizer.

https://automeris.io/WebPlotDigitizer/
https://github.com/ankitrohatgi/WebPlotDigitizer

"""
import numpy as np
from typing import Dict


def read_webplotdigitizer_csv(filename) -> Dict[str, np.ndarray]:
    delimiter = ","
    with open(filename, "r") as f:
        lines = f.readlines()

    titles = lines[0].split(delimiter)[::2]

    def string_to_float(s: str) -> float:
        try:
            return float(s)
        except ValueError:
            return np.NaN

    data = [
        [string_to_float(item) for item in line.split(delimiter)]
        for line in lines[2:]
    ]

    data = np.array(data, dtype=float)

    def remove_nan_rows(a: np.ndarray) -> np.ndarray:
        nan_rows = np.any(np.isnan(a), axis=1)
        return a[~nan_rows, :]

    return {
        title: remove_nan_rows(data[:, 2 * i:2 * i + 2])
        for i, title in enumerate(titles)
    }
