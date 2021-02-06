import aerosandbox as asb
from aerosandbox import cas
import aerosandbox.numpy as np
import pytest

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(palette=sns.color_palette("husl"))

"""
This test solves the 2-dimensional Rosenbrock problem constrained to a circle centered on the origin:

----------

Minimize:
(1 - x) ** 2 + (y - x ** 2) ** 2

Subject to:
x ** 2 + y ** 2 <= r

"""


def test_rosenbrock_constrained(plot=False):
    opti = asb.Opti()

    x = opti.variable(init_guess=0)
    y = opti.variable(init_guess=0)
    r = opti.parameter()

    f = (1 - x) ** 2 + (y - x ** 2) ** 2
    opti.minimize(f)
    con = x ** 2 + y ** 2 <= r
    dual = opti.subject_to(con)

    r_values = np.linspace(1, 3)

    sols = [
        opti.solve({r: r_value})
        for r_value in r_values
    ]
    fs = [
        sol.value(f)
        for sol in sols
    ]
    duals = [
        sol.value(dual)  # Ensure the dual can be evaluated
        for sol in sols
    ]

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
        plt.plot(r_values, fs, label="$f$")
        plt.plot(r_values, duals, label=r"Dual var. ($\frac{df}{dr}$)")
        plt.legend()
        plt.xlabel("$r$")
        plt.show()

    assert dual is not None  # The dual should be a real value
    assert r_values[0] == pytest.approx(1)
    assert duals[0] == pytest.approx(0.10898760051521068, abs=1e-6)


if __name__ == '__main__':
    pytest.main()
