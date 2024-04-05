"""
Suppose you have some function f(x)=exp(-x) on the range x in [0, 1].

You want to pick n points to linearly interpolate this such that the L2-error is minimized. How do you do it?
"""
import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt

opti = asb.Opti()

n = 101

x = opti.variable(np.linspace(0, 10, n))

opti.subject_to([
    x[0] == 0,
    x[-1] == 10,
])

y = np.exp(-x)

x1 = x[:-1]
x2 = x[1:]
errors = -x1 * np.exp(-x2) / 2 - x1 * np.exp(-x1) / 2 + x2 * np.exp(-x2) / 2 + x2 * np.exp(-x1) / 2 + np.exp(
    -x2) - np.exp(-x1)
error = np.sum(errors)

opti.minimize(error * 1e3)
sol = opti.solve()

xopt = sol(x)
minerror = sol(error)
