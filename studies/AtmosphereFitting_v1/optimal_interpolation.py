"""
Suppose you have some function f(x)=exp(-x) on the range x in [0, 1].

You want to pick n points to linearly interpolate this such that the L2-error is minimized. How do you do it?
"""
import numpy as np
import casadi as cas
import matplotlib.pyplot as plt

opti = cas.Opti()

n = 101

x = opti.variable(n)  # cas.linspace(0,1,n)

opti.set_initial(x, cas.linspace(0, 10, n))

opti.subject_to([
    x[0] == 0,
    x[-1] == 10,
])

y = cas.exp(-x)

x1 = x[:-1]
x2 = x[1:]
errors = -x1 * cas.exp(-x2) / 2 - x1 * cas.exp(-x1) / 2 + x2 * cas.exp(-x2) / 2 + x2 * cas.exp(-x1) / 2 + cas.exp(-x2) - cas.exp(-x1)
error = cas.sum1(errors)

opti.minimize(error * 1e3)
opti.solver('ipopt')
sol = opti.solve()

xopt = sol.value(x)
minerror = sol.value(error)
