import aerosandbox as asb
import aerosandbox.numpy as np
from math import gamma


opti = asb.Opti()
x = opti.variable(init_guess=2, lower_bound=0)

# import casadi as _cas
# var = _cas.MX(2)

model = asb.BlackBoxModel(
    lambda x: gamma(x)
)

# model(var)

opti.minimize(model(x))
sol = opti.solve()
print(sol.value(x))