import aerosandbox as asb
import aerosandbox.numpy as np

n_timesteps = 301
mass_block = 1

opti = asb.Opti()

time = np.cosspace(0, 1, n_timesteps)

position = opti.variable(
    init_guess=np.linspace(0, 1, n_timesteps)
)

velocity = opti.derivative_of(
    position,
    with_respect_to=time,
    derivative_init_guess=1,
)

force = opti.variable(
    init_guess=np.linspace(2, -2, n_timesteps),
    n_vars=n_timesteps
)

opti.constrain_derivative(
    derivative=force / mass_block,
    variable=velocity,
    with_respect_to=time,
)

effort_expended = np.sum(
    np.trapz(force ** 2) * np.diff(time)
)

opti.minimize(effort_expended)

### Boundary conditions
opti.subject_to([
    position[0] == 0,
    position[-1] == 1,
    velocity[0] == 0,
    velocity[-1] == 0,
])

sol = opti.solve()

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
plt.plot(sol.value(time), sol.value(position))
plt.xlabel(r"Time")
plt.ylabel(r"Position")
plt.title(r"Position")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
plt.plot(sol.value(time), sol.value(velocity))
plt.xlabel(r"Time")
plt.ylabel(r"Velocity")
plt.title(r"Velocity")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
plt.plot(sol.value(time), sol.value(force), "-")
# plt.plot(sol.value(np.trapz(time)), sol.value(np.diff(velocity) / np.diff(time)), ".-")
plt.show()

# fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
# plt.plot(sol.value(time), sol.value(velocity))
# plt.plot(sol.value(np.trapz(time)), sol.value(np.diff(position) / np.diff(time)), ".-")
# plt.show()
