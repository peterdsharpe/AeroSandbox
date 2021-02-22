# from inviscid_analysis import N, gamma
import aerosandbox as asb
from aerosandbox import cas
import numpy as np

N = 100
ue = cas.linspace(1, 1, N)
x = cas.linspace(0, 1, N)
nu = 1e-5

opti = asb.Opti()

theta = opti.variable(
    n_vars=N,
    init_guess=np.linspace(1e-3, 1e-3, N),
)
H = opti.variable(
    n_vars=N,
    init_guess=2,
)

Re_theta = ue * theta / nu

H_star = cas.if_else(
    H < 4,
    1.515 + 0.076 * (H - 4) ** 2 / H,
    1.515 + 0.040 * (H - 4) ** 2 / H
)  # From AVF Eq. 4.53
c_f = 2 / Re_theta * cas.if_else(
    H < 6.2,
    -0.066 + 0.066 * cas.fabs(6.2 - H) ** 1.5 / (H - 1),
    -0.066 + 0.066 * (H - 6.2) ** 2 / (H - 4) ** 2
)  # From AVF Eq. 4.54
c_D = H_star / 2 / Re_theta * cas.if_else(
    H < 4,
    0.207 + 0.00205 * cas.fabs(4 - H) ** 5.5,
    0.207 - 0.100 * (H - 4) ** 2 / H ** 2
)  # From AVF Eq. 4.55
Re_theta_o = 10 ** (
        2.492 / (H - 1) ** 0.43 +
        0.7 * (
                cas.tanh(
                    14 / (H - 1) - 9.24
                ) + 1
        )
)  # From AVF Eq. 6.38

d_theta_dx = cas.diff(theta) / cas.diff(x)
d_ue_dx = cas.diff(ue) / cas.diff(x)
d_H_star_dx = cas.diff(H_star) / cas.diff(x)


def int(x):
    return (x[1:] + x[:-1]) / 2

def logint(x):
    # return int(x)
    logx = cas.log(x)
    return cas.exp(
        (logx[1:] + logx[:-1]) / 2
    )


### Add governing equations
opti.subject_to(
    d_theta_dx == logint(c_f) / 2 - logint(
        (H + 2) * theta / ue
    ) * d_ue_dx,
)
opti.subject_to(
    logint(theta / H_star) * d_H_star_dx ==
    int(2 * c_D / H_star - c_f / 2) +
    logint(
        (H - 1) * theta / ue
    ) * d_ue_dx
)

### Add initial conditions
opti.subject_to([
    theta[0] == 1e-3,
    H[0] == 2.5824454903566063  # Equilibrium value
])

### Solve
sol = opti.solve()
theta = sol.value(theta)
H = sol.value(H)

### Plot
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(palette=sns.color_palette("husl"))
fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
plt.plot(x, theta)
plt.xlabel(r"$x$")
plt.ylabel(r"$\theta$")
plt.title(r"$\theta$")
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
plt.plot(x, H)
plt.xlabel(r"$x$")
plt.ylabel(r"$H$")
plt.title(r"$H$")
plt.tight_layout()
plt.show()
