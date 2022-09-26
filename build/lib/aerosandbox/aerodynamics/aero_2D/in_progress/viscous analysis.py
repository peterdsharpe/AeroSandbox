# from inviscid_analysis import N, gamma
import aerosandbox as asb
import aerosandbox.numpy as np

N = 1000
ue = np.linspace(1, 1, N)
x = np.linspace(0, 1, N)
nu = 1e-5
theta_0 = 1e-3
H_0 = 2.6

opti = asb.Opti()

theta = opti.variable(
    init_guess=theta_0,
    n_vars=N,
)
H = opti.variable(
    H_0,
    n_vars=N,
)

Re_theta = ue * theta / nu

H_star = np.where(
    H < 4,
    1.515 + 0.076 * (H - 4) ** 2 / H,
    1.515 + 0.040 * (H - 4) ** 2 / H
)  # From AVF Eq. 4.53
c_f = 2 / Re_theta * np.where(
    H < 6.2,
    -0.066 + 0.066 * np.abs(6.2 - H) ** 1.5 / (H - 1),
    -0.066 + 0.066 * (H - 6.2) ** 2 / (H - 4) ** 2
)  # From AVF Eq. 4.54
c_D = H_star / 2 / Re_theta * np.where(
    H < 4,
    0.207 + 0.00205 * np.abs(4 - H) ** 5.5,
    0.207 - 0.100 * (H - 4) ** 2 / H ** 2
)  # From AVF Eq. 4.55
Re_theta_o = 10 ** (
        2.492 / (H - 1) ** 0.43 +
        0.7 * (
                np.tanh(
                    14 / (H - 1) - 9.24
                ) + 1
        )
)  # From AVF Eq. 6.38

d_theta_dx = np.diff(theta) / np.diff(x)
d_ue_dx = np.diff(ue) / np.diff(x)
d_H_star_dx = np.diff(H_star) / np.diff(x)


def int(x):
    return (x[1:] + x[:-1]) / 2


def logint(x):
    # return int(x)
    logx = np.log(x)
    return np.exp(
        (logx[1:] + logx[:-1]) / 2
    )


### Add governing equations
opti.subject_to(
    d_theta_dx == int(c_f) / 2 - int(
        (H + 2) * theta / ue
    ) * d_ue_dx,  # From AVF Eq. 4.51
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
    theta[0] == theta_0,
    H[0] == H_0  # Equilibrium value
])

### Solve
sol = opti.solve()
theta = sol.value(theta)
H = sol.value(H)

### Plot
from aerosandbox.tools.pretty_plots import plt, show_plot

fig, ax = plt.subplots()
plt.plot(x, theta)
show_plot(r"$\theta$", "$x$", r"$\theta$")
fig, ax = plt.subplots()
plt.plot(x, H)
show_plot("$H$", "$x$", "$H$")
