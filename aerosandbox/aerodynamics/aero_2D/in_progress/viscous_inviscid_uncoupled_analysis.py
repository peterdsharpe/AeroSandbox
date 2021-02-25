from aerosandbox.optimization import Opti
from aerosandbox.geometry.airfoil import *
from aerosandbox import cas
import numpy as np
from numpy import pi
from aerodynamics.singularities.linear_strength_line_singularities import calculate_induced_velocity_line_singularities
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(palette=sns.color_palette("husl"))

### Set up givens

airfoil = Airfoil("naca4408").repanel(n_points_per_side=50)
alpha_deg = 5

### Get some basic information about the airfoil
x_panels = airfoil.x()
y_panels = airfoil.y()

N = len(x_panels)  # number of airfoil nodes
N_w = round(N / 8 + 2)  # number of wake nodes

### Set up optimization/analysis framework and unknowns
opti = Opti()
gamma = opti.variable(
    n_vars=N,
    init_guess=0
)
sigma = np.zeros(N)

### Compute freestream velocity components
alpha_rad = alpha_deg * pi / 180
u_freestream = cas.cos(alpha_rad)
v_freestream = cas.sin(alpha_rad)


### Make a function to calculate the local velocity at arbitrary points
def calculate_velocity(
        x_field,
        y_field,
        gamma,
        sigma,
        backend="numpy"
):
    u_field_induced, v_field_induced = calculate_induced_velocity_line_singularities(
        x_field=x_field,
        y_field=y_field,
        x_panels=x_panels,
        y_panels=y_panels,
        gamma=gamma,
        sigma=sigma,
        backend=backend
    )

    # u_field_induced_mirror, v_field_induced_mirror = calculate_induced_velocity(
    #     x_field=x_field,
    #     y_field=-0.2 - y_field,
    #     x_panels=x_panel,
    #     y_panels=y_panel,
    #     gamma=-gamma,
    #     sigma=sigma,
    #     backend=backend
    # )
    # u_field_induced += u_field_induced_mirror
    # v_field_induced += v_field_induced_mirror

    u_field = u_field_induced + u_freestream
    v_field = v_field_induced + v_freestream

    return u_field, v_field


### Compute normal velocities at the middle of each panel
x_midpoints = (x_panels[1:] + x_panels[:-1]) / 2
y_midpoints = (y_panels[1:] + y_panels[:-1]) / 2

u_midpoints, v_midpoints = calculate_velocity(
    x_field=x_midpoints,
    y_field=y_midpoints,
    gamma=gamma,
    sigma=sigma,
    backend="casadi"
)

panel_dx = np.diff(x_panels)
panel_dy = np.diff(y_panels)
panel_length = (panel_dx ** 2 + panel_dy ** 2) ** 0.5

xp_hat_x = panel_dx / panel_length  # x-coordinate of the xp_hat vector
xp_hat_y = panel_dy / panel_length  # y-coordinate of the yp_hat vector

yp_hat_x = -xp_hat_y
yp_hat_y = xp_hat_x

normal_velocities = u_midpoints * yp_hat_x + v_midpoints * yp_hat_y

### Add in flow tangency constraint
opti.subject_to(normal_velocities == 0)

### Add in Kutta condition
opti.subject_to(gamma[0] + gamma[-1] == 0)

### Solve
sol = opti.solve()
gamma = sol.value(gamma)
sigma = sol.value(sigma)

### Calculate lift coefficient
total_vorticity = np.sum(
    (gamma[1:] + gamma[:-1]) / 2 *
    panel_length
)
Cl = 2 * total_vorticity
print(f"Cl: {Cl}")

### Plot the flowfield
fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=200)

margin = 0.4
res = 100
x = np.linspace(-margin, 1 + margin, round(res * (1 + 2 * margin) / (2 * margin)))
y = np.linspace(-margin, margin, res)
X, Y = np.meshgrid(
    x,
    y,
    # indexing='ij',
)
X = X.flatten()
Y = Y.flatten()

U, V = calculate_velocity(
    x_field=X,
    y_field=Y,
    gamma=gamma,
    sigma=sigma,
)
speed = (U ** 2 + V ** 2) ** 0.5
# plt.quiver(
#     X, Y, U, V, speed,
#     scale=30
# )

from palettable.colorbrewer.diverging import RdBu_4 as streamplot_colormap

plt.streamplot(
    x,
    y,
    U.reshape(len(y), len(x)),
    V.reshape(len(y), len(x)),
    color=speed.reshape(len(y), len(x)),
    density=2,
    arrowsize=0,
    cmap=streamplot_colormap.mpl_colormap,
)

plt.fill(x_panels, y_panels, "k", linewidth=0, zorder=4)
CB = plt.colorbar(
    orientation="horizontal",
    shrink=0.8,
    aspect=40,
)
CB.set_label(r"$U/U_\infty$")
plt.xlim(min(x), max(x))
plt.ylim(min(y), max(y))
plt.clim(0.7, 1.3)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel(r"$x/c$")
plt.ylabel(r"$y/c$")
plt.title(rf"Flow Field around {airfoil.name} Airfoil at $\alpha = {alpha_deg}\degree$")
plt.tight_layout()
plt.show()

### Plot C_p
surface_speeds = (gamma[1:] + gamma[:-1]) / 2
C_p = 1 - surface_speeds ** 2

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
plt.plot(x_midpoints, C_p)
plt.gca().invert_yaxis()
plt.xlabel(r"$x/c$")
plt.ylabel(r"$C_p$")
plt.title(r"$C_p$ on Surface")
plt.tight_layout()
plt.show()
