import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.geometry.airfoil import airfoil_families
from tqdm import tqdm


@np.vectorize
def get_dalpha(
        alpha=0.,
        deflection=0.,
        hinge_x=0.75,
        Re=1e6,
        camber=0.,
        thickness=0.12,
):
    global pbar
    pbar.update(1)

    try:
        af = asb.Airfoil(
            name="testaf",
            coordinates=airfoil_families.get_NACA_coordinates(
                n_points_per_side=80,
                max_camber=camber,
                camber_loc=0.40,
                thickness=thickness
            )
        )
        res_deflected = asb.XFoil(
            airfoil=af.add_control_surface(
                deflection=deflection,
                hinge_point_x=hinge_x
            ),
            Re=Re,
            max_iter=20,
            timeout=0.5,
        ).alpha(alpha=alpha)

        if any(np.isnan(res_deflected["CL"])):
            return np.nan

        res_undeflected = asb.XFoil(
            airfoil=af,
            Re=Re,
            max_iter=20,
            timeout=0.5,
        ).cl(cl=res_deflected['CL'])

        delta_alpha = float(res_undeflected['alpha'] - res_deflected['alpha'])

        return delta_alpha

    except (FileNotFoundError, TypeError):
        return np.nan


Alpha, Deflection, Hinge_x, Re, Camber, Thickness = np.meshgrid(
    np.linspace(0, 15, 6),
    np.linspace(1, 25, 11),
    np.linspace(0, 1, 11),
    np.array([1e5, 2e5, 3e5, 4e5, 5e5, 1e6, 5e6]),
    0,
    0.10
)

global pbar
with tqdm(total=Alpha.size) as pbar:
    Dalpha = get_dalpha(
        Alpha,
        Deflection,
        Hinge_x,
        Re,
        Camber,
        Thickness
    )

data = {
        "alpha": Alpha.flatten(),
        "deflection": Deflection.flatten(),
        "hinge_x": Hinge_x.flatten(),
        "Re": Re.flatten(),
        "camber": Camber.flatten(),
        "thickness": Thickness.flatten(),
        "dalpha": Dalpha.flatten()
    }
mask = np.logical_not(np.isnan(Dalpha))


import pandas as pd
df = pd.DataFrame(
    data={
        k: v.flatten()[mask.flatten()]
        for k, v in data.items()
    }
)
df.to_csv("data.csv")



#
#
# N = 10000
#
# def get_xy(s, h):
#     theta = np.linspace(0, np.pi / 2, N)
#     st = np.sin(theta)
#     ct = np.cos(theta)
#
#     ct[-1] = 0
#     st[0] = 0
#
#     x = ct ** (2/s)
#     y = st ** (2/s)
#
#     y *= h
#
#     return x, y
#
# def plot_xy(s, h):
#     x, y = get_xy(s, h)
#     fig, ax = plt.subplots()
#     plt.plot(x, y)
#     p.equal()
#     p.show_plot()
#
# ### Generate data
# @np.vectorize
# def get_arc_length(s, h):
#     x, y = get_xy(s, h)
#
#     dx = np.diff(x)
#     dy = np.diff(y)
#
#     darc = (dx ** 2 + dy ** 2) ** 0.5
#
#     arc_length = np.sum(darc)
#
#     return arc_length
#
# s = np.concatenate([
#     np.linspace(1, 3, 50),
#     np.sinspace(3, 50, 20)[1:]
# ])
# h = np.geomspace(1, 100, 50)
#
# S, H = np.meshgrid(s, h)
#
# Arc_lengths = get_arc_length(S, H)
