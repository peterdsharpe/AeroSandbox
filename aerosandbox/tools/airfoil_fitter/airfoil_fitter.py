"""
Functions to fit automatic-differentiable models to aerodynamic data from an airfoil.
Requires the xfoil package from PyPI
"""
from aerosandbox.geometry import *
from aerosandbox.tools.casadi_tools import *
import plotly.express as px
import plotly.graph_objects as go
import dash


class AirfoilFitter():
    def __init__(self,
                 airfoil,  # type: Airfoil
                 ):
        self.airfoil = airfoil

    def get_xfoil_data(self,
                       a_start=-6,
                       a_end=12,
                       a_step=0.5,
                       a_init=0,
                       Re_start=1e4,
                       Re_end=5e6,
                       n_Res=30,
                       mach=0,
                       max_iter=30,
                       repanel=True,
                       ):
        """
        Pulls XFoil data for a particular airfoil and both writes it to self.xfoil_data and returns it.
        Does a 2D grid sweep of the alpha-Reynolds space at a particular mach number.
        :param a_start: Lower bound of angle of attack [deg]
        :param a_end: Upper bound of angle of attack [deg]
        :param a_step: Angle of attack increment size [deg]
        :param a_init: Angle of attack to initialize runs at. Should solve easily (0 recommended) [deg]
        :param Re_start:
        :param Re_end:
        :param n_Res:
        :param mach:
        :param max_iter:
        :param repanel:
        :return:
        """
        assert a_init > a_start
        assert a_init < a_end
        assert Re_start < Re_end
        assert n_Res >= 1
        assert mach >= 0

        Res = np.logspace(np.log10(Re_start), np.log10(Re_end), n_Res)

        runs_data = []
        for i, Re in enumerate(Res):
            run_data_upper = self.airfoil.xfoil_aseq(
                a_start=a_init + a_step,
                a_end=a_end,
                a_step=a_step,
                Re=Re,
                repanel=repanel,
                max_iter=max_iter,
                M=mach,
                reset_bls=True,
            )
            run_data_lower = self.airfoil.xfoil_aseq(
                a_start=a_init,
                a_end=a_start,
                a_step=-a_step,
                Re=Re,
                repanel=repanel,
                max_iter=max_iter,
                M=mach,
                reset_bls=True,
            )
            run_data = {
                k: np.hstack((
                    run_data_lower[k][::-1],
                    run_data_upper[k]
                )) for k in run_data_upper.keys()
            }
            runs_data.append(run_data)

        xfoil_data_2D = {}
        for k in runs_data[0].keys():
            xfoil_data_2D[k] = np.vstack([
                d[k]
                for d in runs_data
            ])
        xfoil_data_2D["Re"] = np.tile(Res, (
            xfoil_data_2D["alpha"].shape[1],
            1
        )).T
        np.place(
            arr=xfoil_data_2D["Re"],
            mask=np.isnan(xfoil_data_2D["alpha"]),
            vals=np.NaN
        )

        self.xfoil_data_2D = xfoil_data_2D

        # 1-dimensionalize it and remove NaNs
        remove_nans = lambda x: x[~np.isnan(x)]
        xfoil_data_1D = {
            k: remove_nans(xfoil_data_2D[k].reshape(-1))
            for k in xfoil_data_2D.keys()
        }
        self.xfoil_data_1D = xfoil_data_1D

    def plot_xfoil_data(self):
        import matplotlib.pyplot as plt
        import matplotlib.style as style
        import matplotlib.colors as colors
        import matplotlib
        import seaborn as sns
        sns.set(font_scale=1)

        d = self.xfoil_data_1D  # data

        fig = plt.figure(figsize=(10, 8), dpi=200)

        ax = fig.add_subplot(311)
        coords = self.airfoil.coordinates
        plt.plot(coords[:, 0], coords[:, 1], '.-', color='#280887')
        plt.xlabel(r"$x/c$")
        plt.ylabel(r"$y/c$")
        plt.title(r"XFoil Data for %s Airfoil" % self.airfoil.name)
        plt.axis("equal")

        style.use("default")

        ax = fig.add_subplot(323)
        x = d["Re"]
        y = d["alpha"]
        z = d["Cl"]
        levels = np.linspace(-0.5, 1.5, 21)
        norm = None
        CF = ax.tricontourf(x, y, z, levels=levels, norm=norm, cmap="plasma", extend="both")
        C = ax.tricontour(x, y, z, levels=levels, norm=norm, colors='k', extend="both", linewidths=0.5)
        cbar = plt.colorbar(CF, format='%.2f')
        cbar.set_label(r"$C_l$")
        plt.grid(False)
        plt.xlabel(r"$Re$")
        plt.ylabel(r"$\alpha$")
        plt.title(r"$C_l$ from $Re$, $\alpha$")
        ax.set_xscale('log')

        ax = fig.add_subplot(324)
        x = d["Re"]
        y = d["alpha"]
        z = d["Cd"]
        levels = np.logspace(-2.5, -1, 21)
        norm = colors.PowerNorm(gamma=1 / 2, vmin=np.min(levels), vmax=np.max(levels))
        CF = ax.tricontourf(x, y, z, levels=levels, norm=norm, cmap="plasma", extend="both")
        C = ax.tricontour(x, y, z, levels=levels, norm=norm, colors='k', extend="both", linewidths=0.5)
        cbar = plt.colorbar(CF, format='%.3f')
        cbar.set_label(r"$C_d$")
        plt.grid(False)
        plt.xlabel(r"$Re$")
        plt.ylabel(r"$\alpha$")
        plt.title(r"$C_d$ from $Re$, $\alpha$")
        ax.set_xscale('log')

        ax = fig.add_subplot(325)
        x = d["Re"]
        y = d["alpha"]
        z = d["Cl"] / d["Cd"]
        x = x[d["alpha"] >= 0]
        y = y[d["alpha"] >= 0]
        z = z[d["alpha"] >= 0]
        levels = np.logspace(1, np.log10(150), 21)
        norm = colors.PowerNorm(gamma=1 / 2, vmin=np.min(levels), vmax=np.max(levels))
        CF = ax.tricontourf(x, y, z, levels=levels, norm=norm, cmap="plasma", extend="both")
        C = ax.tricontour(x, y, z, levels=levels, norm=norm, colors='k', extend="both", linewidths=0.5)
        cbar = plt.colorbar(CF, format='%.1f')
        cbar.set_label(r"$L/D$")
        plt.grid(False)
        plt.xlabel(r"$Re$")
        plt.ylabel(r"$\alpha$")
        plt.title(r"$L/D$ from $Re$, $\alpha$")
        ax.set_xscale('log')

        ax = fig.add_subplot(326)
        x = d["Re"]
        y = d["alpha"]
        z = d["Cm"]
        levels = np.linspace(-0.15, 0, 21)  # np.logspace(1, np.log10(150), 21)
        norm = None  # colors.PowerNorm(gamma=1 / 2, vmin=np.min(levels), vmax=np.max(levels))
        CF = ax.tricontourf(x, y, z, levels=levels, norm=norm, cmap="plasma", extend="both")
        C = ax.tricontour(x, y, z, levels=levels, norm=norm, colors='k', extend="both", linewidths=0.5)
        cbar = plt.colorbar(CF, format='%.2f')
        cbar.set_label(r"$C_m$")
        plt.grid(False)
        plt.xlabel(r"$Re$")
        plt.ylabel(r"$\alpha$")
        plt.title(r"$C_m$ from $Re$, $\alpha$")
        ax.set_xscale('log')

        plt.tight_layout()
        plt.savefig("C:/Users/User/Downloads/temp.svg")
        plt.show()

    # def fit_xfoil_data(self):
    #


import dill as pickle

try:
    with open("af.pkl", "rb") as f:
        af = pickle.load(f)
except:
    # af = AirfoilFitter(Airfoil(name="HALE_thiqboi_02", coordinates="C:/Users/User/Downloads/HALE_02.dat"))
    af = AirfoilFitter(Airfoil(name="naca0010"))
    af.get_xfoil_data(
        a_step=0.5,
        n_Res=30,
    )

    with open("af.pkl", "wb+") as f:
        pickle.dump(af, f)

af.plot_xfoil_data()
self = af

supercritical_threshold = 1e6
subcritical_threshold = 1e4

### Fit utilities, data extraction, plotting tools
d = self.xfoil_data_1D  # data
raw_sigmoid = lambda x: x / (1 + x ** 4) ** (1 / 4)
sigmoid = lambda x, x_cent, x_scale, y_cent, y_scale: y_cent + y_scale * raw_sigmoid((x - x_cent) / x_scale)


def plot_fit_alpha_Re(
        model,
        x_data,
        y_data,
        params_solved,
        title=None,
        show=True
):
    """
    See the docstring of the "fit" function in aerosandbox.tools.casadi_tools for syntax.
    :param model:
    :param x_data:
    :param y_data:
    :param params_solved:
    :param title:
    :param show:
    :return:
    """
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x_data['alpha'],
                y=x_data['Re'],
                z=y_data,
                mode="markers",
                marker=dict(
                    size=2,
                    color="black"
                )
            ),
            go.Mesh3d(
                x=x_data['alpha'],
                y=x_data['Re'],
                z=np.array(model(d, params_solved)).reshape(-1),
                intensity=np.array(model(d, params_solved)).reshape(-1),
                colorscale="plasma",
                flatshading=True
            ),
        ],
        layout=go.Layout(
            scene=dict(
                xaxis=dict(
                    title="Alpha"
                ),
                yaxis=dict(
                    type='log',
                    title="Re"
                ),
                zaxis=dict(
                    title="f(alpha, Re)"
                ),
            ),
            title=title
        )
    )
    if show:
        fig.show()
    return fig


### Fit the supercritical data
def model_Cl_turbulent(x, p):
    Cl_turbulent = (
        sigmoid(x['alpha'], p['clt_a_c'], p['clt_a_s'], p['clt_cl_c'], p['clt_cl_s'])
    )
    return Cl_turbulent


Cl_turbulent_params_guess = {
    'clt_a_c' : 0,
    'clt_a_s' : 12,
    'clt_cl_c': 0,
    'clt_cl_s': 1.5,
}
Cl_turbulent_param_bounds = {
    'clt_a_c' : (0, None),
    'clt_a_s' : (0, 20),
    'clt_cl_c': (0, None),
    'clt_cl_s': (0, 2.5),
}

Cl_turbulent_params_solved = fit(
    model=model_Cl_turbulent,
    x_data=d,
    y_data=d['Cl'],
    param_guesses=Cl_turbulent_params_guess,
    param_bounds=Cl_turbulent_param_bounds,
    weights=np.logical_and(d['Re'] >= supercritical_threshold, True).astype('int')
)

# plot_fit_alpha_Re(
#     model=model_Cl_turbulent,
#     x_data=d,
#     y_data=d['Cl'],
#     params_solved=Cl_turbulent_params_solved,
#     title="Fit: Lift Coefficient (Turbulent)"
# )


### Fit the subcritical data
def model_Cl_laminar(x, p):
    Cl_laminar = (
            sigmoid(x['alpha'], p['cll_a_c'], p['cll_a_s'], p['cll_cl_c'], p['cll_cl_s']) +
            sigmoid(x['alpha'], p['clld_a_c'], p['clld_a_s'], p['clld_cl_c'], p['clld_cl_s'])
    )
    return Cl_laminar


Cl_laminar_params_guess = {
    'cll_a_c'  : 0,
    'cll_a_s'  : 12,
    'cll_cl_c' : 0,
    'cll_cl_s' : 1.5,
    'clld_a_c' : 0,
    'clld_a_s' : 1,
    'clld_cl_c': 0,
    'clld_cl_s': 0.1,
}
Cl_laminar_param_bounds = {
    'cll_a_c'  : (0, None),
    'cll_a_s'  : (0, 20),
    'cll_cl_c' : (0, None),
    'cll_cl_s' : (0, 2.5),
    'clld_a_c' : (-8, 8),
    'clld_a_s' : (0, 8),
    'clld_cl_c': (0, None),
    'clld_cl_s': (0, 0.4),
}

Cl_laminar_params_solved = fit(
    model=model_Cl_laminar,
    x_data=d,
    y_data=d['Cl'],
    param_guesses=Cl_laminar_params_guess,
    param_bounds=Cl_laminar_param_bounds,
    weights=np.logical_and(d['Re'] <= subcritical_threshold, True).astype('int')
)

plot_fit_alpha_Re(
    model=model_Cl_laminar,
    x_data=d,
    y_data=d['Cl'],
    params_solved=Cl_laminar_params_solved,
    title="Fit: Lift Coefficient (Laminar)"
)


# Fit the blend
def model_Cl_blend(x, p):
    log10_Re = cas.log10(x['Re'])
    blend_input = -p['clb_hardness'] * (
            p['clb_a_scale'] * (5 + (x['alpha'] - p['clb_a_0']) ** 2) ** 0.5 + p['clb_re_0']
            - log10_Re
    )
    blend = sigmoid(blend_input, 0, 1, 0.5, 0.5)
    Cl = blend * model_Cl_turbulent(x, p) + (1 - blend) * model_Cl_laminar(x, p)
    return Cl


Cl_blend_params_guess = {
    **Cl_turbulent_params_solved,
    **Cl_laminar_params_solved,
    'clb_hardness' : 1.7,
    'clb_a_scale'  : 0.3,
    'clb_a_0'      : 6,
    'clb_re_0'     : 3.5,

}
Cl_blend_param_bounds = {
    # **{k: (v, v) for k, v in Cl_laminar_params_solved.items()},
    # **{k: (v, v) for k, v in Cl_turbulent_params_solved.items()},
    **Cl_laminar_param_bounds,
    **Cl_turbulent_param_bounds,
    'clb_hardness' : (1e-3, 10),
    'clb_a_scale'  : (0, 1),
    'clb_a_0'      : (-4, 12),
    'clb_re_0'     : (3, 6),
}

Cl_blend_params_solved = fit(
    model=model_Cl_blend,
    x_data=d,
    y_data=d['Cl'],
    param_guesses=Cl_blend_params_guess,
    param_bounds=Cl_blend_param_bounds,
    # weights=(d['Cl'] >= 0).astype('int')
)

plot_fit_alpha_Re(
    model=model_Cl_blend,
    x_data=d,
    y_data=d['Cl'],
    params_solved=Cl_blend_params_solved,
    title="Fit: Lift Coefficient (Blend)"
)
