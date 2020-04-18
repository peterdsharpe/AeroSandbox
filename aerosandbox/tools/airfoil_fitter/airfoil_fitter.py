"""
Functions to fit automatic-differentiable models to aerodynamic data from an airfoil.
Requires the xfoil package from PyPI
"""
from aerosandbox.geometry import *


class AirfoilFitter():
    def __init__(self,
                 airfoil,  # type: Airfoil
                 ):
        self.airfoil = airfoil

    def get_xfoil_data(self,
                       a_start=-5,
                       a_end=12,
                       a_step=1,
                       a_init=0,
                       Re_start=5e4,
                       Re_end=5e6,
                       n_Res=10,
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
            xfoil_data_2D["a"].shape[1],
            1
        )).T
        np.place(
            arr=xfoil_data_2D["Re"],
            mask=np.isnan(xfoil_data_2D["a"]),
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
        from matplotlib import ticker
        import matplotlib.colors as colors
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

        ax = fig.add_subplot(323)
        x = d["Re"]
        y = d["cl"]
        z = d["cd"]
        levels = np.logspace(-2.5, -1, 21)
        norm = colors.PowerNorm(gamma=1 / 2, vmin=np.min(levels), vmax=np.max(levels))
        CF = ax.tricontourf(x, y, z, levels=levels, norm=norm, cmap="plasma", extend="both")
        C = ax.tricontour(x, y, z, levels=levels, norm=norm, colors='k', extend="both", linewidths=0.5)
        cbar = plt.colorbar(CF, format='%.3f')
        cbar.set_label(r"$C_d$")
        plt.grid(False)
        plt.xlabel(r"$Re$")
        plt.ylabel(r"$C_l$")
        plt.title(r"$C_d$ from $Re$, $C_l$")
        ax.set_xscale('log')

        ax = fig.add_subplot(324)
        x = d["Re"]
        y = d["a"]
        z = d["cl"]
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

        ax = fig.add_subplot(325)
        x = d["Re"]
        y = d["a"]
        z = d["cl"] / d["cd"]
        x = x[d["a"] >= 0]
        y = y[d["a"] >= 0]
        z = z[d["a"] >= 0]
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
        y = d["a"]
        z = d["cd"]
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

        plt.tight_layout()
        plt.savefig("C:/Users/User/Downloads/temp.svg")
        plt.show()


if __name__ == '__main__':
    af = AirfoilFitter(Airfoil(name="HALE_02", coordinates="C:/Users/User/Downloads/HALE_02.dat"))
    af.get_xfoil_data(repanel=True)
    af.plot_xfoil_data()
