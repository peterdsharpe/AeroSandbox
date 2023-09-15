from aerosandbox.common import *
from aerosandbox.geometry import Airfoil
from aerosandbox.performance import OperatingPoint
from aerosandbox.aerodynamics.aero_2D.singularities import calculate_induced_velocity_line_singularities
import aerosandbox.numpy as np
from typing import Union, List, Optional


class AirfoilInviscid(ImplicitAnalysis):
    """
    An implicit analysis for inviscid analysis of an airfoil (or family of airfoils).

    Key outputs:

        * AirfoilInviscid.Cl

    """

    @ImplicitAnalysis.initialize
    def __init__(self,
                 airfoil: Union[Airfoil, List[Airfoil]],
                 op_point: OperatingPoint,
                 ground_effect: bool = False,
                 ):
        if isinstance(airfoil, Airfoil):
            self.airfoils = [airfoil]
        else:
            self.airfoils = airfoil

        self.op_point = op_point

        self.ground_effect = ground_effect

        self._setup_unknowns()
        self._enforce_governing_equations()
        self._calculate_forces()

    def __repr__(self):
        return self.__class__.__name__ + "(\n\t" + "\n\t".join([
            f"airfoils={self.airfoils}",
            f"op_point={self.op_point}",
        ]) + "\n)"

    def _setup_unknowns(self):
        for airfoil in self.airfoils:
            airfoil.gamma = self.opti.variable(
                init_guess=0,
                scale=self.op_point.velocity,
                n_vars=airfoil.n_points()
            )
            airfoil.sigma = np.zeros(airfoil.n_points())

    def calculate_velocity(self,
                           x_field,
                           y_field,
                           ) -> [np.ndarray, np.ndarray]:
        ### Analyze the freestream
        u_freestream = self.op_point.velocity * np.cosd(self.op_point.alpha)
        v_freestream = self.op_point.velocity * np.sind(self.op_point.alpha)

        u_field = u_freestream
        v_field = v_freestream

        for airfoil in self.airfoils:

            ### Add in the influence of the vortices and sources on the airfoil surface
            u_field_induced, v_field_induced = calculate_induced_velocity_line_singularities(
                x_field=x_field,
                y_field=y_field,
                x_panels=airfoil.x(),
                y_panels=airfoil.y(),
                gamma=airfoil.gamma,
                sigma=airfoil.sigma,
            )

            u_field = u_field + u_field_induced
            v_field = v_field + v_field_induced

            ### Add in the influence of a source across the open trailing-edge panel.
            if airfoil.TE_thickness() != 0:
                u_field_induced_TE, v_field_induced_TE = calculate_induced_velocity_line_singularities(
                    x_field=x_field,
                    y_field=y_field,
                    x_panels=[airfoil.x()[0], airfoil.x()[-1]],
                    y_panels=[airfoil.y()[0], airfoil.y()[-1]],
                    gamma=[0, 0],
                    sigma=[airfoil.gamma[0], airfoil.gamma[0]]
                )

                u_field = u_field + u_field_induced_TE
                v_field = v_field + v_field_induced_TE

            if self.ground_effect:

                ### Add in the influence of the vortices and sources on the airfoil surface
                u_field_induced, v_field_induced = calculate_induced_velocity_line_singularities(
                    x_field=x_field,
                    y_field=y_field,
                    x_panels=airfoil.x(),
                    y_panels=-airfoil.y(),
                    gamma=-airfoil.gamma,
                    sigma=airfoil.sigma,
                )

                u_field = u_field + u_field_induced
                v_field = v_field + v_field_induced

                ### Add in the influence of a source across the open trailing-edge panel.
                if airfoil.TE_thickness() != 0:
                    u_field_induced_TE, v_field_induced_TE = calculate_induced_velocity_line_singularities(
                        x_field=x_field,
                        y_field=y_field,
                        x_panels=[airfoil.x()[0], airfoil.x()[-1]],
                        y_panels=-1 * np.array([airfoil.y()[0], airfoil.y()[-1]]),
                        gamma=[0, 0],
                        sigma=[airfoil.gamma[0], airfoil.gamma[0]]
                    )

                    u_field = u_field + u_field_induced_TE
                    v_field = v_field + v_field_induced_TE

        return u_field, v_field

    def _enforce_governing_equations(self):

        for airfoil in self.airfoils:
            ### Compute normal velocities at the middle of each panel
            x_midpoints = np.trapz(airfoil.x())
            y_midpoints = np.trapz(airfoil.y())

            u_midpoints, v_midpoints = self.calculate_velocity(
                x_field=x_midpoints,
                y_field=y_midpoints,
            )

            panel_dx = np.diff(airfoil.x())
            panel_dy = np.diff(airfoil.y())
            panel_length = (panel_dx ** 2 + panel_dy ** 2) ** 0.5

            xp_hat_x = panel_dx / panel_length  # x-coordinate of the xp_hat vector
            xp_hat_y = panel_dy / panel_length  # y-coordinate of the yp_hat vector

            yp_hat_x = -xp_hat_y
            yp_hat_y = xp_hat_x

            normal_velocities = u_midpoints * yp_hat_x + v_midpoints * yp_hat_y

            ### Add in flow tangency constraint
            self.opti.subject_to(normal_velocities == 0)

            ### Add in Kutta condition
            self.opti.subject_to(airfoil.gamma[0] + airfoil.gamma[-1] == 0)

    def _calculate_forces(self):

        for airfoil in self.airfoils:
            panel_dx = np.diff(airfoil.x())
            panel_dy = np.diff(airfoil.y())
            panel_length = (panel_dx ** 2 + panel_dy ** 2) ** 0.5

            ### Sum up the vorticity on this airfoil by integrating
            airfoil.vorticity = np.sum(
                (airfoil.gamma[1:] + airfoil.gamma[:-1]) / 2 *
                panel_length
            )

            airfoil.Cl = 2 * airfoil.vorticity  # TODO normalize by chord and freestream velocity etc.

        self.total_vorticity = sum([airfoil.vorticity for airfoil in self.airfoils])
        self.Cl = 2 * self.total_vorticity

    def draw_streamlines(self, res=200, show=True):
        import matplotlib.pyplot as plt
        import aerosandbox.tools.pretty_plots as p

        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)

        plt.xlim(-0.5, 1.5)
        plt.ylim(-0.5, 0.5)
        xrng = np.diff(np.array(ax.get_xlim()))
        yrng = np.diff(np.array(ax.get_ylim()))

        x = np.linspace(*ax.get_xlim(), int(np.round(res * xrng / yrng)))
        y = np.linspace(*ax.get_ylim(), res)

        X, Y = np.meshgrid(x, y)
        shape = X.shape
        X = X.flatten()
        Y = Y.flatten()

        U, V = self.calculate_velocity(X, Y)
        X = X.reshape(shape)
        Y = Y.reshape(shape)
        U = U.reshape(shape)
        V = V.reshape(shape)

        # NaN out any points inside the airfoil
        for airfoil in self.airfoils:
            contains = airfoil.contains_points(X, Y)
            U[contains] = np.nan
            V[contains] = np.nan

        speed = (U ** 2 + V ** 2) ** 0.5
        Cp = 1 - speed ** 2

        ### Draw the airfoils
        for airfoil in self.airfoils:
            plt.fill(airfoil.x(), airfoil.y(), "k", linewidth=0, zorder=4)

        plt.streamplot(
            x,
            y,
            U,
            V,
            color=speed,
            density=2.5,
            arrowsize=0,
            cmap=p.mpl.colormaps.get_cmap('coolwarm_r'),
        )
        CB = plt.colorbar(
            orientation="horizontal",
            shrink=0.8,
            aspect=40,
        )
        CB.set_label(r"Relative Airspeed ($U/U_\infty$)")
        plt.clim(0.6, 1.4)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel(r"$x/c$")
        plt.ylabel(r"$y/c$")
        plt.title(rf"Inviscid Airfoil: Flow Field")
        plt.tight_layout()
        if show:
            plt.show()

    def draw_cp(self, show=True):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
        for airfoil in self.airfoils:
            surface_speeds = airfoil.gamma
            C_p = 1 - surface_speeds ** 2

            plt.plot(airfoil.x(), C_p)

        plt.ylim(-4, 1.1)
        plt.gca().invert_yaxis()
        plt.xlabel(r"$x/c$")
        plt.ylabel(r"$C_p$")
        plt.title(r"$C_p$ on Surface")
        plt.tight_layout()
        if show:
            plt.show()


if __name__ == '__main__':
    a = AirfoilInviscid(
        airfoil=[
            # Airfoil("naca4408")
            #     .repanel(50)
            Airfoil("e423")
            .repanel(n_points_per_side=50),
            Airfoil("naca6408")
            .repanel(n_points_per_side=50)
            .scale(0.4, 0.4)
            .rotate(np.radians(-25))
            .translate(0.9, -0.05),
        ],
        op_point=OperatingPoint(
            velocity=1,
            alpha=5,
        )
    )
    a.draw_streamlines()
    a.draw_cp()

    from aerosandbox import Opti

    opti2 = Opti()
    b = AirfoilInviscid(
        airfoil=Airfoil("naca4408"),
        op_point=OperatingPoint(
            velocity=1,
            alpha=5
        ),
        opti=opti2
    )
