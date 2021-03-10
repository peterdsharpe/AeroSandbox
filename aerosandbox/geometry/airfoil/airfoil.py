import aerosandbox.numpy as np
from aerosandbox import AeroSandboxObject
from aerosandbox.geometry.polygon import Polygon, stack_coordinates
from aerosandbox.geometry.airfoil.airfoil_families import get_NACA_coordinates, get_UIUC_coordinates, \
    get_kulfan_coordinates, get_file_coordinates
from aerosandbox.geometry.airfoil.default_airfoil_aerodynamics import default_CL_function, default_CD_function, \
    default_CM_function
from scipy.interpolate import interp1d
from aerosandbox.visualization.matplotlib import plt
from aerosandbox.visualization.plotly import go, px
from typing import Callable, Union


class Airfoil(Polygon):
    """
    An airfoil. See constructor docstring for usage details.
    """
    def __init__(self,
                 name: str = "Untitled",
                 coordinates: Union[None, str, np.ndarray] = None,
                 CL_function: Callable[[float, float, float, float], float] = default_CL_function,
                 CD_function: Callable[[float, float, float, float], float] = default_CD_function,
                 CM_function: Callable[[float, float, float, float], float] = default_CM_function,
                 ):
        """
        Creates an Airfoil object.

        Args:

            name: Name of the airfoil [string]. Can also be used to auto-generate coordinates; see docstring for
            `coordinates` below.

            coordinates: A representation of the coordinates that define the airfoil. Can be one of several types of
            input; the following sequence of operations is used to interpret the meaning of the parameter:

                If `coordinates` is an Nx2 array of the [x, y] coordinates that define the airfoil, these are used
                as-is. Points are expected to be provided in standard airfoil order:

                    * Points should start on the upper surface at the trailing edge, continue forward over the upper
                    surface, wrap around the nose, continue aft over the lower surface, and then end at the trailing
                    edge on the lower surface.

                    * The trailing edge need not be closed, but many analyses implicitly assume that this gap is small.

                    * Take care to ensure that the point at the leading edge of the airfoil, usually (0, 0),
                    is not duplicated.

                If `coordinates` is provided as a string, it assumed to be the filepath to a *.dat file containing
                the coordinates; we attempt to load coordinates from this.

                If the coordinates are not specified and instead left as None, the constructor will attempt to
                auto-populate the coordinates based on the `name` parameter provided, in the following order of
                priority:

                    * If `name` is a 4-digit NACA airfoil (e.g. "naca2412"), coordinates will be created based on the
                    analytical equation.

                    * If `name` is the name of an airfoil in the UIUC airfoil database (e.g. "s1223", "e216",
                    "dae11"), coordinates will be loaded from that. Note that the string you provide must be exactly
                    the name of the associated *.dat file in the UIUC database.

            CL_function: A function that gives the sectional lift coefficient of the airfoil as a function of several
            parameters.

                Must be a callable with the exact syntax:

                >>> def my_function(alpha, Re, mach, deflection)

                where:

                    * `alpha` is the local angle of attack, in degrees

                    * `Re` is the local Reynolds number

                    * `mach` is the local mach number

                    * `deflection` is the deflection of any control surface on the airfoil, given in degrees.

            CD_function: A function that gives the sectional drag coefficient of the airfoil as a function of
            several parameters.

                Has the exact same syntax as `CL_function`, see above.

            Cm_function: A function that gives the sectional moment coefficient of the airfoil (about the
            quarter-chord) as a function of several parameters.

                Has the exact same syntax as `CL_function`, see above.

        """

        ### Handle the airfoil name
        self.name = name

        ### Handle the coordinates
        if coordinates is None:  # If no coordinates are given
            try:  # See if it's a NACA airfoil
                coordinates = get_NACA_coordinates(name=self.name)
            except:
                try:  # See if it's in the UIUC airfoil database
                    coordinates = get_UIUC_coordinates(name=self.name)
                except:
                    pass
        elif isinstance(coordinates, str):  # If coordinates is a string, assume it's a filepath to a .dat file
            coordinates = get_file_coordinates(filepath=coordinates)

        self.coordinates = coordinates

        ### Handle other arguments
        self.CL_function = CL_function
        self.CD_function = CD_function
        self.CM_function = CM_function

    def __repr__(self):  # String representation
        return f"Airfoil {self.name} ({self.n_points()} points)"

    def local_camber(self, x_over_c=np.linspace(0, 1, 101)):
        """
        Returns the local camber of the airfoil at a given point or points.
        :param x_over_c: The x/c locations to calculate the camber at [1D array, more generally, an iterable of floats]
        :return: Local camber of the airfoil (y/c) [1D array].
        """
        # TODO casadify?
        upper = self.upper_coordinates()[::-1]
        lower = self.lower_coordinates()

        upper_interpolated = np.interp(
            x_over_c,
            upper[:, 0],
            upper[:, 1],
        )
        lower_interpolated = np.interp(
            x_over_c,
            lower[:, 0],
            lower[:, 1],
        )

        return (upper_interpolated + lower_interpolated) / 2

    def local_thickness(self, x_over_c=np.linspace(0, 1, 101)):
        """
        Returns the local thickness of the airfoil at a given point or points.
        :param x_over_c: The x/c locations to calculate the thickness at [1D array, more generally, an iterable of floats]
        :return: Local thickness of the airfoil (y/c) [1D array].
        """
        # TODO casadify?
        upper = self.upper_coordinates()[::-1]
        lower = self.lower_coordinates()

        upper_interpolated = np.interp(
            x_over_c,
            upper[:, 0],
            upper[:, 1],
        )
        lower_interpolated = np.interp(
            x_over_c,
            lower[:, 0],
            lower[:, 1],
        )

        return upper_interpolated - lower_interpolated

    def draw(self, draw_mcl=True, backend="plotly", show=True):
        """
        Draw the airfoil object.
        :param draw_mcl: Should we draw the mean camber line (MCL)? [boolean]
        :param backend: Which backend should we use? "plotly" or "matplotlib"
        :return: None
        """
        x = np.array(self.x()).reshape(-1)
        y = np.array(self.y()).reshape(-1)
        if draw_mcl:
            x_mcl = np.linspace(np.min(x), np.max(x), len(x))
            y_mcl = self.local_camber(x_mcl)
        if backend == "plotly":
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    name="Airfoil",
                    fill="toself",
                    line=dict(
                        color="blue"
                    )
                ),
            )
            if draw_mcl:
                fig.add_trace(
                    go.Scatter(
                        x=x_mcl,
                        y=y_mcl,
                        mode="lines+markers",
                        name="Mean Camber Line (MCL)",
                        line=dict(
                            color="navy"
                        )
                    )
                )
            fig.update_layout(
                xaxis_title="x/c",
                yaxis_title="y/c",
                yaxis=dict(scaleanchor="x", scaleratio=1),
                title="%s Airfoil" % self.name
            )
            if show:
                fig.show()
            else:
                return fig
        elif backend == "matplotlib":
            fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
            plt.plot(x, y, ".-", zorder=11, color='#280887')
            if draw_mcl:
                plt.plot(x_mcl, y_mcl, "-", zorder=4, color='#28088744')
            plt.axis("equal")
            plt.xlabel(r"$x/c$")
            plt.ylabel(r"$y/c$")
            plt.title("%s Airfoil" % self.name)
            plt.tight_layout()
            if show:
                plt.show()
            else:
                return fig, ax

    def LE_index(self) -> int:
        """
        Returns the index of the leading-edge point.
        """
        return np.argmin(self.x())

    def lower_coordinates(self) -> np.ndarray:
        """
        Returns an Nx2 ndarray of [x, y] coordinates that describe the lower surface of the airfoil.

        Order is from the leading edge to the trailing edge.

        Includes the leading edge point; be careful about duplicates if using this method in conjunction with
        Airfoil.upper_coordinates().
        """
        return self.coordinates[self.LE_index():, :]

    def upper_coordinates(self) -> np.ndarray:
        """
        Returns an Nx2 ndarray of [x, y] coordinates that describe the upper surface of the airfoil.

        Order is from the trailing edge to the leading edge.

        Includes the leading edge point; be careful about duplicates if using this method in conjunction with
        Airfoil.lower_coordinates().
        """
        return self.coordinates[:self.LE_index() + 1, :]

    def TE_thickness(self) -> float:
        """
        Returns the thickness of the trailing edge of the airfoil.
        """
        return self.local_thickness(x_over_c=1)

    def TE_angle(self) -> float:
        """
        Returns the trailing edge angle of the airfoil, in degrees
        """
        upper_TE_vec = self.coordinates[0, :] - self.coordinates[1, :]
        lower_TE_vec = self.coordinates[-1, :] - self.coordinates[-2, :]

        return 180 / np.pi * (np.arctan2(
            upper_TE_vec[0] * lower_TE_vec[1] - upper_TE_vec[1] * lower_TE_vec[0],
            upper_TE_vec[0] * lower_TE_vec[0] + upper_TE_vec[1] * upper_TE_vec[1]
        ))

    def repanel(self,
                n_points_per_side: int = 100,
                ) -> 'Airfoil':
        """
        Returns a repaneled version of the airfoil with cosine-spaced coordinates on the upper and lower surfaces.
        :param n_points_per_side: Number of points per side (upper and lower) of the airfoil [int]
            Notes: The number of points defining the final airfoil will be n_points_per_side*2-1,
            since one point (the leading edge point) is shared by both the upper and lower surfaces.
        :return: Returns the new airfoil.
        """

        upper_original_coors = self.upper_coordinates()  # Note: includes leading edge point, be careful about duplicates
        lower_original_coors = self.lower_coordinates()  # Note: includes leading edge point, be careful about duplicates

        # Find distances between coordinates, assuming linear interpolation
        upper_distances_between_points = (
                                                 (upper_original_coors[:-1, 0] - upper_original_coors[1:, 0]) ** 2 +
                                                 (upper_original_coors[:-1, 1] - upper_original_coors[1:, 1]) ** 2
                                         ) ** 0.5
        lower_distances_between_points = (
                                                 (lower_original_coors[:-1, 0] - lower_original_coors[1:, 0]) ** 2 +
                                                 (lower_original_coors[:-1, 1] - lower_original_coors[1:, 1]) ** 2
                                         ) ** 0.5
        upper_distances_from_TE = np.hstack((0, np.cumsum(upper_distances_between_points)))
        lower_distances_from_LE = np.hstack((0, np.cumsum(lower_distances_between_points)))
        upper_distances_from_TE_normalized = upper_distances_from_TE / upper_distances_from_TE[-1]
        lower_distances_from_LE_normalized = lower_distances_from_LE / lower_distances_from_LE[-1]

        distances_from_TE_normalized = np.hstack((
            upper_distances_from_TE_normalized,
            1 + lower_distances_from_LE_normalized[1:]
        ))

        # Generate a cosine-spaced list of points from 0 to 1
        cosspaced_points = np.cosspace(0, 1, n_points_per_side)
        s = np.hstack((
            cosspaced_points,
            1 + cosspaced_points[1:],
        ))

        # Check that there are no duplicate points in the airfoil.
        if np.any(np.diff(distances_from_TE_normalized) == 0):
            raise ValueError(
                "This airfoil has a duplicated point (i.e. two adjacent points with the same (x, y) coordinates), so you can't repanel it!")

        x = interp1d(
            distances_from_TE_normalized,
            self.x(),
            kind="cubic",
        )(s)
        y = interp1d(
            distances_from_TE_normalized,
            self.y(),
            kind="cubic",
        )(s)

        return Airfoil(
            name=self.name,
            coordinates=stack_coordinates(x, y)
        )

    def add_control_surface(
            self,
            deflection: float = 0.,
            hinge_point_x: float = 0.75,
    ) -> 'Airfoil':
        """
        Returns a version of the airfoil with a control surface added at a given point. Implicitly repanels the airfoil as part of this operation.
        :param deflection: deflection angle [degrees]. Downwards-positive.
        :param hinge_point_x: location of the hinge, as a fraction of chord [float].
        :return: The new airfoil.
        """

        # Make the rotation matrix for the given angle.
        rotation_matrix = np.rotations.rotation_matrix_2D(-np.pi / 180 * deflection)

        # Find the hinge point
        hinge_point_y = self.local_camber(hinge_point_x)

        # Find the coordinates of a rotated airfoil
        rotated_airfoil = self.rotate(
            angle=-np.pi / 180 * deflection,
            x_center=hinge_point_x,
            y_center=hinge_point_y,
        )

        # Merge the two sets of coordinates

        coordinates = np.copy(self.coordinates)
        is_past_hinge = self.x() > hinge_point_x  # TODO fix hinge self-intersecting paneling issue for large deflection
        coordinates[is_past_hinge] = rotated_airfoil.coordinates[is_past_hinge]

        return Airfoil(
            name=self.name,
            coordinates=coordinates
        )

    def scale(self,
              scale_x: float = 1.,
              scale_y: float = 1.,
              ) -> 'Airfoil':
        """
        Scales an Airfoil about the origin.
        Args:
            scale_x: Amount to scale in the x-direction.
            scale_y: Amount to scale in the y-direction.

        Returns: The scaled Airfoil.
        """
        x = self.x() * scale_x
        y = self.y() * scale_y

        if scale_y < 0:
            x = x[::-1]
            y = y[::-1]

        return Airfoil(
            name=self.name,
            coordinates=stack_coordinates(x, y)
        )

    def translate(self,
                  translate_x: float = 0.,
                  translate_y: float = 0.,
                  ) -> 'Airfoil':
        """
        Translates an Airfoil by a given amount.
        Args:
            translate_x: Amount to translate in the x-direction
            translate_y: Amount to translate in the y-direction

        Returns: The translated Airfoil.

        """
        x = self.x() + translate_x
        y = self.y() + translate_y

        return Airfoil(
            name=self.name,
            coordinates=stack_coordinates(x, y)
        )

    def rotate(self,
               angle: float,
               x_center: float = 0.,
               y_center: float = 0.
               ) -> 'Airfoil':
        """
        Rotates the airfoil clockwise by the specified amount, in radians.

        Rotates about the point (x_center, y_center), which is (0, 0) by default.

        Args:
            angle: Angle to rotate, counterclockwise, in radians.

            x_center: The x-coordinate of the center of rotation.

            y_center: The y-coordinate of the center of rotation.

        Returns: The rotated Airfoil.

        """

        coordinates = np.copy(self.coordinates)

        ### Translate
        translation = np.array([x_center, y_center])
        coordinates -= translation

        ### Rotate
        rotation_matrix = np.rotation_matrix_2D(
            angle=angle,
        )
        coordinates = (rotation_matrix @ coordinates.T).T

        ### Translate
        coordinates += translation

        return Airfoil(
            name=self.name,
            coordinates=coordinates
        )

    # def normalize(self):
    #     pass  # TODO finish me

    def write_dat(self,
                  filepath: str
                  ):
        """
        Writes a .dat file corresponding to this airfoil to a filepath.

        Args:
            filepath: filepath (including the filename and .dat extension) [string]

        Returns: None

        """
        with open(filepath, "w+") as f:
            f.writelines(
                [self.name + "\n"] +
                [f"%f %f\n" % tuple(coordinate) for coordinate in self.coordinates]
            )

    def write_sldcrv(self,
                     filepath: str
                     ):
        """
        Writes a .sldcrv (SolidWorks curve) file corresponding to this airfoil to a filepath.
        Args:
            filepath: A filepath (including the filename and .sldcrv extension) [string]

        Returns: None

        """
        with open(filepath, "w+") as f:
            for i, coordinate in enumerate(self.coordinates):
                f.write(
                    f"{coordinate[0]} {coordinate[1]} 0"
                )
                if i < self.n_points() - 1:
                    f.write(
                        f"\n"
                    )

    # def get_xfoil_data(self,
    #                    a_start=-6,  # type: float
    #                    a_end=12,  # type: float
    #                    a_step=0.5,  # type: float
    #                    a_init=0,  # type: float
    #                    Re_start=1e4,  # type: float
    #                    Re_end=1e7,  # type: float
    #                    n_Res=30,  # type: int
    #                    mach=0,  # type: float
    #                    max_iter=20,  # type: int
    #                    repanel=False,  # type: bool
    #                    parallel=True,  # type: bool
    #                    verbose=True,  # type: bool
    #                    ):
    #     """ # TODO finish docstring
    #     Calculates aerodynamic performance data for a particular airfoil with XFoil.
    #     Does a 2D grid sweep of the alpha-Reynolds space at a particular Mach number.
    #     Populates two new instance variables:
    #         * self.xfoil_data_1D: A dict of XFoil data at all calculated operating points (1D arrays, NaNs removed)
    #         * self.xfoil_data_2D: A dict of XFoil data at all calculated operating points (2D arrays, NaNs present)
    #     :param a_start: Lower bound of angle of attack [deg]
    #     :param a_end: Upper bound of angle of attack [deg]
    #     :param a_step: Angle of attack increment size [deg]
    #     :param a_init: Angle of attack to initialize runs at. Should solve easily (0 recommended) [deg]
    #     :param Re_start: Reynolds number to begin sweep at. [unitless]
    #     :param Re_end: Reynolds number to end sweep at. [unitless]
    #     :param n_Res: Number of Reynolds numbers to sweep. Points are log-spaced.
    #     :param mach: Mach number to sweep at.
    #     :param max_iter: Maximum number of XFoil iterations per op-point.
    #     :param repanel: Should we interally repanel the airfoil within XFoil before running? [boolean]
    #         Consider disabling this if you try to do optimization based on this data (for smoothness reasons).
    #         Otherwise, it's generally a good idea to leave this on.
    #     :param parallel: Should we run in parallel? Generally results in significant speedup, but might not run
    #         correctly on some machines. Disable this if it's a problem. [boolean]
    #     :param verbose: Should we do verbose output? [boolean]
    #     :return: self (in-place operation that creates self.xfoil_data_1D and self.xfoil_data_2D)
    #     """
    #     assert a_init > a_start
    #     assert a_init < a_end
    #     assert Re_start < Re_end
    #     assert n_Res >= 1
    #     assert mach >= 0
    #
    #     Res = np.logspace(np.log10(Re_start), np.log10(Re_end), n_Res)
    #
    #     def get_xfoil_data_at_Re(Re):
    #
    #         import aerosandbox.numpy as np  # needs to be imported here to support parallelization
    #
    #         run_data_upper = self.xfoil_aseq(
    #             a_start=a_init + a_step,
    #             a_end=a_end,
    #             a_step=a_step,
    #             Re=Re,
    #             repanel=repanel,
    #             max_iter=max_iter,
    #             M=mach,
    #             reset_bls=True,
    #         )
    #         run_data_lower = self.xfoil_aseq(
    #             a_start=a_init,
    #             a_end=a_start,
    #             a_step=-a_step,
    #             Re=Re,
    #             repanel=repanel,
    #             max_iter=max_iter,
    #             M=mach,
    #             reset_bls=True,
    #         )
    #         run_data = {
    #             k: np.hstack((
    #                 run_data_lower[k][::-1],
    #                 run_data_upper[k]
    #             )) for k in run_data_upper.keys()
    #         }
    #         return run_data
    #
    #     if verbose:
    #         print("Running XFoil sweeps on Airfoil %s..." % self.name)
    #         import time
    #         start_time = time.time()
    #
    #     if not parallel:
    #         runs_data = [get_xfoil_data_at_Re(Re) for Re in Res]
    #     else:
    #         import multiprocess as mp
    #         pool = mp.Pool(mp.cpu_count())
    #         runs_data = pool.map(get_xfoil_data_at_Re, Res)
    #         pool.close()
    #
    #     if verbose:
    #         run_time = time.time() - start_time
    #         print("XFoil Runtime: %.3f sec" % run_time)
    #
    #     xfoil_data_2D = {}
    #     for k in runs_data[0].keys():
    #         xfoil_data_2D[k] = np.vstack([
    #             d[k]
    #             for d in runs_data
    #         ])
    #     xfoil_data_2D["Re"] = np.tile(Res, (
    #         xfoil_data_2D["alpha"].shape[1],
    #         1
    #     )).T
    #     np.place(
    #         arr=xfoil_data_2D["Re"],
    #         mask=np.isnan(xfoil_data_2D["alpha"]),
    #         vals=np.NaN
    #     )
    #     xfoil_data_2D["alpha_indices"] = np.arange(a_start, a_end + a_step / 2, a_step)
    #     xfoil_data_2D["Re_indices"] = Res
    #
    #     self.xfoil_data_2D = xfoil_data_2D
    #
    #     # 1-dimensionalize it and remove NaNs
    #     xfoil_data_1D = {
    #         k: remove_nans(xfoil_data_2D[k].reshape(-1))
    #         for k in xfoil_data_2D.keys()
    #     }
    #     self.xfoil_data_1D = xfoil_data_1D
    #
    #     return self
    #
    # def has_xfoil_data(self, raise_exception_if_absent=True):
    #     """
    #     Runs a quick check to see if this airfoil has XFoil data.
    #     :param raise_exception_if_absent: Boolean flag to raise an Exception if XFoil data is not found.
    #     :return: Boolean of whether or not XFoil data is present.
    #     """
    #     data_present = (
    #             hasattr(self, 'xfoil_data_1D') and
    #             hasattr(self, 'xfoil_data_2D')
    #     )
    #     if not data_present and raise_exception_if_absent:
    #         raise Exception(
    #             """This Airfoil %s does not yet have XFoil data,
    #             so you can't run the function you've called.
    #             To get XFoil data, first call:
    #                 Airfoil.get_xfoil_data()
    #             which will perform an in-place update that
    #             provides the data.""" % self.name
    #         )
    #     return data_present
    #
    # def plot_xfoil_data_contours(self):  # TODO add docstring
    #     self.has_xfoil_data()  # Ensure data is present.
    #     from matplotlib import colors
    #
    #     d = self.xfoil_data_1D  # data
    #
    #     fig = plt.figure(figsize=(10, 8), dpi=200)
    #
    #     ax = fig.add_subplot(311)
    #     coords = self.coordinates
    #     plt.plot(coords[:, 0], coords[:, 1], '.-', color='#280887')
    #     plt.xlabel(r"$x/c$")
    #     plt.ylabel(r"$y/c$")
    #     plt.title(r"XFoil Data for %s Airfoil" % self.name)
    #     plt.axis("equal")
    #
    #     with plt.style.context("default"):
    #         ax = fig.add_subplot(323)
    #         x = d["Re"]
    #         y = d["alpha"]
    #         z = d["Cl"]
    #         levels = np.linspace(-0.5, 1.5, 21)
    #         norm = None
    #         CF = ax.tricontourf(x, y, z, levels=levels, norm=norm, cmap="plasma", extend="both")
    #         C = ax.tricontour(x, y, z, levels=levels, norm=norm, colors='k', extend="both", linewidths=0.5)
    #         cbar = plt.colorbar(CF, format='%.2f')
    #         cbar.set_label(r"$C_l$")
    #         plt.grid(False)
    #         plt.xlabel(r"$Re$")
    #         plt.ylabel(r"$\alpha$")
    #         plt.title(r"$C_l$ from $Re$, $\alpha$")
    #         ax.set_xscale('log')
    #
    #         ax = fig.add_subplot(324)
    #         x = d["Re"]
    #         y = d["alpha"]
    #         z = d["Cd"]
    #         levels = np.logspace(-2.5, -1, 21)
    #         norm = colors.PowerNorm(gamma=1 / 2, vmin=np.min(levels), vmax=np.max(levels))
    #         CF = ax.tricontourf(x, y, z, levels=levels, norm=norm, cmap="plasma", extend="both")
    #         C = ax.tricontour(x, y, z, levels=levels, norm=norm, colors='k', extend="both", linewidths=0.5)
    #         cbar = plt.colorbar(CF, format='%.3f')
    #         cbar.set_label(r"$C_d$")
    #         plt.grid(False)
    #         plt.xlabel(r"$Re$")
    #         plt.ylabel(r"$\alpha$")
    #         plt.title(r"$C_d$ from $Re$, $\alpha$")
    #         ax.set_xscale('log')
    #
    #         ax = fig.add_subplot(325)
    #         x = d["Re"]
    #         y = d["alpha"]
    #         z = d["Cl"] / d["Cd"]
    #         x = x[d["alpha"] >= 0]
    #         y = y[d["alpha"] >= 0]
    #         z = z[d["alpha"] >= 0]
    #         levels = np.logspace(1, np.log10(150), 21)
    #         norm = colors.PowerNorm(gamma=1 / 2, vmin=np.min(levels), vmax=np.max(levels))
    #         CF = ax.tricontourf(x, y, z, levels=levels, norm=norm, cmap="plasma", extend="both")
    #         C = ax.tricontour(x, y, z, levels=levels, norm=norm, colors='k', extend="both", linewidths=0.5)
    #         cbar = plt.colorbar(CF, format='%.1f')
    #         cbar.set_label(r"$L/D$")
    #         plt.grid(False)
    #         plt.xlabel(r"$Re$")
    #         plt.ylabel(r"$\alpha$")
    #         plt.title(r"$L/D$ from $Re$, $\alpha$")
    #         ax.set_xscale('log')
    #
    #         ax = fig.add_subplot(326)
    #         x = d["Re"]
    #         y = d["alpha"]
    #         z = d["Cm"]
    #         levels = np.linspace(-0.15, 0, 21)  # np.logspace(1, np.log10(150), 21)
    #         norm = None  # colors.PowerNorm(gamma=1 / 2, vmin=np.min(levels), vmax=np.max(levels))
    #         CF = ax.tricontourf(x, y, z, levels=levels, norm=norm, cmap="plasma", extend="both")
    #         C = ax.tricontour(x, y, z, levels=levels, norm=norm, colors='k', extend="both", linewidths=0.5)
    #         cbar = plt.colorbar(CF, format='%.2f')
    #         cbar.set_label(r"$C_m$")
    #         plt.grid(False)
    #         plt.xlabel(r"$Re$")
    #         plt.ylabel(r"$\alpha$")
    #         plt.title(r"$C_m$ from $Re$, $\alpha$")
    #         ax.set_xscale('log')
    #
    #     plt.tight_layout()
    #     plt.show()
    #
    #     return self
    #
    # def plot_xfoil_data_all_polars(self,
    #                                n_lines_max=20,
    #                                Cd_plot_max=0.04,
    #                                ):
    #     """
    #     Plots the existing XFoil data found by running self.get_xfoil_data().
    #     :param n_lines_max: Maximum number of Reynolds numbers to plot. Useful if you ran a sweep with tons of Reynolds numbers.
    #     :param Cd_plot_max: Upper limit of Cd to plot [float]
    #     :return: self (makes plot)
    #     """
    #
    #     self.has_xfoil_data()  # Ensure data is present.
    #
    #     n_lines_max = min(n_lines_max, len(self.xfoil_data_2D["Re_indices"]))
    #
    #     fig, ax = plt.subplots(1, 1, figsize=(7, 6), dpi=200)
    #     indices = np.array(
    #         np.round(np.linspace(0, len(self.xfoil_data_2D["Re_indices"]) - 1, n_lines_max)),
    #         dtype=int
    #     )
    #     indices_worth_plotting = [
    #         np.min(remove_nans(self.xfoil_data_2D["Cd"][index, :])) < Cd_plot_max
    #         for index in indices
    #     ]
    #     indices = indices[indices_worth_plotting]
    #
    #     colors = plt.cm.rainbow(np.linspace(0, 1, len(indices)))[::-1]
    #     for i, Re in enumerate(self.xfoil_data_2D["Re_indices"][indices]):
    #         Cds = remove_nans(self.xfoil_data_2D["Cd"][indices[i], :])
    #         Cls = remove_nans(self.xfoil_data_2D["Cl"][indices[i], :])
    #         Cd_min = np.min(Cds)
    #         if Cd_min < Cd_plot_max:
    #             plt.plot(
    #                 Cds * 1e4,
    #                 Cls,
    #                 label="Re = %s" % eng_string(Re),
    #                 color=colors[i],
    #             )
    #     plt.xlim(0, Cd_plot_max * 1e4)
    #     plt.ylim(0, 2)
    #     plt.xlabel(r"$C_d \cdot 10^4$")
    #     plt.ylabel(r"$C_l$")
    #     plt.title("XFoil Polars for %s Airfoil" % self.name)
    #     plt.tight_layout()
    #     plt.legend()
    #     plt.show()
    #
    #     return self
    #
    # def plot_xfoil_data_polar(self,
    #                           Res,  # type: list
    #                           Cd_plot_max=0.04,
    #                           repanel=False,
    #                           parallel=True,
    #                           max_iter=40,
    #                           verbose=True,
    #                           ):
    #     """
    #     Plots CL-CD polar for a single Reynolds number or a variety of Reynolds numbers.
    #     :param Res: Reynolds number to plot polars at. Either a single float or an iterable (list, 1D ndarray, etc.)
    #     :param Cd_plot_max: Upper limit of Cd to plot [float]
    #     :param cl_step: Cl increment for XFoil runs. Trades speed vs. plot resolution. [float]
    #     :param repanel: Should we repanel the airfoil within XFoil? [boolean]
    #     :param parallel: Should we run different Res in parallel? [boolean]
    #     :param max_iter: Maximum number of iterations for XFoil to run. [int]
    #     :param verbose: Should we print information as we run the sweeps? [boolean]
    #     :return: self (makes plot)
    #     """
    #
    #     try:  # If it's not an iterable, make it one.
    #         Res[0]
    #     except TypeError:
    #         Res = [Res]
    #
    #     fig, ax = plt.subplots(1, 1, figsize=(7, 6), dpi=200)
    #     colors = plt.cm.rainbow(np.linspace(0, 1, len(Res)))[::-1]
    #
    #     def get_xfoil_data_at_Re(Re):
    #
    #         xfoil_data = self.xfoil_aseq(
    #             a_start=0,
    #             a_end=15,
    #             a_step=0.25,
    #             Re=Re,
    #             M=0,
    #             reset_bls=True,
    #             repanel=repanel,
    #             max_iter=max_iter,
    #             verbose=False,
    #         )
    #         Cd = remove_nans(xfoil_data["Cd"])
    #         Cl = remove_nans(xfoil_data["Cl"])
    #         return {"Cl": Cl, "Cd": Cd}
    #
    #     if verbose:
    #         print("Running XFoil sweeps...")
    #         import time
    #         start_time = time.time()
    #
    #     if not parallel:
    #         runs_data = [get_xfoil_data_at_Re(Re) for Re in Res]
    #     else:
    #         import multiprocess as mp
    #         pool = mp.Pool(mp.cpu_count())
    #         runs_data = pool.map(get_xfoil_data_at_Re, Res)
    #         pool.close()
    #
    #     if verbose:
    #         run_time = time.time() - start_time
    #         print("XFoil Runtime: %.3f sec" % run_time)
    #
    #     for i, Re in enumerate(Res):
    #         plt.plot(
    #             runs_data[i]["Cd"] * 1e4,
    #             runs_data[i]["Cl"],
    #             label="Re = %s" % eng_string(Re),
    #             color=colors[i],
    #         )
    #     plt.xlim(0, Cd_plot_max * 1e4)
    #     plt.ylim(0, 2)
    #     plt.xlabel(r"$C_d \cdot 10^4$")
    #     plt.ylabel(r"$C_l$")
    #     plt.title("XFoil Polars for %s Airfoil" % self.name)
    #     plt.tight_layout()
    #     plt.legend()
    #     plt.show()
    #
    #     return self
