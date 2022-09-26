import aerosandbox.numpy as np
from aerosandbox import AeroSandboxObject
from aerosandbox.geometry.polygon import Polygon, stack_coordinates
from aerosandbox.geometry.airfoil.airfoil_families import get_NACA_coordinates, get_UIUC_coordinates, \
    get_kulfan_coordinates, get_file_coordinates
from aerosandbox.geometry.airfoil.default_airfoil_aerodynamics import default_CL_function, default_CD_function, \
    default_CM_function
from aerosandbox.library.aerodynamics import transonic
from aerosandbox.modeling.splines.hermite import linear_hermite_patch, cubic_hermite_patch
from scipy import interpolate
import matplotlib.pyplot as plt
from typing import Callable, Union, Any, Dict
import json
from pathlib import Path


class Airfoil(Polygon):
    """
    An airfoil. See constructor docstring for usage details.
    """

    def __init__(self,
                 name: str = "Untitled",
                 coordinates: Union[None, str, np.ndarray] = None,
                 generate_polars: bool = False,
                 CL_function: Callable[[float, float, float, float], float] = None,
                 CD_function: Callable[[float, float, float, float], float] = None,
                 CM_function: Callable[[float, float, float, float], float] = None,
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

                Must be a callable with that takes exactly these parameters as follows:

                >>> def CL_function(alpha, Re, mach, deflection)

                where:

                    * `alpha` is the local angle of attack, in degrees

                    * `Re` is the local Reynolds number

                    * `mach` is the local Mach number

                    * `deflection` is the deflection of any control surface on the airfoil, given in degrees. Following
                    convention, a positive control surface deflections corresponds to a downwards deflection of a trailing-edge surface.

            CD_function: A function that gives the sectional drag coefficient of the airfoil as a function of
            several parameters.

                Has the exact same syntax as `CL_function`, see above.

            CM_function: A function that gives the sectional moment coefficient of the airfoil (about the
            quarter-chord) as a function of several parameters.

                Has the exact same syntax as `CL_function`, see above.

        """

        ### Handle the airfoil name
        self.name = name

        ### Handle the coordinates
        self.coordinates = None
        if coordinates is None:  # If no coordinates are given
            try:  # See if it's a NACA airfoil
                self.coordinates = get_NACA_coordinates(name=self.name)
            except:
                try:  # See if it's in the UIUC airfoil database
                    self.coordinates = get_UIUC_coordinates(name=self.name)
                except:
                    pass
        else:

            try:  # If coordinates is a string, assume it's a filepath to a .dat file
                self.coordinates = get_file_coordinates(filepath=coordinates)
            except (OSError, FileNotFoundError, TypeError, UnicodeDecodeError):
                try:
                    shape = coordinates.shape
                    assert len(shape) == 2
                    assert shape[0] == 2 or shape[1] == 2
                    if not shape[1] == 2:
                        coordinates = np.transpose(shape)

                    self.coordinates = coordinates
                except AttributeError:
                    pass

        if self.coordinates is None:
            import warnings
            warnings.warn(
                f"Airfoil {self.name} had no coordinates assigned, and could not parse the `coordinates` input!",
                stacklevel=2,
            )

        ### Handle getting default polars
        if generate_polars:
            self.generate_polars()
        else:
            self.CL_function = default_CL_function
            self.CD_function = default_CD_function
            self.CM_function = default_CM_function

        ### Overwrite any default polars with those provided
        if CL_function is not None:
            self.CL_function = CL_function

        if CD_function is not None:
            self.CD_function = CD_function

        if CM_function is not None:
            self.CM_function = CM_function

    def __repr__(self):  # String representation
        return f"Airfoil {self.name} ({self.n_points()} points)"

    def generate_polars(self,
                        alphas=np.linspace(-15, 15, 21),
                        Res=np.geomspace(1e4, 1e7, 10),
                        cache_filename: str = None,
                        xfoil_kwargs: Dict[str, Any] = None,
                        unstructured_interpolated_model_kwargs: Dict[str, Any] = None,
                        include_compressibility_effects: bool = True,
                        transonic_buffet_lift_knockdown: float = 0.3,
                        make_symmetric_polars: bool = False,
                        ) -> None:
        """
        Generates airfoil polars (CL, CD, CM functions) and assigns them in-place to this Airfoil's polar functions.

        In other words, when this function is run, the following functions will be added (or overwritten) to the instance:
            * Airfoil.CL_function(alpha, Re, mach, deflection)
            * Airfoil.CD_function(alpha, Re, mach, deflection)
            * Airfoil.CM_function(alpha, Re, mach, deflection)

        Where alpha is in degrees. Right now, deflection is not used.

        Args:

            alphas: The range of alphas to sample from XFoil at.

            Res: The range of Reynolds numbers to sample from XFoil at.

            cache_filename: A path-like filename (ideally a "*.json" file) that can be used to cache the XFoil
            results, making it much faster to regenerate the results.

                * If the file does not exist, XFoil will be run, and a cache file will be created.

                * If the file does exist, XFoil will not be run, and the cache file will be read instead.

            xfoil_kwargs: Keyword arguments to pass into the AeroSandbox XFoil module. See the aerosandbox.XFoil
            constructor for options.

            unstructured_interpolated_model_kwargs: Keyword arguments to pass into the UnstructuredInterpolatedModels
            that contain the polars themselves. See the aerosandbox.UnstructuredInterpolatedModel constructor for
            options.

            include_compressibility_effects: Includes compressibility effects in the polars, such as wave drag,
            mach tuck, CL effects across normal shocks. Note that accuracy here is dubious in the transonic regime
            and above - you should really specify your own CL/CD/CM models

        Warning: In-place operation! Modifies this Airfoil object by setting Airfoil.CL_function, etc. to the new
        polars.

        Returns: None (in-place)

        """
        if self.coordinates is None:
            raise ValueError("Cannot generate polars for an airfoil that you don't have the coordinates of!")

        ### Set defaults
        if xfoil_kwargs is None:
            xfoil_kwargs = {}
        if unstructured_interpolated_model_kwargs is None:
            unstructured_interpolated_model_kwargs = {}

        xfoil_kwargs = {  # See asb.XFoil for documentation on these.
            "verbose"      : False,
            "max_iter"     : 20,
            "xfoil_repanel": True,
            **xfoil_kwargs
        }

        unstructured_interpolated_model_kwargs = {  # These were tuned heuristically as defaults!
            "resampling_interpolator_kwargs": {
                "degree"   : 0,
                # "kernel": "linear",
                "kernel"   : "multiquadric",
                "epsilon"  : 3,
                "smoothing": 0.01,
                # "kernel": "cubic"
            },
            **unstructured_interpolated_model_kwargs
        }

        ### Retrieve XFoil Polar Data from cache, if it exists.
        data = None
        if cache_filename is not None:
            try:
                with open(cache_filename, "r") as f:
                    data = {
                        k: np.array(v)
                        for k, v in json.load(f).items()
                    }
            except FileNotFoundError:
                pass

        ### Analyze airfoil with XFoil, if needed
        if data is None:

            from aerosandbox.aerodynamics.aero_2D import XFoil

            def get_run_data(Re):  # Get the data for an XFoil alpha sweep at one specific Re.
                run_data = XFoil(
                    airfoil=self,
                    Re=Re,
                    **xfoil_kwargs
                ).alpha(alphas)
                run_data["Re"] = Re * np.ones_like(run_data["alpha"])
                return run_data  # Data is a dict where keys are figures of merit [str] and values are 1D ndarrays.

            from tqdm import tqdm

            run_datas = [  # Get a list of dicts, where each dict is the result of an XFoil run at a particular Re.
                get_run_data(Re)
                for Re in tqdm(
                    Res,
                    desc=f"Running XFoil to generate polars for Airfoil '{self.name}':",
                )
            ]
            data = {  # Merge the dicts into one big database of all runs.
                k: np.concatenate(
                    tuple([run_data[k] for run_data in run_datas])
                )
                for k in run_datas[0].keys()
            }

            if make_symmetric_polars:  # If the airfoil is known to be symmetric, duplicate all data across alpha.
                keys_symmetric_across_alpha = ['CD', 'CDp', 'Re']

                data = {
                    k: np.concatenate([v, v if k in keys_symmetric_across_alpha else -v])
                    for k, v in data.items()
                }

            if cache_filename is not None:  # Cache the accumulated data for later use, if it doesn't already exist.
                with open(cache_filename, "w+") as f:
                    json.dump(
                        {k: v.tolist() for k, v in data.items()},
                        f,
                        indent=4
                    )

        ### Save the raw data as an instance attribute for later use
        self.xfoil_data = data

        ### Make the interpolators for attached aerodynamics
        from aerosandbox.modeling import UnstructuredInterpolatedModel

        alpha_resample = np.concatenate([
            np.array([-180, -150, -120, -90, -60, -30]),
            alphas[::2],
            np.array([30, 60, 90, 120, 150, 180])
        ])  # This is the list of points that we're going to resample from the XFoil runs for our InterpolatedModel, using an RBF.
        Re_resample = np.concatenate([
            np.array([1e0, 1e1, 1e2, 1e3]),
            Res,
            np.array([1e8, 1e9, 1e10, 1e11, 1e12])
        ])  # This is the list of points that we're going to resample from the XFoil runs for our InterpolatedModel, using an RBF.

        x_data = {
            "alpha": data["alpha"],
            "ln_Re": np.log(data["Re"]),
        }
        x_data_resample = {
            "alpha": alpha_resample,
            "ln_Re": np.log(Re_resample)
        }

        CL_attached_interpolator = UnstructuredInterpolatedModel(
            x_data=x_data,
            y_data=data["CL"],
            x_data_resample=x_data_resample,
            **unstructured_interpolated_model_kwargs
        )
        log10_CD_attached_interpolator = UnstructuredInterpolatedModel(
            x_data=x_data,
            y_data=np.log10(data["CD"]),
            x_data_resample=x_data_resample,
            **unstructured_interpolated_model_kwargs
        )
        CM_attached_interpolator = UnstructuredInterpolatedModel(
            x_data=x_data,
            y_data=data["CM"],
            x_data_resample=x_data_resample,
            **unstructured_interpolated_model_kwargs
        )

        ### Determine if separated
        alpha_stall_positive = np.max(data["alpha"])  # Across all Re
        alpha_stall_negative = np.min(data["alpha"])  # Across all Re

        def separation_parameter(alpha, Re=0):
            """
            Positive if separated, negative if attached.

            This will be an input to a tanh() sigmoid blend via asb.numpy.blend(), so a value of 1 means the flow is
            ~90% separated, and a value of -1 means the flow is ~90% attached.
            """
            return 0.5 * np.softmax(
                alpha - alpha_stall_positive,
                alpha_stall_negative - alpha
            )

        ### Make the interpolators for separated aerodynamics
        from aerosandbox.aerodynamics.aero_2D.airfoil_polar_functions import airfoil_coefficients_post_stall

        CL_if_separated, CD_if_separated, CM_if_separated = airfoil_coefficients_post_stall(
            airfoil=self,
            alpha=alpha_resample
        )

        CD_if_separated = CD_if_separated + np.median(data["CD"])
        # The line above effectively ensures that separated CD will never be less than attached CD. Not exactly, but generally close. A good heuristic.

        CL_separated_interpolator = UnstructuredInterpolatedModel(
            x_data=alpha_resample,
            y_data=CL_if_separated
        )
        log10_CD_separated_interpolator = UnstructuredInterpolatedModel(
            x_data=alpha_resample,
            y_data=np.log10(CD_if_separated)
        )
        CM_separated_interpolator = UnstructuredInterpolatedModel(
            x_data=alpha_resample,
            y_data=CM_if_separated
        )

        def CL_function(alpha, Re, mach=0, deflection=0):
            alpha = np.mod(alpha + 180, 360) - 180  # Keep alpha in the valid range.
            CL_attached = CL_attached_interpolator({
                "alpha": alpha,
                "ln_Re": np.log(Re),
            })
            CL_separated = CL_separated_interpolator(alpha)

            CL_mach_0 = np.blend(
                separation_parameter(alpha, Re),
                CL_separated,
                CL_attached
            )

            if include_compressibility_effects:
                prandtl_glauert_beta_squared_ideal = 1 - mach ** 2

                prandtl_glauert_beta = np.softmax(
                    prandtl_glauert_beta_squared_ideal,
                    -prandtl_glauert_beta_squared_ideal,
                    hardness=2.0  # Empirically tuned to data
                ) ** 0.5

                CL = CL_mach_0 / prandtl_glauert_beta

                mach_crit = transonic.mach_crit_Korn(
                    CL=CL,
                    t_over_c=self.max_thickness(),
                    sweep=0,
                    kappa_A=0.95
                )

                ### Accounts approximately for the lift drop due to buffet.
                buffet_factor = np.blend(
                    40 * (mach - mach_crit - (0.1 / 80) ** (1 / 3) - 0.06) * (mach - 1.1),
                    1,
                    transonic_buffet_lift_knockdown
                )

                ### Accounts for the fact that theoretical CL_alpha goes from 2 * pi (subsonic) to 4 (supersonic),
                # following linearized supersonic flow on a thin airfoil.
                cla_supersonic_ratio_factor = np.blend(
                    10 * (mach - 1),
                    4 / (2 * np.pi),
                    1,
                )

                return CL * buffet_factor * cla_supersonic_ratio_factor

            else:
                return CL_mach_0

        def CD_function(alpha, Re, mach=0, deflection=0):
            alpha = np.mod(alpha + 180, 360) - 180  # Keep alpha in the valid range.
            log10_CD_attached = log10_CD_attached_interpolator({
                "alpha": alpha,
                "ln_Re": np.log(Re),
            })
            log10_CD_separated = log10_CD_separated_interpolator(alpha)

            log10_CD_mach_0 = np.blend(
                separation_parameter(alpha, Re),
                log10_CD_separated,
                log10_CD_attached,
            )

            if include_compressibility_effects:

                CL_attached = CL_attached_interpolator({
                    "alpha": alpha,
                    "ln_Re": np.log(Re),
                })
                CL_separated = CL_separated_interpolator(alpha)

                CL_mach_0 = np.blend(
                    separation_parameter(alpha, Re),
                    CL_separated,
                    CL_attached
                )
                prandtl_glauert_beta_squared_ideal = 1 - mach ** 2

                prandtl_glauert_beta = np.softmax(
                    prandtl_glauert_beta_squared_ideal,
                    -prandtl_glauert_beta_squared_ideal,
                    hardness=2.0  # Empirically tuned to data
                ) ** 0.5

                CL = CL_mach_0 / prandtl_glauert_beta

                t_over_c = self.max_thickness()

                mach_crit = transonic.mach_crit_Korn(
                    # CL=CL_function(alpha, Re, mach, deflection),
                    CL=CL,
                    t_over_c=t_over_c,
                    sweep=0,
                    kappa_A=0.92
                )
                mach_dd = mach_crit + (0.1 / 80) ** (1 / 3)
                CD_wave = np.where(
                    mach < mach_crit,
                    0,
                    np.where(
                        mach < mach_dd,
                        20 * (mach - mach_crit) ** 4,
                        np.where(
                            mach < 0.97,
                            cubic_hermite_patch(
                                mach,
                                x_a=mach_dd,
                                x_b=0.97,
                                f_a=20 * (0.1 / 80) ** (4 / 3),
                                f_b=0.8 * t_over_c,
                                dfdx_a=0.1,
                                dfdx_b=0.8 * t_over_c * 8
                            ),
                            np.where(
                                mach < 1.1,
                                cubic_hermite_patch(
                                    mach,
                                    x_a=0.97,
                                    x_b=1.1,
                                    f_a=0.8 * t_over_c,
                                    f_b=0.8 * t_over_c,
                                    dfdx_a=0.8 * t_over_c * 8,
                                    dfdx_b=-0.8 * t_over_c * 8,
                                ),
                                np.blend(
                                    8 * 2 * (mach - 1.1) / (1.2 - 0.8),
                                    0.8 * 0.8 * t_over_c,
                                    1.2 * 0.8 * t_over_c,
                                )
                            )
                        )
                    )
                )

                # CD_wave = transonic.approximate_CD_wave(
                #     mach=mach,
                #     mach_crit=mach_crit,
                #     CD_wave_at_fully_supersonic=0.90 * self.max_thickness()
                # )

                return 10 ** log10_CD_mach_0 + CD_wave


            else:
                return 10 ** log10_CD_mach_0

        def CM_function(alpha, Re, mach=0, deflection=0):
            alpha = np.mod(alpha + 180, 360) - 180  # Keep alpha in the valid range.
            CM_attached = CM_attached_interpolator({
                "alpha": alpha,
                "ln_Re": np.log(Re),
            })
            CM_separated = CM_separated_interpolator(alpha)

            CM_mach_0 = np.blend(
                separation_parameter(alpha, Re),
                CM_separated,
                CM_attached
            )
            if include_compressibility_effects:
                prandtl_glauert_beta_squared_ideal = 1 - mach ** 2

                prandtl_glauert_beta = np.softmax(
                    prandtl_glauert_beta_squared_ideal,
                    -prandtl_glauert_beta_squared_ideal,
                    hardness=2.0  # Empirically tuned to data
                ) ** 0.5

                CM = CM_mach_0 / prandtl_glauert_beta

                return CM
            else:
                return CM_mach_0

        self.CL_function = CL_function
        self.CD_function = CD_function
        self.CM_function = CM_function

    def local_camber(self,
                     x_over_c: Union[float, np.ndarray] = np.linspace(0, 1, 101)
                     ) -> Union[float, np.ndarray]:
        """
        Returns the local camber of the airfoil at a given point or points.
        :param x_over_c: The x/c locations to calculate the camber at [1D array, more generally, an iterable of floats]
        :return: Local camber of the airfoil (y/c) [1D array].
        """
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

    def local_thickness(self,
                        x_over_c: Union[float, np.ndarray] = np.linspace(0, 1, 101)
                        ) -> Union[float, np.ndarray]:
        """
        Returns the local thickness of the airfoil at a given point or points.
        :param x_over_c: The x/c locations to calculate the thickness at [1D array, more generally, an iterable of floats]
        :return: Local thickness of the airfoil (y/c) [1D array].
        """
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

    def max_camber(self,
                   x_over_c_sample: np.ndarray = np.linspace(0, 1, 101)
                   ) -> float:
        """
        Returns the maximum camber of the airfoil.

        Args:
            x_over_c_sample: Where should the airfoil be sampled to determine the max camber?

        Returns: The maximum thickness, as a fraction of chord.

        """
        return np.max(self.local_camber(x_over_c=x_over_c_sample))

    def max_thickness(self,
                      x_over_c_sample: np.ndarray = np.linspace(0, 1, 101)
                      ) -> float:
        """
        Returns the maximum thickness of the airfoil.

        Args:
            x_over_c_sample: Where should the airfoil be sampled to determine the max thickness?

        Returns: The maximum thickness, as a fraction of chord.

        """
        return np.max(self.local_thickness(x_over_c=x_over_c_sample))

    def draw(self,
             draw_mcl=True,
             backend="matplotlib",
             show=True
             ):
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

        if backend == "matplotlib":
            color = '#280887'
            plt.plot(x, y, ".-", zorder=11, color=color)
            plt.fill(x, y, zorder=10, color=color, alpha=0.2)
            if draw_mcl:
                plt.plot(x_mcl, y_mcl, "-", zorder=4, color=color, alpha=0.4)
            plt.axis("equal")
            plt.xlabel(r"$x/c$")
            plt.ylabel(r"$y/c$")
            plt.title(f"{self.name} Airfoil")
            plt.tight_layout()
            if show:
                plt.show()

        elif backend == "plotly":
            from aerosandbox.visualization.plotly import go
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
                title=f"{self.name} Airfoil"
            )
            if show:
                fig.show()
            else:
                return fig

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
        Returns the trailing edge angle of the airfoil, in degrees.
        """
        upper_TE_vec = self.coordinates[0, :] - self.coordinates[1, :]
        lower_TE_vec = self.coordinates[-1, :] - self.coordinates[-2, :]

        return 180 / np.pi * (np.arctan2(
            upper_TE_vec[0] * lower_TE_vec[1] - upper_TE_vec[1] * lower_TE_vec[0],
            upper_TE_vec[0] * lower_TE_vec[0] + upper_TE_vec[1] * upper_TE_vec[1]
        ))

    # def LE_radius(self) -> float:
    #     """
    #     Gives the approximate leading edge radius of the airfoil, in chord-normalized units.
    #     """ # TODO finish me

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

        x = interpolate.PchipInterpolator(
            distances_from_TE_normalized,
            self.x(),
        )(s)
        y = interpolate.PchipInterpolator(
            distances_from_TE_normalized,
            self.y(),
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

    def blend_with_another_airfoil(self,
                                   airfoil: "Airfoil",
                                   blend_fraction: float = 0.5,
                                   n_points_per_side: int = 100,
                                   ) -> "Airfoil":
        """
        Blends this airfoil with another airfoil. Merges both the coordinates and the aerodynamic functions.

        Args:

            airfoil: The other airfoil to blend with.

            blend_fraction: The fraction of the other airfoil to use when blending. Defaults to 0.5 (50%).

                * A blend fraction of 0 will return an identical airfoil to this one (self).

                * A blend fraction of 1 will return an identical airfoil to the other one (`airfoil` parameter).

            n_points_per_side: The number of points per side to use when blending the coordinates of the two airfoils.

        Returns: A new airfoil that is a blend of this airfoil and another one.

        """
        this_foil = self.repanel(n_points_per_side=n_points_per_side)
        that_foil = airfoil.repanel(n_points_per_side=n_points_per_side)
        this_fraction = 1 - blend_fraction
        that_fraction = blend_fraction

        name = f"{this_fraction * 100:.0f}% {self.name}, {that_fraction * 100:.0f}% {airfoil.name}"

        coordinates = (
                this_fraction * this_foil.coordinates +
                that_fraction * that_foil.coordinates
        )

        def CL_function(alpha, Re, mach, deflection):
            return (
                    this_fraction * this_foil.CL_function(alpha, Re, mach, deflection) +
                    that_fraction * that_foil.CL_function(alpha, Re, mach, deflection)
            )

        def CD_function(alpha, Re, mach, deflection):
            return (
                    this_fraction * this_foil.CD_function(alpha, Re, mach, deflection) +
                    that_fraction * that_foil.CD_function(alpha, Re, mach, deflection)
            )

        def CM_function(alpha, Re, mach, deflection):
            return (
                    this_fraction * this_foil.CM_function(alpha, Re, mach, deflection) +
                    that_fraction * that_foil.CM_function(alpha, Re, mach, deflection)
            )

        return Airfoil(
            name=name,
            coordinates=coordinates,
            CL_function=CL_function,
            CD_function=CD_function,
            CM_function=CM_function,
        )

    # def normalize(self):
    #     pass  # TODO finish me

    def write_dat(self,
                  filepath: Union[str, Path] = None,
                  include_name: bool = True,
                  ) -> str:
        """
        Writes a .dat file corresponding to this airfoil to a filepath.

        Args:
            filepath: filepath (including the filename and .dat extension) [string]
                If None, this function returns the .dat file as a string.

            include_name: Should the name be included in the .dat file? (In a standard *.dat file, it usually is.)

        Returns: None

        """
        contents = []

        if include_name:
            contents += [self.name]

        contents += ["%f %f" % tuple(coordinate) for coordinate in self.coordinates]

        string = "\n".join(contents)

        if filepath is not None:
            with open(filepath, "w+") as f:
                f.write(string)

        return string

    def write_sldcrv(self,
                     filepath: str = None
                     ):
        """
        Writes a .sldcrv (SolidWorks curve) file corresponding to this airfoil to a filepath.

        Args:
            filepath: A filepath (including the filename and .sldcrv extension) [string]
                if None, this function returns the .sldcrv file as a string.

        Returns: None

        """
        string = "\n".join(
            [
                "%f %f 0" % tuple(coordinate)
                for coordinate in self.coordinates
            ]
        )

        if filepath is not None:
            with open(filepath, "w+") as f:
                f.write(string)

        return string

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
    #         vals=np.nan
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
    # def plot_xfoil_data_contours(self):
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
