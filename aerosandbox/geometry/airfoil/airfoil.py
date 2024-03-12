import aerosandbox.numpy as np
from aerosandbox.geometry.polygon import Polygon
from aerosandbox.geometry.airfoil.airfoil_families import (
    get_NACA_coordinates,
    get_UIUC_coordinates,
    get_file_coordinates
)
from aerosandbox.library.aerodynamics import transonic
from aerosandbox.modeling.splines.hermite import linear_hermite_patch, cubic_hermite_patch
from scipy import interpolate
from typing import Callable, Union, Any, Dict, List
import json
from pathlib import Path
import os


class Airfoil(Polygon):
    """
    An airfoil. See constructor docstring for usage details.
    """

    def __init__(self,
                 name: str = "Untitled",
                 coordinates: Union[None, str, Path, np.ndarray] = None,
                 **deprecated_keyword_arguments
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

        """
        ### Handle the airfoil name
        self.name = name

        ### Handle the coordinates
        self.coordinates = None
        if coordinates is None:  # If no coordinates are given
            try:  # See if it's a NACA airfoil
                self.coordinates = get_NACA_coordinates(name=self.name)
            except (ValueError, NotImplementedError):
                try:  # See if it's in the UIUC airfoil database
                    self.coordinates = get_UIUC_coordinates(name=self.name)
                except FileNotFoundError:
                    pass
                except UnicodeDecodeError:
                    import warnings
                    warnings.warn(
                        f"Airfoil {self.name} was found in the UIUC airfoil database, but could not be parsed.\n"
                        f"Check for any non-Unicode-compatible characters in the file, or specify the airfoil "
                        f"coordinates yourself.",
                    )
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
                UserWarning,
                stacklevel=2,
            )

        ### Handle deprecated keyword arguments
        if len(deprecated_keyword_arguments) > 0:
            import warnings
            warnings.warn(
                "The `generate_polars`, `CL_function`, `CD_function`, and `CM_function` keyword arguments to the "
                "Airfoil constructor will be deprecated in an upcoming release. Their functionality is replaced"
                "by `Airfoil.get_aero_from_neuralfoil()`, which is faster and has better properties for optimization.",
                DeprecationWarning
            )

            generate_polars = deprecated_keyword_arguments.get("generate_polars", False)
            CL_function = deprecated_keyword_arguments.get("CL_function", None)
            CD_function = deprecated_keyword_arguments.get("CD_function", None)
            CM_function = deprecated_keyword_arguments.get("CM_function", None)

            ### Handle getting default polars
            if generate_polars:
                self.generate_polars()
            else:

                from aerosandbox.library.aerodynamics.viscous import Cf_flat_plate

                def print_default_warning():
                    warnings.warn("\n".join([
                        "Warning: Using a placeholder aerodynamics model for this Airfoil!",
                        "It's highly recommended that you either:",
                        "\ta) Specify polar functions in the Airfoil constructor, or",
                        "\tb) Call Airfoil.generate_polars() to auto-generate these polar functions with XFoil."
                    ]), stacklevel=3)

                def default_CL_function(alpha, Re, mach=0, deflection=0):
                    """
                    Lift coefficient.
                    """
                    print_default_warning()
                    Cl_inc = np.pi * np.sind(2 * alpha)
                    beta = (1 - mach) ** 2

                    Cl = Cl_inc * beta
                    return Cl

                def default_CD_function(alpha, Re, mach=0, deflection=0):
                    """
                    Drag coefficient.
                    """
                    print_default_warning()
                    Cf = Cf_flat_plate(Re_L=Re, method="hybrid-sharpe-convex")

                    ### Form factor model from Raymer, "Aircraft Design". Section 12.5, Eq. 12.30
                    t_over_c = 0.12
                    FF = 1 + 2 * t_over_c * 100 * t_over_c ** 4

                    Cd_inc = 2 * Cf * FF * (
                            1 + (np.sind(alpha) * 180 / np.pi / 5) ** 2
                    )
                    beta = (1 - mach) ** 2

                    Cd = Cd_inc * beta
                    return Cd

                def default_CM_function(alpha, Re, mach=0, deflection=0):
                    """
                    Pitching moment coefficient, as measured about quarter-chord.
                    """
                    print_default_warning()
                    return np.zeros_like(alpha)

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

    def __repr__(self) -> str:
        return f"Airfoil {self.name} ({self.n_points()} points)"

    def __eq__(self, other: "Airfoil") -> bool:
        """
        Checks if two airfoils are equal. Two airfoils are equal if they have the same name, coordinates, and
        polar functions.

        Args:
            other: The other airfoil to compare to.

        Returns:
            True if the two airfoils are equal, False otherwise.
        """
        if other is self:  # If they're the same object in memory, they're equal
            return True

        if not type(self) == type(other):  # If the types are different, they're not equal
            return False

        # At this point, we know that the types are the same, so we can compare the attributes
        if self.name != other.name:  # If the names are different, they're not equal
            return False

        if self.coordinates.shape != other.coordinates.shape:  # If the coordinates are different shapes, they're not equal
            return False

        try:
            return np.allclose(
                self.coordinates,
                other.coordinates
            )
        except Exception:
            return False

    def to_kulfan_airfoil(self,
                          n_weights_per_side: int = 8,
                          N1: float = 0.5,
                          N2: float = 1.0,
                          normalize_coordinates: bool = True,
                          use_leading_edge_modification: bool = True,
                          ) -> "KulfanAirfoil":

        from aerosandbox.geometry.airfoil.kulfan_airfoil import KulfanAirfoil
        from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters

        parameters = get_kulfan_parameters(
            coordinates=self.coordinates,
            n_weights_per_side=n_weights_per_side,
            N1=N1,
            N2=N2,
            normalize_coordinates=normalize_coordinates,
            use_leading_edge_modification=use_leading_edge_modification,
        )

        return KulfanAirfoil(
            name=self.name,
            lower_weights=parameters["lower_weights"],
            upper_weights=parameters["upper_weights"],
            leading_edge_weight=parameters["leading_edge_weight"],
            TE_thickness=parameters["TE_thickness"],
            N1=N1,
            N2=N2,
        )

    def generate_polars(self,
                        alphas=np.linspace(-13, 13, 27),
                        Res=np.geomspace(1e3, 1e8, 12),
                        cache_filename: str = None,
                        xfoil_kwargs: Dict[str, Any] = None,
                        unstructured_interpolated_model_kwargs: Dict[str, Any] = None,
                        include_compressibility_effects: bool = True,
                        transonic_buffet_lift_knockdown: float = 0.3,
                        make_symmetric_polars: bool = False,
                        ) -> None:
        """
        Generates airfoil polar surrogate models (CL, CD, CM functions) from XFoil data and assigns them in-place to
        this Airfoil's polar functions.

        In other words, when this function is run, the following functions will be added (or overwritten) to the instance:
            * Airfoil.CL_function(alpha, Re, mach)
            * Airfoil.CD_function(alpha, Re, mach)
            * Airfoil.CM_function(alpha, Re, mach)

        Where alpha is in degrees.

        Warning: In-place operation! Modifies this Airfoil object by setting Airfoil.CL_function, etc. to the new
        polars.

        Args:

            alphas: The range of alphas to sample from XFoil at. Given in degrees.

            Res: The range of Reynolds numbers to sample from XFoil at. Dimensionless.

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

        Returns: None (in-place), adds the following functions to the instance:

            * Airfoil.CL_function(alpha, Re, mach)
            * Airfoil.CD_function(alpha, Re, mach)
            * Airfoil.CM_function(alpha, Re, mach)

        """
        if self.coordinates is None:
            raise ValueError("Cannot generate polars for an airfoil that you don't have the coordinates of!")

        ### Set defaults
        if xfoil_kwargs is None:
            xfoil_kwargs = {}
        if unstructured_interpolated_model_kwargs is None:
            unstructured_interpolated_model_kwargs = {}

        xfoil_kwargs = {  # See asb.XFoil for the documentation on these.
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

        ### Retrieve XFoil Polar Data from the cache, if it exists.
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
            ### If a cache filename is given, ensure that the directory exists.
            if cache_filename is not None:
                os.makedirs(os.path.dirname(cache_filename), exist_ok=True)

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
                keys_symmetric_across_alpha = ['CD', 'CDp', 'Re']  # Assumes the rest are antisymmetric

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

        attached_alphas_to_use = (
            alphas[::2] if len(alphas) > 20 else alphas
        )

        alpha_resample = np.concatenate([
            np.linspace(-180, attached_alphas_to_use.min(), 10)[:-1],
            attached_alphas_to_use,
            np.linspace(attached_alphas_to_use.max(), 180, 10)[1:],
        ])  # This is the list of points that we're going to resample from the XFoil runs for our InterpolatedModel, using an RBF.
        Re_resample = np.concatenate([
            Res.min() / 10 ** np.arange(1, 5)[::-1],
            Res,
            Res.max() * 10 ** np.arange(1, 5),
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

        def CL_function(alpha, Re, mach=0):

            alpha = np.mod(alpha + 180, 360) - 180  # Keep alpha in the valid range.
            CL_attached = CL_attached_interpolator({
                "alpha": alpha,
                "ln_Re": np.log(Re),
            })
            CL_separated = CL_separated_interpolator(alpha)  # Lift coefficient if separated

            CL_mach_0 = np.blend(  # Lift coefficient at mach = 0
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

        def CD_function(alpha, Re, mach=0):

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

        def CM_function(alpha, Re, mach=0):

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

    def get_aero_from_neuralfoil(self,
                                 alpha: Union[float, np.ndarray],
                                 Re: Union[float, np.ndarray],
                                 mach: Union[float, np.ndarray] = 0.,
                                 n_crit: Union[float, np.ndarray] = 9.0,
                                 xtr_upper: Union[float, np.ndarray] = 1.0,
                                 xtr_lower: Union[float, np.ndarray] = 1.0,
                                 model_size: str = "large",
                                 control_surfaces: List["ControlSurface"] = None,
                                 include_360_deg_effects: bool = True,
                                 ) -> Dict[str, Union[float, np.ndarray]]:

        airfoil_normalization = self.normalize(return_dict=True)

        kulfan_airfoil = airfoil_normalization["airfoil"].to_kulfan_airfoil(
            n_weights_per_side=8,
            N1=0.5,
            N2=1.0,
        )

        return kulfan_airfoil.get_aero_from_neuralfoil(
            alpha=alpha + airfoil_normalization["rotation_angle"],
            Re=Re / airfoil_normalization["scale_factor"],
            mach=mach,
            n_crit=n_crit,
            xtr_upper=xtr_upper,
            xtr_lower=xtr_lower,
            model_size=model_size,
            control_surfaces=control_surfaces,
            include_360_deg_effects=include_360_deg_effects
        )

    def plot_polars(self,
                    alphas: Union[np.ndarray, List[float]] = np.linspace(-20, 20, 500),
                    Res: Union[np.ndarray, List[float]] = 10 ** np.arange(3, 9),
                    mach: float = 0.,
                    show: bool = True,
                    Re_colors=None,
                    ) -> None:
        import matplotlib.pyplot as plt
        import aerosandbox.tools.pretty_plots as p

        fig, ax = plt.subplots(2, 2, figsize=(8, 7))
        plt.sca(ax[0, 0])
        plt.title("Lift Coefficient")
        plt.xlabel(r"Angle of Attack $\alpha$ [deg]")
        plt.ylabel(r"Lift Coefficient $C_L$")
        p.set_ticks(5, 1, 0.5, 0.1)
        plt.sca(ax[0, 1])
        plt.title("Drag Coefficient")
        plt.xlabel(r"Angle of Attack $\alpha$ [deg]")
        plt.ylabel(r"Drag Coefficient $C_D$")
        plt.ylim(bottom=0, top=0.05)
        p.set_ticks(5, 1, 0.01, 0.002)
        plt.sca(ax[1, 0])
        plt.title("Moment Coefficient")
        plt.xlabel(r"Angle of Attack $\alpha$ [deg]")
        plt.ylabel(r"Moment Coefficient $C_m$")
        p.set_ticks(5, 1, 0.05, 0.01)
        plt.sca(ax[1, 1])
        plt.title("Lift-to-Drag Ratio")
        plt.xlabel(r"Angle of Attack $\alpha$ [deg]")
        plt.ylabel(r"Lift-to-Drag Ratio $C_L/C_D$")
        p.set_ticks(5, 1, 20, 5)

        if Re_colors is None:
            Re_colors = p.mpl.colormaps.get_cmap('rainbow')(np.linspace(0, 1, len(Res)))
            Re_colors = [
                p.adjust_lightness(color, 0.7)
                for color in Re_colors
            ]

        for i, Re in enumerate(Res):
            kwargs = dict(
                alpha=alphas,
                Re=Re,
                mach=mach
            )

            plt.sca(ax[0, 0])
            plt.plot(
                alphas,
                self.CL_function(**kwargs),
                color=Re_colors[i],
                alpha=0.7
            )

            plt.sca(ax[0, 1])
            plt.plot(
                alphas,
                self.CD_function(**kwargs),
                color=Re_colors[i],
                alpha=0.7
            )

            plt.sca(ax[1, 0])
            plt.plot(
                alphas,
                self.CM_function(**kwargs),
                color=Re_colors[i],
                alpha=0.7
            )

            plt.sca(ax[1, 1])
            plt.plot(
                alphas,
                self.CL_function(**kwargs) / self.CD_function(**kwargs),
                color=Re_colors[i],
                alpha=0.7
            )

        from aerosandbox.tools.string_formatting import eng_string

        plt.sca(ax[0, 0])
        plt.legend(
            title="Reynolds Number",
            labels=[eng_string(Re) for Re in Res],
            ncol=2,
            # Note: `ncol` is old syntax; preserves backwards-compatibility with matplotlib 3.5.x.
            # New matplotlib versions use `ncols` instead.
            fontsize=8,
            loc='lower right'
        )

        if show:
            p.show_plot(
                f"Polar Functions for {self.name} Airfoil",
                legend=False,
            )

    def local_camber(self,
                     x_over_c: Union[float, np.ndarray] = np.linspace(0, 1, 101)
                     ) -> Union[float, np.ndarray]:
        """
        Returns the local camber of the airfoil at a given point or points.

        Args:
            x_over_c: The x/c locations to calculate the camber at [1D array, more generally, an iterable of floats]

        Returns:
            Local camber of the airfoil (y/c) [1D array].
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

        Args:
            x_over_c: The x/c locations to calculate the thickness at [1D array, more generally, an iterable of floats]

        Returns:
            Local thickness of the airfoil (y/c) [1D array].
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
             draw_mcl=False,
             draw_markers=True,
             backend="matplotlib",
             show=True
             ) -> None:
        """
        Draw the airfoil object.

        Args:
            draw_mcl: Should we draw the mean camber line (MCL)? [boolean]

            backend: Which backend should we use? "plotly" or "matplotlib"

            show: Should we show the plot? [boolean]

        Returns: None
        """
        x = np.reshape(np.array(self.x()), -1)
        y = np.reshape(np.array(self.y()), -1)
        if draw_mcl:
            x_mcl = np.linspace(np.min(x), np.max(x), len(x))
            y_mcl = self.local_camber(x_mcl)

        if backend == "matplotlib":
            import matplotlib.pyplot as plt
            import aerosandbox.tools.pretty_plots as p

            color = '#280887'
            plt.plot(
                x, y,
                ".-" if draw_markers else "-",
                zorder=11, color=color)
            plt.fill(x, y, zorder=10, color=color, alpha=0.2)
            if draw_mcl:
                plt.plot(x_mcl, y_mcl, "-", zorder=4, color=color, alpha=0.4)
            plt.axis("equal")
            if show:
                p.show_plot(
                    title=f"{self.name} Airfoil",
                    xlabel=r"$x/c$",
                    ylabel=r"$y/c$",

                )

        elif backend == "plotly":
            from aerosandbox.visualization.plotly import go
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers" if draw_markers else "lines",
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
                        mode="lines",
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
        Returns the index of the leading edge point in the airfoil coordinates.
        """
        return int(np.argmin(self.x()))

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
        x_gap = self.coordinates[0, 0] - self.coordinates[-1, 0]
        y_gap = self.coordinates[0, 1] - self.coordinates[-1, 1]

        return (
                x_gap ** 2 +
                y_gap ** 2
        ) ** 0.5

    def TE_angle(self) -> float:
        """
        Returns the trailing edge angle of the airfoil, in degrees.
        """
        upper_TE_vec = self.coordinates[0, :] - self.coordinates[1, :]
        lower_TE_vec = self.coordinates[-1, :] - self.coordinates[-2, :]

        return np.arctan2d(
            upper_TE_vec[0] * lower_TE_vec[1] - upper_TE_vec[1] * lower_TE_vec[0],
            upper_TE_vec[0] * lower_TE_vec[0] + upper_TE_vec[1] * upper_TE_vec[1]
        )

    # def LE_radius(self) -> float:
    #     """
    #     Gives the approximate leading edge radius of the airfoil, in chord-normalized units.
    #     """ # TODO finish me

    def repanel(self,
                n_points_per_side: int = 100,
                spacing_function_per_side=np.cosspace,
                ) -> 'Airfoil':
        """
        Returns a repaneled copy of the airfoil with cosine-spaced coordinates on the upper and lower surfaces.

        Args:

            n_points_per_side: Number of points per side (upper and lower) of the airfoil [int]

                Notes: The number of points defining the final airfoil will be `n_points_per_side * 2 - 1`,
                since one point (the leading edge point) is shared by both the upper and lower surfaces.

            spacing_function_per_side: Determines how to space the points on each side of the airfoil. Can be
                `np.linspace` or `np.cosspace`, or any other function of the call signature `f(a, b, n)` that returns
                a spaced array of `n` points between `a` and `b`. [function]

        Returns: A copy of the airfoil with the new coordinates.
        """

        old_upper_coordinates = self.upper_coordinates()  # Note: includes leading edge point, be careful about duplicates
        old_lower_coordinates = self.lower_coordinates()  # Note: includes leading edge point, be careful about duplicates

        # Find the streamwise distances between coordinates, assuming linear interpolation
        upper_distances_between_points = np.linalg.norm(np.diff(old_upper_coordinates, axis=0), axis=1)
        lower_distances_between_points = np.linalg.norm(np.diff(old_lower_coordinates, axis=0), axis=1)
        upper_distances_from_TE = np.concatenate(([0], np.cumsum(upper_distances_between_points)))
        lower_distances_from_LE = np.concatenate(([0], np.cumsum(lower_distances_between_points)))

        try:
            new_upper_coordinates = interpolate.CubicSpline(
                x=upper_distances_from_TE,
                y=old_upper_coordinates,
                axis=0,
                bc_type=(
                    (2, (0, 0)),
                    (1, (0, -1)),
                )
            )(spacing_function_per_side(0, upper_distances_from_TE[-1], n_points_per_side))

            new_lower_coordinates = interpolate.CubicSpline(
                x=lower_distances_from_LE,
                y=old_lower_coordinates,
                axis=0,
                bc_type=(
                    (1, (0, -1)),
                    (2, (0, 0)),
                )
            )(spacing_function_per_side(0, lower_distances_from_LE[-1], n_points_per_side))

        except ValueError as e:
            if not (
                    (np.all(np.diff(upper_distances_from_TE)) > 0) and
                    (np.all(np.diff(lower_distances_from_LE)) > 0)
            ):
                raise ValueError(
                    "It looks like your Airfoil has a duplicate point. Try removing the duplicate point and "
                    "re-running Airfoil.repanel()."
                )
            else:
                raise e

        return Airfoil(
            name=self.name,
            coordinates=np.concatenate((new_upper_coordinates, new_lower_coordinates[1:, :]), axis=0),
        )

    def normalize(
            self,
            return_dict: bool = False,
    ) -> Union['Airfoil', Dict[str, Union['Airfoil', float]]]:
        """
        Returns a copy of the Airfoil with a new set of `coordinates`, such that:
            - The leading edge (LE) is at (0, 0)
            - The trailing edge (TE) is at (1, 0)
            - The chord length is equal to 1

        The trailing-edge (TE) point is defined as the midpoint of the line segment connecting the first and last coordinate points (upper and lower surface TE points, respectively). The TE point is not necessarily one of the original points in the airfoil coordinates (`Airfoil.coordinates`); in general, it will not be one of the points if the TE thickness is nonzero.

        The leading-edge (LE) point is defined as the coordinate point with the largest Euclidian distance from the trailing edge. (In other words, if you were to center a circle on the trailing edge and progressively grow it, what's the last coordinate point that it would intersect?) The LE point is always one of the original points in the airfoil coordinates.

        The chord is defined as the Euclidian distance between the LE and TE points.

        Coordinate modifications to achieve the constraints described above (LE @ origin, TE at (1, 0), and chord of 1) are done by means of a translation and rotation.

        Args:

            return_dict: Determines the output type of the function.
                - If `False` (default), returns a copy of the Airfoil with the new coordinates.
                - If `True`, returns a dictionary with keys:
                
                        - "airfoil": a copy of the Airfoil with the new coordinates

                        - "x_translation": the amount by which the airfoil's LE was translated in the x-direction

                        - "y_translation": the amount by which the airfoil's LE was translated in the y-direction

                        - "scale_factor": the amount by which the airfoil was scaled (if >1, the airfoil had to get
                            bigger)

                        - "rotation_angle": the angle (in degrees) by which the airfoil was rotated about the LE.
                            Sign convention is that positive angles rotate the airfoil counter-clockwise.

                    All of thes values represent the "required change", e.g.:

                        - "x_translation" is the amount by which the airfoil's LE had to be translated in the
                            x-direction to get it to the origin.

                        - "rotation_angle" is the angle (in degrees) by which the airfoil had to be rotated (CCW).

        Returns: Depending on the value of `return_dict`, either:

            - A copy of the airfoil with the new coordinates (default), or

            - A dictionary with keys "airfoil", "x_translation", "y_translation", "scale_factor", and "rotation_angle".
                documentation for `return_tuple` for more information.
        """

        ### Step 1: Translate so that the LE point is at (0, 0).
        x_te = (self.x()[0] + self.x()[-1]) / 2
        y_te = (self.y()[0] + self.y()[-1]) / 2

        distance_to_te = (
                                 (self.x() - x_te) ** 2 +
                                 (self.y() - y_te) ** 2
                         ) ** 0.5

        le_index = np.argmax(distance_to_te)

        x_translation = -self.x()[le_index]
        y_translation = -self.y()[le_index]

        newfoil = self.translate(
            translate_x=x_translation,
            translate_y=y_translation,
        )

        ### Step 2: Scale so that the chord length is 1.
        scale_factor = 1 / distance_to_te[le_index]

        newfoil = newfoil.scale(
            scale_x=scale_factor,
            scale_y=scale_factor,
        )

        ### Step 3: Rotate so that the trailing edge is at (1, 0).

        x_te = (newfoil.x()[0] + newfoil.x()[-1]) / 2
        y_te = (newfoil.y()[0] + newfoil.y()[-1]) / 2

        rotation_angle = -np.arctan2(y_te, x_te)

        newfoil = newfoil.rotate(
            angle=rotation_angle,
        )

        if not return_dict:
            return newfoil
        else:
            return {
                "airfoil"       : newfoil,
                "x_translation" : x_translation,
                "y_translation" : y_translation,
                "scale_factor"  : scale_factor,
                "rotation_angle": np.degrees(rotation_angle),
            }

    def add_control_surface(
            self,
            deflection: float = 0.,
            hinge_point_x: float = 0.75,
            modify_coordinates: bool = True,
            modify_polars: bool = True,
    ) -> 'Airfoil':
        """
        Returns a version of the airfoil with a trailing-edge control surface added at a given point. Implicitly
        repanels the airfoil as part of this operation.

        Args:
            deflection: Deflection angle [degrees]. Downwards-positive.
            hinge_point_x: Chordwise location of the hinge, as a fraction of chord (x/c) [float]

        Returns: an Airfoil object with the new control deflection.

        """
        if modify_coordinates:
            # Find the hinge point
            hinge_point_y = np.where(
                deflection > 0,
                self.local_camber(hinge_point_x) - self.local_thickness(hinge_point_x) / 2,
                self.local_camber(hinge_point_x) + self.local_thickness(hinge_point_x) / 2,
            )

            # hinge_point_y = self.local_camber(hinge_point_x)
            hinge_point = np.reshape(
                np.array([hinge_point_x, hinge_point_y]),
                (1, 2)
            )

            def is_behind_hinge(xy: np.ndarray) -> np.ndarray:
                return (
                        (xy[:, 0] - hinge_point_x) * np.cosd(deflection / 2) -
                        (xy[:, 1] - hinge_point_y) * np.sind(deflection / 2)
                        > 0
                )

            orig_u = self.upper_coordinates()
            orig_l = self.lower_coordinates()[1:, :]

            rotation_matrix = np.rotation_matrix_2D(
                angle=-np.radians(deflection),
            )

            def T(xy):
                return np.transpose(xy)

            hinge_point_u = np.tile(hinge_point, (np.length(orig_u), 1))
            hinge_point_l = np.tile(hinge_point, (np.length(orig_l), 1))

            rot_u = T(rotation_matrix @ T(orig_u - hinge_point_u)) + hinge_point_u
            rot_l = T(rotation_matrix @ T(orig_l - hinge_point_l)) + hinge_point_l

            coordinates_x = np.concatenate([
                np.where(
                    is_behind_hinge(rot_u),
                    rot_u[:, 0],
                    orig_u[:, 0]
                ),
                np.where(
                    is_behind_hinge(rot_l),
                    rot_l[:, 0],
                    orig_l[:, 0]
                )
            ])
            coordinates_y = np.concatenate([
                np.where(
                    is_behind_hinge(rot_u),
                    rot_u[:, 1],
                    orig_u[:, 1]
                ),
                np.where(
                    is_behind_hinge(rot_l),
                    rot_l[:, 1],
                    orig_l[:, 1]
                )
            ])

            coordinates = np.stack([
                coordinates_x,
                coordinates_y
            ], axis=1)
        else:
            coordinates = self.coordinates

        if modify_polars:
            effectiveness = 1 - np.maximum(0, hinge_point_x + 1e-16) ** 2.751428551177291
            dalpha = deflection * effectiveness

            def CL_function(alpha: float, Re: float, mach: float) -> float:
                return self.CL_function(
                    alpha=alpha + dalpha,
                    Re=Re,
                    mach=mach,
                )

            def CD_function(alpha: float, Re: float, mach: float) -> float:
                return self.CD_function(
                    alpha=alpha + dalpha,
                    Re=Re,
                    mach=mach,
                )

            def CM_function(alpha: float, Re: float, mach: float) -> float:
                return self.CM_function(
                    alpha=alpha + dalpha,
                    Re=Re,
                    mach=mach,
                )

        else:
            CL_function = self.CL_function
            CD_function = self.CD_function
            CM_function = self.CM_function

        return Airfoil(
            name=self.name,
            coordinates=coordinates,
            CL_function=CL_function,
            CD_function=CD_function,
            CM_function=CM_function,
        )

    def set_TE_thickness(self,
                         thickness: float = 0.,
                         ) -> 'Airfoil':
        """
        Creates a modified copy of the Airfoil that has a specified trailing-edge thickness.

        Note that the trailing-edge thickness is given nondimensionally (e.g., as a fraction of chord).

        Args:
            thickness: The target trailing-edge thickness, given nondimensionally (e.g., as a fraction of chord).

        Returns: The modified airfoil.

        """
        ### Compute existing trailing-edge properties
        x_gap = self.coordinates[0, 0] - self.coordinates[-1, 0]
        y_gap = self.coordinates[0, 1] - self.coordinates[-1, 1]

        s_gap = (
                        x_gap ** 2 +
                        y_gap ** 2
                ) ** 0.5

        s_adjustment = (thickness - self.TE_thickness()) / 2

        ### Determine how much the trailing edge should move by in X and Y.
        if s_gap != 0:
            x_adjustment = s_adjustment * x_gap / s_gap
            y_adjustment = s_adjustment * y_gap / s_gap
        else:
            x_adjustment = 0
            y_adjustment = s_adjustment

        ### Decompose the existing airfoil coordinates to upper and lower sides, and x and y.
        u = self.upper_coordinates()
        ux = u[:, 0]
        uy = u[:, 1]

        le_x = ux[-1]

        l = self.lower_coordinates()[1:]
        lx = l[:, 0]
        ly = l[:, 1]

        te_x = (ux[0] + lx[-1]) / 2

        ### Create modified versions of the upper and lower coordinates
        new_u = np.stack(
            arrays=[
                ux + x_adjustment * (ux - le_x) / (te_x - le_x),
                uy + y_adjustment * (ux - le_x) / (te_x - le_x)
            ],
            axis=1
        )
        new_l = np.stack(
            arrays=[
                lx - x_adjustment * (lx - le_x) / (te_x - le_x),
                ly - y_adjustment * (lx - le_x) / (te_x - le_x)
            ],
            axis=1
        )

        ### If the desired thickness is zero, ensure that is precisely reached.
        if thickness == 0:
            new_l[-1] = new_u[0]

        ### Combine the upper and lower surface coordinates into a single array.
        new_coordinates = np.concatenate(
            [
                new_u,
                new_l
            ],
            axis=0
        )

        ### Return a new Airfoil with the desired coordinates.
        return Airfoil(
            name=self.name,
            coordinates=new_coordinates
        )

    def scale(self,
              scale_x: float = 1.,
              scale_y: float = 1.,
              ) -> 'Airfoil':
        """
        Scales an Airfoil about the origin.

        Args:

            scale_x: Amount to scale in the x-direction.

            scale_y: Amount to scale in the y-direction. Scaling by a negative y-value will result in coordinates
                being re-ordered such that the order of the coordinates is still correct (i.e., starts from the
                upper-surface trailing edge, continues along the upper surface to the nose, then continues along the
                lower surface to the trailing edge).

        Returns: A copy of the Airfoil with appropriate scaling applied.
        """
        x = self.x() * scale_x
        y = self.y() * scale_y

        if scale_y < 0:
            x = x[::-1]
            y = y[::-1]

        return Airfoil(
            name=self.name,
            coordinates=np.stack((x, y), axis=1)
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
            coordinates=np.stack((x, y), axis=1)
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
        foil_a = self.repanel(n_points_per_side=n_points_per_side)
        foil_b = airfoil.repanel(n_points_per_side=n_points_per_side)
        a_fraction = 1 - blend_fraction
        b_fraction = blend_fraction

        name = f"{a_fraction * 100:.0f}% {self.name}, {b_fraction * 100:.0f}% {airfoil.name}"

        coordinates = (
                a_fraction * foil_a.coordinates +
                b_fraction * foil_b.coordinates
        )

        return Airfoil(
            name=name,
            coordinates=coordinates,
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


if __name__ == '__main__':
    af = Airfoil("dae11")

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots(4, 2, figsize=(6.4, 6.4), dpi=200)

    alpha = np.linspace(-90, 90, 500)
    sizes = ["xxsmall", "xsmall", "small", "medium", "large", "xlarge", "xxlarge", "xxxlarge"]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(sizes)))[::-1]

    for i, ms in enumerate(sizes):
        aero = af.get_aero_from_neuralfoil(
            alpha=alpha,
            Re=1e6,
            mach=0.3,
            model_size=ms,
        )

        kwargs = dict(
            alpha=0.5,
            color=colors[i],
        )
        for a, key in zip(ax.T.flatten(), ["CL", "CD", "CM", "Cpmin", "mach_crit", "Top_Xtr", "Bot_Xtr", "Cpmin_0"]):
            a.plot(alpha, aero[key], **kwargs)
            if key == "CD":
                a.set_yscale('log')
            a.set_ylabel(key)

    p.show_plot()

    # af.draw()
    # af.generate_polars(
    #     alphas=np.linspace(-10, 15, 61),
    # )
    # af.plot_polars(
    #     Res=np.geomspace(1e4, 1e6, 6)
    # )
