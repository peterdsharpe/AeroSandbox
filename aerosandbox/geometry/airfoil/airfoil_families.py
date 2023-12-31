import aerosandbox.numpy as np
from scipy.special import comb
import re
from typing import Union
import os
from typing import List, Optional, Dict

_default_n_points_per_side = 200


def get_NACA_coordinates(
        name: str = None,
        n_points_per_side: int = _default_n_points_per_side,
        max_camber: float = None,
        camber_loc: float = None,
        thickness: float = None,
) -> np.ndarray:
    """
    Returns the coordinates of a 4-series NACA airfoil.

    Can EITHER specify `name`, or all three of `max_camber`, `camber_loc`, and `thickness` - not both.

    Args:

        Either:

            * name: Name of the NACA airfoil, as a string (e.g., "naca2412")

        Or:

            * All three of:

                max_camber: Maximum camber of the airfoil, as a fraction of chord (e.g., 0.02)

                camber_loc: The location of maximum camber, as a fraction of chord (e.g., 0.40)

                thickness: The maximum thickness of the airfoil, as a fraction of chord (e.g., 0.12)

        n_points_per_side: Number of points per side of the airfoil (top/bottom).

    Returns: The coordinates of the airfoil as a Nx2 ndarray [x, y]

    """
    ### Validate inputs
    name_specified = name is not None
    params_specified = [
        (max_camber is not None),
        (camber_loc is not None),
        (thickness is not None)
    ]

    if name_specified:
        if any(params_specified):
            raise ValueError(
                "Cannot specify both `name` and (`max_camber`, `camber_loc`, `thickness`) parameters - must be one or the other.")

        name = name.lower().strip()

        if not "naca" in name:
            raise ValueError("Not a NACA airfoil - name must start with 'naca'!")

        nacanumber = name.split("naca")[1]
        if not nacanumber.isdigit():
            raise ValueError("Couldn't parse the number of the NACA airfoil!")

        if not len(nacanumber) == 4:
            raise NotImplementedError("Only 4-digit NACA airfoils are currently supported!")

        # Parse
        max_camber = int(nacanumber[0]) * 0.01
        camber_loc = int(nacanumber[1]) * 0.1
        thickness = int(nacanumber[2:]) * 0.01

    else:
        if not all(params_specified):
            raise ValueError(
                "Must specify either `name` or all three (`max_camber`, `camber_loc`, `thickness`) parameters.")

    # Referencing https://en.wikipedia.org/wiki/NACA_airfoil#Equation_for_a_cambered_4-digit_NACA_airfoil
    # from here on out

    # Make uncambered coordinates
    x_t = np.cosspace(0, 1, n_points_per_side)  # Generate some cosine-spaced points
    y_t = 5 * thickness * (
            + 0.2969 * x_t ** 0.5
            - 0.1260 * x_t
            - 0.3516 * x_t ** 2
            + 0.2843 * x_t ** 3
            - 0.1015 * x_t ** 4  # 0.1015 is original, #0.1036 for sharp TE
    )

    if camber_loc == 0:
        camber_loc = 0.5  # prevents divide by zero errors for things like naca0012's.

    # Get camber
    y_c = np.where(
        x_t <= camber_loc,
        max_camber / camber_loc ** 2 * (2 * camber_loc * x_t - x_t ** 2),
        max_camber / (1 - camber_loc) ** 2 * ((1 - 2 * camber_loc) + 2 * camber_loc * x_t - x_t ** 2)
    )

    # Get camber slope
    dycdx = np.where(
        x_t <= camber_loc,
        2 * max_camber / camber_loc ** 2 * (camber_loc - x_t),
        2 * max_camber / (1 - camber_loc) ** 2 * (camber_loc - x_t)
    )
    theta = np.arctan(dycdx)

    # Combine everything
    x_U = x_t - y_t * np.sin(theta)
    x_L = x_t + y_t * np.sin(theta)
    y_U = y_c + y_t * np.cos(theta)
    y_L = y_c - y_t * np.cos(theta)

    # Flip upper surface so it's back to front
    x_U, y_U = x_U[::-1], y_U[::-1]

    # Trim 1 point from lower surface so there's no overlap
    x_L, y_L = x_L[1:], y_L[1:]

    x = np.concatenate((x_U, x_L))
    y = np.concatenate((y_U, y_L))

    return np.stack((x, y), axis=1)


def get_kulfan_coordinates(
        lower_weights: np.ndarray = -0.2 * np.ones(8),
        upper_weights: np.ndarray = 0.2 * np.ones(8),
        leading_edge_weight: float = 0.,
        TE_thickness: float = 0.,
        n_points_per_side: int = _default_n_points_per_side,
        N1: float = 0.5,
        N2: float = 1.0,
        **deprecated_kwargs
) -> np.ndarray:
    """
    Given a set of Kulfan parameters, computes the coordinates of the resulting airfoil.

    This function is the inverse of `get_kulfan_parameters()`.

    Kulfan parameters are a highly-efficient and flexible way to parameterize the shape of an airfoil. The particular
    flavor of Kulfan parameterization used in AeroSandbox is the "CST with LEM" method, which is described in various
    papers linked below. In total, the Kulfan parameterization consists of:

    * A vector of weights corresponding to the lower surface of the airfoil
    * A vector of weights corresponding to the upper surface of the airfoil
    * A scalar weight corresponding to the strength of a leading-edge camber mode shape of the airfoil (optional)
    * The trailing-edge (TE) thickness of the airfoil (optional)

    These Kulfan parameters are also referred to as CST (Class/Shape Transformation) parameters.

    References on Kulfan (CST) airfoils:

    * Kulfan, Brenda "Universal Parametric Geometry Representation Method" (2008). AIAA Journal of Aircraft.
        Describes the basic Kulfan (CST) airfoil parameterization.
        Mirrors:
            * https://arc.aiaa.org/doi/10.2514/1.29958
            * https://www.brendakulfan.com/_files/ugd/169bff_6738e0f8d9074610942c53dfaea8e30c.pdf
            * https://www.researchgate.net/publication/245430684_Universal_Parametric_Geometry_Representation_Method

    * Kulfan, Brenda "Modification of CST Airfoil Representation Methodology" (2020). Unpublished note:
        Describes the optional "Leading-Edge Modification" (LEM) addition to the Kulfan (CST) airfoil parameterization.
        Mirrors:
            * https://www.brendakulfan.com/_files/ugd/169bff_16a868ad06af4fea946d299c6028fb13.pdf
            * https://www.researchgate.net/publication/343615711_Modification_of_CST_Airfoil_Representation_Methodology

    * Masters, D.A. "Geometric Comparison of Aerofoil Shape Parameterization Methods" (2017). AIAA Journal.
        Compares the Kulfan (CST) airfoil parameterization to other airfoil parameterizations. Also has further notes
        on the LEM addition.
        Mirrors:
            * https://arc.aiaa.org/doi/10.2514/1.J054943
            * https://research-information.bris.ac.uk/ws/portalfiles/portal/91793513/SP_Journal_RED.pdf

    Notes on N1, N2 (shape factor) combinations:
        * 0.5, 1: Conventional airfoil
        * 0.5, 0.5: Elliptic airfoil
        * 1, 1: Biconvex airfoil
        * 0.75, 0.75: Sears-Haack body (radius distribution)
        * 0.75, 0.25: Low-drag projectile
        * 1, 0.001: Cone or wedge airfoil
        * 0.001, 0.001: Rectangle, circular duct, or circular rod.

    To make a Kulfan (CST) airfoil, use the following syntax:

    >>> import aerosandbox as asb
    >>> asb.Airfoil("My Airfoil Name", coordinates=asb.get_kulfan_coordinates(*args))

    Args:

        lower_weights (iterable): The Kulfan weights to use for the lower surface.

        upper_weights (iterable): The Kulfan weights to use for the upper surface.

        TE_thickness (float): The trailing-edge thickness to add, in terms of y/c.

        n_points_per_side (int): The number of points to discretize with, when generating the coordinates.

        N1 (float): The shape factor corresponding to the leading edge of the airfoil. See above for examples.

        N2 (float): The shape factor corresponding to the trailing edge of the airfoil. See above for examples.

    Returns:
        np.ndarray: The coordinates of the airfoil as a Nx2 array.
    """
    if len(deprecated_kwargs) > 0:
        import warnings
        warnings.warn(
            "The following arguments are deprecated and will be removed in a future version:\n"
            f"{deprecated_kwargs}",
            DeprecationWarning
        )

        if deprecated_kwargs.get("enforce_continuous_LE_radius", False):
            lower_weights[0] = -1 * upper_weights[0]

    x = np.cosspace(0, 1, n_points_per_side)  # Generate some cosine-spaced points

    # Class function
    C = (x) ** N1 * (1 - x) ** N2

    def shape_function(w):
        # Shape function (Bernstein polynomials)
        N = np.length(w) - 1  # Order of Bernstein polynomials

        K = comb(N, np.arange(N + 1))  # Bernstein polynomial coefficients

        dims = (np.length(w), np.length(x))

        def wide(vector):
            return np.tile(np.reshape(vector, (1, dims[1])), (dims[0], 1))

        def tall(vector):
            return np.tile(np.reshape(vector, (dims[0], 1)), (1, dims[1]))

        S_matrix = (
                tall(K) * wide(x) ** tall(np.arange(N + 1)) *
                wide(1 - x) ** tall(N - np.arange(N + 1))
        )  # Bernstein polynomial coefficients * weight matrix
        S_x = np.sum(tall(w) * S_matrix, axis=0)

        # Calculate y output
        y = C * S_x
        return y

    y_lower = shape_function(lower_weights)
    y_upper = shape_function(upper_weights)

    # Add trailing-edge (TE) thickness
    y_lower -= x * TE_thickness / 2
    y_upper += x * TE_thickness / 2

    # Add Kulfan's leading-edge-modification (LEM)
    y_lower += leading_edge_weight * (x) * (1 - x) ** (np.length(lower_weights) + 0.5)
    y_upper += leading_edge_weight * (x) * (1 - x) ** (np.length(upper_weights) + 0.5)

    x = np.concatenate((x[::-1], x[1:]))
    y = np.concatenate((y_upper[::-1], y_lower[1:]))
    coordinates = np.stack((x, y), axis=1)

    return coordinates


def get_kulfan_parameters(
        coordinates: np.ndarray,
        n_weights_per_side: int = 8,
        N1: float = 0.5,
        N2: float = 1.0,
        n_points_per_side: int = _default_n_points_per_side,
        normalize_coordinates: bool = True,
        use_leading_edge_modification: bool = True,
        method: str = "least_squares",
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Given a set of airfoil coordinates, reconstructs the Kulfan parameters that would recreate that airfoil. Uses a
    curve fitting (optimization) process.

    This function is the inverse of `get_kulfan_coordinates()`.

    Kulfan parameters are a highly-efficient and flexible way to parameterize the shape of an airfoil. The particular
    flavor of Kulfan parameterization used in AeroSandbox is the "CST with LEM" method, which is described in various
    papers linked below. In total, the Kulfan parameterization consists of:

    * A vector of weights corresponding to the lower surface of the airfoil
    * A vector of weights corresponding to the upper surface of the airfoil
    * A scalar weight corresponding to the strength of a leading-edge camber mode shape of the airfoil (optional)
    * The trailing-edge (TE) thickness of the airfoil (optional)

    These Kulfan parameters are also referred to as CST (Class/Shape Transformation) parameters.

    References on Kulfan (CST) airfoils:

    * Kulfan, Brenda "Universal Parametric Geometry Representation Method" (2008). AIAA Journal of Aircraft.
        Describes the basic Kulfan (CST) airfoil parameterization.
        Mirrors:
            * https://arc.aiaa.org/doi/10.2514/1.29958
            * https://www.brendakulfan.com/_files/ugd/169bff_6738e0f8d9074610942c53dfaea8e30c.pdf
            * https://www.researchgate.net/publication/245430684_Universal_Parametric_Geometry_Representation_Method

    * Kulfan, Brenda "Modification of CST Airfoil Representation Methodology" (2020). Unpublished note:
        Describes the optional "Leading-Edge Modification" (LEM) addition to the Kulfan (CST) airfoil parameterization.
        Mirrors:
            * https://www.brendakulfan.com/_files/ugd/169bff_16a868ad06af4fea946d299c6028fb13.pdf
            * https://www.researchgate.net/publication/343615711_Modification_of_CST_Airfoil_Representation_Methodology

    * Masters, D.A. "Geometric Comparison of Aerofoil Shape Parameterization Methods" (2017). AIAA Journal.
        Compares the Kulfan (CST) airfoil parameterization to other airfoil parameterizations. Also has further notes
        on the LEM addition.
        Mirrors:
            * https://arc.aiaa.org/doi/10.2514/1.J054943
            * https://research-information.bris.ac.uk/ws/portalfiles/portal/91793513/SP_Journal_RED.pdf

    Notes on N1, N2 (shape factor) combinations:
        * 0.5, 1: Conventional airfoil
        * 0.5, 0.5: Elliptic airfoil
        * 1, 1: Biconvex airfoil
        * 0.75, 0.75: Sears-Haack body (radius distribution)
        * 0.75, 0.25: Low-drag projectile
        * 1, 0.001: Cone or wedge airfoil
        * 0.001, 0.001: Rectangle, circular duct, or circular rod.

    The following demonstrates the reversibility of this function:

    >>> import aerosandbox as asb
    >>> from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters
    >>>
    >>> af = asb.Airfoil("dae11")  # A conventional airfoil
    >>> params = get_kulfan_parameters(
    >>>     coordinates=af.coordinates,
    >>> )
    >>> af_reconstructed = asb.Airfoil(
    >>>     name="Reconstructed Airfoil",
    >>>     coordinates=get_kulfan_coordinates(
    >>>         **params
    >>>     )

    Args:

        coordinates (np.ndarray): The coordinates of the airfoil as a Nx2 array.

        n_weights_per_side (int): The number of Kulfan weights to use per side of the airfoil.

        N1 (float): The shape factor corresponding to the leading edge of the airfoil. See above for examples.

        N2 (float): The shape factor corresponding to the trailing edge of the airfoil. See above for examples.

        n_points_per_side (int): The number of points to discretize with, when formulating the curve-fitting
            optimization problem.

    Returns:
        A dictionary containing the Kulfan parameters. The keys are:
            * "lower_weights" (np.ndarray): The weights corresponding to the lower surface of the airfoil.
            * "upper_weights" (np.ndarray): The weights corresponding to the upper surface of the airfoil.
            * "TE_thickness" (float): The trailing-edge thickness of the airfoil.
            * "leading_edge_weight" (float): The strength of the leading-edge camber mode shape of the airfoil.

        These can be passed directly into `get_kulfan_coordinates()` to reconstruct the airfoil.
    """
    from aerosandbox.geometry.airfoil import Airfoil

    if method == "opti":

        target_airfoil = Airfoil(
            name="Target Airfoil",
            coordinates=coordinates
        ).repanel(
            n_points_per_side=n_points_per_side
        )

        if normalize_coordinates:
            target_airfoil = target_airfoil.normalize()

        x = np.cosspace(0, 1, n_points_per_side)
        target_thickness = target_airfoil.local_thickness(x_over_c=x)
        target_camber = target_airfoil.local_camber(x_over_c=x)

        target_y_upper = target_camber + target_thickness / 2
        target_y_lower = target_camber - target_thickness / 2

        # Class function
        C = (x) ** N1 * (1 - x) ** N2

        def shape_function(w):
            # Shape function (Bernstein polynomials)
            N = np.length(w) - 1  # Order of Bernstein polynomials

            K = comb(N, np.arange(N + 1))  # Bernstein polynomial coefficients

            dims = (np.length(w), np.length(x))

            def wide(vector):
                return np.tile(np.reshape(vector, (1, dims[1])), (dims[0], 1))

            def tall(vector):
                return np.tile(np.reshape(vector, (dims[0], 1)), (1, dims[1]))

            S_matrix = (
                    tall(K) * wide(x) ** tall(np.arange(N + 1)) *
                    wide(1 - x) ** tall(N - np.arange(N + 1))
            )  # Bernstein polynomial coefficients * weight matrix
            S_x = np.sum(tall(w) * S_matrix, axis=0)

            # Calculate y output
            y = C * S_x
            return y

        opti = asb.Opti()
        lower_weights = opti.variable(init_guess=0, n_vars=n_weights_per_side)
        upper_weights = opti.variable(init_guess=0, n_vars=n_weights_per_side)
        TE_thickness = opti.variable(init_guess=0, lower_bound=0)
        if use_leading_edge_modification:
            leading_edge_weight = opti.variable(init_guess=0)
        else:
            leading_edge_weight = 0

        y_lower = shape_function(lower_weights)
        y_upper = shape_function(upper_weights)

        # Add trailing-edge (TE) thickness
        y_lower -= x * TE_thickness / 2
        y_upper += x * TE_thickness / 2

        # Add Kulfan's leading-edge-modification (LEM)
        y_lower += leading_edge_weight * (x) * (1 - x) ** (np.length(lower_weights) + 0.5)
        y_upper += leading_edge_weight * (x) * (1 - x) ** (np.length(upper_weights) + 0.5)

        opti.minimize(
            np.sum((y_lower - target_y_lower) ** 2) +
            np.sum((y_upper - target_y_upper) ** 2)
        )

        sol = opti.solve(
            verbose=False
        )

        return {
            "lower_weights"      : sol.value(lower_weights),
            "upper_weights"      : sol.value(upper_weights),
            "TE_thickness"       : sol.value(TE_thickness),
            "leading_edge_weight": sol.value(leading_edge_weight),
        }

    elif method == "least_squares":

        """
        
        The goal here is to set up this fitting problem as a least-squares problem (likely an overconstrained one, 
        but keeping it general for now. This will then be solved with np.linalg.lstsq(A, b), where A will (likely) 
        not be square.
        
        The columns of the A matrix will correspond to our unknowns, which are going to be a 1D vector `x` packed in as:
            * upper_weights from 0 to n_weights_per_side - 1
            * lower_weights from 0 to n_weights_per_side - 1
            * leading_edge_weight
            * trailing_edge_thickness
            
        See `get_kulfan_coordinates()` for more details on the meaning of these variables.
        
        The rows of the A matrix will correspond to each row of the given airfoil coordinates (i.e., a single vertex 
        on the airfoil). The idea here is to express each vertex as a linear combination of the unknowns, and then
        solve for the unknowns that minimize the error between the given airfoil coordinates and the reconstructed
        airfoil coordinates.
        
        """

        if normalize_coordinates:
            coordinates = Airfoil(
                name="Target Airfoil",
                coordinates=coordinates
            ).normalize().coordinates

        n_coordinates = np.length(coordinates)

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        LE_index = np.argmin(x)
        is_upper = np.arange(len(x)) <= LE_index

        # Class function
        C = (x) ** N1 * (1 - x) ** N2

        # Shape function (Bernstein polynomials)
        N = n_weights_per_side - 1  # Order of Bernstein polynomials

        K = comb(N, np.arange(N + 1))  # Bernstein polynomial coefficients

        dims = (n_weights_per_side, n_coordinates)

        def wide(vector):
            return np.tile(np.reshape(vector, (1, dims[1])), (dims[0], 1))

        def tall(vector):
            return np.tile(np.reshape(vector, (dims[0], 1)), (1, dims[1]))

        S_matrix = (
                tall(K) * wide(x) ** tall(np.arange(N + 1)) *
                wide(1 - x) ** tall(N - np.arange(N + 1))
        )  # Bernstein polynomial coefficients * weight matrix

        leading_edge_weight_row = x * np.maximum(1 - x, 0) ** (n_weights_per_side + 0.5)

        trailing_edge_thickness_row = np.where(
            is_upper,
            x / 2,
            -x / 2
        )

        A = np.concatenate([
            np.where(wide(is_upper), 0, wide(C) * S_matrix).T,
            np.where(wide(is_upper), wide(C) * S_matrix, 0).T,
            np.reshape(leading_edge_weight_row, (n_coordinates, 1)),
            np.reshape(trailing_edge_thickness_row, (n_coordinates, 1)),
        ], axis=1)

        b = y

        # Solve least-squares problem
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        lower_weights = x[:n_weights_per_side]
        upper_weights = x[n_weights_per_side:2 * n_weights_per_side]
        leading_edge_weight = x[-2]
        trailing_edge_thickness = x[-1]

        # If you got a negative trailing-edge thickness, then resolve the problem with a TE_thickness = 0 constraint.
        if trailing_edge_thickness < 0:

            x, _, _, _ = np.linalg.lstsq(A[:, :-1], b, rcond=None)

            lower_weights = x[:n_weights_per_side]
            upper_weights = x[n_weights_per_side:2 * n_weights_per_side]
            leading_edge_weight = x[-1]
            trailing_edge_thickness = 0

        return {
            "lower_weights"      : lower_weights,
            "upper_weights"      : upper_weights,
            "TE_thickness"       : trailing_edge_thickness,
            "leading_edge_weight": leading_edge_weight,
        }

    else:
        raise ValueError(f"Invalid method '{method}'.")


def get_coordinates_from_raw_dat(
        raw_text: List[str]
) -> np.ndarray:
    """
    Returns a Nx2 ndarray of airfoil coordinates from the raw text of a airfoil *.dat file.

    Args:

        raw_text: A list of strings, where each string is one line of the *.dat file. One good way to get this input
            is to read the file via the `with open(file, "r") as file:`, `file.readlines()` interface.

    Returns: A Nx2 ndarray of airfoil coordinates [x, y].

    """
    raw_coordinates = []

    def is_number(s: str) -> bool:
        # Determines whether a string is representable as a float
        try:
            float(s)
        except ValueError:
            return False
        return True

    def parse_line(line: str) -> Optional[List[float]]:
        # Given a single line of a `*.dat` file, tries to parse it into a list of two floats [x, y].
        # If not possible, returns None.
        line_split = re.split(r'[;|,|\s|\t]', line)
        line_items = [s for s in line_split if s != ""]
        if len(line_items) == 2 and all([is_number(item) for item in line_items]):
            return line_items
        else:
            return None

    for line in raw_text:
        parsed_line = parse_line(line)
        if parsed_line is not None:
            raw_coordinates.append(parsed_line)

    if len(raw_coordinates) == 0:
        raise ValueError("Could not read any coordinates from the `raw_text` input!")

    coordinates = np.array(raw_coordinates, dtype=float)

    return coordinates


def get_file_coordinates(
        filepath: Union[str, os.PathLike]
):
    possible_errors = (FileNotFoundError, UnicodeDecodeError)

    if isinstance(filepath, np.ndarray):
        raise TypeError("`filepath` should be a string or os.PathLike object.")

    try:
        with open(filepath, "r") as f:
            raw_text = f.readlines()
    except possible_errors as e:
        try:
            with open(f"{filepath}.dat", "r") as f:
                raw_text = f.readlines()
        except possible_errors as e:
            raise FileNotFoundError(
                f" Neither '{filepath}' nor '{filepath}.dat' were found and readable."
            ) from e

    try:
        return get_coordinates_from_raw_dat(raw_text)
    except ValueError:
        raise ValueError("File was found, but could not read any coordinates!")


def get_UIUC_coordinates(
        name: str = 'dae11'
) -> np.ndarray:
    """
    Returns the coordinates of a specified airfoil in the UIUC airfoil database.
    Args:
        name: Name of the airfoil to retrieve from the UIUC database.

    Returns: The coordinates of the airfoil as a Nx2 ndarray [x, y]
    """
    from aerosandbox import _asb_root

    airfoil_database_root = _asb_root / "geometry" / "airfoil" / "airfoil_database"

    try:
        with open(airfoil_database_root / name) as f:
            raw_text = f.readlines()
    except FileNotFoundError as e:
        try:
            with open(airfoil_database_root / f"{name}.dat") as f:
                raw_text = f.readlines()
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Neither '{name}' nor '{name}.dat' were found in the UIUC airfoil database."
            ) from e

    return get_coordinates_from_raw_dat(raw_text)


if __name__ == '__main__':
    import aerosandbox as asb
    import aerosandbox.numpy as np

    af = asb.Airfoil("e377").normalize()
    af.draw(backend="plotly")

    kulfan_params = get_kulfan_parameters(
        coordinates=af.coordinates,
        n_weights_per_side=8,
    )

    af_reconstructed = asb.Airfoil(
        name="Reconstructed Airfoil",
        coordinates=get_kulfan_coordinates(
            **kulfan_params
        ),
    )
    af_reconstructed.draw(backend="plotly")

    print(kulfan_params)
    print(af.jaccard_similarity(af_reconstructed))
