import aerosandbox.numpy as np
from scipy.special import comb
from aerosandbox.geometry.polygon import stack_coordinates
import re

_default_n_points_per_side = 200


def get_NACA_coordinates(
        name: str = 'naca2412',
        n_points_per_side: int = _default_n_points_per_side
) -> np.ndarray:
    """
    Returns the coordinates of a specified 4-digit NACA airfoil.
    Args:
        name: Name of the NACA airfoil.
        n_points_per_side: Number of points per side of the airfoil (top/bottom).

    Returns: The coordinates of the airfoil as a Nx2 ndarray [x, y]

    """
    name = name.lower().strip()

    if not "naca" in name:
        raise ValueError("Not a NACA airfoil!")

    nacanumber = name.split("naca")[1]
    if not nacanumber.isdigit():
        raise ValueError("Couldn't parse the number of the NACA airfoil!")

    if not len(nacanumber) == 4:
        raise NotImplementedError("Only 4-digit NACA airfoils are currently supported!")

    # Parse
    max_camber = int(nacanumber[0]) * 0.01
    camber_loc = int(nacanumber[1]) * 0.1
    thickness = int(nacanumber[2:]) * 0.01

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

    x = np.hstack((x_U, x_L))
    y = np.hstack((y_U, y_L))

    return stack_coordinates(x, y)


def get_kulfan_coordinates(
        lower_weights=-0.2 * np.ones(5),  # type: np.ndarray
        upper_weights=0.2 * np.ones(5),  # type: np.ndarray
        enforce_continuous_LE_radius=True,
        TE_thickness=0.,  # type: float
        n_points_per_side=_default_n_points_per_side,  # type: int
        N1=0.5,  # type: float
        N2=1.0,  # type: float
) -> np.ndarray:
    """
    Calculates the coordinates of a Kulfan (CST) airfoil.
    To make a Kulfan (CST) airfoil, use the following syntax:

    asb.Airfoil("My Airfoil Name", coordinates = asb.kulfan_coordinates(*args))

    More on Kulfan (CST) airfoils: http://brendakulfan.com/docs/CST2.pdf
    Notes on N1, N2 (shape factor) combinations:
        * 0.5, 1: Conventional airfoil
        * 0.5, 0.5: Elliptic airfoil
        * 1, 1: Biconvex airfoil
        * 0.75, 0.75: Sears-Haack body (radius distribution)
        * 0.75, 0.25: Low-drag projectile
        * 1, 0.001: Cone or wedge airfoil
        * 0.001, 0.001: Rectangle, circular duct, or circular rod.
    :param lower_weights:
    :param upper_weights:
    :param enforce_continuous_LE_radius: Enforces a continous leading-edge radius by throwing out the first lower weight.
    :param TE_thickness:
    :param n_points_per_side:
    :param N1: LE shape factor
    :param N2: TE shape factor
    :return:
    """

    if enforce_continuous_LE_radius:
        lower_weights[0] = -1 * upper_weights[0]

    x_lower = np.cosspace(0, 1, n_points_per_side)
    x_upper = x_lower[::-1]

    x_lower = x_lower[1:] # Trim off the nose coordinate so there are no duplicates

    def shape(w, x):
        # Class function
        C = x ** N1 * (1 - x) ** N2

        # Shape function (Bernstein polynomials)
        n = len(w) - 1  # Order of Bernstein polynomials

        K = comb(n, np.arange(n + 1))  # Bernstein polynomial coefficients

        S_matrix = (
                w * K * np.expand_dims(x, 1) ** np.arange(n + 1) *
                np.expand_dims(1 - x, 1) ** (n - np.arange(n + 1))
        )  # Polynomial coefficient * weight matrix
        # S = np.sum(S_matrix, axis=1)
        S = np.array([np.sum(S_matrix[i,:]) for i in range(S_matrix.shape[0])])

        # Calculate y output
        y = C * S
        return y

    y_lower = shape(lower_weights, x_lower)
    y_upper = shape(upper_weights, x_upper)

    # TE thickness
    y_lower -= x_lower * TE_thickness / 2
    y_upper += x_upper * TE_thickness / 2

    x = np.concatenate([x_upper, x_lower])
    y = np.concatenate([y_upper, y_lower])
    coordinates = np.vstack((x, y)).T

    return coordinates


def get_coordinates_from_raw_dat(raw_text) -> np.ndarray:
    """
    Returns a Nx2 ndarray of airfoil coordinates from the raw text of a airfoil *.dat file.
    Args:
        raw_text: The raw text of the *.dat file, as read by file.readlines()

    Returns: A Nx2 ndarray of airfoil coordinates [x, y].

    """
    raw_coordinates = []

    def is_number(s):  # determines whether a string is representable as a float
        try:
            float(s)
        except ValueError:
            return False
        return True

    for line in raw_text:
        try:
            line_split = re.split(r'[; |, |\*|\n]', line)
            line_items = [s for s in line_split
                          if s != "" and is_number(s)
                          ]
            if len(line_items) == 2:
                raw_coordinates.append(line_items)
        except:
            pass

    if len(raw_coordinates) == 0:
        raise ValueError("File was found, but could not read any coordinates!")

    coordinates = np.array(raw_coordinates, dtype=float)

    return coordinates


def get_file_coordinates(
        filepath
):
    try:
        with open(filepath, "r") as f:
            raw_text = f.readlines()
    except FileNotFoundError as e:
        try:
            with open(f"{filepath}.dat", "r") as f:
                raw_text = f.readlines()
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f" Neither '{filepath}' nor '{filepath}.dat' were found."
            ) from e

    return get_coordinates_from_raw_dat(raw_text)


def get_UIUC_coordinates(
        name: str = 'dae11'
) -> np.ndarray:
    """
    Returns the coordinates of a specified airfoil in the UIUC airfoil database.
    Args:
        name: Name of the airfoil to retrieve from the UIUC database.

    Returns: The coordinates of the airfoil as a Nx2 ndarray [x, y]
    """

    name = name.lower().strip()

    import importlib.resources
    from aerosandbox.geometry.airfoil import airfoil_database

    try:
        with importlib.resources.open_text(airfoil_database, name) as f:
            raw_text = f.readlines()
    except FileNotFoundError as e:
        try:
            with importlib.resources.open_text(airfoil_database, name + '.dat') as f:
                raw_text = f.readlines()
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Neither '{name}' nor '{name}.dat' were found in the UIUC airfoil database."
            ) from e

    return get_coordinates_from_raw_dat(raw_text)

