from aerosandbox.geometry import Airfoil
from aerosandbox.performance import OperatingPoint
import aerosandbox.numpy as np
import aerosandbox.library.aerodynamics as aerolib


def airfoil_coefficients_post_stall(
        airfoil: Airfoil,
        alpha: float,
):
    """
    Estimates post-stall aerodynamics of an airfoil.

    Uses methods given in:

    Truong, V. K. "An analytical model for airfoil aerodynamic characteristics over the entire 360deg angle of attack
    range". J. Renewable Sustainable Energy. 2020. doi: 10.1063/1.5126055

    Args:
        airfoil:
        op_point:

    Returns:

    """
    sina = np.sind(alpha)
    cosa = np.cosd(alpha)

    ##### Normal force calulation
    # Cd90_fp = aerolib.Cd_flat_plate_normal() # TODO implement
    # Cd90_0 = Cd90_fp - 0.83 * airfoil.LE_radius() - 1.46 / 2 * airfoil.max_thickness() + 1.46 * airfoil.max_camber()
    # Cd270_0 = Cd90_fp - 0.83 * airfoil.LE_radius() - 1.46 / 2 * airfoil.max_thickness() - 1.46 * airfoil.max_camber()

    ### Values for NACA0012
    Cd90_0 = 2.08
    pn2_star = 8.36e-2
    pn3_star = 4.06e-1
    pt1_star = 9.00e-2
    pt2_star = -1.78e-1
    pt3_star = -2.98e-1

    Cd90 = Cd90_0 + pn2_star * cosa + pn3_star * cosa ** 2
    CN = Cd90 * sina

    ##### Tangential force calculation
    CT = (pt1_star + pt2_star * cosa + pt3_star * cosa ** 3) * sina ** 2

    ##### Conversion to wind axes
    CL = CN * cosa + CT * sina
    CD = CN * sina - CT * cosa
    CM = np.zeros_like(CL)  # TODO

    return CL, CD, CM


if __name__ == '__main__':
    af = Airfoil("naca0012")
    alpha = np.linspace(0, 360, 721)
    CL, CD, CM = airfoil_coefficients_post_stall(
        af, alpha
    )
    from aerosandbox.tools.pretty_plots import plt, show_plot, set_ticks

    fig, ax = plt.subplots(1, 2, figsize=(8, 5))
    plt.sca(ax[0])
    plt.plot(alpha, CL)
    plt.xlabel("AoA")
    plt.ylabel("CL")
    set_ticks(45, 15, 0.5, 0.1)
    plt.sca(ax[1])
    plt.plot(alpha, CD)
    plt.xlabel("AoA")
    plt.ylabel("CD")
    set_ticks(45, 15, 0.5, 0.1)
    show_plot()
