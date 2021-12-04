from aerosandbox.geometry import *
from aerosandbox.performance import OperatingPoint
import aerosandbox.numpy as np
import aerosandbox.library.aerodynamics as aerolib


def jorgensen_eta(fineness_ratio: float) -> float:
    """
    A fit for the eta parameter (crossflow lift multiplier) of a fuselage, as described in:

    Jorgensen, Leland Howard. "Prediction of Static Aerodynamic Characteristics for Slender Bodies
    Alone and with Lifting Surfaces to Very High Angles of Attack". NASA TR R-474. 1977.

    Fits performed in /studies/JorgensenEtaFitting/

    Args:
        fineness_ratio: The fineness ratio of the fuselage.

    Returns: An estimate of eta.

    """
    x = fineness_ratio
    p = {
        '1scl': 23.009059965179222,
        '1cen': -122.76900250914575,
        '2scl': 13.006453125841258,
        '2cen': -24.367562906887436
    }
    return 1 - p["1scl"] / (x - p["1cen"]) - (p["2scl"] / (x - p["2cen"])) ** 2


def fuselage_aerodynamics(
        fuselage: Fuselage,
        op_point: OperatingPoint,
):
    """
    Estimates the aerodynamic forces, moments, and derivatives on a fuselage in isolation.

    Assumes:
        * The fuselage is a body of revolution aligned with the x_b axis.
        * The angle between the nose and the freestream is less than 90 degrees.

    Moments are given with the reference at Fuselage [0, 0, 0].

    Uses methods from Jorgensen, Leland Howard. "Prediction of Static Aerodynamic Characteristics for Slender Bodies
    Alone and with Lifting Surfaces to Very High Angles of Attack". NASA TR R-474. 1977.

    Args:

        fuselage: A Fuselage object that you wish to analyze.

        op_point: The OperatingPoint that you wish to analyze the fuselage at.

    Returns:

    """

    fuselage.Re = op_point.reynolds(reference_length=fuselage.length())

    ####### Reference quantities (Set these 1 here, just so we can follow Jorgensen syntax.)
    # Outputs of this function should be invariant of these quantities, if normalization has been done correctly.
    S_ref = 1  # m^2
    c_ref = 1  # m

    ####### Fuselage zero-lift drag estimation

    ### Forebody drag
    C_f_forebody = aerolib.Cf_flat_plate(
        Re_L=fuselage.Re
    )

    ### Base Drag
    C_D_base = 0.029 / np.sqrt(C_f_forebody) * fuselage.area_base() / S_ref

    ### Skin friction drag
    C_D_skin = C_f_forebody * fuselage.area_wetted() / S_ref

    ### Total zero-lift drag
    C_D_zero_lift = C_D_skin + C_D_base

    ####### Jorgensen model

    ### First, merge the alpha and beta into a single "generalized alpha", which represents the degrees between the fuselage axis and the freestream.
    x_w, y_w, z_w = op_point.convert_axes(
        1, 0, 0, from_axes="body", to_axes="wind"
    )
    generalized_alpha = np.arccosd(x_w / (1 + 1e-14))
    sin_generalized_alpha = np.sind(generalized_alpha)
    cos_generalized_alpha = x_w

    # ### Limit generalized alpha to -90 < alpha < 90, for now.
    # generalized_alpha = np.clip(generalized_alpha, -90, 90)
    # # TODO make the drag/moment functions not give negative results for alpha > 90.

    alpha_fractional_component = -z_w / np.sqrt(
        y_w ** 2 + z_w ** 2 + 1e-16)  # The fraction of any "generalized lift" to be in the direction of alpha
    beta_fractional_component = y_w / np.sqrt(
        y_w ** 2 + z_w ** 2 + 1e-16)  # The fraction of any "generalized lift" to be in the direction of beta

    ### Compute normal quantities
    ### Note the (N)ormal, (A)ligned coordinate system. (See Jorgensen for definitions.)
    # M_n = sin_generalized_alpha * op_point.mach()
    Re_n = sin_generalized_alpha * fuselage.Re
    # V_n = sin_generalized_alpha * op_point.velocity
    q = op_point.dynamic_pressure()
    x_nose = fuselage.xsecs[0].xyz_c[0]
    x_m = 0 - x_nose
    x_c = fuselage.x_centroid_projected() - x_nose

    ##### Potential flow crossflow model
    C_N_p = (  # Normal force coefficient due to potential flow. (Jorgensen Eq. 2.12, part 1)
            fuselage.area_base() / S_ref * np.sind(2 * generalized_alpha) * np.cosd(generalized_alpha / 2)
    )
    C_m_p = (
            (
                    fuselage.volume() - fuselage.area_base() * (fuselage.length() - x_m)
            ) / (
                    S_ref * c_ref
            ) * np.sind(2 * generalized_alpha) * np.cosd(generalized_alpha / 2)
    )

    ##### Viscous crossflow model
    C_d_n = np.where(
        Re_n != 0,
        aerolib.Cd_cylinder(Re_D=Re_n),  # Replace with 1.20 from Jorgensen Table 1 if not working well
        0
    )
    eta = jorgensen_eta(fuselage.fineness_ratio())

    C_N_v = (  # Normal force coefficient due to viscous crossflow. (Jorgensen Eq. 2.12, part 2)
            eta * C_d_n * fuselage.area_projected() / S_ref * sin_generalized_alpha ** 2
    )
    C_m_v = (
            eta * C_d_n * fuselage.area_projected() / S_ref * (x_m - x_c) / c_ref * sin_generalized_alpha ** 2
    )

    ##### Total C_N model
    C_N = C_N_p + C_N_v
    C_m_generalized = C_m_p + C_m_v

    ##### Total C_A model
    C_A = C_D_zero_lift * cos_generalized_alpha * np.abs(cos_generalized_alpha)

    ##### Convert to lift, drag
    C_L_generalized = C_N * cos_generalized_alpha - C_A * sin_generalized_alpha
    C_D = C_N * sin_generalized_alpha + C_A * cos_generalized_alpha

    ### Set proper directions

    C_L = C_L_generalized * alpha_fractional_component
    C_Y = -C_L_generalized * beta_fractional_component
    C_l = 0
    C_m = C_m_generalized * alpha_fractional_component
    C_n = -C_m_generalized * beta_fractional_component

    ### Un-normalize
    L = C_L * q * S_ref
    Y = C_Y * q * S_ref
    D = C_D * q * S_ref
    l_w = C_l * q * S_ref * c_ref
    m_w = C_m * q * S_ref * c_ref
    n_w = C_n * q * S_ref * c_ref

    ### Convert to axes coordinates for reporting
    F_w = (
        -D,
        Y,
        -L
    )
    F_b = op_point.convert_axes(*F_w, from_axes="wind", to_axes="body")
    F_g = op_point.convert_axes(*F_b, from_axes="body", to_axes="geometry")
    M_w = (
        l_w,
        m_w,
        n_w,
    )
    M_b = op_point.convert_axes(*M_w, from_axes="wind", to_axes="body")
    M_g = op_point.convert_axes(*M_b, from_axes="body", to_axes="geometry")

    return {
        "F_g": F_g,
        "F_b": F_b,
        "F_w": F_w,
        "M_g": M_g,
        "M_b": M_b,
        "M_w": M_w,
        "L"  : -F_w[2],
        "Y"  : F_w[1],
        "D"  : -F_w[0],
        "l_b": M_b[0],
        "m_b": M_b[1],
        "n_b": M_b[2]
    }


if __name__ == '__main__':
    fuselage = Fuselage(
        xsecs=[
            FuselageXSec(
                xyz_c=[s, 0, 0],
                radius=Airfoil("naca0010").local_thickness(0.8 * s)
            )
            for s in np.cosspace(0, 1, 20)
        ]
    )
    # Airplane(fuselages=[fuselage]).draw()
    aero = fuselage_aerodynamics(
        fuselage=fuselage,
        op_point=OperatingPoint(
            velocity=10,
            alpha=10,
            beta=5
        )
    )
    print(aero)

    from aerosandbox.tools.pretty_plots import plt, show_plot, contour, equal

    fig, ax = plt.subplots(1, 2)
    alpha = np.linspace(-60, 60, 1000)
    aero = fuselage_aerodynamics(
        fuselage=fuselage,
        op_point=OperatingPoint(
            velocity=10,
            alpha=alpha,
        )
    )
    plt.sca(ax[0])
    plt.plot(alpha, aero["L"])
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("Lift Force [N]")

    plt.sca(ax[1])
    plt.plot(alpha, aero["D"])
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("Drag Force [N]")

    show_plot(
        "Fuselage Aerodynamics"
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    Beta, Alpha = np.meshgrid(np.linspace(-90, 90, 500), np.linspace(-90, 90, 500))
    aero = fuselage_aerodynamics(
        fuselage=fuselage,
        op_point=OperatingPoint(
            velocity=10,
            alpha=Alpha,
            beta=Beta,
        )
    )
    from aerosandbox.tools.string_formatting import eng_string

    contour(
        Beta, Alpha, aero["L"],
        levels=30, colorbar_label="Lift $L$ [N]",
        linelabels_format=lambda s: eng_string(s, unit="N")
    )
    equal()
    show_plot("3D Fuselage Lift", r"$\beta$ [deg]", r"$\alpha$ [deg]")
