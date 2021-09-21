from aerosandbox.geometry import *
from aerosandbox.performance import OperatingPoint
import aerosandbox.numpy as np
from aerosandbox.library.aerodynamics import Cf_flat_plate, Cd_cylinder


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
    S_ref = 1  # m^2
    c_ref = 1  # m

    ####### Fuselage zero-lift drag estimation

    ### Forebody drag
    C_f_forebody = Cf_flat_plate(
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
    generalized_alpha = np.sqrt(  # Strictly-speaking only valid for small alpha, beta. TODO update to be rigorous in 3D.
        op_point.alpha ** 2 +
        op_point.beta ** 2
    )
    alpha_fractional_component = op_point.alpha / generalized_alpha  # The fraction of any "generalized lift" to be in the direction of alpha
    beta_fractional_component = op_point.beta / generalized_alpha  # The fraction of any "generalized lift" to be in the direction of beta

    generalized_alpha = np.clip(generalized_alpha, -90, 90) # TODO make the drag/moment functions not give negative results for alpha > 90.

    ### Compute normal quantities
    ### Note the (N)ormal, (A)ligned coordinate system. (See Jorgensen for definitions.)
    sina = np.abs(np.sind(generalized_alpha))
    M_n = sina * op_point.mach()
    Re_n = sina * fuselage.Re
    V_n = sina * op_point.velocity
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
        Cd_cylinder(Re_D=Re_n),  # Replace with 1.20 from Jorgensen Table 1 if not working well
        0
    )
    eta = 0.7  # TODO make this a function of fineness ratio, per Figure 4 of Jorgensen

    C_N_v = (  # Normal force coefficient due to viscous crossflow. (Jorgensen Eq. 2.12, part 2)
            eta * C_d_n * fuselage.area_projected() / S_ref * np.sind(generalized_alpha) ** 2
    )
    C_m_v = (
            eta * C_d_n * fuselage.area_projected() / S_ref * (x_m - x_c) / c_ref * np.sind(generalized_alpha) ** 2
    )

    ##### Total C_N model
    C_N = C_N_p + C_N_v
    C_m_generalized = C_m_p + C_m_v

    ##### Total C_A model
    C_A = C_D_zero_lift * np.cosd(generalized_alpha) ** 2

    ##### Convert to lift, drag
    C_L_generalized = C_N * np.cosd(generalized_alpha) - C_A * np.sind(generalized_alpha)
    C_D = C_N * np.sind(generalized_alpha) + C_A * np.cosd(generalized_alpha)

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
    l = C_l * q * S_ref * c_ref
    m = C_m * q * S_ref * c_ref
    n = C_n * q * S_ref * c_ref

    return {
        "L": L,
        "Y": Y,
        "D": D,
        "l": l,
        "m": m,
        "n": n
    }


if __name__ == '__main__':
    fuselage = Fuselage(
        xyz_le=np.array([0, 0, 0]),
        xsecs=[
            FuselageXSec(
                xyz_c=[s, 0, 0],
                radius=Airfoil("naca0012").local_thickness(0.8 * s)
            )
            for s in np.cosspace(0, 1, 20)
        ]
    )
    # Airplane(fuselages=[fuselage]).draw()
    aero = fuselage_aerodynamics(
        fuselage=fuselage,
        op_point=OperatingPoint(
            velocity=10,
            alpha=-5,
            beta=-5
        )
    )
    print(aero)

    from aerosandbox.tools.pretty_plots import plt, show_plot

    fig, ax = plt.subplots(1,2)
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
