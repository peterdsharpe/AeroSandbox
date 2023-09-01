from aerosandbox.geometry.airplane import Airplane
from aerosandbox.performance import OperatingPoint
from aerosandbox.weights import MassProperties
import aerosandbox.numpy as np


def get_modes(
        airplane: Airplane,
        op_point: OperatingPoint,
        mass_props: MassProperties,
        aero,
        g=9.81,
):
    Q = op_point.dynamic_pressure()
    S = airplane.s_ref
    c = airplane.c_ref
    b = airplane.b_ref
    QS = Q * S
    m = mass_props.mass
    Ixx = mass_props.Ixx
    Iyy = mass_props.Iyy
    Izz = mass_props.Izz
    u0 = op_point.velocity

    Cxu = -2 * aero["CD"]
    Czu = -2 * aero["CL"]

    Cmq = aero["Cmq"]
    Cma = aero["Cma"]

    X_u = QS / m / u0 * Cxu  # Units: 1/sec
    # X_w = QS / m / u0 * Cxa
    # X_q = QS / m * c / (2 * u0) * Cxq
    Z_u = QS / m / u0 * Czu  # Units: 1/sec
    # Z_w = QS / m / u0 * Cza
    # Z_q = QS / m * c / (2 * u0) * Czq
    # M_u = QS * c / Iyy / u0 * Cmu
    M_w = QS * c / Iyy / u0 * Cma # Units: 1/(meter * sec)
    M_q = QS * c / Iyy * c / (2 * u0) * Cmq # Units: 1/sec

    modes = {}

    def get_mode_info(
            sigma,
            omega_squared,
    ):
        is_oscillatory = omega_squared > 0
        return {
            "eigenvalue_real": sigma + np.where(
                is_oscillatory,
                0,
                np.abs(omega_squared + 1e-100) ** 0.5,
            ),
            "eigenvalue_imag": np.where(
                is_oscillatory,
                np.abs(omega_squared + 1e-100) ** 0.5,
                0,
            ),
        }

    ##### Longitudinal modes

    ### Phugoid
    modes['phugoid'] = {  # FVA, Eq. 9.55-9.57
        **get_mode_info(
            sigma=X_u / 2,
            omega_squared=-(X_u ** 2) / 4 - g * Z_u / u0
        ),
        "eigenvalue_imag_approx": 2 ** 0.5 * g / u0,
        "damping_ratio_approx"  : 2 ** -0.5 * aero["CD"] / aero["CL"],
    }
    #
    # ### Short-period
    modes['short_period'] = get_mode_info(
        sigma=0.5 * M_q,
        omega_squared=-(M_q ** 2) / 4 - u0 * M_w
    )

    ##### Lateral modes

    ### Roll subsidence
    modes['roll_subsidence'] = {  # FVA, Eq. 9.63
        "eigenvalue_real": (
                QS * b ** 2 / (2 * Ixx * u0) * aero["Clp"]
        ),
        "eigenvalue_imag": 0,
        "damping_ratio"  : 1,
    }

    ### Dutch roll

    modes['dutch_roll'] = get_mode_info(  # FVA, Eq. 9.68
        sigma=(
                QS * b ** 2 /
                (2 * Izz * u0) *
                (aero["Cnr"] + Izz / (m * b ** 2) * aero["CYb"])
        ),
        omega_squared=(
                QS * b / Izz *
                (
                        aero["Cnb"] + (
                        op_point.atmosphere.density() * S * b / (4 * m) *
                        (aero["CYb"] * aero["Cnr"] - aero["Cnb"] * aero["CYr"])
                )
                )
        )
    )

    ### Spiral
    spiral_parameter = (aero["Cnr"] - aero["Cnb"] * aero["Clr"] / aero["Clb"])  # FVA, Eq. 9.66
    modes['spiral'] = {  # FVA, Eq. 9.66
        "eigenvalue_real": (
                QS * b ** 2 / (2 * Izz * u0) *
                spiral_parameter
        ),
        "eigenvalue_imag": 0,
    }

    ### Compute damping ratios of all modes
    for mode_name, mode_data in modes.items():
        modes[mode_name]['damping_ratio'] = (
                -mode_data['eigenvalue_real'] /
                (mode_data['eigenvalue_real'] ** 2 + mode_data['eigenvalue_imag'] ** 2) ** 0.5
        )

    return modes


if __name__ == '__main__':
    import aerosandbox as asb
    import aerosandbox.numpy as np

    from pprint import pprint

    airplane = asb.Airplane(
        s_ref=9,
        c_ref=0.90,
        b_ref=10,
    )
    op_point = asb.OperatingPoint(velocity=10)
    aero = dict(
        CL=0.46,
        CD=0.11,
        Cm=0.141,
        CLa=5.736,
        # CYa = 0,
        # Cla = 0,
        Cma=-1.59,
        # Cna = 0,
        # CLb = 0,
        CYb=-0.380,
        Clb=-0.208,
        # Cmb=0,
        Cnb=0.0294,
        # CLp =0,
        CYp=-0.325,
        Clp=-0.593,
        # Cmp=0,
        Cnp=-0.041,
        CLq=10.41,
        # CYq=0,
        # Clq=0,
        Cmq=-25.05,
        # Cnq=0,
        # CLr=0,
        CYr=0.194,
        Clr=0.143,
        # Cmr=0,
        Cnr=-0.048
    )

    mass_props = asb.mass_properties_from_radius_of_gyration(
        mass=op_point.dynamic_pressure() * airplane.s_ref * aero["CL"] / 9.81,
        radius_of_gyration_x=0.5 * airplane.b_ref,
        radius_of_gyration_y=3 * airplane.c_ref,
        radius_of_gyration_z=0.5 * airplane.b_ref,
    )

    pprint(
        get_modes(
            airplane=airplane,
            op_point=op_point,
            mass_props=mass_props,
            aero=aero
        )
    )
