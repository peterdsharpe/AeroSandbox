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
    M_w = QS * c / Iyy / u0 * Cma  # Units: 1/(meter * sec)
    M_q = QS * c / Iyy * c / (2 * u0) * Cmq  # Units: 1/sec

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
    from aerosandbox.tools import units as u

    from pprint import pprint

    # Numbers below are from:
    # Caughey, David A., "Introduction to Aircraft Stability and Control, Course Notes for M&AE 5070", 2011
    # https://courses.cit.cornell.edu/mae5070/Caughey_2011_04.pdf
    airplane = asb.Airplane(
        name='Boeing 737-800',
        s_ref=1260 * u.foot ** 2,
        c_ref=11 * u.foot,
        b_ref=113 * u.foot,
    )

    aero = dict(
        CL=1.83443,
        CD=0.13037,
        Cm=0,
        CLa=5.542930,
        Cma=-2.044696,
        CYb=-1.103873,
        Clb=-0.374933,
        Cnb=0.239877,
        CYp=0.800161,
        Clp=-0.449404,
        Cnp=-0.255028,
        CLq=18.973344,
        Cmq=-74.997742,
        CYr=0.796001,
        Clr=0.364638,
        Cnr=-0.434410
    )

    mass_props_TOGW = asb.MassProperties(
        mass=77146,
        x_cg=65.2686 * u.foot,
        y_cg=0.0,
        z_cg=1.16559 * u.foot,
        Ixx=706684,
        Iyy=0.270824e7,
        Izz=0.330763e7,
        Ixy=0.0,
        Iyz=0.0,
        Ixz=26994.4,
    )

    op_point = asb.OperatingPoint(
        atmosphere=asb.Atmosphere(
            altitude=2438.399975619396,
            method='differentiable',
        ),
        velocity=85.64176936131635,
    )

    assert np.allclose(
        aero["CL"],
        mass_props_TOGW.mass * 9.81 / op_point.dynamic_pressure() / airplane.s_ref,
        rtol=0.001
    )

    eigenvalues_from_AVL = {
        'phugoid'        : -0.0171382 + 0.145072j, # Real is wrong (2x)
        'short_period'   : -0.439841 + 0.842195j, # Pretty close
        'roll_subsidence': -1.35132, # get_modes says -1.81
        'dutch_roll'     : -0.385418 + 1.52695j, # Imag is wrong (1.5x)
        'spiral'         : -0.0573017, # Too small, get_modes says -0.17
    }

    pprint(
        get_modes(
            airplane=airplane,
            op_point=op_point,
            mass_props=mass_props_TOGW,
            aero=aero
        )
    )
