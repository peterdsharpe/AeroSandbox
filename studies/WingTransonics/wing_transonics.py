import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.geometry.airfoil.airfoil_families import get_NACA_coordinates


def CL_function(alpha, Re, mach, deflection):
    return 2 * np.pi * np.radians(alpha)


def CD_function(alpha, Re, mach, deflection):
    return 0 * Re ** 0


def CM_function(alpha, Re, mach, deflection):
    return 0 * alpha


ideal_airfoil = asb.Airfoil(
    name="Ideal Airfoil",
    coordinates=get_NACA_coordinates('naca0012'),
    CL_function=CL_function,
    CD_function=CD_function,
    CM_function=CM_function
)

wing = asb.Wing(
    xsecs=[
        asb.WingXSec(
            xyz_le=[0, y_i, 0],
            chord=1,
            airfoil=ideal_airfoil
        )
        for y_i in [0, 10]
    ],
    symmetric=True
)

airplane = asb.Airplane(
    wings=[wing]
)

op_point = asb.OperatingPoint(
    velocity=340,
    alpha=0
)

aero = asb.AeroBuildup(
    airplane=airplane,
    op_point=op_point
).run()

from pprint import pprint

pprint(aero)
