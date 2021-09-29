import aerosandbox as asb
import aerosandbox.numpy as np

opti = asb.Opti()

alpha = opti.variable(1, lower_bound=-90, upper_bound=90)
beta = opti.variable(0)

fuselage = asb.Fuselage(
    xsecs=[
        asb.FuselageXSec(
            xyz_c=[xi, 0, 0],
            radius=asb.Airfoil("naca0010").local_thickness(xi)
        )
        for xi in np.cosspace(0, 1, 20)
    ],
)

from aerosandbox.aerodynamics.aero_3D.aero_buildup_submodels import fuselage_aerodynamics

aero = fuselage_aerodynamics(
    fuselage,
    op_point=asb.OperatingPoint(
        velocity=10,
        alpha=alpha,
        beta=beta
    )
)

opti.minimize(-aero["L"] / aero["D"])

sol = opti.solve()
print(sol.value(alpha), sol.value(beta))

