from aerosandbox import *
from autograd import grad, jacobian


def moments(alphabeta):
    alpha = alphabeta[0]
    beta = alphabeta[1]

    a = Airplane(
        name="Conventional",
        xyz_ref=[0.0, 0, 0],
        wings=[
            Wing(
                name="Main Wing",
                xyz_le=[0, 0, 0],
                symmetric=True,
                xsecs=[
                    WingXSec(  # Root
                        xyz_le=[0, 0, 0],
                        chord=0.18,
                        twist=2,
                        airfoil=Airfoil(name="naca4412"),
                        control_surface_type='symmetric',
                        control_surface_deflection=0,
                        control_surface_hinge_point=0.75
                    ),
                    WingXSec(  # Mid
                        xyz_le=[0.01, 0.5, 0],
                        chord=0.16,
                        twist=0,
                        airfoil=Airfoil(name="naca4412")
                    ),
                    WingXSec(  # Tip
                        xyz_le=[0.08, 1, 0.1],
                        chord=0.08,
                        twist=-2,
                        airfoil=Airfoil(name="naca4412"),
                        control_surface_type='symmetric',
                        control_surface_deflection=0,
                        control_surface_hinge_point=0.75
                    )
                ]
            ),
            Wing(
                name="Horizontal Stabilizer",
                xyz_le=[0.6, 0, 0.1],
                symmetric=True,
                xsecs=[
                    WingXSec(  # root
                        xyz_le=[0, 0, 0],
                        chord=0.1,
                        twist=-10,
                        airfoil=Airfoil(name="naca0012")
                    ),
                    WingXSec(  # tip
                        xyz_le=[0.02, 0.17, 0],
                        chord=0.08,
                        twist=-10,
                        airfoil=Airfoil(name="naca0012")
                    )
                ]
            ),
            Wing(
                name="Vertical Stabilizer",
                xyz_le=[0.6, 0, 0.15],
                symmetric=False,
                xsecs=[
                    WingXSec(
                        xyz_le=[0, 0, 0],
                        chord=0.1,
                        twist=0,
                        airfoil=Airfoil(name="naca0012")
                    ),
                    WingXSec(
                        xyz_le=[0.04, 0, 0.15],
                        chord=0.06,
                        twist=0,
                        airfoil=Airfoil(name="naca0012")
                    )
                ]
            )
        ]
    )

    ap = vlm2(
        airplane=a,
        op_point=OperatingPoint(velocity=10,
                                alpha=alpha,
                                beta=beta),
    )
    ap.run(verbose=False)
    return np.array((ap.Cl, ap.Cm, ap.Cn))

stability_derivatives = jacobian(moments)

alpha_at_trim = 5.0
beta_at_trim = 0.0

stability_derivatives_at_trim = stability_derivatives(np.array((alpha_at_trim, beta_at_trim)))

stability_derivatives_at_trim = stability_derivatives_at_trim * 180/np.pi # put it in units of 1/radian

print("d(Cl)/d(alpha): ", stability_derivatives_at_trim[0,0])
print("d(Cm)/d(alpha): ", stability_derivatives_at_trim[1,0])
print("d(Cn)/d(alpha): ", stability_derivatives_at_trim[2,0])
print("d(Cl)/d(beta): ", stability_derivatives_at_trim[0,1])
print("d(Cm)/d(beta): ", stability_derivatives_at_trim[1,1])
print("d(Cn)/d(beta): ", stability_derivatives_at_trim[2,1])

# print(stability_derivatives_at_trim)