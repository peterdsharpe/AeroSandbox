from aerosandbox import *
from autograd import grad, elementwise_grad


def performance(design_var):
    a = Airplane(
        name="Conventional",
        xyz_ref=[0.05, 0, 0],
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
                        airfoil=Airfoil(name="naca4412")
                    ),
                    WingXSec(  # Mid
                        xyz_le=[0.01, 0.5, 0],
                        chord=0.16,
                        twist=0,
                        airfoil=Airfoil(name="naca4412")
                    ),
                    WingXSec(  # Tip
                        xyz_le=[0.08, 1, 0.1],
                        chord=design_var,
                        twist=-2,
                        airfoil=Airfoil(name="naca4412")
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
    a.set_ref_dims_from_wing()

    ap = vlm1(
        airplane=a,
        op_point=OperatingPoint(velocity=10,
                                alpha=5,
                                beta=0),
    )
    ap.run()

    performance = ap.CL_over_CDi
    return -np.real(performance)

nominal_value = 0.08
epsilon = 1e-8

# Autograd
gradfunc = grad(performance)
dPdv_auto = gradfunc(nominal_value)

# Finite differencing
dPdv_finite = (performance(nominal_value + epsilon) - performance(nominal_value)) / epsilon


# Complex Step
dPdv_complex = np.imag(performance(nominal_value + epsilon * 1j)) / epsilon

print("")
print("Derivatives:")
print("dPdv_auto = ", dPdv_auto)
print("dPdv_finite = ", dPdv_finite)
print("dPdv_complex = ", dPdv_complex)



# x = np.linspace(0.01,0.06,20)
# y = np.zeros(x.shape)
# for i in range(len(x)):
#     y[i]=performance(x[i])
#
# plt.plot(x,y)
# plt.show()



# import scipy.optimize as sp_opt
# x_opt = sp_opt.minimize(
#     fun = performance,
#     x0 = 0.04,
#     method='SLSQP',
#     )
# print(x_opt)