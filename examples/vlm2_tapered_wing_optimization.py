from aerosandbox import *
import autograd.numpy as np
from autograd import grad
import scipy.optimize as sp_opt


def get_ap(inputs):
    # Gets the AeroProblem object for a given set of inputs
    taper_ratio = inputs[0]
    alpha = inputs[1]

    a = Airplane(
        name="Single Wing",
        xyz_ref=[0, 0, 0],
        wings=[
            Wing(
                name="Wing",
                xyz_le=[0, 0, 0],
                symmetric=True,
                xsecs=[
                    WingXSec(
                        xyz_le=[-0.25 / (taper_ratio + 1), 0, 0],
                        chord=1 / (taper_ratio + 1),
                        # These values are set so that the taper ratio is changed, but the aspect ratio is not.
                        twist=0,
                        airfoil=Airfoil(name="naca0012")
                    ),
                    WingXSec(
                        xyz_le=[-0.25 * taper_ratio / (taper_ratio + 1), 1, 0],
                        chord=taper_ratio / (taper_ratio + 1),
                        # These values are set so that the taper ratio is changed, but the aspect ratio is not.
                        twist=0,
                        airfoil=Airfoil(name="naca0012")
                    )
                ]
            )
        ]
    )
    a.set_ref_dims_from_wing()

    ap = vlm2(
        airplane=a,
        op_point=OperatingPoint(velocity=10,
                                alpha=alpha,
                                beta=0),
    )
    ap.run(verbose=False)

    return ap


def objective_function(inputs):
    ap = get_ap(inputs)
    return ap.CDi


def constraint(inputs):
    ap = get_ap(inputs)
    return ap.CL - 0.5


objective_function_derivative = grad(objective_function)
constraint_derivative = grad(constraint)

# Graph it so you can see the design space
# x = np.linspace(0.01, 0.99, 50)
# y = np.array([objective_function(xi) for xi in x])
# plt.plot(x, y)

# Optimization
initial_guess = [0.25, 3]
# eq_cons = {
#     'type':'eq',
#     'fun': lambda x: constraint(x)
# }
ineq_cons = {
    'type': 'ineq',
    'fun': lambda x: constraint(x),
    'jac': lambda x: constraint_derivative(x)
}

res = sp_opt.minimize(
    fun=objective_function,
    x0=initial_guess,
    method='SLSQP',
    jac=objective_function_derivative,
    constraints=ineq_cons,
    options={
        'disp': True
    },
    bounds=[(0.001, 1), (0, 10)]
)
x_opt = res.x

print(res)
