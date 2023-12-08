# from casadi import *
# from casadi.tools import *
#
# #! An SX graph
# a = SX.sym("a")
# b = SX.sym("b")
#
# c = sin(a**5 + b)
#
# c = c - b/ sqrt(fabs(c))
# print(c)
#
# dotdraw(c)
#
# #! An SX
#
# dotdraw(SX.sym("x",Sparsity.lower(3)))
#
# dotdraw(SX.sym("x",Sparsity.lower(3))**2)
#
# #! An MX graph
# x = MX.sym("x",Sparsity.lower(2))
# y = MX.sym("y",Sparsity.lower(2))
#
# z = MX.sym("z",4,2)
#
# zz = x+y
#
# dotdraw(zz)
#
# f = Function("magic", [z,y],[z+x[0,0],x-y])
#
# z,z2 = f(vertcat(x,y),zz.T)
#
# z = z[:2,:] +x + cos(x) - sin(x) / tan(z2)
#
# dotdraw(z)

import aerosandbox as asb
import aerosandbox.numpy as np
import casadi as cas

CL_multipoint_targets = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
CL_multipoint_weights = np.array([5, 6, 7, 8, 9, 10])

Re = 500e3 * (CL_multipoint_targets / 1.25) ** -0.5
mach = 0.03

initial_guess_airfoil = asb.KulfanAirfoil("naca0012")
initial_guess_airfoil.name = "Initial Guess (NACA0012)"

optimized_airfoil = asb.KulfanAirfoil(
    name="Optimized",
    lower_weights=cas.SX.sym("lower_weights", 8, 1),
    upper_weights=cas.SX.sym("upper_weights", 8, 1),
    leading_edge_weight=cas.SX.sym("leading_edge_weight", 1, 1),
    TE_thickness=cas.SX.sym("TE_thickness", 1, 1),
)

alpha = cas.SX.sym("alpha", 1, 1)

aero = optimized_airfoil.get_aero_from_neuralfoil(
    alpha=alpha,
    Re=Re,
    mach=mach,
    # model_size="xxsmall",
    model_size="xxxlarge",
)

from casadi.tools import dotdraw, dotsave, dotgraph

# dotsave(cas.cse(aero["CL"]))

# optimized_airfoil = sol(optimized_airfoil)
# aero = sol(aero)
