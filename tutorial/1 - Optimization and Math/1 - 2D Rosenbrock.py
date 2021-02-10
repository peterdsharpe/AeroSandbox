"""
AeroSandbox is fundamentally a tool for solving optimization problems. Therefore, the most important part of
AeroSandbox is the Opti stack, which allows you formulate and solve an optimization problem in natural mathematical
syntax.

The `Opti` class inherits directly from the `Opti` class of CasADi, an underlying tool for algorithmic
differentiation - huge credit goes to the CasADi team for putting together such an elegant and easy-to-use interface.

AeroSandbox's Opti class acts as a wrapper for this interface with syntax tailored specifically for engineering
design, letting users easily implement problem scaling, common transformations, bounds constraints,
freezing variables, categorizing variables, and more. We'll explore more of these features later!

For now, let's show a basic optimization problem by solving the 2D Rosenbrock problem:

The Rosenbrock problem is a classic optimization test case.
https://en.wikipedia.org/wiki/Rosenbrock_function

Mathematically, it is stated as:

    With decision variables x and y:
    Minimize: (a-x)**2 + b*(y-x**2)**2
    for a = 1, b = 100.

"""

import aerosandbox as asb  # This is the standard AeroSandbox import convention

opti = asb.Opti()  # Initialize a new optimization environment; convention is to name it `opti`.

# Define optimization variables
x = opti.variable(init_guess=0)  # You must provide initial guesses.
y = opti.variable(init_guess=0)

# Define objective
f = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2  # You can construct nonlinear functions of variables...
opti.minimize(f)  # ...and then optimize them.

# Optimize
sol = opti.solve()  # This is the conventional syntax to solve the optimization problem.

# Extract values at the optimum
x_opt = sol.value(x) # Evaluates x at the point where the solver converged.
y_opt = sol.value(y)

# Print values
print(f"x = {x_opt}")
print(f"y = {y_opt}")

"""
The solution is found to be (1, 1), which can be shown to be the optimal value via hand calcs.

(To show this, we would calculate the gradient of our objective function, set it to zero, and solve for x and y.)

"""
