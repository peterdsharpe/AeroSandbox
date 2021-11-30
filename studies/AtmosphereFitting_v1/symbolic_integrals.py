import sympy as sp

x = sp.symbols("x")

y_exact = sp.exp(-x)

x1, x2 = sp.symbols("x1 x2")

# x1 = 0
# x2 = 1

y1 = sp.exp(-x1)
y2 = sp.exp(-x2)

y_approx = (y2 - y1)/(x2 - x1) * (x - x1) + y1
y_approx = sp.simplify(y_approx)

difference = y_approx - y_exact
difference = sp.simplify(difference)

L1error = sp.integrate(difference,(x,x1,x2))
L1error = sp.simplify(L1error)
# L2error = sp.integrate(difference**2,[x,x1,x2])