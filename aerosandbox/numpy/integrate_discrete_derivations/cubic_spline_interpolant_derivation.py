import sympy as s
from sympy import init_printing
init_printing()

# Gets the integral from x2 to x3, looking at the cubic spline interpolant from x1...x4

# Define the symbols
x1, x2, x3, x4 = s.symbols('x1 x2 x3 x4', real=True)
f1, f2, f3, f4 = s.symbols('f1 f2 f3 f4', real=True)

h = x3 - x2
hm = x2 - x1
hp = x4 - x3

q = s.symbols('q')  # Normalized space for a Bernstein basis.
# Mapping from x-space to q-space has x=x2 -> q=0, x=x3 -> q=1.

q2 = 0
q3 = 1
# q1 = q2 - hm / h
# q4 = q3 + hp / h
q1, q4 = s.symbols('q1 q4', real=True)

# Define the Bernstein basis polynomials
b1 = (1 - q) ** 3
b2 = 3 * q * (1 - q) ** 2
b3 = 3 * q ** 2 * (1 - q)
b4 = q ** 3

c1, c2, c3, c4 = s.symbols('c1 c2 c3 c4', real=True)

# Can solve for c2 and c3 exactly
c1 = f2
c4 = f3

f = c1 * b1 + c2 * b2 + c3 * b3 + c4 * b4

f1_cubic = f.subs(q, q1)#.simplify()
f4_cubic = f.subs(q, q4)#.simplify()

factors = [q1, q4]
# factors = [f1, f2, f3, f4]

# Solve for c2 and c3
sol = s.solve(
    [
        f1_cubic - f1,
        f4_cubic - f4,
    ],
    [
        c2,
        c3,
    ],
)
c2 = sol[c2].factor(factors).simplify()
c3 = sol[c3].factor(factors).simplify()



integral = (c1 + c2 + c3 + c4) / 4 # God I love Bernstein polynomials
# integral = s.simplify(integral)

integral = integral.factor(factors).simplify()

parsimony = len(str(integral))
print(s.pretty(integral, num_columns=100))
print(f"Parsimony: {parsimony}")