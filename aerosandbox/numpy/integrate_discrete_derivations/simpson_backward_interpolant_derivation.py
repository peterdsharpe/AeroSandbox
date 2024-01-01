import sympy as s
from sympy import init_printing
init_printing()

# "Backward" -> gets the integral from x2 to x3, by analogy to backward Euler

# Define the symbols
x1, x2, x3 = s.symbols('x1 x2 x3', real=True)
f1, f2, f3 = s.symbols('f1 f2 f3', real=True)

h = x3 - x2
hm = x2 - x1

q = s.symbols('q')  # Normalized space for a Bernstein basis.
# Mapping from x-space to q-space has x=x2 -> q=0, x=x3 -> q=1.

q1 = s.symbols('q1', real=True)
q2 = 0
q3 = 1

# Define the Bernstein basis polynomials
b1 = (1 - q) ** 2
b2 = 2 * q * (1 - q)
b3 = q ** 2

c1, c2, c3 = s.symbols('c1 c2 c3', real=True)

# Can solve for c2 and c3 exactly
c1 = f2
c3 = f3

f = c1 * b1 + c2 * b2 + c3 * b3

f1_cubic = f.subs(q, q1)#.simplify()

factors = [q1]
# factors = [f1, f2, f3, f4]

# Solve for c2 and c3
sol = s.solve(
    [
        f1_cubic - f1,
    ],
    [
        c2,
    ],
)
c2 = sol[c2].factor(factors).simplify()


f = c1 * b1 + c2 * b2 + c3 * b3

integral = (c1 + c2 + c3) / 3 # God I love Bernstein polynomials
# integral = s.simplify(integral)

integral = integral.factor(factors).simplify()

parsimony = len(str(integral))
print(s.pretty(integral, num_columns=100))
print(f"Parsimony: {parsimony}")