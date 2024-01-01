import sympy as s
from sympy import init_printing
init_printing()

# Reconstructs a quadratic interpolant from x1...x3, then gets the derivative at x2

# Define the symbols
x1, x2, x3 = s.symbols('x1 x2 x3', real=True)
f1, f2, f3 = s.symbols('f1 f2 f3', real=True)

# hm = x2 - x1
# hp = x3 - x2
hm, hp = s.symbols("hm hp")

q = s.symbols('q')  # Normalized space for a Bernstein basis.
# Mapping from x-space to q-space has x=x2 -> q=0, x=x3 -> q=1.

q1 = 0
# q2 = s.symbols('q2', real=True) # (x2 - x1) / (x3 - x1)
q2 = hm / (hm + hp)
q3 = 1

# Define the Bernstein basis polynomials
b1 = (1 - q) ** 2
b2 = 2 * q * (1 - q)
b3 = q ** 2

c1, c2, c3 = s.symbols('c1 c2 c3', real=True)

# Can solve for c2 and c3 exactly
c1 = f1
c3 = f3

f = c1 * b1 + c2 * b2 + c3 * b3

f2_quadratic = f.subs(q, q2)#.simplify()

factors = [q2]
# factors = [f1, f2, f3, f4]

# Solve for c2 and c3
sol = s.solve(
    [
        f2_quadratic - f2,
    ],
    [
        c2,
    ],
)
c2 = sol[c2].factor(factors).simplify()


f = c1 * b1 + c2 * b2 + c3 * b3
dfdq = f.diff(q).simplify()
# dqdx = 1 / (x3 - x1)
dqdx = 1 / (x3 - x1)
dfdx = dfdq * dqdx

df2dx = (
    dfdx.diff(q) / (x3 - x1)
)

dfm, dfp = s.symbols("dfm dfp")

def simplify(expr):
    import copy
    original_expr = copy.copy(expr)
    expr = expr.subs({
        f3 - f2: dfp,
        f2 - f1: dfm,
        f3 - f1: dfp + dfm,
        x3 - x1: hm + hp,
    })
    expr = expr.subs({
        f3 - f2: dfp,
        f2 - f1: dfm,
        f3 - f1: dfp + dfm,
        x3 - x1: hm + hp,
        f1 - 2 * f2 + f3: dfp - dfm,
    })
    expr = expr.factor([
        hp,
        hm
    ]).simplify()
    if expr != original_expr:
        expr = simplify(expr)
    return expr

dfdx_q1 = simplify(dfdx.subs(q, q1))
dfdx_q2 = simplify(dfdx.subs(q, q2))
dfdx_q3 = simplify(dfdx.subs(q, q3))

df2dx = simplify(df2dx)


# integral = (c1 + c2 + c3) / 3 # God I love Bernstein polynomials



# integral = s.simplify(integral)

parsimony = len(str(df2dx))
print(s.pretty(df2dx, num_columns=100))
print(f"Parsimony: {parsimony}")