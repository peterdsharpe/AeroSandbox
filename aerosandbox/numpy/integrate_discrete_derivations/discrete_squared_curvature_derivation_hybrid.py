import sympy as s
from sympy import init_printing

init_printing()

def simplify(expr, maxdepth=10, _depth=0):
    import copy
    original_expr = copy.copy(expr)
    print(f"Depth: {_depth} | Parsimony: {len(str(expr))}")
    expr = expr.simplify()
    if expr != original_expr:
        if len(str(expr)) < len(str(original_expr)):
            if _depth < maxdepth:
                return simplify(expr, maxdepth=maxdepth, _depth=_depth + 1)
            else:
                return expr
        else:
            return original_expr
    else:
        return expr
##### Now, a new problem:
##### Do a cubic reconstruction of an interval using values and first derivatives at the endpoints.

# x1, x2 = s.symbols("x1 x2", real=True)
h = s.symbols("h", real=True)

x1 = 0
x2 = h

f1, f2 = s.symbols("f1 f2", real=True)
dfdx1, dfdx2 = s.symbols("dfdx1 dfdx2", real=True)

q = s.symbols("q", real=True)

q1 = 0
q2 = 1
dqdx = 1 / (x2 - x1)

# Define the Bernstein basis polynomials
b1 = (1 - q) ** 3
b2 = 3 * q * (1 - q) ** 2
b3 = 3 * q ** 2 * (1 - q)
b4 = q ** 3

c1, c2, c3, c4 = s.symbols('c1 c2 c3 c4', real=True)

# Can solve for c1 and c4 exactly
c1 = f1
c4 = f2

f = c1 * b1 + c2 * b2 + c3 * b3 + c4 * b4
dfdx = f.diff(q) * dqdx
# Solve for c2 and c3
sol = s.solve(
    [
        dfdx.subs(q, q1) - dfdx1,
        dfdx.subs(q, q2) - dfdx2,
    ],
    [
        c2,
        c3,
    ],
)
c2_sol = simplify(sol[c2])
c3_sol = simplify(sol[c3])

f = c1 * b1 + c2_sol * b2 + c3_sol * b3 + c4 * b4

dfdx = f.diff(q) * dqdx
df2dx = dfdx.diff(q) * dqdx

res = s.integrate(df2dx ** 2, (q, 0, 1)) * h
res = simplify(res)

cse = s.cse(
    res,
    symbols=s.numbered_symbols("s"),
    list=False
)

parsimony = len(str(res))
print("\nSquared Curvature:")
print(f"Parsimony: {parsimony}")

for i, (var, expr) in enumerate(cse[0]):
    print(f"{var} = {expr}")
print(f"res = {cse[1]}")
