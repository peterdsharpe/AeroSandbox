import sympy as s
from sympy import init_printing

init_printing()

# Gets the integral from x2 to x3, looking at the cubic spline interpolant from x1...x4

# Define the symbols
hm, h, hp = s.symbols("hm h hp", real=True)
dfm, df, dfp = s.symbols("dfm df dfp", real=True)

x1, x2, x3, x4 = s.symbols('x1 x2 x3 x4', real=True)
f1, f2, f3, f4 = s.symbols('f1 f2 f3 f4', real=True)

x = s.symbols('x', real=True)


def simplify(expr, maxdepth=10, _depth=0):
    import copy
    original_expr = copy.copy(expr)
    print(f"Depth: {_depth} | Parsimony: {len(str(expr))}")
    maps = {
        f4 - f3: dfp,
        f3 - f2: df,
        f2 - f1: dfm,
        f4 - f2: dfp + df,
        f3 - f1: df + dfm,
        f4 - f1: dfp + df + dfm,
        x4 - x3: hp,
        x3 - x2: h,
        x2 - x1: hm,
        x4 - x2: hp + h,
        x3 - x1: h + hm,
        x4 - x1: hp + h + hm,
    }
    print("hi1")
    # expr = expr.factor(list(maps.keys()))
    print("hi2")
    expr = expr.subs(maps)
    print("hi3")
    # expr = expr.simplify()
    print("hi4")
    if expr != original_expr:
        if _depth < maxdepth:
            expr = simplify(expr, maxdepth=maxdepth, _depth=_depth + 1)
    return expr


f = (
        f1 * (x - x2) * (x - x3) * (x - x4) / ((x1 - x2) * (x1 - x3) * (x1 - x4)) +
        f2 * (x - x1) * (x - x3) * (x - x4) / ((x2 - x1) * (x2 - x3) * (x2 - x4)) +
        f3 * (x - x1) * (x - x2) * (x - x4) / ((x3 - x1) * (x3 - x2) * (x3 - x4)) +
        f4 * (x - x1) * (x - x2) * (x - x3) / ((x4 - x1) * (x4 - x2) * (x4 - x3))
)
f = simplify(f)

dfdx = f.diff(x)
dfdx = simplify(dfdx)

df2dx = f.diff(x, 2)
df2dx = simplify(df2dx)

res = s.integrate(df2dx ** 2, (x, x2, x3))
res = simplify(res.factor([f1, f2, f3, f4]))

parsimony = len(str(res))
print(s.pretty(res, num_columns=100))
print(f"Parsimony: {parsimony}")

#
# q = s.symbols('q')  # Normalized space for a Bernstein basis.
# # Mapping from x-space to q-space has x=x2 -> q=0, x=x3 -> q=1.
#
# q2 = 0
# q3 = 1
# # q1 = q2 - hm / h
# # q4 = q3 + hp / h
# q1, q4 = s.symbols('q1 q4', real=True)
#
# # Define the Bernstein basis polynomials
# b1 = (1 - q) ** 3
# b2 = 3 * q * (1 - q) ** 2
# b3 = 3 * q ** 2 * (1 - q)
# b4 = q ** 3
#
# c1, c2, c3, c4 = s.symbols('c1 c2 c3 c4', real=True)
#
# # Can solve for c2 and c3 exactly
# c1 = f2
# c4 = f3
#
# f = c1 * b1 + c2 * b2 + c3 * b3 + c4 * b4
#
# f1_cubic = f.subs(q, q1)#.simplify()
# f4_cubic = f.subs(q, q4)#.simplify()
#
# factors = [q1, q4]
# # factors = [f1, f2, f3, f4]
#
# # Solve for c2 and c3
# sol = s.solve(
#     [
#         f1_cubic - f1,
#         f4_cubic - f4,
#     ],
#     [
#         c2,
#         c3,
#     ],
# )
# c2 = sol[c2].factor(factors).simplify()
# c3 = sol[c3].factor(factors).simplify()
#
# f = c1 * b1 + c2 * b2 + c3 * b3 + c4 * b4
#
# integral = (c1 + c2 + c3 + c4) / 4 # God I love Bernstein polynomials
# # integral = s.simplify(integral)
#
# integral = integral.factor(factors).simplify()
#
# parsimony = len(str(integral))
# print(s.pretty(integral, num_columns=100))
# print(f"Parsimony: {parsimony}")
