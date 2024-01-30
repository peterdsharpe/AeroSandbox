import sympy as s
from sympy import init_printing

init_printing()

# Gets the integral from x2 to x3, looking at the cubic spline interpolant from x1...x4

# Define the symbols
hm, h, hp = s.symbols("hm h hp", real=True)

x1, x2, x3, x4 = s.symbols('x1 x2 x3 x4', real=True)
# f1, f2, f3, f4 = s.symbols('f1 f2 f3 f4', real=True)

dfm, df, dfp = s.symbols("dfm df dfp", real=True)

f1 = s.symbols("f1", real=True)
f2 = f1 + dfm
f3 = f2 + df
f4 = f3 + dfp


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
    expr = expr.factor([h, hm, hp])
    expr = expr.subs(maps)
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


q = s.symbols('q')  # Normalized space for a Bernstein basis.
# Mapping from x-space to q-space has x=x2 -> q=0, x=x3 -> q=1.

q2 = 0
q3 = 1
q1 = q2 - hm / h
q4 = q3 + hp / h
# q1, q4 = s.symbols('q1 q4', real=True)

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

f1_cubic = f.subs(q, q1)  # .simplify()
f4_cubic = f.subs(q, q4)  # .simplify()

factors = [dfm, df, dfp]
# factors = [ddfm, ddfp]
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
c2_sol = simplify(sol[c2].factor(factors))
c3_sol = simplify(sol[c3].factor(factors))

# f = c1 * b1 + c2_sol * b2 + c3_sol * b3 + c4 * b4

dfdx = f.diff(q) / h
df2dx = dfdx.diff(q) / h

res = s.integrate(df2dx ** 2, (q, q2, q3)) * h
res = simplify(res.factor(factors))
res = simplify(res)

res = simplify(res.subs({c2: c2_sol, c3: c3_sol}).factor(factors))

cse = s.cse(
    res,
    symbols=s.numbered_symbols("s"),
    list=False)

parsimony = len(str(res))
# print(s.pretty(res, num_columns=100))
print(f"Parsimony: {parsimony}")

for i, (var, expr) in enumerate(cse[0]):
    print(f"{var} = {expr}")
print(f"res = {cse[1]}")

if __name__ == '__main__':
    x = s.symbols('x', real=True)

    a = 0
    b = 4


    def example_f(x):
        return x ** 3 + 1


    h_val = b - a
    hm_val = 1
    hp_val = 1
    df_val = example_f(b) - example_f(a)
    dfm_val = example_f(a) - example_f(a - hm_val)
    dfp_val = example_f(b + hp_val) - example_f(b)
    ddfm_val = df_val - dfm_val
    ddfp_val = dfp_val - df_val

    subs = {
        h  : h_val,
        hm : hm_val,
        hp : hp_val,
        df : df_val,
        dfm: dfm_val,
        dfp: dfp_val,
        # ddfm: ddfm_val,
        # ddfp: ddfp_val,
    }

    exact = s.N(
        s.integrate(
            example_f(x).diff(x, 2) ** 2,
            (x, a, b)
        )
    )

    eqn = s.N(
        res.subs(subs)
    )

    print(f"exact: {exact}")
    print(f"eqn: {eqn}")
    print(f"ratio: {exact / eqn}")
