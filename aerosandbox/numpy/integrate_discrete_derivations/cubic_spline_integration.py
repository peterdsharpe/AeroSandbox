def x2_to_x3_integral(
        x1,
        x2,
        x3,
        x4,
        f1,
        f2,
        f3,
        f4,
):
    h = x3 - x2
    hm = x2 - x1
    hp = x4 - x3

    q1 = -hm / h
    q4 = 1 + hp / h

    avg_f = (
                               6 * q1 ** 3 * q4 ** 2 * (f2 + f3)
                               - 4 * q1 ** 3 * q4 * (2 * f2 + f3)
                               + 2 * q1 ** 3 * (f2 - f4)
                               - 6 * q1 ** 2 * q4 ** 3 * (f2 + f3)
                               + 3 * q1 ** 2 * q4 * (3 * f2 + f3)
                               + 3 * q1 ** 2 * (-f2 + f4)
                               + 4 * q1 * q4 ** 3 * (2 * f2 + f3)
                               - 3 * q1 * q4 ** 2 * (3 * f2 + f3)
                               + q1 * (f2 - f4)
                               + 2 * q4 ** 3 * (f1 - f2)
                               + 3 * q4 ** 2 * (-f1 + f2)
                               + q4 * (f1 - f2)
                       ) / (
                               12 * q1 * q4 * (q1 - 1) * (q1 - q4) * (q4 - 1)
                       )

    return avg_f * h


def f(x):
    return (
            -0.2 * x ** 3
            + x ** 2
            - x
            + 1
    )


a = 1
b = 3

from scipy import integrate

exact = integrate.quad(
    f,
    a,
    b,
)[0]
print(f"exact: {exact}")

x1 = a - 1
x2 = a
x3 = b
x4 = b + 1
f1 = f(x1)
f2 = f(x2)
f3 = f(x3)
f4 = f(x4)

approx = x2_to_x3_integral(
    x1,
    x2,
    x3,
    x4,
    f1,
    f2,
    f3,
    f4,
)
print(f"approx: {approx}")
