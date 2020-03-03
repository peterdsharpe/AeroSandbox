def first_derivative(xs, ys, mode='centered'):
    if mode == 'forward':
        x = xs[:-1]
        y = ys[:-1]
        x_next = xs[1:]
        y_next = ys[1:]
        derivs = (y_next - y) / (x_next - x)
    elif mode == 'centered':
        x1 = xs[:-2]
        y1 = ys[:-2]
        x2 = xs[1:-1]
        y2 = ys[1:-1]
        x3 = xs[2:]
        y3 = ys[2:]
        derivs = (
                         x1 ** 2 * y2 + x2 ** 2 * y1 - x1 ** 2 * y3 + x3 ** 2 * y1
                         - x2 ** 2 * y3 - x3 ** 2 * y2 - 2 * x1 * x2 * y2 + 2 * x1 * x2 * y3
                         - 2 * x2 * x3 * y1 + 2 * x2 * x3 * y2
                 ) / (
                         (x1 - x2) * (x1 - x3) * (x2 - x3)
                 )
    else:
        raise Exception("Bad input for the ''mode'' argument.")
    return derivs


def second_derivative(xs, ys, mode='centered'):
    if mode == 'centered':
        x1 = xs[:-2]
        y1 = ys[:-2]
        x2 = xs[1:-1]
        y2 = ys[1:-1]
        x3 = xs[2:]
        y3 = ys[2:]
        derivs = -2 * (
                x1 * y2 - x2 * y1 - x1 * y3 + x3 * y1 + x2 * y3 - x3 * y2
        ) / (
                         (x1 - x2) * (x1 - x3) * (x2 - x3)
                 )
    else:
        raise Exception("Bad input for the ''mode'' argument.")
    return derivs


if __name__ == "__main__":
    import numpy as np

    x = np.linspace(0, 1, 11)**2
    y = x ** 2
    dydx = first_derivative(x, y, mode='forward')
    d2ydx2 = second_derivative(x, y)
    import matplotlib.pyplot as plt

    plt.plot(x, y, '.-')
