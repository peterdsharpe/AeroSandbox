from aerosandbox.modeling.interpolation import InterpolatedModel
import pytest
import aerosandbox as asb
import aerosandbox.numpy as np


def underlying_function(x):  # Softmax of three linear functions
    def f1(x):
        return -3 * x - 4

    def f2(x):
        return -0.25 * x + 1

    def f3(x):
        return 2 * x - 12

    return np.softmax(
        f1(x),
        f2(x),
        f3(x),
        hardness=1
    )


def get_interpolated_model(res=51):
    x_samples = np.linspace(-10, 10, res)
    f_samples = underlying_function(x_samples)

    return InterpolatedModel(
        x_data_coordinates=x_samples,
        y_data_structured=f_samples
    )


def plot_underlying_function():
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(palette=sns.color_palette("husl"))
    x = np.linspace(-10, 10, 500)
    f = underlying_function(x)

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
    plt.plot(x, f)
    plt.title("Underlying Function")
    plt.show()


def plot_interpolated_model(interpolated_model=get_interpolated_model()):
    interpolated_model.plot()


def test_solve_actual_function():
    opti = asb.Opti()

    x = opti.variable(init_guess=-6)

    opti.minimize(underlying_function(x))

    sol = opti.solve()

    assert sol.value(x) == pytest.approx(4.85358, abs=1e-3)


def test_solve_interpolated_unbounded(
        interpolated_model=get_interpolated_model()
):
    opti = asb.Opti()

    x = opti.variable(init_guess=-5)

    opti.minimize(interpolated_model(x))

    sol = opti.solve()

    assert sol.value(x) == pytest.approx(4.85358, abs=0.1)


# def test_solve_interpolated_infeasible_start_but_bounded(
#         interpolated_model=get_interpolated_model()
# ):
#     opti = asb.Opti()
#
#     x = opti.variable(init_guess=-11, lower_bound=-10, upper_bound=10)
#
#     opti.minimize(interpolated_model(x))
#
#     sol = opti.solve()
#
#     assert sol.value(x) == pytest.approx(4.85358, abs=0.1)


if __name__ == '__main__':
    # plot_underlying_function()
    # plot_interpolated_model(interpolated_model())
    pytest.main()
