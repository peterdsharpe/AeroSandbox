import aerosandbox.numpy as np


def haack_series(
        x_over_L: np.ndarray,
        C=1 / 3
):
    theta = np.arccos(1 - 2 * x_over_L)
    radius = ((
                      theta - np.sin(2 * theta) / 2 + C * np.sin(theta) ** 3
              ) / np.pi) ** 0.5
    return radius


def karman(
        x_over_L: np.ndarray,
):
    return haack_series(
        x_over_L=x_over_L,
        C=0
    )


def LV_haack(
        x_over_L: np.ndarray,
):
    return haack_series(
        x_over_L=x_over_L,
        C=1 / 3
    )


def tangent(
        x_over_L: np.ndarray,
):
    return haack_series(
        x_over_L=x_over_L,
        C=2 / 3
    )


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots()

    x_over_L = np.cosspace(0, 1, 2000)
    FR = 3

    plt.plot(x_over_L * FR, karman(x_over_L), label="$C=0$ (LD-Haack, Karman)")
    plt.plot(x_over_L * FR, LV_haack(x_over_L), label="$C=1/3$ (LV-Haack)")
    plt.plot(x_over_L * FR, tangent(x_over_L), label="$C=2/3$ (Tangent)")

    p.equal()
    p.show_plot(
        f"Nosecone Haack Series\nFineness Ratio $FR = {FR}$",
        "Length $x$",
        "Radius $r$"
    )
