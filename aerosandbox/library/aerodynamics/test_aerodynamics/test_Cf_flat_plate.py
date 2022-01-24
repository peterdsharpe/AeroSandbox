import matplotlib.pyplot as plt
import seaborn as sns
import aerosandbox.numpy as np
import aerosandbox.library.aerodynamics as aero


def plot_Cf_flat_plates():
    from aerosandbox.tools.pretty_plots import plt, show_plot

    Res = np.geomspace(1e3, 1e8, 500)
    for method in [
        "blasius",
        "turbulent",
        "hybrid-cengel",
        "hybrid-schlichting",
        "hybrid-sharpe-convex",
        "hybrid-sharpe-nonconvex",
    ]:
        plt.loglog(
            Res,
            aero.Cf_flat_plate(Res, method=method),
            label=method
        )
    plt.ylim(1e-3, 1e-1)
    show_plot(
        "Models for Mean Skin Friction Coefficient of Flat Plate",
        r"$Re$",
        r"$C_f$",
    )


if __name__ == '__main__':
    plot_Cf_flat_plates()
