import matplotlib.pyplot as plt
import seaborn as sns
import aerosandbox.numpy as np
import aerosandbox.library.aerodynamics as aero


def plot_Cf_flat_plates():
    sns.set(palette=sns.color_palette("husl"))
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
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
    plt.xlabel(r"$Re$")
    plt.ylabel(r"$C_f$")
    plt.ylim(1e-3, 1e-1)
    plt.title(r"Models for Mean Skin Friction Coefficient of Flat Plate")
    plt.tight_layout()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_Cf_flat_plates()
