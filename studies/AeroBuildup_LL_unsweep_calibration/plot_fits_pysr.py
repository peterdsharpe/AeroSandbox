import aerosandbox as asb
import aerosandbox.numpy as np
import pandas as pd
from scipy import interpolate

df = pd.read_csv("data.csv")

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots()
    plt.plot(
        df["sweep"],
        ((df["ab_xnp"] - df["vlm_xnp"]) / (df["MAC"])).values,
        ".k",
        alpha=0.2
    )
    sweep_plot = np.linspace(-90, 90, 500)

    AR = 10
    sweep_rad = np.radians(sweep_plot)
    taper = 0.5

    a = AR / (AR + 2)
    s = sweep_rad
    t = np.exp(-taper)

    pysr_model = """
((((3.557726 ^ (a ^ 2.8443985)) * ((((s * a) + (t * 1.9149417)) + -1.4449639) * s)) + (a + -0.89228547)) * -0.16073418)
    """

    dev = eval(pysr_model.replace("^", "**").replace("\n", ""))

    # dev = a ** 2.2836578 * (
    #         -0.4994786 * s * (s + (t ** (-0.46121752) + 0.6455717) * (t - 0.7528881)) - 0.13291761) + 0.10509591

    # dev = ((((sweep_rad - taper) * -0.30672163) * sweep_rad) * np.tan(ARf))
    # dev = (((((sweep_rad * ARf) - (taper - 0.29707846)) * sweep_rad) * (np.tan(ARf) * -0.35278562)) + (
    #         0.017927283 / ARf))

    interpolator = interpolate.LinearNDInterpolator(
        np.vstack([df["AR"], df["sweep"], df["taper"]]).T,
        ((df["ab_xnp"] - df["vlm_xnp"]) / (df["MAC"])).values,
        rescale=True
        # neighbors=100,
        # kernel='linear',
    )

    plt.plot(
        sweep_plot,
        interpolator(
            np.vstack([
                AR * np.ones_like(sweep_plot),
                sweep_plot,
                taper * np.ones_like(sweep_plot)
            ]).T
        ),
        label="Data"
    )

    plt.plot(
        sweep_plot,
        dev,
        label="Model"
    )
    plt.ylim(-0.5, 0.3)
    p.show_plot()
