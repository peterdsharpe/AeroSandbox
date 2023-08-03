import aerosandbox as asb
import aerosandbox.numpy as np

af = asb.Airfoil(
    "naca0008",
    CL_function=lambda alpha, Re, mach: 2 * np.pi * alpha,
    CD_function=lambda alpha, Re, mach: np.zeros_like(alpha),
    CM_function=lambda alpha, Re, mach: np.zeros_like(alpha),
)


def assert_equal(a, b, abs=1e-4, rel=1e-4):
    if np.all(np.abs(a - b) < abs):
        return
    if np.all(np.abs(a / b - 1) < rel):
        return
    else:
        raise AssertionError


@np.vectorize
def get_xnps(
        AR=10,
        taper=0.5,
        sweep_deg=0,
        root_chord=10,
):
    print(AR, taper, sweep_deg)
    alpha = 1

    wing = asb.Wing(
        symmetric=True,
        xsecs=[
            asb.WingXSec(
                xyz_le=[
                    -0.25 * root_chord,
                    0,
                    0,
                ],
                chord=root_chord,
                airfoil=af
            ),
            asb.WingXSec(
                xyz_le=[
                    (
                            -0.25 * taper * root_chord +
                            np.tand(sweep_deg) * (AR / 2) * (1 + taper) / 2 * root_chord
                    ),
                    AR / 2 * (1 + taper) / 2 * root_chord,
                    0
                ],
                chord=taper * root_chord,
                airfoil=af
            )
        ]
    )

    assert_equal(AR, wing.aspect_ratio())
    assert_equal(taper, wing.taper_ratio())
    assert_equal(sweep_deg, wing.mean_sweep_angle())
    assert_equal(root_chord, wing.xsecs[0].chord)

    airplane = asb.Airplane(
        wings=[wing],
        c_ref=root_chord,
        b_ref=wing.span(),
        s_ref=wing.area()
    )

    op_point = asb.OperatingPoint(
        velocity=10,
        alpha=alpha
    )

    xyz_ref = [
        -0.01 * root_chord,
        0,
        0
    ]

    vlm_aero = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=op_point,
        xyz_ref=xyz_ref,
        spanwise_spacing='cosine',
        spanwise_resolution=50,
        vortex_core_radius=1e-16
    ).run_with_stability_derivatives(
        alpha=True,
        beta=False,
        p=False,
        q=False,
        r=False
    )
    vlm_xnp = vlm_aero["x_np"]

    ab_aero = asb.AeroBuildup(
        airplane=airplane,
        op_point=op_point,
        xyz_ref=xyz_ref,
    ).run_with_stability_derivatives(
        alpha=True,
        beta=False,
        p=False,
        q=False,
        r=False
    )
    ab_xnp = ab_aero["x_np"]

    return vlm_xnp, ab_xnp, wing


if __name__ == '__main__':

    aspect_ratio = np.geomspace(0.25, 64, 9)
    taper = np.geomspace(0.1, 10, 20)
    sweep_deg = np.linspace(-75, 75, 21)
    root_chord = 10

    Aspect_ratio, Taper, Sweep_deg = np.meshgrid(
        aspect_ratio,
        taper,
        sweep_deg,
    )

    vlm_xnps, ab_xnps, wings = get_xnps(
        AR=Aspect_ratio,
        taper=Taper,
        sweep_deg=Sweep_deg,
        root_chord=root_chord
    )

    import pandas as pd
    df = pd.DataFrame(
        {
            "AR": Aspect_ratio.flatten(),
            "taper": Taper.flatten(),
            "sweep": Sweep_deg.flatten(),
            "root_chord": root_chord * np.ones_like(Aspect_ratio.flatten()),
            "vlm_xnp": vlm_xnps.flatten(),
            "ab_xnp": ab_xnps.flatten(),
            "span": np.vectorize(lambda w: w.span())(wings).flatten(),
            "MAC": np.vectorize(lambda w: w.mean_aerodynamic_chord())(wings).flatten()
        }
    )
    df.to_csv("data.csv",index=False)

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    # fig, ax = plt.subplots()
    # plt.plot(sweep_deg, vlm_xnps, ".-", label="VLM")
    # plt.plot(sweep_deg, ab_xnps, ".-", label="AB")
    # p.show_plot(
    #     "",
    #     "Sweep [deg]",
    #     "$x_{np}$ [m]"
    # )

    cm = plt.cm.get_cmap('rainbow')
    clim = (0, 1)

    fig, ax = plt.subplots()
    for i in range(Aspect_ratio.shape[0]):
        for j in range(Aspect_ratio.shape[1]):
            # for k in range(Aspect_ratio.shape[2]):
            plt.plot(
                Sweep_deg[i, j,  :],
                (ab_xnps - vlm_xnps)[i, j, :] / root_chord,
                ".-",
                color=cm((Taper[i, j, 0] - clim[0]) / (clim[1] - clim[0]))
            )
    p.show_plot(
        "",
        "Sweep",
        "$\Delta x_{\\rm np} / c_{\\rm ref}$ [m]"
    )
