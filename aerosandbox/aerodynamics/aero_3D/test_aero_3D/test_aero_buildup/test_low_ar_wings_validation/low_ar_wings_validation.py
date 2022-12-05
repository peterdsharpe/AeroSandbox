import aerosandbox as asb
import aerosandbox.numpy as np

AR = 2
taper = 0.1

af = asb.Airfoil("naca0010")
af.generate_polars(
    cache_filename="../cache/naca0010.json"
)

wing = asb.Wing(
    symmetric=True,
    xsecs=[
        asb.WingXSec(
            xyz_le=[
                -0.25 * taper ** -0.5,
                0,
                0
            ],
            chord=taper ** -0.5,
            airfoil=af
        ),
        asb.WingXSec(
            xyz_le=[
                -0.25 * taper ** 0.5,
                AR,
                0
            ],
            chord=taper ** 0.5,
            airfoil=af
        )
    ]
).subdivide_sections(5)

airplane = asb.Airplane(
    wings=[wing]
)

alphas = np.linspace(0, 90, 300)

op_point = asb.OperatingPoint(
    atmosphere=asb.Atmosphere(altitude=0),
    velocity=10,
    alpha=alphas
)

aero = asb.AeroBuildup(
    airplane=airplane,
    op_point=op_point
).run()

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots()
    plt.plot(
        alphas,
        aero["CL"],
        label="AeroBuildup"
    )

    from aerosandbox.tools.webplotdigitizer_reader import read_webplotdigitizer_csv

    polhamus_data = read_webplotdigitizer_csv(
        filename="data/wpd_datasets.csv"
    )

    polhamus_data_AR = polhamus_data['2.0']

    plt.plot(
        polhamus_data_AR[:, 0],
        polhamus_data_AR[:, 1],
        ".",
        label="Experiment (Polhamus)"
    )

    p.set_ticks(10, 2, 0.1, 0.05)

    p.show_plot(
        r"AeroBuildup for Low-Aspect-Ratio, Swept Wings",
        r"Angle of Attack $\alpha$ [deg]",
        r"Lift Coefficient $C_L$ [-]"
    )
