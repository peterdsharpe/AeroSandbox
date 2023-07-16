import aerosandbox as asb
import aerosandbox.numpy as np

AR = 2
taper = 0.1

sweep_position_x_over_c = 1

wing = asb.Wing(
    symmetric=True,
    xsecs=[
        asb.WingXSec(
            xyz_le=[
                -sweep_position_x_over_c * taper ** -0.5,
                0,
                0
            ],
            chord=taper ** -0.5,
            airfoil=asb.Airfoil("naca0010")
        ),
        asb.WingXSec(
            xyz_le=[
                -sweep_position_x_over_c * taper ** 0.5,
                AR,
                0
            ],
            chord=taper ** 0.5,
            airfoil=asb.Airfoil("naca0010")
        )
    ]
).subdivide_sections(5)

airplane = asb.Airplane(
    wings=[wing]
)

alphas = np.linspace(0, 90, 500)

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

    polhamus_data_AR = polhamus_data[f'{AR:.1f}']

    plt.plot(
        polhamus_data_AR[:, 0],
        polhamus_data_AR[:, 1],
        ".",
        label="Experiment (Polhamus)"
    )
    plt.xlim(alphas.min(), alphas.max())
    plt.ylim(0, 1.5)

    p.set_ticks(10, 2, 0.1, 0.05)

    p.show_plot(
        f"AeroBuildup for Low-Aspect-Ratio, Swept Wings\n$AR={AR:.1f}$",
        r"Angle of Attack $\alpha$ [deg]",
        r"Lift Coefficient $C_L$ [-]"
    )
