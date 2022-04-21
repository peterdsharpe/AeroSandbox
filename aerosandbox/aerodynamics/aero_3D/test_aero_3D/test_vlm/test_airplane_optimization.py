import aerosandbox as asb
import aerosandbox.numpy as np
import pytest


def test_airplane_optimization():
    def get_aero(alpha, taper_ratio):
        airplane = asb.Airplane(
            wings=[
                asb.Wing(
                    symmetric=True,
                    xsecs=[
                        asb.WingXSec(
                            xyz_le=[-0.25, 0, 0],
                            chord=1,
                        ),
                        asb.WingXSec(
                            xyz_le=[-0.25 * taper_ratio, 1, 0],
                            chord=taper_ratio,
                        )
                    ]
                )
            ]
        )
        op_point = asb.OperatingPoint(
            velocity=1,
            alpha=alpha,
        )

        vlm = asb.VortexLatticeMethod(
            airplane,
            op_point,
            chordwise_resolution=6,
            spanwise_resolution=6,
        )
        return vlm.run()

    # tr = np.linspace(0.01, 1)
    # aeros = np.vectorize(get_aero)(tr)
    # import matplotlib.pyplot as plt
    # import aerosandbox.tools.pretty_plots as p
    # fig, ax = plt.subplots()
    # plt.plot(tr, [a['CL'] / a['CD'] for a in aeros])
    # p.show_plot()

    opti = asb.Opti()
    alpha = opti.variable(0, lower_bound=-90, upper_bound=90)
    taper_ratio = opti.variable(0.7, lower_bound=0.0001, upper_bound=1)
    aero = get_aero(alpha, taper_ratio)
    CD0 = 0.01
    LD = aero["CL"] / (aero["CD"] + CD0)
    opti.minimize(-LD)
    sol = opti.solve()

    assert sol.value(alpha) == pytest.approx(5.5, abs=1)


if __name__ == '__main__':

    pytest.main()
