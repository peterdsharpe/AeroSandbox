import aerosandbox as asb
import aerosandbox.numpy as np
import pytest

airplane = asb.Airplane(
    wings=[
        asb.Wing(
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=1,
                ),
                asb.WingXSec(
                    xyz_le=[0, 1, 0],
                    chord=1,
                )
            ]
        )
    ]
)


def LD_from_alpha(alpha):
    op_point = asb.OperatingPoint(
        velocity=1,
        alpha=alpha,
    )

    vlm = asb.VortexLatticeMethod(
        airplane,
        op_point,
    )
    aero = vlm.run()

    CD0 = 0.01

    LD = aero["CL"] / (aero["CD"] + CD0)
    return LD


def test_vlm_optimization_operating_point():
    opti = asb.Opti()
    alpha = opti.variable(init_guess=0, lower_bound=-90, upper_bound=90)
    LD = LD_from_alpha(alpha)
    opti.minimize(-LD)
    sol = opti.solve(
        verbose=True,
        # callback=lambda _: print(f"alpha = {opti.debug.value(alpha)}")
    )
    assert sol.value(alpha) == pytest.approx(5.85, abs=0.1)


if __name__ == '__main__':
    LD_from_alpha(6)
    test_vlm_optimization_operating_point()
    # pytest.main()
