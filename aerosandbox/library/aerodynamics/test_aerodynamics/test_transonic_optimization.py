import aerosandbox as asb
import aerosandbox.numpy as np
import pytest

from aerosandbox.library.aerodynamics import transonic


def test_simple_scalar_optimization():
    opti = asb.Opti()
    mach = opti.variable(init_guess=0.8, )
    CD_induced = 0.1 / mach
    CD_wave = transonic.approximate_CD_wave(
        mach=mach,
        mach_crit=0.7,
        CD_wave_at_fully_supersonic=1,
    )
    CD = CD_induced + CD_wave
    opti.minimize(CD)
    sol = opti.solve()


if __name__ == '__main__':
    pytest.main()
