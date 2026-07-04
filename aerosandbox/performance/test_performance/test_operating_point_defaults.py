import pytest

import aerosandbox as asb


def test_default_atmospheres_are_not_shared():
    ### Regression test: the default Atmosphere used to be a single shared
    ### instance evaluated at import time, so mutating one OperatingPoint's
    ### atmosphere silently corrupted all other default-constructed instances.
    op1 = asb.OperatingPoint()
    op2 = asb.OperatingPoint()

    assert op1.atmosphere is not op2.atmosphere

    op1.atmosphere.altitude = 10000
    assert op2.atmosphere.altitude == 0


def test_default_atmosphere_is_sea_level():
    for op in [
        asb.OperatingPoint(),
        asb.OperatingPoint(atmosphere=None),
    ]:
        assert op.atmosphere.altitude == 0
        assert op.atmosphere.pressure() == pytest.approx(101325, rel=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
