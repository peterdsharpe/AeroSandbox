import aerosandbox as asb
import pytest

from aerosandbox.structures.legacy.beams import TubeBeam1


def test_setup_with_bending():
    opti = asb.Opti()
    beam = TubeBeam1(
        opti=opti,
        length=30,
        points_per_point_load=20,
        bending=True,
        torsion=False,
    )
    beam.add_uniform_load(force=1000)
    beam.setup()

    assert hasattr(beam, "stress")
    assert hasattr(beam, "stress_axial")
    assert hasattr(beam, "u")


def test_setup_with_bending_false():
    """
    Regression test: TubeBeam1(bending=False).setup() used to crash with
    `AttributeError: 'TubeBeam1' object has no attribute 'stress_axial'`, because the
    stress constraints (which depend on the bending analysis) were applied
    unconditionally.
    """
    opti = asb.Opti()
    beam = TubeBeam1(
        opti=opti,
        length=30,
        points_per_point_load=20,
        bending=False,
        torsion=True,
    )
    beam.add_uniform_load(force=1000)
    beam.setup()  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__])
