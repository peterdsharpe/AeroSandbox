import aerosandbox as asb
import pytest


def test_roundtrip_conversion_similarity():

    for name in [
        "dae11",
        "rae2822",
        "naca0012",
    ]:
        af = asb.Airfoil(name)
        kaf = af.to_kulfan_airfoil()
        kaf_a = kaf.to_airfoil()

        similarity = af.jaccard_similarity(kaf_a)
        print(f"Airfoil: {name}, Similarity: {similarity}")

        assert similarity > 0.995


def test_TE_angle():
    kaf = asb.KulfanAirfoil("naca4412")
    af = kaf.to_airfoil()

    assert kaf.TE_angle() == pytest.approx(af.TE_angle(), rel=0.1)


def test_LE_radius():
    kaf = asb.KulfanAirfoil("naca4412")
    af = kaf.to_airfoil()

    assert kaf.LE_radius() == pytest.approx(af.LE_radius(), rel=0.1)


def test_area():
    kaf = asb.KulfanAirfoil("naca4412")
    af = kaf.to_airfoil()

    assert kaf.area() == pytest.approx(af.area(), rel=0.01)


if __name__ == "__main__":
    test_roundtrip_conversion_similarity()
    pytest.main([__file__])
