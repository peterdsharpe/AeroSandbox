import pytest
import aerosandbox.numpy as np
from aerosandbox.structures.buckling import plate_buckling_critical_load

"""
Validates `plate_buckling_critical_load()` against the plate-buckling-stress equation from
NACA TN 3781 ("Handbook of Structural Stability, Part I: Buckling of Flat Plates",
Gerard & Becker, 1957):

    sigma_cr = k_c * pi^2 * E / (12 * (1 - nu^2)) * (t/b)^2    [NACA TN 3781, Eq. (1)]

with buckling coefficients for infinitely long plates in compression [NACA TN 3781, Table 7]:

    k_c = 4.00   (simply supported on all edges)
    k_c = 6.98   (clamped on all edges)
    k_c = 0.425  (one unloaded edge simply supported, other free; long-plate limit of Eq. (18) at nu = 0.3)

The critical load is the critical stress times the loaded cross-sectional area (width * thickness).

This is a regression test for a historical bug where the tabulated coefficients already
included the pi^2 / (12 * (1 - nu^2)) factor (at nu = 0.3), which was then multiplied in
again, making results ~9.5% too low.
"""


def expected_critical_load(k_c, width, wall_thickness, elastic_modulus, poissons_ratio):
    sigma_cr = (
        k_c
        * np.pi**2
        * elastic_modulus
        / (12 * (1 - poissons_ratio**2))
        * (wall_thickness / width) ** 2
    )
    return sigma_cr * width * wall_thickness


def test_plate_buckling_pin_pin_handbook_value():
    """
    Hand-computed handbook value:

    E = 70 GPa, nu = 0.3, t = 2 mm, b = 200 mm, simply supported ("pin-pin") sides:

        sigma_cr = 4.00 * pi^2 * 70e9 / (12 * (1 - 0.09)) * (0.002 / 0.2)^2 = 2.5307e7 Pa
        P_cr = sigma_cr * (0.2 * 0.002) = 1.0123e4 N
    """
    load = plate_buckling_critical_load(
        length=1.0,
        width=0.2,
        wall_thickness=0.002,
        elastic_modulus=70e9,
        poissons_ratio=0.3,
        side_boundary_condition_type="pin-pin",
    )
    assert load == pytest.approx(10122.7, rel=1e-4)


@pytest.mark.parametrize(
    "side_boundary_condition_type, k_c",
    [
        ("pin-pin", 4.00),
        ("clamp-clamp", 6.98),
        ("free-free", 0.425),
    ],
)
def test_plate_buckling_matches_naca_tn3781(side_boundary_condition_type, k_c):
    kwargs = dict(
        length=1.0,
        width=0.1,
        wall_thickness=0.001,
        elastic_modulus=228e9,
        poissons_ratio=0.33,
    )
    load = plate_buckling_critical_load(
        side_boundary_condition_type=side_boundary_condition_type, **kwargs
    )
    assert load == pytest.approx(
        expected_critical_load(
            k_c=k_c,
            width=kwargs["width"],
            wall_thickness=kwargs["wall_thickness"],
            elastic_modulus=kwargs["elastic_modulus"],
            poissons_ratio=kwargs["poissons_ratio"],
        ),
        rel=1e-12,
    )


def test_plate_buckling_invalid_boundary_condition_raises():
    with pytest.raises(ValueError):
        plate_buckling_critical_load(
            length=1.0,
            width=0.1,
            wall_thickness=0.001,
            elastic_modulus=70e9,
            side_boundary_condition_type="clamp-pin",
        )


def test_plate_buckling_casadi_backend():
    """
    The function should also work symbolically with the CasADi backend and give
    identical numerics.
    """
    import casadi as cas

    float_load = plate_buckling_critical_load(
        length=1.0,
        width=0.2,
        wall_thickness=0.002,
        elastic_modulus=70e9,
        poissons_ratio=0.3,
    )
    cas_load = plate_buckling_critical_load(
        length=1.0,
        width=0.2,
        wall_thickness=0.002,
        elastic_modulus=cas.MX(70e9),
        poissons_ratio=0.3,
    )
    assert float(cas.evalf(cas_load)) == pytest.approx(float_load, rel=1e-12)


if __name__ == "__main__":
    pytest.main()
