import tempfile
import aerosandbox as asb


def w(name: str, x0: float, twist: float) -> asb.Wing:
    """
    Creates a symmetric wing with MH60 airfoil sections and control surfaces.

    Args:
        name: Name identifier for the wing component.
        x0: X-axis position offset for the wing leading edge [m].
        twist: Twist angle at the wing tip [deg]. Root twist is 5 degrees less.

    Returns:
        Wing object with two cross-sections (root and tip), positioned at the specified x-offset.
        The wing spans 4 meters (2m each side from symmetry) with chords of 0.5m (root) and 0.4m (tip).
    """
    return asb.Wing(
        name=name,
        xsecs=[
            asb.WingXSec(
                xyz_le=asb.np.array([0, 0, 0]),
                chord=0.5,
                twist=twist - 5,
                airfoil=asb.Airfoil("mh60"),
                control_surfaces=[asb.ControlSurface(symmetric=True)],
            ),
            asb.WingXSec(
                xyz_le=asb.np.array([0, 2, 0]),
                chord=0.4,
                twist=twist,
                airfoil=asb.Airfoil("mh60"),
                control_surfaces=[asb.ControlSurface(symmetric=True)],
            ),
        ],
        symmetric=True,
    ).translate(asb.np.array([x0, 0, 0]))


def f(name: str, x0: float) -> asb.Fuselage:
    """
    Creates a fuselage with a smooth profile that expands and contracts.

    Args:
        name: Name identifier for the fuselage component.
        x0: X-axis position offset for the fuselage start [m].

    Returns:
        Fuselage object with 5 cross-sections spanning 4 meters. The radius profile starts at
        0.1m, expands to 0.4m, maintains that width, then tapers back down through 0.2m to 0.1m.
    """
    return asb.Fuselage(
        name=name,
        xsecs=[
            asb.FuselageXSec(xyz_c=[x0 + x, 0, 0], radius=r)
            for x, r in [(0, 0.1), (1, 0.4), (2, 0.4), (3, 0.2), (4, 0.1)]
        ],
    )


def test_cadquery_export() -> None:
    """
    Tests that airplane geometry can be exported to STEP CAD format with proper component naming.

    Creates an airplane with two wings and a fuselage, exports it to a temporary STEP file,
    and verifies that all component names are preserved in the exported CAD geometry.
    This ensures that the CadQuery export pipeline correctly maintains component identity
    for downstream CAD/CAM workflows.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        fname = f"{tmpdirname}/test.step"
        airplane = asb.Airplane(
            wings=[w("FrontWing", 1, 0), w("BackWing", 3, 5)],
            fuselages=[f("Fuselage", 0)],
        )
        airplane.export_cadquery_geometry(fname, split_leading_edge=False)
        step_file = open(fname).read()
        assert "'FrontWing'" in step_file
        assert "'BackWing'" in step_file
        assert "'Fuselage'" in step_file


if __name__ == "__main__":
    test_cadquery_export()
