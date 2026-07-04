import re

import numpy as onp
import pytest

import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.geometry.openvsp_io.asb_to_openvsp.fuselage_vspscript_generator import (
    generate_fuselage,
)
from aerosandbox.geometry.openvsp_io.asb_to_openvsp.propulsor_vspscript_generator import (
    generate_propulsor,
)


def _get_parm_vals(script: str, parm: str) -> dict[str, float]:
    """
    Extracts `SetParmVal( fid, "<parm>", "<group>", <value> );` entries from a
    VSPScript string, returning a dict of {group: value}.
    """
    pattern = re.compile(
        r'SetParmVal\(\s*\w+,\s*"' + re.escape(parm) + r'",\s*"(\w+)",\s*([^)]+?)\s*\);'
    )
    return {group: float(value) for group, value in pattern.findall(script)}


def make_fuselage() -> asb.Fuselage:
    return asb.Fuselage(
        name="Fuse",
        xsecs=[
            asb.FuselageXSec(
                xyz_c=[xi, 0, 0.05 * xi**2],
                width=0.5,
                height=0.4,
                shape=2,
            )
            for xi in [0.0, 0.5, 2.0]
        ],
    )


def test_fuselage_loc_percents_at_origin():
    fuse = make_fuselage()
    script = generate_fuselage(fuse, include_main=False)

    x_percents = _get_parm_vals(script, "XLocPercent")
    y_percents = _get_parm_vals(script, "YLocPercent")
    z_percents = _get_parm_vals(script, "ZLocPercent")

    length = 2.0
    for i, xsec in enumerate(fuse.xsecs):
        assert x_percents[f"XSec_{i}"] == pytest.approx(xsec.xyz_c[0] / length)
        assert y_percents[f"XSec_{i}"] == pytest.approx(0.0)
        assert z_percents[f"XSec_{i}"] == pytest.approx(xsec.xyz_c[2] / length)


def test_fuselage_loc_percents_translated():
    """
    The fuselage's overall translation is already applied via the
    X/Y/Z_Rel_Location parameters, so the per-section *LocPercent values must
    be relative to the first cross-section (not absolute), or the offset gets
    double-counted.
    """
    translation = [5.0, 1.0, 2.0]
    fuse = make_fuselage().translate(translation)
    script = generate_fuselage(fuse, include_main=False)

    ### The overall translation should show up in the Rel_Location parameters...
    for axis, offset in zip("XYZ", translation):
        assert f'SetParmVal( fid, "{axis}_Rel_Location", "XForm", {offset} );' in script

    ### ...and therefore must NOT also show up in the *LocPercent parameters.
    script_untranslated = generate_fuselage(make_fuselage(), include_main=False)
    for parm in ["XLocPercent", "YLocPercent", "ZLocPercent"]:
        assert _get_parm_vals(script, parm) == pytest.approx(
            _get_parm_vals(script_untranslated, parm)
        )


def test_fuselage_zero_length_raises():
    fuse = asb.Fuselage(
        name="Disk",
        xsecs=[
            asb.FuselageXSec(xyz_c=[0, 0, 0], width=0.5, height=0.5, shape=2),
            asb.FuselageXSec(xyz_c=[0, 0, 0], width=1.0, height=1.0, shape=2),
        ],
    )
    with pytest.raises(ValueError):
        generate_fuselage(fuse)


def _get_propulsor_rotations(xyz_normal) -> tuple[float, float]:
    """
    Generates the VSPScript for a Propulsor with the given normal, and parses
    out the (y_rot_deg, z_rot_deg) values that were written to the script.
    """
    script = generate_propulsor(
        asb.Propulsor(xyz_normal=np.array(xyz_normal, dtype=float), radius=0.5),
        include_main=False,
    )
    return (
        _get_parm_vals(script, "Y_Rel_Rotation")["XForm"],
        _get_parm_vals(script, "Z_Rel_Rotation")["XForm"],
    )


def _propulsor_normal_from_rotations(y_rot_deg: float, z_rot_deg: float) -> np.ndarray:
    """
    Reconstructs the direction that the propulsor's default normal [-1, 0, 0]
    points after applying the y- and z-rotations written to the VSPScript.
    """
    rot = np.rotation_matrix_3D(
        angle=np.radians(y_rot_deg), axis="y"
    ) @ np.rotation_matrix_3D(angle=np.radians(z_rot_deg), axis="z")
    return rot @ np.array([-1.0, 0.0, 0.0])


def test_propulsor_rotations_align_normal_random_orientations():
    """
    For randomized orientations, the rotations written to the VSPScript must
    map the propulsor's default normal [-1, 0, 0] onto the desired normal.

    This replaces (and checks equivalence with) the previous implementation,
    which solved the same alignment problem with an iterative optimizer; see
    `test_propulsor_rotations_match_optimization_approach` below.
    """
    rng = onp.random.default_rng(42)
    for _ in range(100):
        desired_normal = rng.normal(size=3)
        desired_normal /= onp.linalg.norm(desired_normal)

        y_rot_deg, z_rot_deg = _get_propulsor_rotations(desired_normal)
        achieved_normal = _propulsor_normal_from_rotations(y_rot_deg, z_rot_deg)

        assert achieved_normal == pytest.approx(desired_normal, abs=1e-6)


def test_propulsor_rotations_edge_cases():
    for desired_normal in [
        [-1, 0, 0],  # Default (forward-facing thrust)
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        [1, 1, 1],
        # Aft-facing normals, where the previous optimization-based
        # implementation could converge to a local optimum:
        [0.45, -0.31, 0.84],
        [0.44, 0.32, 0.84],
    ]:
        desired_normal = onp.array(desired_normal, dtype=float)
        desired_normal /= onp.linalg.norm(desired_normal)

        y_rot_deg, z_rot_deg = _get_propulsor_rotations(desired_normal)
        achieved_normal = _propulsor_normal_from_rotations(y_rot_deg, z_rot_deg)

        assert achieved_normal == pytest.approx(desired_normal, abs=1e-6)

    ### The exact [1, 0, 0] case is special-cased in the generator:
    assert _get_propulsor_rotations([1, 0, 0]) == (180.0, 0.0)


def test_propulsor_rotations_match_optimization_approach():
    """
    Checks that the closed-form rotation solution matches the previous
    optimization-based implementation (maximize dot(normal, desired_normal)
    over y- and z-rotations), wherever the optimizer finds the true optimum.
    """
    rng = onp.random.default_rng(0)
    for _ in range(5):
        desired_normal = rng.normal(size=3)
        desired_normal /= onp.linalg.norm(desired_normal)

        ### The previous implementation, reproduced verbatim:
        opti = asb.Opti()
        y_rot = opti.variable(init_guess=0, lower_bound=-np.pi, upper_bound=np.pi)
        z_rot = opti.variable(init_guess=0, lower_bound=-np.pi, upper_bound=np.pi)
        rot = np.rotation_matrix_3D(angle=y_rot, axis="y") @ np.rotation_matrix_3D(
            angle=z_rot, axis="z"
        )
        normal = rot @ np.array([-1, 0, 0])
        opti.maximize(np.dot(normal, desired_normal))
        sol = opti.solve(verbose=False)
        optimized_normal = _propulsor_normal_from_rotations(
            np.degrees(sol(y_rot)), np.degrees(sol(z_rot))
        )
        if not onp.allclose(optimized_normal, desired_normal, atol=1e-6):
            continue  # Optimizer landed on a local optimum; nothing to compare.

        ### Both approaches must orient the propulsor identically:
        closed_form_normal = _propulsor_normal_from_rotations(
            *_get_propulsor_rotations(desired_normal)
        )
        assert closed_form_normal == pytest.approx(optimized_normal, abs=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
