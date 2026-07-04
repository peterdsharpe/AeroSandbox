import re

import pytest

import aerosandbox as asb
from aerosandbox.geometry.openvsp_io.asb_to_openvsp.fuselage_vspscript_generator import (
    generate_fuselage,
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


if __name__ == "__main__":
    pytest.main([__file__])
