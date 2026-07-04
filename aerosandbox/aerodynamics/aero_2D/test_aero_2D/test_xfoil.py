"""
Tests for the AeroSandbox XFoil interface.

Most of these tests exercise the keystroke-generation and output-parsing logic directly,
and hence do NOT require the XFoil binary. Tests that do require the binary are gated
behind a skipif and will skip cleanly if XFoil is not on PATH.
"""

import shutil
import sys

import pytest

import aerosandbox as asb
import aerosandbox.numpy as np

xfoil_present = shutil.which("xfoil") is not None


def make_airfoil() -> asb.Airfoil:
    return asb.Airfoil("naca2412")


def capture_run_commands(xf: asb.XFoil):
    """
    Replaces xf._run_xfoil with a stub that records the keystroke commands it receives,
    so that run-scheduling logic can be tested without the XFoil binary.

    Returns a list; after calling xf.alpha() / xf.cl(), the list contains the individual
    keystroke lines that would have been sent to XFoil.
    """
    captured_lines = []

    def fake_run_xfoil(run_command: str, read_bl_data_from=None):
        captured_lines.clear()
        captured_lines.extend(run_command.split("\n"))
        return {"alpha": np.array([]), "CL": np.array([])}

    xf._run_xfoil = fake_run_xfoil
    return captured_lines


### hinge_point_x handling


def test_keystrokes_with_hinge_point():
    xf = asb.XFoil(airfoil=make_airfoil(), hinge_point_x=0.75)
    keystrokes = xf._default_keystrokes("airfoil.dat", "output.txt")

    assert "hinc" in keystrokes
    assert any(k.startswith("fnew 0.75") for k in keystrokes)
    assert "fmom" in keystrokes


def test_keystrokes_with_hinge_point_none():
    """
    The docstring promises that hinge_point_x=None disables the hinge-moment
    calculation; previously this raised a TypeError instead. (Regression test.)
    """
    xf = asb.XFoil(airfoil=make_airfoil(), hinge_point_x=None)
    keystrokes = xf._default_keystrokes("airfoil.dat", "output.txt")

    assert "hinc" not in keystrokes
    assert not any(k.startswith("fnew") for k in keystrokes)
    assert "fmom" not in keystrokes


def test_alpha_with_hinge_point_none_schedules_no_fmom():
    xf = asb.XFoil(airfoil=make_airfoil(), hinge_point_x=None)
    commands = capture_run_commands(xf)

    xf.alpha(5)

    assert "fmom" not in commands


### Tests requiring the XFoil binary


@pytest.fixture(scope="session")
def functional_xfoil():
    """
    Skips the requesting test if no functional XFoil binary is available on PATH.

    (Some XFoil builds are present on PATH but unable to run - e.g., builds that crash
    with a floating-point exception at startup - so this checks with a smoke test.)
    """
    if not xfoil_present:
        pytest.skip("XFoil is not on PATH.")
    try:
        result = asb.XFoil(airfoil=make_airfoil(), Re=1e6).alpha(3)
    except Exception as e:
        pytest.skip(f"The XFoil binary on PATH is not functional ({type(e).__name__}).")
    if len(result["alpha"]) == 0:
        pytest.skip("The XFoil binary on PATH did not converge a trivial case.")


def test_xfoil_alpha_with_hinge_point_none(functional_xfoil):
    xf = asb.XFoil(airfoil=make_airfoil(), Re=1e6, hinge_point_x=None)
    result = xf.alpha(3)

    assert len(result["alpha"]) == 1
    assert result["alpha"][0] == pytest.approx(3, abs=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
