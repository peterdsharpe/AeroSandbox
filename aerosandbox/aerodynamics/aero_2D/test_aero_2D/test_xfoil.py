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


### Run-scheduling / ordering logic


def scheduled_values(commands, prefix: str):
    """
    Extracts the numeric values of scheduled runs (e.g., "a 5.0" -> 5.0) from a
    list of keystroke lines.
    """
    return [
        float(line[len(prefix) :])
        for line in commands
        if line.startswith(prefix) and line != "init"
    ]


def test_alpha_order_preserved_when_start_at_is_none():
    """
    The docstring promises that with start_at=None, 'the alpha inputs are run as a
    single sequence in the order given'. Previously, inputs were always sorted first.
    (Regression test.)
    """
    xf = asb.XFoil(airfoil=make_airfoil())
    commands = capture_run_commands(xf)

    xf.alpha([5, 3, 4], start_at=None)

    assert scheduled_values(commands, "a ") == [5, 3, 4]


def test_cl_order_preserved_when_start_at_is_none():
    xf = asb.XFoil(airfoil=make_airfoil())
    commands = capture_run_commands(xf)

    xf.cl([0.9, 0.5, 0.7], start_at=None)

    assert scheduled_values(commands, "cl ") == [0.9, 0.5, 0.7]


def test_alpha_start_at_split_with_list_input():
    xf = asb.XFoil(airfoil=make_airfoil())
    commands = capture_run_commands(xf)

    xf.alpha([-5, 5, -10, 10], start_at=0)

    # Two sweeps diverging from start_at=0: upward first, then (after re-init) downward.
    assert scheduled_values(commands, "a ") == [5, 10, -5, -10]
    assert "init" in commands


def test_alpha_start_at_split_with_unsorted_array_input():
    """
    Points crossing start_at must be split into two diverging sweeps, each run exactly
    once, regardless of the input ordering.
    """
    xf = asb.XFoil(airfoil=make_airfoil())
    commands = capture_run_commands(xf)

    xf.alpha(np.array([5, -5, 10, -10]), start_at=0)

    assert scheduled_values(commands, "a ") == [5, 10, -5, -10]


def test_cl_start_at_split():
    xf = asb.XFoil(airfoil=make_airfoil())
    commands = capture_run_commands(xf)

    xf.cl([-0.5, 0.25, 0.75], start_at=0)

    assert scheduled_values(commands, "cl ") == [0.25, 0.75, -0.5]
    assert "init" in commands


### Process-failure diagnostics


def make_fake_xfoil_executable(tmp_path, body: str):
    """
    Creates a small POSIX shell script that stands in for the XFoil executable.
    The script consumes stdin (the keystrokes), then executes `body`.
    """
    script = tmp_path / "fake_xfoil.sh"
    script.write_text("#!/bin/sh\ncat > /dev/null\n" + body + "\n")
    script.chmod(0o755)
    return str(script)


def test_missing_executable_raises_xfoil_error():
    """
    A nonexistent XFoil executable should raise a descriptive XFoilError, not a raw
    FileNotFoundError. (Regression test.)
    """
    xf = asb.XFoil(
        airfoil=make_airfoil(),
        xfoil_command="nonexistent_xfoil_binary_a8f3b2",
    )
    with pytest.raises(asb.XFoil.XFoilError, match="PATH"):
        xf.alpha(5)


@pytest.mark.skipif(sys.platform == "win32", reason="Uses a POSIX shell script.")
def test_segfault_return_code_raises_descriptive_error(tmp_path):
    """
    A segfault-style return code should produce the curated XFoilError message.
    Previously this diagnostic was dead code (guarded by an except clause that could
    never fire), so users got a generic 'no output file' error. (Regression test.)
    """
    xf = asb.XFoil(
        airfoil=make_airfoil(),
        xfoil_command=make_fake_xfoil_executable(tmp_path, "exit 139"),
    )
    with pytest.raises(asb.XFoil.XFoilError, match="segmentation"):
        xf.alpha(5)


@pytest.mark.skipif(sys.platform == "win32", reason="Uses a POSIX shell script.")
def test_floating_point_exception_return_code_raises_descriptive_error(tmp_path):
    xf = asb.XFoil(
        airfoil=make_airfoil(),
        xfoil_command=make_fake_xfoil_executable(tmp_path, "exit 136"),
    )
    with pytest.raises(asb.XFoil.XFoilError, match="floating point"):
        xf.alpha(5)


### Output-parsing logic


@pytest.mark.skipif(sys.platform == "win32", reason="Uses a POSIX shell script.")
def test_malformed_output_file_raises_xfoil_error(tmp_path):
    """
    An output file with no separator line (e.g., truncated output) should raise the
    intended XFoilError; previously it raised UnboundLocalError instead.
    (Regression test.)
    """
    xf = asb.XFoil(
        airfoil=make_airfoil(),
        xfoil_command=make_fake_xfoil_executable(
            tmp_path,
            "printf 'garbage\\nno separator here\\n' > output.txt",
        ),
    )
    with pytest.raises(asb.XFoil.XFoilError, match="malformed"):
        xf.alpha(5)


FAKE_POLAR = """\
       XFOIL         Version 6.99

 Calculated polar for: NACA 2412

 1 1 Reynolds number fixed          Mach number fixed

 xtrf =   1.000 (top)        1.000 (bottom)
 Mach =   0.000     Re =     1.000 e 6     Ncrit =   9.000

   alpha    CL        CD       CDp       CM      Cpmin    Xcpmin   Chinge   Top_Xtr  Bot_Xtr
  ------ -------- --------- --------- -------- -------- -------- -------- -------- --------
   5.000   0.9200   0.00800   0.00150  -0.0500  -1.5000   0.0300  -0.0100   0.4500   0.9000
   3.000   0.7010   0.00612   0.00098  -0.0525  -1.1000   0.0400  -0.0080   0.5715   0.9457
"""


@pytest.mark.skipif(sys.platform == "win32", reason="Uses a POSIX shell script.")
def test_polar_parsing_without_binary(tmp_path):
    """
    Parses a realistic (fixture) XFoil polar file, without needing the XFoil binary.
    Also checks that results are returned sorted by alpha, as documented.
    """
    polar_file = tmp_path / "polar.txt"
    polar_file.write_text(FAKE_POLAR)

    xf = asb.XFoil(
        airfoil=make_airfoil(),
        xfoil_command=make_fake_xfoil_executable(
            tmp_path,
            f"cp '{polar_file}' output.txt",
        ),
    )
    result = xf.alpha([3, 5])

    assert result["alpha"] == pytest.approx([3, 5])
    assert result["CL"] == pytest.approx([0.7010, 0.9200])
    assert result["CD"] == pytest.approx([0.00612, 0.00800])


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
