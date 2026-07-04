"""
Tests for the AeroSandbox MSES interface that do not require the MSES/MSET/MPLOT binaries.

These tests monkeypatch `subprocess.run` to simulate the external programs, so that the
control-flow logic in MSES.run() can be tested on any machine.
"""

import subprocess

import pytest

import aerosandbox as asb


def make_fake_subprocess_run(
    call_log,
    mset_side_effect=None,
    mses_stdout="",
):
    """
    Returns a callable that fakes `subprocess.run` for MSET / MSES / MPLOT invocations.

    Args:
        call_log: A list, to which the name of each faked program call is appended.

        mset_side_effect: If not None, an exception instance raised when MSET is invoked.

        mses_stdout: The stdout that the faked MSES call should produce.
    """

    def fake_run(command, *args, **kwargs):
        if '"mset"' in command:
            call_log.append("mset")
            if mset_side_effect is not None:
                raise mset_side_effect
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="", stderr=""
            )
        elif '"mses"' in command:
            call_log.append("mses")
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout=mses_stdout, stderr=""
            )
        elif '"mplot"' in command:
            call_log.append("mplot")
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="", stderr=""
            )
        else:
            raise ValueError(f"Unexpected command: {command}")

    return fake_run


def make_mses(**kwargs) -> asb.MSES:
    return asb.MSES(
        airfoil=asb.Airfoil("naca0012"),
        use_xvfb=False,
        verbosity=0,
        **kwargs,
    )


def test_terminate_behavior_stops_sweep_even_when_silent(monkeypatch):
    """
    With behavior_after_unconverged_run="terminate" and verbosity=0, an unconverged run
    must stop the sweep. (Regression test for a `break` that was misindented under a
    verbosity check, causing 'terminate' to be silently ignored when verbosity=0.)
    """
    call_log = []
    monkeypatch.setattr(
        subprocess,
        "run",
        make_fake_subprocess_run(
            call_log,
            mses_stdout="Some MSES output without the convergence marker",
        ),
    )

    ms = make_mses(behavior_after_unconverged_run="terminate")
    result = ms.run(alpha=[1, 2, 3])

    assert call_log.count("mses") == 1  # Should have terminated after the first run
    assert result == {}


def test_no_converged_runs_returns_empty_dict(monkeypatch):
    """
    If zero runs converge, MSES.run() should return an empty dictionary rather than
    raising a confusing KeyError('Ma'). (Regression test.)
    """
    call_log = []
    monkeypatch.setattr(
        subprocess,
        "run",
        make_fake_subprocess_run(
            call_log,
            mses_stdout="Some MSES output without the convergence marker",
        ),
    )

    ms = make_mses(behavior_after_unconverged_run="reinitialize")
    result = ms.run(alpha=[1, 2])

    assert result == {}


def test_mset_failure_is_reraised(monkeypatch):
    """
    A non-X11-related MSET failure must propagate, rather than being silently swallowed
    (which previously let run() continue without a mesh and fail confusingly later).
    """
    call_log = []
    monkeypatch.setattr(
        subprocess,
        "run",
        make_fake_subprocess_run(
            call_log,
            mset_side_effect=subprocess.CalledProcessError(
                returncode=1,
                cmd="mset",
                output="",
                stderr="Some unrelated MSET failure",
            ),
        ),
    )

    ms = make_mses()
    with pytest.raises(subprocess.CalledProcessError):
        ms.run(alpha=3)


def test_mset_x11_failure_raises_descriptive_error(monkeypatch):
    """
    An MSET failure due to a missing X11 display should raise the descriptive
    RuntimeError pointing the user towards Xvfb.
    """
    call_log = []
    monkeypatch.setattr(
        subprocess,
        "run",
        make_fake_subprocess_run(
            call_log,
            mset_side_effect=subprocess.CalledProcessError(
                returncode=1,
                cmd="mset",
                output="",
                stderr="Xlib: BadName (named color or font does not exist)",
            ),
        ),
    )

    ms = make_mses()
    with pytest.raises(RuntimeError, match="X11"):
        ms.run(alpha=3)


if __name__ == "__main__":
    pytest.main([__file__])
