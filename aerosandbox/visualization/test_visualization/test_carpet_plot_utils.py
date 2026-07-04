import signal
import time

import numpy as onp
import pytest

from aerosandbox.visualization.carpet_plot_utils import time_limit, patch_nans

needs_sigalrm = pytest.mark.skipif(
    not hasattr(signal, "SIGALRM"),
    reason="signal.SIGALRM is only available on POSIX systems.",
)


@needs_sigalrm
def test_time_limit_restores_previous_handler():
    def previous_handler(signum, frame):
        pass  # pragma: no cover

    old_handler = signal.signal(signal.SIGALRM, previous_handler)
    try:
        with time_limit(10):
            pass
        assert signal.getsignal(signal.SIGALRM) is previous_handler
    finally:
        signal.signal(signal.SIGALRM, old_handler)


@needs_sigalrm
def test_time_limit_completes_without_timeout():
    with time_limit(10):
        x = 1 + 1
    assert x == 2


@needs_sigalrm
def test_time_limit_accepts_float_and_times_out():
    with pytest.raises(TimeoutError):
        with time_limit(0.1):
            time.sleep(5)


def test_patch_nans_simple_hole():
    array = onp.ones((5, 5))
    array[2, 2] = onp.nan

    patched = patch_nans(array, verbose=False)

    assert not onp.any(onp.isnan(patched))
    assert patched[2, 2] == pytest.approx(1.0)


def test_patch_nans_does_not_mutate_input():
    array = onp.ones((5, 5))
    array[2, 2] = onp.nan

    patch_nans(array, verbose=False)

    assert onp.isnan(array[2, 2])  # The caller's array should be untouched


def test_patch_nans_verbose_flag(capsys):
    array = onp.ones((4, 4))
    array[1, 1] = onp.nan

    patch_nans(array, verbose=False)
    assert capsys.readouterr().out == ""

    patch_nans(array)  # verbose=True is the default
    assert "Bridging" in capsys.readouterr().out


def test_patch_nans_bridging_prefers_first_valid_pair():
    """
    In the bridging stage, candidate neighbor-pairs are listed in priority
    order (orthogonal pairs before diagonal ones), and the first valid pair
    should win. A dead-code `continue` used to let the *last* valid
    (diagonal) pair overwrite the intended value.

    This uses an 'hourglass' of two NaN blobs joined at a pinch cell whose
    horizontal neighbor-pair average (0.0) differs strongly from its
    diagonal pair average (200.0); the wrong bridged value measurably leaks
    through the later diffusion stage. The expected value below was
    computed with the corrected implementation; the pre-fix implementation
    gives 1.2869 at the probed cell.
    """
    blob_h, blob_w = 16, 12
    r, c = blob_h + 2, blob_w + 2  # The pinch cell
    array = onp.zeros((2 * blob_h + 5, blob_w + c + 3))
    array[r - 1, c - 1] = 200.0
    array[r + 1, c + 1] = 200.0
    array[r - blob_h : r, c : c + blob_w] = onp.nan  # Upper blob
    array[r + 1 : r + 1 + blob_h, c - blob_w + 1 : c + 1] = onp.nan  # Lower blob
    array[r, c] = onp.nan  # The pinch cell joining the blobs

    patched = patch_nans(array, verbose=False)

    assert not onp.any(onp.isnan(patched))
    assert patched[25, 8] == pytest.approx(1.1573949105, abs=0.04)


if __name__ == "__main__":
    pytest.main([__file__])
