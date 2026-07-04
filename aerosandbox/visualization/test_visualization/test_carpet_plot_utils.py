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


if __name__ == "__main__":
    pytest.main([__file__])
