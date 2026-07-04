import pytest

from aerosandbox.tools.code_benchmarking import Timer, time_function


def test_timer_context_manager_binds_self():
    """`with Timer() as t:` should bind the Timer, so t.runtime is readable."""
    with Timer(verbose=False) as t:
        pass

    assert isinstance(t, Timer)
    assert t.runtime >= 0


def test_timer_nested():
    with Timer("a", verbose=False) as a:
        with Timer("b", verbose=False) as b:
            pass

    assert a.runtime >= b.runtime >= 0
    assert Timer.number_running == 0


def test_time_function():
    runtime, result = time_function(lambda: 42, repeats=3)
    assert result == 42
    assert runtime >= 0


if __name__ == "__main__":
    pytest.main([__file__])
