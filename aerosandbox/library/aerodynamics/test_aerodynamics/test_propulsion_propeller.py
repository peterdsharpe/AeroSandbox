import os
import subprocess
import sys
from pathlib import Path

import pytest

import aerosandbox.library
from aerosandbox.library.propulsion_propeller import mass_gearbox

LIBRARY_DIR = Path(aerosandbox.library.__file__).parent


def test_mass_gearbox():
    assert mass_gearbox(power=3000, rpm_in=6000, rpm_out=600) == pytest.approx(
        1.315036029805168
    )


def test_demo_block_runs():
    """
    Regression test: the __main__ demo block used style.use('seaborn'), an
    alias removed in matplotlib 3.8 (renamed 'seaborn-v0_8'), so running the
    module as a script crashed with OSError.
    """
    pytest.importorskip("matplotlib")
    env = {**os.environ, "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, str(LIBRARY_DIR / "propulsion_propeller.py")],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    pytest.main([__file__])
