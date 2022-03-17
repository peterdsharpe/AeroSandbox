import aerosandbox as asb
import aerosandbox.numpy as np
from pathlib import Path
from pprint import pprint

this_dir = Path(__file__).parent

af = asb.Airfoil("naca2412").repanel()

xf = asb.XFoil(
    airfoil=af,
    # Re=1e6,
    # full_potential=True,
    # xfoil_repanel=False,
    # xfoil_command = 'wsl "/mnt/c/AeroTools/drela_tools/xfoil"',
    xfoil_command = 'wsl "/mnt/c/AeroTools/xfoil7.02/bin/xfoil7"',
    # working_directory=str(this_dir / "debug_directory"),
    # verbose=True
)

res = xf.alpha(np.arange(10))
pprint(res)