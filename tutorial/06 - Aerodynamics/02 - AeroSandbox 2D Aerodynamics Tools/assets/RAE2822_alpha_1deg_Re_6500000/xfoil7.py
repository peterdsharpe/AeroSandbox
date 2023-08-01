import aerosandbox as asb
import aerosandbox.numpy as np

af = asb.Airfoil("rae2822")

xf = asb.XFoil(
    airfoil=af,
    Re=0,
    mach=0.3,
    xfoil_command="/mnt/c/AeroTools/xfoil7.02/bin/xfoil7",
    verbose=True,
    full_potential=True
)

print(xf.alpha(3))

## TODO not finished