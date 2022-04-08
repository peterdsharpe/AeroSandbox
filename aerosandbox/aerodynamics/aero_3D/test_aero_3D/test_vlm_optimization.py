import aerosandbox as asb
import aerosandbox.numpy as np

airplane = asb.Airplane(
    wings=[
        asb.Wing(
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=1,
                ),
                asb.WingXSec(
                    xyz_le=[0, 1, 0],
                    chord=1,
                )
            ]
        )
    ]
)

def LD_from_alpha(alpha):
    op_point = asb.OperatingPoint(
        velocity=1,
        alpha=alpha,
    )

    vlm = asb.VortexLatticeMethod(
        airplane,
        op_point,
    )
    aero = vlm.run()

    CDp = 0.01

    LD = aero["CL"] / (aero["CD"] + CDp)
    return LD

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
alphas = np.linspace(-15, 15, 50)
LDs = np.vectorize(LD_from_alpha)(alphas)
plt.plot(alphas, LDs, ".-")
p.show_plot()

# opti = asb.Opti()
# alpha = opti.variable(init_guess=0)
# LD = LD_from_alpha(alpha)
# opti.minimize(-LD)
# sol = opti.solve()