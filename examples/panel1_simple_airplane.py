from aerosandbox import *

a = Airplane(
    name="Single Wing",
    xyz_ref=[0, 0, 0],
    wings=[
        Wing(
            name="Wing",
            xyz_le=[0, 0, 0],
            symmetric=True,
            xsecs=[
                WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=0.5,
                    twist=0,
                    airfoil=Airfoil(name="naca4412")
                ),
                WingXSec(
                    xyz_le=[0, 1, 0],
                    chord=0.5,
                    twist=0,
                    airfoil=Airfoil(name="naca4412")
                )
            ]
        )
    ]
)
a.set_ref_dims_from_wing()

a.set_paneling_everywhere(n_chordwise_panels=30, n_spanwise_panels=30)

ap = panel1(
    airplane=a,
    op_point=OperatingPoint(velocity=10,
                            alpha=15,
                            beta=0),
)
ap.run()
ap.draw(
    shading_type="doublet_strengths",
    streamlines_type=np.vstack((
        linspace_3D((-0.05, -0.95, -0.15), (-0.05, -0.95, 0.03), 30),
        linspace_3D((-0.05, 0, -0.15), (-0.05, 0, 0.03), 30),
    )),
    points_type=None
)

# Answer you should get: (XFLR5)
# CL = 0.320
# CDi = 0.008
# CL/CDi = 40.157
# Cm = -0.074
