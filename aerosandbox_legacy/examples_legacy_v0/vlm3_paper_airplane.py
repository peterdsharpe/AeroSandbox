from aerosandbox_legacy_v0 import *

paper_airplane = Airplane(
    name="Conventional",
    xyz_ref=[0, 0, 0], # CG location
    wings=[
        Wing(
            name="Main Wing",
            xyz_le=[0, 0, 0], # Coordinates of the wing's leading edge
            symmetric=True,
            xsecs=[ # The wing's cross ("X") sections
                WingXSec(  # Root
                    xyz_le=[0, 0, 0], # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    chord=0.0254*11,
                    twist=0, # degrees
                    airfoil=Airfoil(name="naca0003"),
                    control_surface_type='symmetric',  # Flap # Control surfaces are applied between a given XSec and the next one.
                    control_surface_deflection=0, # degrees
                    control_surface_hinge_point=0.75 # as chord fraction
                ),
                WingXSec(  # Mid
                    xyz_le=[0.0254*1.349, 0.0254*0.25, 0.0254*0.5],
                    chord=0.0254*9.651,
                    twist=0,
                    airfoil=Airfoil(name="naca0003"),
                    control_surface_type='symmetric',  # Aileron
                    control_surface_deflection=0,
                    control_surface_hinge_point=0.75
                ),
                WingXSec(  # Tip
                    xyz_le=[0.0254*10.260, 0.0254*3.907, 0.0254*1],
                    chord=0.0254*0.740,
                    twist=0,
                    airfoil=Airfoil(name="naca0003"),
                )
            ]
        )
    ]
)
paper_airplane.set_paneling_everywhere(20, 20)

ap = vlm3(
    airplane=paper_airplane,
    op_point=OperatingPoint(
        velocity=10,
        alpha=5,
        beta=0,
        p=0,
        q=0,
        r=0,
    ),
)
ap.run()
ap.draw()

# Answer you should get: (XFLR5)
# CL = 0.797
# CDi = 0.017
# CL/CDi = 47.211
# Cm = -0.184
