from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.conventional import airplane
import aerosandbox as asb
import aerosandbox.numpy as np


op_point = asb.OperatingPoint(
    velocity=25,
    alpha=3
)

vlm = asb.VortexLatticeMethod(
    airplane,
    op_point,
    align_trailing_vortices_with_wind=True,
    chordwise_resolution=12,
    spanwise_resolution=12,
)
vlm_aero = vlm.run()

ab = asb.AeroBuildup(
    airplane,
    op_point
)
ab_aero = ab.run()

avl = asb.AVL(
    airplane,
    op_point,
)
avl_aero = avl.run()

for k, v in {
    "VLM": vlm_aero,
    "AVL": avl_aero,
    "AB" : ab_aero
}.items():
    print(f"{k}:")
    for f in ["CL", "CD", "Cm"]:
        print(f"\t{f} : {v[f]}")
    print(f"\tL/D : {v['CL'] / v['CD']}")
