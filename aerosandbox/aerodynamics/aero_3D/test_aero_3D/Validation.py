import aerosandbox as asb
import aerosandbox.numpy as np
from pathlib import Path

assets = Path("assets")

af = asb.Airfoil(
    name="n64_1_A612",
    coordinates=assets/ "n64_1_A612.dat"
)
af.generate_polars(
    cache_filename=assets / "n64_1_A612.json"
)

airplane = asb.Airplane(
    name="NACA_RM_A50K27 Wing",
    xyz_ref=[0.58508889, 0, 0],  # CG location
    wings=[
        asb.Wing(
            name="Main Wing",
            symmetric=True,  # Should this wing be mirrored across the XZ plane?
            xsecs=[  # The wing's cross ("X") sections
                asb.WingXSec(  # Root
                    xyz_le=[0.0, 0, 0],  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    chord=0.4120,
                    twist=0.0,  # degrees
                    airfoil=af,  # Airfoils are blended between a given XSec and the next one.
                ),
                asb.WingXSec(
                    xyz_le=[1.1362, 1.5490, 0],
                    chord=0.2060,
                    twist=0,
                    airfoil=af,
                )
            ]
        ).subdivide_sections(10)
    ],
    c_ref=0.32
)

airplane.draw_three_view()

op_point = asb.OperatingPoint(
    atmosphere=asb.Atmosphere(altitude=0),
    velocity=91.3,  # m/s
    alpha=np.linspace(-10, 10, 11)
)

xyz_ref = [0.585, 0, 0]

aerobuildup_aero = asb.AeroBuildup(
    airplane=airplane,
    op_point=op_point,
    xyz_ref=xyz_ref
).run()

vlm_aeros = [
    asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=op,
        xyz_ref=xyz_ref,
        spanwise_resolution=10,
        chordwise_resolution= 10,
    ).run()
    for op in op_point
]

vlm_aero = {}

for k in vlm_aeros[0].keys():
    vlm_aero[k] = np.array([
        aero[k]
        for aero in vlm_aeros
    ])

LL_aeros = [
    asb.NlLiftingLine(
        airplane=airplane,
        op_point=op,
        xyz_ref=xyz_ref,
        spanwise_resolution=4,
        verbose=True
    ).run()
    for op in op_point
]

LL_aero = {}

for k in LL_aeros[0].keys():
    LL_aero[k] = np.array([
        aero[k]
        for aero in LL_aeros
    ])

name_data_paths = {
    # "AVL 3.35 (inviscid)"   : assets / "avl.csv",
    "AVL + XFoil"         : assets / "avl_and_xfoil.csv",
    "Panel + IBL"         : assets / "panel_and_IBL.csv",
    "OpenVSP 3.31.1 Panel": assets / "openvsp_panel.csv",
}

import pandas as pd

name_data = {
    k: pd.read_csv(
        v,

    )
    for k, v in name_data_paths.items()
}

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots(3, 1, figsize=(8, 12), dpi=200)

for name, aero in {
     f"ASB {asb.__version__} AeroBuildup": aerobuildup_aero,
     f"ASB {asb.__version__} VLM (inviscid)": vlm_aero,
     f"ASB {asb.__version__} NL LL": LL_aero
}.items():
    plt.sca(ax[0])

    p.plot_smooth(
        op_point.alpha,
        aero["CL"],
        ".-",
        label=name
    )

    plt.sca(ax[1])

    p.plot_smooth(
        aero["CD"],
        aero["CL"],
        ".-",
        label=name
    )

    plt.sca(ax[2])

    p.plot_smooth(
        aero["CL"],
        aero["Cm"],
        ".-",
        label=name
    )

for name, data in name_data.items():
    plt.sca(ax[0])

    p.plot_smooth(
        data["alpha"],
        data["CL"],
        ".-",
        label=name
    )

    plt.sca(ax[1])


    p.plot_smooth(
        data["CD"],
        data["CL"],
        ".-",
        label=name
    )

    plt.sca(ax[2])

    p.plot_smooth(
        data["CL"],
        data["Cm"],
        ".-",
        label=name
    )

ax[0].set_title("Lift Polar")
ax[0].set_xlabel("Angle of Attack $\\alpha$ [deg]")
ax[0].set_ylabel("Lift Coefficient $C_L$ [-]")
plt.sca(ax[0])
p.set_ticks(2,0.5,0.2,0.05)

ax[1].set_title("Drag Polar")
ax[1].set_xlabel("Drag Coefficient $C_D$ [-]")
ax[1].set_ylabel("Lift Coefficient $C_L$ [-]")
ax[1].set_xlim(left=0)
plt.sca(ax[1])
p.set_ticks(0.01, 0.002, 0.2, 0.05)

ax[2].set_title("Moment Polar")
ax[2].set_xlabel("Lift Coefficient $C_L$ [-]")
ax[2].set_ylabel("Moment Coefficient $C_m$ [-]")
ax[2].set_ylim(top=0.01)  # Keep zero in view
plt.sca(ax[2])
p.set_ticks(0.2, 0.05, 0.02, 0.01)

plt.sca(ax[0])
p.show_plot(savefig="Validation")
