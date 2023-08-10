import aerosandbox as asb
import aerosandbox.numpy as np
import pytest

wing_airfoil = asb.Airfoil("naca0010")  # asb.Airfoil("sd7037")
tail_airfoil = asb.Airfoil("naca0010")

### Define the 3D geometry you want to analyze/optimize.
# Here, all distances are in meters and all angles are in degrees.
airplane = asb.Airplane(
    name="Peter's Glider",
    xyz_ref=[0.18 * 0.32, 0, 0],  # CG location
    s_ref=0.292,
    c_ref=0.151,
    b_ref=2,
    wings=[
        asb.Wing(
            name="Main Wing",
            symmetric=True,  # Should this wing be mirrored across the XZ plane?
            xsecs=[  # The wing's cross ("X") sections
                asb.WingXSec(  # Root
                    xyz_le=[0, 0, 0],  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    chord=0.18,
                    twist=0,  # degrees
                    airfoil=wing_airfoil,  # Airfoils are blended between a given XSec and the next one.
                ),
                asb.WingXSec(  # Mid
                    xyz_le=[0.01, 0.5, 0],
                    chord=0.16,
                    twist=0,
                    airfoil=wing_airfoil,
                ),
                asb.WingXSec(  # Tip
                    xyz_le=[0.08, 1, 0.1],
                    chord=0.08,
                    twist=-0,
                    airfoil=wing_airfoil,
                ),
            ]
        ),
        asb.Wing(
            name="Horizontal Stabilizer",
            symmetric=True,
            xsecs=[
                asb.WingXSec(  # root
                    xyz_le=[0, 0, 0],
                    chord=0.1,
                    twist=-10,
                    airfoil=tail_airfoil,
                ),
                asb.WingXSec(  # tip
                    xyz_le=[0.02, 0.17, 0],
                    chord=0.08,
                    twist=-10,
                    airfoil=tail_airfoil
                )
            ]
        ).translate([0.6, 0, 0.06]),
        asb.Wing(
            name="Vertical Stabilizer",
            symmetric=False,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=0.1,
                    twist=0,
                    airfoil=tail_airfoil,
                ),
                asb.WingXSec(
                    xyz_le=[0.04, 0, 0.15],
                    chord=0.06,
                    twist=0,
                    airfoil=tail_airfoil
                )
            ]
        ).translate([0.6, 0, 0.07])
    ],
    fuselages=[
        # asb.Fuselage(
        #     name="Fuselage",
        #     xsecs=[
        #         asb.FuselageXSec(
        #             xyz_c=[0.8 * xi - 0.1, 0, 0.1 * xi - 0.03],
        #             radius=0.6 * asb.Airfoil("dae51").local_thickness(x_over_c=xi)
        #         )
        #         for xi in np.cosspace(0, 1, 30)
        #     ]
        # )
    ]
)

op_point = asb.OperatingPoint(
    velocity=100,
    alpha=5,
    beta=0,
)

ab = asb.AeroBuildup(
    airplane,
    op_point,
    xyz_ref=airplane.xyz_ref
).run_with_stability_derivatives()

av = asb.AVL(
    airplane,
    op_point,
    xyz_ref=airplane.xyz_ref
).run()

vl = asb.VortexLatticeMethod(
    airplane,
    op_point,
    xyz_ref=airplane.xyz_ref
).run_with_stability_derivatives()

keys = set()
keys.update(ab.keys())
keys.update(av.keys())
keys = list(keys)
keys.sort()

titles = [
    'Output',
    'AeroBuildup',
    'AVL      ',
    'VLM      ',
    'AB & AVL Significantly Different?'
]


def println(*data):
    print(
        " | ".join([
            d.ljust(len(t)) if isinstance(d, str) else f"{{0:{len(t)}.3g}}".format(d)
            for d, t in zip(data, titles)
        ])
    )


println(*titles)
print("-" * 80)
for k in keys:
    try:
        rel = 0.20
        abs = 0.01

        if 'l' in k or 'm' in k or 'n' in k:
            rel = 0.5
            abs = 0.05

        differences = ab[k] != pytest.approx(av[k], rel=rel, abs=abs)
        differences_text = '*' if differences else ''
        if 'D' in k:
            differences_text = 'Expected'

        println(
            k,
            float(ab[k]),
            float(av[k]),
            float(vl[k]) if k in vl.keys() else ' ' * 5 + '-',
            differences_text
        )

    except (KeyError, TypeError):
        pass
