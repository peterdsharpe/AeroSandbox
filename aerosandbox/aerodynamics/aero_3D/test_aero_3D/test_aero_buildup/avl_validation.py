from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.conventional import airplane
import aerosandbox as asb
import aerosandbox.numpy as np
import pytest

for i, wing in enumerate(airplane.wings):
    for j, xsec in enumerate(wing.xsecs):
        af = xsec.airfoil
        af.generate_polars(
            cache_filename=f"cache/{af.name}.json",
            xfoil_kwargs=dict(
                xfoil_repanel="naca" in af.name.lower()
            )
        )
        airplane.wings[i].xsecs[j].airfoil = af

wing = airplane.wings[0]
wing.xsecs=[wing.xsecs[0], wing.xsecs[1]]
airplane.wings = [wing]
airplane.fuselages = []

op_point = asb.OperatingPoint(
    velocity=100,
    alpha=0,
)

ab = asb.AeroBuildup(
    airplane,
    op_point,
).run_with_stability_derivatives()

# from pprint import pprint
#
# pprint(ab)

av = asb.AVL(
    airplane,
    op_point
).run()

keys = set()
keys.update(ab.keys())
keys.update(av.keys())
keys = list(keys)
keys.sort()

title = f"{'Output':10} | {'AeroBuild':10} | {'AVL':10} | Significantly Different?"
print(title)
print("-" * len(title))
for k in keys:
    try:
        is_correct = ab[k] == pytest.approx(av[k], rel=0.75, abs=0.01)
        print(f"{k:10} | {ab[k]:10.4g} | {av[k]:10.4g} | {'*' if not is_correct else ''}")
    except (KeyError, TypeError):
        pass
