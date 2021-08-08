import aerosandbox as asb
import pytest
from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.conventional import airplane

def test_vortex_lattice_method():
    analysis = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(),
    )
    return analysis.run()

if __name__ == '__main__':
    test_vortex_lattice_method()
    # pytest.main()