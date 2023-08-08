import aerosandbox as asb
import aerosandbox.numpy as np
import pytest
from typing import Type

def test_wing_aero_3D_matches_2D_in_high_AR_limit():
    airfoil = asb.Airfoil("naca2412")

    wing = asb.Wing(
        xsecs=[
            asb.WingXSec(
                xyz_le=np.array([0, 0, 0]),
                chord=1,
                airfoil=airfoil,
            ),
            asb.WingXSec(
                xyz_le=np.array([0, 0.5e12, 0]),
                chord=1,
                airfoil=airfoil,
            )
        ]
    )

    airplane = asb.Airplane(
        wings=[wing],
    )

    Q = 1000

    op_point = asb.OperatingPoint(
        velocity=(2 * Q / 1.225) ** 0.5,
        alpha=5.,
    )

    xyz_ref = np.array([-1, 0, 0])

    ### Do 2D predictions
    airfoil_only_aero = airfoil.get_aero_from_neuralfoil(
        alpha=op_point.alpha,
        Re=op_point.reynolds(reference_length=1),
        mach=op_point.mach(),
        model_size="large"
    )
    expected_aero = {
        "CL": airfoil_only_aero["CL"],
        "CD": airfoil_only_aero["CD"],
        "Cm": (
                airfoil_only_aero["CM"]
                - (0.25 - xyz_ref[0]) * airfoil_only_aero["CL"]
        ),
    }

    ### Do 3D predictions
    aero = asb.AeroBuildup(
        airplane=airplane,
        op_point=op_point,
        xyz_ref=xyz_ref,
        model_size="large"
    ).run()

    ### Compare
    print(expected_aero)
    print({k: v for k, v in aero.items() if k in expected_aero})

    rtol = 0.05

    assert aero["CL"] == pytest.approx(expected_aero["CL"], rel=rtol)

    # assert aero["CD"] == pytest.approx(expected_aero["CD"], rel=rtol)
    """
    Drag convergence in the high-AR limit is NOT checked, which may appear surprising.
    
    The classical "intro-level" equation for induced drag, C_D_i = C_L^2 / (pi e AR), implies that induced drag goes to
    zero as the aspect ratio goes to infinity. 
    
    However, more advanced text consistently espouse a more nuanced view, where the induced drag actually does *not* 
    go to zero in the high-AR limit. (It does decrease, but the limit behavior is not exactly zero.) This is because 
    the induced drag is a function of the Oswald's efficiency (span efficiency) factor, e, which is itself a function 
    of aspect ratio.
    
    I was initially surprised to learn this and somewhat skeptical, but this view actually appears supported by a 
    rather-large fraction of the world's most elite aerodynamicists. For example:
    
        * Obert, "Aerodynamic Design of Transport Aircraft", 2009
        * Kroo, "Aircraft Design: Synthesis and Analysis", 2001
        * Stinton, "The Design of the Aeroplane", 2001
        * Hoerner, "Fluid-Dynamic Drag", 1965
    
    The models for e (Oswald's efficiency factor) that are given by these aerodynamicists are (partially) a function 
    of aspect ratio, and based on their form, the denominator of the induced drag equation (pi e AR) does not go to 
    zero in the high-AR limit.
    
    For a review paper of this phenomenon, which is counterintuitive based on the classical "intro-level" equation,
    see:
    
        * "Estimating the Oswald Factor from Basic Aircraft Geometrical Parameters" by M. Nita, D. Scholz; Hamburg 
        Univ. of Applied Sciences, 2012. https://www.fzt.haw-hamburg.de/pers/Scholz/OPerA/OPerA_PUB_DLRK_12-09-10.pdf

    """

    assert aero["Cm"] == pytest.approx(expected_aero["Cm"], rel=rtol)


wing = asb.Wing(
    symmetric=True,
    xsecs=[
        asb.WingXSec(
            xyz_le=np.array([0., 0, 0]),
            chord=1.,
            airfoil=asb.Airfoil("naca2412"),
        ),
        asb.WingXSec(
            xyz_le=np.array([0., 5, 0]),
            chord=1.,
            airfoil=asb.Airfoil("naca2412"),
        ),
    ]
)

airplane = asb.Airplane(
    wings=[wing],
)


def test_simple_wing_stability_derivatives(
        AeroAnalysis: Type = asb.AeroBuildup,
):
    analysis = AeroAnalysis(
        airplane=airplane,
        op_point=asb.OperatingPoint(
            velocity=100.,
            alpha=0.,
            beta=5.,
        ),
        xyz_ref=np.array([
            0.,
            0.,
            0.
        ]),
    )

    try:
        aero = analysis.run_with_stability_derivatives()
    except AttributeError:
        aero = analysis.run()

    print(f"Aerodynamic coefficients with {AeroAnalysis.__name__}:")
    for key in ["CL", "CD", "CY", "Cl", "Cm", "Cn", "Cma"]:
        print(f"{key.rjust(10)}: {float(aero[key]):20.4f}")


if __name__ == '__main__':
    test_wing_aero_3D_matches_2D_in_high_AR_limit()
    test_simple_wing_stability_derivatives()
