from aerosandbox.geometry import *
from aerosandbox.performance import OperatingPoint
import aerosandbox.numpy as np
import aerosandbox.library.aerodynamics as aerolib


def wing_aerodynamics(
        wing: Wing,
        op_point: OperatingPoint,
):
    """
    Estimates the aerodynamic forces, moments, and derivatives on a wing in isolation.

    Assumes:
        * The fuselage is a body of revolution aligned with the x_b axis.
        * The angle between the nose and the freestream is less than 90 degrees.

    Moments are given with the reference at Wing [0, 0, 0].

    Args:

        wing: A Wing object that you wish to analyze.

        op_point: The OperatingPoint that you wish to analyze the fuselage at.

    Returns:

    """
    sweep = wing.mean_sweep_angle()
    AR = wing.aspect_ratio()
    mach = op_point.mach()
    CL_over_Cl = aerolib.CL_over_Cl(
        aspect_ratio=AR,
        mach=mach,
        sweep=sweep
    )
    oswalds_efficiency = aerolib.oswalds_efficiency(
        taper_ratio=wing.taper_ratio(),
        aspect_ratio=AR,
        sweep=sweep,
        fuselage_diameter_to_span_ratio=0  # an assumption
    )

    for section_id in range(len(wing.xsecs) - 1):
        xsec_a = wing.xsecs[section_id]
        xsec_b = wing.xsecs[section_id + 1]
        a_weight = xsec_a.chord / (xsec_a.chord + xsec_b.chord)
        b_weight = xsec_b.chord / (xsec_a.chord + xsec_b.chord)

        xg_local, yg_local, zg_local = wing._compute_frame_of_section(section_id)

        section_normal_w = op_point.convert_axes(
            x_from=zg_local[0], y_from=zg_local[1], z_from=zg_local[2],
            from_axes="geometry",
            to_axes="wind",
        )

        section_alpha_generalized = np.arccosd(section_normal_w[0]) - 90
        print(section_alpha_generalized)

        xsec_a_Cl_incompressible = xsec_a.airfoil.CL_function(
            alpha=section_alpha_generalized,
            Re=op_point.reynolds(xsec_a.chord),
            mach=0,  # Note: this is correct, mach correction happens in 2D -> 3D step
            deflection=xsec_a.control_surface_deflection  # TODO treat symmetry
        )
        xsec_b_Cl_incompressible = xsec_b.airfoil.CL_function(
            alpha=section_alpha_generalized,
            Re=op_point.reynolds(xsec_b.chord),
            mach=0,  # Note: this is correct, mach correction happens in 2D -> 3D step
            deflection=xsec_a.control_surface_deflection
        )

        section_CL_approx = (
                                    xsec_a_Cl_incompressible * a_weight +
                                    xsec_b_Cl_incompressible * b_weight
                            ) * CL_over_Cl

        xsec_a_Cd_profile = xsec_a.airfoil.CD_function(
            alpha=section_alpha_generalized,
            Re=op_point.reynolds(xsec_a.chord),
            mach=mach,
            deflection=xsec_a.control_surface_deflection
        )
        xsec_b_Cd_profile = xsec_b.airfoil.CD_function(
            alpha=section_alpha_generalized,
            Re=op_point.reynolds(xsec_b.chord),
            mach=mach,
            deflection=xsec_a.control_surface_deflection
        )

        section_CDp_approx = (
                xsec_a_Cd_profile * a_weight +
                xsec_b_Cd_profile * b_weight
        )

        section_CDi_approx = (
                section_CL_approx ** 2 / (np.pi * AR * oswalds_efficiency)
        )

        section_CD_approx = section_CDp_approx + section_CDi_approx

        print(section_CL_approx, section_CD_approx)


if __name__ == '__main__':
    wing = Wing(
        xyz_le=[1, 0, 0],
        xsecs=[
            WingXSec(
                xyz_le=[0, 0, 0],
                chord=1,
                airfoil=Airfoil("naca0012"),
                twist=0,
            ),
            WingXSec(
                xyz_le=[0.5, 1, 0],
                chord=0.5,
                airfoil=Airfoil("naca0012"),
                twist=0,
            ),
            WingXSec(
                xyz_le=[0.7, 1, 0.3],
                chord=0.3,
                airfoil=Airfoil("naca0012"),
                twist=0,
            )
        ]
    )
    aero = wing_aerodynamics(
        wing=wing,
        op_point=OperatingPoint(
            velocity=10,
            alpha=10,
            beta=5
        )
    )
