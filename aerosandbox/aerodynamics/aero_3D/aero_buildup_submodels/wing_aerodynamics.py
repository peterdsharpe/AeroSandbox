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

    Moments are given with the reference at Wing [0, 0, 0].

    Args:

        wing: A Wing object that you wish to analyze.

        op_point: The OperatingPoint that you wish to analyze the fuselage at.

    TODO account for wing airfoil pitching moment

    Returns:

    """
    ##### Compute general wing properties and things to be used in sectional analysis.
    sweep = wing.mean_sweep_angle()
    AR = wing.aspect_ratio()
    mach = op_point.mach()
    q = op_point.dynamic_pressure()
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
    areas = wing.area(_sectional=True)
    aerodynamic_centers = wing.aerodynamic_center(_sectional=True)

    F_g = [0, 0, 0]
    M_g = [0, 0, 0]

    ##### Iterate through the wing sections.
    for sect_id in range(len(wing.xsecs) - 1):

        ##### Identify the wing cross sections adjacent to this wing section.
        xsec_a = wing.xsecs[sect_id]
        xsec_b = wing.xsecs[sect_id + 1]

        ##### When linearly interpolating, weight things by the relative chord.
        a_weight = xsec_a.chord / (xsec_a.chord + xsec_b.chord)
        b_weight = xsec_b.chord / (xsec_a.chord + xsec_b.chord)

        ##### Compute the local frame of this section, and put the z (normal) component into wind axes.
        xg_local, yg_local, zg_local = wing._compute_frame_of_section(sect_id)
        sect_aerodynamic_center = aerodynamic_centers[sect_id]

        sect_z_w = op_point.convert_axes(
            x_from=zg_local[0], y_from=zg_local[1], z_from=zg_local[2],
            from_axes="geometry",
            to_axes="wind",
        )

        ##### Compute the generalized angle of attack, so the geometric alpha that the wing section "sees".
        velocity_vector_b_from_freestream = op_point.convert_axes(
            x_from=-op_point.velocity, y_from=0, z_from=0,
            from_axes="wind",
            to_axes="body"
        )
        velocity_vector_b_from_rotation = np.cross(
            op_point.convert_axes(
                sect_aerodynamic_center[0], sect_aerodynamic_center[1], sect_aerodynamic_center[2],
                from_axes="geometry",
                to_axes="body"
            ),
            [op_point.p, op_point.q, op_point.r],
            manual=True
        )
        velocity_vector_b = [
            velocity_vector_b_from_freestream[i] + velocity_vector_b_from_rotation[i]
            for i in range(3)
        ]
        velocity_mag_b = np.sqrt(sum([comp ** 2 for comp in velocity_vector_b]))
        velocity_dir_b = [
            velocity_vector_b[i] / velocity_mag_b
            for i in range(3)
        ]
        sect_z_b = op_point.convert_axes(
            x_from=zg_local[0], y_from=zg_local[1], z_from=zg_local[2],
            from_axes="geometry",
            to_axes="body",
        )
        vel_dot_normal = np.dot(velocity_dir_b, sect_z_b, manual=True)

        sect_alpha_generalized = 90 - np.arccosd(vel_dot_normal)

        ##### Compute sectional lift at cross sections using lookup functions. Merge them linearly to get section CL.
        xsec_a_Cl_incompressible = xsec_a.airfoil.CL_function(
            alpha=sect_alpha_generalized,
            Re=op_point.reynolds(xsec_a.chord),
            mach=0,  # Note: this is correct, mach correction happens in 2D -> 3D step
            deflection=xsec_a.control_surface_deflection  # TODO treat symmetry
        )
        xsec_b_Cl_incompressible = xsec_b.airfoil.CL_function(
            alpha=sect_alpha_generalized,
            Re=op_point.reynolds(xsec_b.chord),
            mach=0,  # Note: this is correct, mach correction happens in 2D -> 3D step
            deflection=xsec_a.control_surface_deflection
        )
        sect_CL = (
                          xsec_a_Cl_incompressible * a_weight +
                          xsec_b_Cl_incompressible * b_weight
                  ) * CL_over_Cl

        ##### Compute sectional drag at cross sections using lookup functions. Merge them linearly to get section CD.
        xsec_a_Cd_profile = xsec_a.airfoil.CD_function(
            alpha=sect_alpha_generalized,
            Re=op_point.reynolds(xsec_a.chord),
            mach=mach,
            deflection=xsec_a.control_surface_deflection
        )
        xsec_b_Cd_profile = xsec_b.airfoil.CD_function(
            alpha=sect_alpha_generalized,
            Re=op_point.reynolds(xsec_b.chord),
            mach=mach,
            deflection=xsec_a.control_surface_deflection
        )
        sect_CDp = (
                xsec_a_Cd_profile * a_weight +
                xsec_b_Cd_profile * b_weight
        )

        ##### Compute induced drag from local CL and full-wing properties (AR, e)
        sect_CDi = (
                sect_CL ** 2 / (np.pi * AR * oswalds_efficiency)
        )

        ##### Total the drag.
        sect_CD = sect_CDp + sect_CDi

        ##### Go to dimensional quantities using the area.
        area = areas[sect_id]
        sect_L = q * area * sect_CL
        sect_D = q * area * sect_CD

        ##### Compute the direction of the lift by projecting the section's normal vector into the plane orthogonal to the freestream.
        sect_L_direction_w = (
            np.zeros_like(sect_z_w[0]),
            sect_z_w[1] / np.sqrt(sect_z_w[1] ** 2 + sect_z_w[2] ** 2),
            sect_z_w[2] / np.sqrt(sect_z_w[1] ** 2 + sect_z_w[2] ** 2)
        )
        sect_L_direction_g = op_point.convert_axes(
            *sect_L_direction_w, from_axes="wind", to_axes="geometry"
        )

        ##### Compute the direction of the drag by aligning the drag vector with the freestream vector.
        sect_D_direction_w = (-1, 0, 0)
        sect_D_direction_g = op_point.convert_axes(
            *sect_D_direction_w, from_axes="wind", to_axes="geometry"
        )

        ##### Compute the force vector in geometry axes.
        sect_F_g = [
            sect_L * sect_L_direction_g[i] + sect_D * sect_D_direction_g[i]
            for i in range(3)
        ]

        ##### Compute the moment vector in geometry axes.
        sect_M_g = np.cross(
            sect_aerodynamic_center,
            sect_F_g,
            manual=True
        )

        ##### Add section forces and moments to overall forces and moments
        F_g = [
            F_g[i] + sect_F_g[i]
            for i in range(3)
        ]
        M_g = [
            M_g[i] + sect_M_g[i]
            for i in range(3)
        ]

        ##### Treat symmetry
        if wing.symmetric:
            ##### Compute the local frame of this section, and put the z (normal) component into wind axes.

            sym_sect_aerodynamic_center = aerodynamic_centers[sect_id]
            sym_sect_aerodynamic_center[1] *= -1

            sym_sect_z_w = op_point.convert_axes(
                x_from=zg_local[0], y_from=-zg_local[1], z_from=zg_local[2],
                from_axes="geometry",
                to_axes="wind",
            )

            ##### Compute the generalized angle of attack, so the geometric alpha that the wing section "sees".
            sym_velocity_vector_b_from_freestream = op_point.convert_axes(
                x_from=-op_point.velocity, y_from=0, z_from=0,
                from_axes="wind",
                to_axes="body"
            )
            sym_velocity_vector_b_from_rotation = np.cross(
                op_point.convert_axes(
                    sym_sect_aerodynamic_center[0], sym_sect_aerodynamic_center[1], sym_sect_aerodynamic_center[2],
                    from_axes="geometry",
                    to_axes="body"
                ),
                [op_point.p, op_point.q, op_point.r],
                manual=True
            )
            sym_velocity_vector_b = [
                sym_velocity_vector_b_from_freestream[i] + sym_velocity_vector_b_from_rotation[i]
                for i in range(3)
            ]
            sym_velocity_mag_b = np.sqrt(sum([comp ** 2 for comp in sym_velocity_vector_b]))
            sym_velocity_dir_b = [
                sym_velocity_vector_b[i] / sym_velocity_mag_b
                for i in range(3)
            ]
            sym_sect_z_b = op_point.convert_axes(
                x_from=zg_local[0], y_from=-zg_local[1], z_from=zg_local[2],
                from_axes="geometry",
                to_axes="body",
            )
            sym_vel_dot_normal = np.dot(sym_velocity_dir_b, sym_sect_z_b, manual=True)

            sym_sect_alpha_generalized = 90 - np.arccosd(sym_vel_dot_normal)

            ##### Compute sectional lift at cross sections using lookup functions. Merge them linearly to get section CL.
            sym_xsec_a_Cl_incompressible = xsec_a.airfoil.CL_function(
                alpha=sym_sect_alpha_generalized,
                Re=op_point.reynolds(xsec_a.chord),
                mach=0,  # Note: this is correct, mach correction happens in 2D -> 3D step
                deflection=xsec_a.control_surface_deflection * (1 if xsec_a.control_surface_is_symmetric else -1)
            )
            sym_xsec_b_Cl_incompressible = xsec_b.airfoil.CL_function(
                alpha=sym_sect_alpha_generalized,
                Re=op_point.reynolds(xsec_b.chord),
                mach=0,  # Note: this is correct, mach correction happens in 2D -> 3D step
                deflection=xsec_a.control_surface_deflection * (1 if xsec_a.control_surface_is_symmetric else -1)
            )
            sym_sect_CL = (
                                  sym_xsec_a_Cl_incompressible * a_weight +
                                  sym_xsec_b_Cl_incompressible * b_weight
                          ) * CL_over_Cl

            ##### Compute sectional drag at cross sections using lookup functions. Merge them linearly to get section CD.
            sym_xsec_a_Cd_profile = xsec_a.airfoil.CD_function(
                alpha=sym_sect_alpha_generalized,
                Re=op_point.reynolds(xsec_a.chord),
                mach=mach,
                deflection=xsec_a.control_surface_deflection * (1 if xsec_a.control_surface_is_symmetric else -1)
            )
            sym_xsec_b_Cd_profile = xsec_b.airfoil.CD_function(
                alpha=sym_sect_alpha_generalized,
                Re=op_point.reynolds(xsec_b.chord),
                mach=mach,
                deflection=xsec_a.control_surface_deflection * (1 if xsec_a.control_surface_is_symmetric else -1)
            )
            sym_sect_CDp = (
                    sym_xsec_a_Cd_profile * a_weight +
                    sym_xsec_b_Cd_profile * b_weight
            )

            ##### Compute induced drag from local CL and full-wing properties (AR, e)
            sym_sect_CDi = (
                    sym_sect_CL ** 2 / (np.pi * AR * oswalds_efficiency)
            )

            ##### Total the drag.
            sym_sect_CD = sym_sect_CDp + sym_sect_CDi

            ##### Go to dimensional quantities using the area.
            area = areas[sect_id]
            sym_sect_L = q * area * sym_sect_CL
            sym_sect_D = q * area * sym_sect_CD

            ##### Compute the direction of the lift by projecting the section's normal vector into the plane orthogonal to the freestream.
            sym_sect_L_direction_w = (
                np.zeros_like(sym_sect_z_w[0]),
                sym_sect_z_w[1] / np.sqrt(sym_sect_z_w[1] ** 2 + sym_sect_z_w[2] ** 2),
                sym_sect_z_w[2] / np.sqrt(sym_sect_z_w[1] ** 2 + sym_sect_z_w[2] ** 2)
            )
            sym_sect_L_direction_g = op_point.convert_axes(
                *sym_sect_L_direction_w, from_axes="wind", to_axes="geometry"
            )

            ##### Compute the direction of the drag by aligning the drag vector with the freestream vector.
            sym_sect_D_direction_w = (-1, 0, 0)
            sym_sect_D_direction_g = op_point.convert_axes(
                *sym_sect_D_direction_w, from_axes="wind", to_axes="geometry"
            )

            ##### Compute the force vector in geometry axes.
            sym_sect_F_g = [
                sym_sect_L * sym_sect_L_direction_g[i] + sym_sect_D * sym_sect_D_direction_g[i]
                for i in range(3)
            ]

            ##### Compute the moment vector in geometry axes.
            sym_sect_M_g = np.cross(
                sym_sect_aerodynamic_center,
                sym_sect_F_g,
                manual=True
            )

            ##### Add section forces and moments to overall forces and moments
            F_g = [
                F_g[i] + sym_sect_F_g[i]
                for i in range(3)
            ]
            M_g = [
                M_g[i] + sym_sect_M_g[i]
                for i in range(3)
            ]

    ##### Convert F_g and M_g to body and wind axes for reporting.
    F_b = op_point.convert_axes(*F_g, from_axes="geometry", to_axes="body")
    F_w = op_point.convert_axes(*F_b, from_axes="body", to_axes="wind")
    M_b = op_point.convert_axes(*M_g, from_axes="geometry", to_axes="body")
    M_w = op_point.convert_axes(*M_b, from_axes="body", to_axes="wind")

    return {
        "F_g": F_g,
        "F_b": F_b,
        "F_w": F_w,
        "M_g": M_g,
        "M_b": M_b,
        "M_w": M_w,
        "L"  : -F_w[2],
        "Y"  : F_w[1],
        "D"  : -F_w[0],
        "l_b": M_b[0],
        "m_b": M_b[1],
        "n_b": M_b[2]
    }


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
            # WingXSec(
            #     xyz_le=[0.7, 1, 0.3],
            #     chord=0.3,
            #     airfoil=Airfoil("naca0012"),
            #     twist=0,
            # )
        ],
        symmetric=True
    )
    aero = wing_aerodynamics(
        wing=wing,
        op_point=OperatingPoint(
            velocity=10,
            alpha=10,
            beta=0,
            p=1
        )
    )
    print(aero)
