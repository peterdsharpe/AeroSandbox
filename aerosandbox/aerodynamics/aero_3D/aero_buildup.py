from aerosandbox import ExplicitAnalysis
from aerosandbox.geometry import *
from aerosandbox.performance import OperatingPoint
import aerosandbox.library.aerodynamics as aero
import aerosandbox.numpy as np
from aerosandbox.aerodynamics.aero_3D.aero_buildup_submodels.fuselage_aerodynamics_utilities import *
from aerosandbox.library.aerodynamics import transonic
import aerosandbox.library.aerodynamics as aerolib


class AeroBuildup(ExplicitAnalysis):
    """
    A workbook-style aerodynamics buildup.
    """
    default_analysis_specific_options = {
        Airplane: dict(),
        Wing    : dict(
            additional_CL=0,
            additional_CD=0,
            additional_CM=0,
            additional_L=0,
            additional_D=0,
            additional_M=0,
        ),
        WingXSec: dict(
            additional_CL=0,
            additional_CD=0,
            additional_CM=0,
            additional_L=0,
            additional_D=0,
            additional_M=0,
        ),
        Fuselage: dict(
            E_wave_drag=2.5,  # Wave drag efficiency factor

            # Defined by Raymer, "Aircraft Design: A Conceptual Approach", 2nd Ed. Chap. 12.5.9 "Supersonic Parasite Drag".
            # Notated there as "E_WD".
            #
            # Various recommendations:
            #   * For a perfect Sears-Haack body, 1.0
            #   * For a clean aircraft with smooth volume distribution (e.g., BWB), 1.2
            #   * For a "more typical supersonic...", 1.8 - 2.2
            #   * For a "poor supersonic design", 2.5 - 3.0
            #   * The F-15 has E_WD = 2.9.

            nose_fineness_ratio=3,  # Fineness ratio (length / diameter) of the nose section of the fuselage.

            # Impacts wave drag calculations, among other things.

            additional_CL=0,
            additional_CD=0,
            additional_CM=0,
            additional_L=0,
            additional_D=0,
            additional_M=0,
        ),
    }

    def __init__(self,
                 airplane: Airplane,
                 op_point: OperatingPoint,
                 include_wave_drag: bool = True,
                 ):
        ### Initialize
        self.airplane = airplane
        self.op_point = op_point
        self.include_wave_drag = include_wave_drag

    def run(self):
        ### Compute the forces on each component
        aero_components = [
                              self.wing_aerodynamics(wing=wing.translate(-self.airplane.xyz_ref)) for wing in
                              self.airplane.wings
                          ] + [
                              self.fuselage_aerodynamics(fuselage=fuse.translate(-self.airplane.xyz_ref)) for fuse in
                              self.airplane.fuselages
                          ]

        ### Sum up the forces
        aero_total = {}

        for k in aero_components[0].keys():
            values = [
                component[k] for component in aero_components
            ]

            try:
                aero_total[k] = sum(values)
            except TypeError:
                aero_total[k] = [
                    sum([
                        value[i]
                        for value in values
                    ])
                    for i in range(3)
                ]

        ##### Add nondimensional forces, and nondimensional quantities.
        if self.airplane.s_ref is not None:
            qS = self.op_point.dynamic_pressure() * self.airplane.s_ref

            aero_total["CL"] = aero_total["L"] / qS
            aero_total["CY"] = aero_total["Y"] / qS
            aero_total["CD"] = aero_total["D"] / qS
            aero_total["Cl"] = aero_total["l_b"] / qS / self.airplane.b_ref
            aero_total["Cm"] = aero_total["m_b"] / qS / self.airplane.c_ref
            aero_total["Cn"] = aero_total["n_b"] / qS / self.airplane.b_ref

            self.output = aero_total

        return aero_total

    def wing_aerodynamics(self,
                          wing: Wing,
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
        ##### Alias a few things for convenience
        op_point = self.op_point
        wing_options = self.get_options(wing)

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

            def get_deflection(xsec):
                n_surfs = len(xsec.control_surfaces)
                if n_surfs == 0:
                    return 0
                elif n_surfs == 1:
                    surf = xsec.control_surfaces[0]
                    return surf.deflection
                else:
                    raise NotImplementedError(
                        "AeroBuildup currently cannot handle multiple control surfaces attached to a given WingXSec.")

            ##### Compute sectional lift at cross sections using lookup functions. Merge them linearly to get section CL.
            xsec_a_Cl_incompressible = xsec_a.airfoil.CL_function(
                alpha=sect_alpha_generalized,
                Re=op_point.reynolds(xsec_a.chord),
                mach=0,  # Note: this is correct, mach correction happens in 2D -> 3D step
                deflection=get_deflection(xsec_a)
            )
            xsec_b_Cl_incompressible = xsec_b.airfoil.CL_function(
                alpha=sect_alpha_generalized,
                Re=op_point.reynolds(xsec_b.chord),
                mach=0,  # Note: this is correct, mach correction happens in 2D -> 3D step
                deflection=get_deflection(xsec_b)
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
                deflection=get_deflection(xsec_a)
            )
            xsec_b_Cd_profile = xsec_b.airfoil.CD_function(
                alpha=sect_alpha_generalized,
                Re=op_point.reynolds(xsec_b.chord),
                mach=mach,
                deflection=get_deflection(xsec_b)
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

                def get_deflection(xsec):
                    n_surfs = len(xsec.control_surfaces)
                    if n_surfs == 0:
                        return 0
                    elif n_surfs == 1:
                        surf = xsec.control_surfaces[0]
                        return surf.deflection if surf.symmetric else -surf.deflection
                    else:
                        raise NotImplementedError(
                            "AeroBuildup currently cannot handle multiple control surfaces attached to a given WingXSec.")

                ##### Compute sectional lift at cross sections using lookup functions. Merge them linearly to get section CL.
                sym_xsec_a_Cl_incompressible = xsec_a.airfoil.CL_function(
                    alpha=sym_sect_alpha_generalized,
                    Re=op_point.reynolds(xsec_a.chord),
                    mach=0,  # Note: this is correct, mach correction happens in 2D -> 3D step
                    deflection=get_deflection(xsec_a)
                )
                sym_xsec_b_Cl_incompressible = xsec_b.airfoil.CL_function(
                    alpha=sym_sect_alpha_generalized,
                    Re=op_point.reynolds(xsec_b.chord),
                    mach=0,  # Note: this is correct, mach correction happens in 2D -> 3D step
                    deflection=get_deflection(xsec_b)
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
                    deflection=get_deflection(xsec_a)
                )
                sym_xsec_b_Cd_profile = xsec_b.airfoil.CD_function(
                    alpha=sym_sect_alpha_generalized,
                    Re=op_point.reynolds(xsec_b.chord),
                    mach=mach,
                    deflection=get_deflection(xsec_b)
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

    def fuselage_aerodynamics(self,
                              fuselage: Fuselage,
                              ):
        """
        Estimates the aerodynamic forces, moments, and derivatives on a fuselage in isolation.

        Assumes:
            * The fuselage is a body of revolution aligned with the x_b axis.
            * The angle between the nose and the freestream is less than 90 degrees.

        Moments are given with the reference at Fuselage [0, 0, 0].

        Uses methods from Jorgensen, Leland Howard. "Prediction of Static Aerodynamic Characteristics for Slender Bodies
        Alone and with Lifting Surfaces to Very High Angles of Attack". NASA TR R-474. 1977.

        Args:

            fuselage: A Fuselage object that you wish to analyze.

        Returns:

        """
        ##### Alias a few things for convenience
        op_point = self.op_point
        Re = op_point.reynolds(reference_length=fuselage.length())
        fuse_options = self.get_options(fuselage)

        ####### Reference quantities (Set these 1 here, just so we can follow Jorgensen syntax.)
        # Outputs of this function should be invariant of these quantities, if normalization has been done correctly.
        S_ref = 1  # m^2
        c_ref = 1  # m

        ####### Fuselage zero-lift drag estimation

        ### Forebody drag
        C_f_forebody = aerolib.Cf_flat_plate(
            Re_L=Re
        )

        ### Base Drag
        C_D_base = 0.029 / np.sqrt(C_f_forebody) * fuselage.area_base() / S_ref

        ### Skin friction drag
        C_D_skin = C_f_forebody * fuselage.area_wetted() / S_ref

        ### Wave drag
        if self.include_wave_drag:
            sears_haack_drag = transonic.sears_haack_drag_from_volume(
                volume=fuselage.volume(),
                length=fuselage.length()
            )
            C_D_wave = transonic.approximate_CD_wave(
                mach=op_point.mach(),
                mach_crit=critical_mach(
                    fineness_ratio_nose=fuse_options["nose_fineness_ratio"]
                ),
                CD_wave_at_fully_supersonic=fuse_options["E_wave_drag"] * sears_haack_drag,
            )
        else:
            C_D_wave = 0

        ### Total zero-lift drag
        C_D_zero_lift = C_D_skin + C_D_base + C_D_wave

        ####### Jorgensen model

        ### First, merge the alpha and beta into a single "generalized alpha", which represents the degrees between the fuselage axis and the freestream.
        x_w, y_w, z_w = op_point.convert_axes(
            1, 0, 0, from_axes="body", to_axes="wind"
        )
        generalized_alpha = np.arccosd(x_w / (1 + 1e-14))
        sin_generalized_alpha = np.sind(generalized_alpha)
        cos_generalized_alpha = x_w

        # ### Limit generalized alpha to -90 < alpha < 90, for now.
        # generalized_alpha = np.clip(generalized_alpha, -90, 90)
        # # TODO make the drag/moment functions not give negative results for alpha > 90.

        alpha_fractional_component = -z_w / np.sqrt(
            y_w ** 2 + z_w ** 2 + 1e-16)  # The fraction of any "generalized lift" to be in the direction of alpha
        beta_fractional_component = y_w / np.sqrt(
            y_w ** 2 + z_w ** 2 + 1e-16)  # The fraction of any "generalized lift" to be in the direction of beta

        ### Compute normal quantities
        ### Note the (N)ormal, (A)ligned coordinate system. (See Jorgensen for definitions.)
        # M_n = sin_generalized_alpha * op_point.mach()
        Re_n = sin_generalized_alpha * Re
        # V_n = sin_generalized_alpha * op_point.velocity
        q = op_point.dynamic_pressure()
        x_nose = fuselage.xsecs[0].xyz_c[0]
        x_m = 0 - x_nose
        x_c = fuselage.x_centroid_projected() - x_nose

        ##### Potential flow crossflow model
        C_N_p = (  # Normal force coefficient due to potential flow. (Jorgensen Eq. 2.12, part 1)
                fuselage.area_base() / S_ref * np.sind(2 * generalized_alpha) * np.cosd(generalized_alpha / 2)
        )
        C_m_p = (
                (
                        fuselage.volume() - fuselage.area_base() * (fuselage.length() - x_m)
                ) / (
                        S_ref * c_ref
                ) * np.sind(2 * generalized_alpha) * np.cosd(generalized_alpha / 2)
        )

        ##### Viscous crossflow model
        C_d_n = np.where(
            Re_n != 0,
            aerolib.Cd_cylinder(Re_D=Re_n),  # Replace with 1.20 from Jorgensen Table 1 if not working well
            0
        )
        eta = jorgensen_eta(fuselage.fineness_ratio())

        C_N_v = (  # Normal force coefficient due to viscous crossflow. (Jorgensen Eq. 2.12, part 2)
                eta * C_d_n * fuselage.area_projected() / S_ref * sin_generalized_alpha ** 2
        )
        C_m_v = (
                eta * C_d_n * fuselage.area_projected() / S_ref * (x_m - x_c) / c_ref * sin_generalized_alpha ** 2
        )

        ##### Total C_N model
        C_N = C_N_p + C_N_v
        C_m_generalized = C_m_p + C_m_v

        ##### Total C_A model
        C_A = C_D_zero_lift * cos_generalized_alpha * np.abs(cos_generalized_alpha)

        ##### Convert to lift, drag
        C_L_generalized = C_N * cos_generalized_alpha - C_A * sin_generalized_alpha
        C_D = C_N * sin_generalized_alpha + C_A * cos_generalized_alpha

        ### Set proper directions

        C_L = C_L_generalized * alpha_fractional_component
        C_Y = -C_L_generalized * beta_fractional_component
        C_l = 0
        C_m = C_m_generalized * alpha_fractional_component
        C_n = -C_m_generalized * beta_fractional_component

        ### Un-normalize
        L = C_L * q * S_ref
        Y = C_Y * q * S_ref
        D = C_D * q * S_ref
        l_w = C_l * q * S_ref * c_ref
        m_w = C_m * q * S_ref * c_ref
        n_w = C_n * q * S_ref * c_ref

        ### Convert to axes coordinates for reporting
        F_w = (
            -D,
            Y,
            -L
        )
        F_b = op_point.convert_axes(*F_w, from_axes="wind", to_axes="body")
        F_g = op_point.convert_axes(*F_b, from_axes="body", to_axes="geometry")
        M_w = (
            l_w,
            m_w,
            n_w,
        )
        M_b = op_point.convert_axes(*M_w, from_axes="wind", to_axes="body")
        M_g = op_point.convert_axes(*M_b, from_axes="body", to_axes="geometry")

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
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.conventional import airplane

    aero = AeroBuildup(
        airplane=airplane,
        op_point=OperatingPoint(alpha=0, beta=1),
    ).run()

    from aerosandbox.tools.pretty_plots import plt, show_plot, contour, equal, set_ticks

    fig, ax = plt.subplots(2, 2)
    alpha = np.linspace(-10, 10, 1000)
    aero = AeroBuildup(
        airplane=airplane,
        op_point=OperatingPoint(
            velocity=100,
            alpha=alpha,
            beta=0
        ),
    ).run()

    plt.sca(ax[0, 0])
    plt.plot(alpha, aero["CL"])
    plt.xlabel(r"$\alpha$ [deg]")
    plt.ylabel(r"$C_L$")
    set_ticks(5, 1, 0.5, 0.1)

    plt.sca(ax[0, 1])
    plt.plot(alpha, aero["CD"])
    plt.xlabel(r"$\alpha$ [deg]")
    plt.ylabel(r"$C_D$")
    set_ticks(5, 1, 0.05, 0.01)
    plt.ylim(bottom=0)

    plt.sca(ax[1, 0])
    plt.plot(alpha, aero["Cm"])
    plt.xlabel(r"$\alpha$ [deg]")
    plt.ylabel(r"$C_m$")
    set_ticks(5, 1, 0.5, 0.1)

    plt.sca(ax[1, 1])
    plt.plot(alpha, aero["CL"] / aero["CD"])
    plt.xlabel(r"$\alpha$ [deg]")
    plt.ylabel(r"$C_L/C_D$")
    set_ticks(5, 1, 10, 2)

    show_plot(
        "`asb.AeroBuildup` Aircraft Aerodynamics"
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    Beta, Alpha = np.meshgrid(np.linspace(-90, 90, 200), np.linspace(-90, 90, 200))
    aero = AeroBuildup(
        airplane=airplane,
        op_point=OperatingPoint(
            velocity=10,
            alpha=Alpha,
            beta=Beta
        ),
    ).run()
    contour(Beta, Alpha, aero["CL"], levels=30)
    equal()
    show_plot("AeroBuildup", r"$\beta$ [deg]", r"$\alpha$ [deg]")
