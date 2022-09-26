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

    Example usage:

    >>> import aerosandbox as asb
    >>> ab = asb.AeroBuildup(  # This sets up the analysis, but doesn't execute calculation
    >>>     airplane=my_airplane,  # type: asb.Airplane
    >>>     op_point=my_operating_point,  # type: asb.OperatingPoint
    >>>     xyz_ref=[0.1, 0.2, 0.3],  # Moment reference and center of rotation.
    >>> )
    >>> aero = ab.run()  # This executes the actual aero analysis.
    >>> aero_with_stability_derivs = ab.run_with_stability_derivatives()  # Same, but also gets stability derivatives.

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
                 xyz_ref: List[float] = None,
                 include_wave_drag: bool = True,
                 ):
        super().__init__()

        ### Set defaults
        if xyz_ref is None:
            xyz_ref = airplane.xyz_ref

        ### Initialize
        self.airplane = airplane
        self.op_point = op_point
        self.xyz_ref = xyz_ref
        self.include_wave_drag = include_wave_drag

    def run(self):
        """
        Computes the aerodynamic forces.

        Returns a dictionary with keys:

            'F_g' : an [x, y, z] list of forces in geometry axes [N]

            'F_b' : an [x, y, z] list of forces in body axes [N]

            'F_w' : an [x, y, z] list of forces in wind axes [N]

            'M_g' : an [x, y, z] list of moments about geometry axes [Nm]

            'M_b' : an [x, y, z] list of moments about body axes [Nm]

            'M_w' : an [x, y, z] list of moments about wind axes [Nm]

            'L' : the lift force [N]. Definitionally, this is in wind axes.

            'Y' : the side force [N]. This is in wind axes.

            'D' : the drag force [N]. Definitionally, this is in wind axes.

            'l_b', the rolling moment, in body axes [Nm]. Positive is roll-right.

            'm_b', the pitching moment, in body axes [Nm]. Positive is pitch-up.

            'n_b', the yawing moment, in body axes [Nm]. Positive is nose-right.

            'CL', the lift coefficient [-]. Definitionally, this is in wind axes.

            'CY', the sideforce coefficient [-]. This is in wind axes.

            'CD', the drag coefficient [-]. Definitionally, this is in wind axes.

            'Cl', the rolling coefficient [-], in body axes

            'Cm', the pitching coefficient [-], in body axes

            'Cn', the yawing coefficient [-], in body axes

        Nondimensional values are nondimensionalized using reference values in the AeroBuildup.airplane object.
        """

        ### Compute the forces on each component
        aero_components = [
                              self.wing_aerodynamics(wing=wing) for wing in
                              self.airplane.wings
                          ] + [
                              self.fuselage_aerodynamics(fuselage=fuse) for fuse in
                              self.airplane.fuselages
                          ]

        ### Sum up the forces
        aero_total = {
            "F_g": [0., 0., 0.],
            "F_b": [0., 0., 0.],
            "F_w": [0., 0., 0.],
            "M_g": [0., 0., 0.],
            "M_b": [0., 0., 0.],
            "M_w": [0., 0., 0.],
            "L"  : 0.,
            "Y"  : 0.,
            "D"  : 0.,
            "l_b": 0.,
            "m_b": 0.,
            "n_b": 0.,
        }

        for k in aero_total.keys():
            for aero_component in aero_components:
                if isinstance(aero_total[k], list):
                    aero_total[k] = [
                        aero_total[k][i] + aero_component[k][i]
                        for i in range(3)
                    ]
                else:
                    aero_total[k] = aero_total[k] + aero_component[k]

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

    def run_with_stability_derivatives(self,
                                       alpha=True,
                                       beta=True,
                                       p=True,
                                       q=True,
                                       r=True,
                                       ):
        abbreviations = {
            "alpha": "a",
            "beta" : "b",
            "p"    : "p",
            "q"    : "q",
            "r"    : "r",
        }
        finite_difference_amounts = {
            "alpha": 0.001,
            "beta" : 0.001,
            "p"    : 0.001 * (2 * self.op_point.velocity) / self.airplane.b_ref,
            "q"    : 0.001 * (2 * self.op_point.velocity) / self.airplane.c_ref,
            "r"    : 0.001 * (2 * self.op_point.velocity) / self.airplane.b_ref,
        }
        scaling_factors = {
            "alpha": np.degrees(1),
            "beta" : np.degrees(1),
            "p"    : (2 * self.op_point.velocity) / self.airplane.b_ref,
            "q"    : (2 * self.op_point.velocity) / self.airplane.c_ref,
            "r"    : (2 * self.op_point.velocity) / self.airplane.b_ref,
        }

        original_op_point = self.op_point

        # Compute the point analysis, which returns a dictionary that we will later add key:value pairs to.
        run_base = self.run()

        # Note for the loops below: here, "derivative numerator" and "... denominator" refer to the quantity being
        # differentiated and the variable of differentiation, respectively. In other words, in the expression df/dx,
        # the "numerator" is f, and the "denominator" is x. I realize that this would make a mathematician cry (as a
        # partial derivative is not a fraction), but the reality is that there seems to be no commonly-accepted name
        # for these terms. (Curiously, this contrasts with integration, where there is an "integrand" and a "variable
        # of integration".)

        for derivative_denominator in abbreviations.keys():
            if not locals()[derivative_denominator]:  # Basically, if the parameter from the function input is not True,
                continue  # Skip this run.
                # This way, you can (optionally) speed up this routine if you only need static derivatives,
                # or longitudinal derivatives, etc.

            # These lines make a copy of the original operating point, incremented by the finite difference amount
            # along the variable defined by derivative_denominator.
            incremented_op_point = copy.copy(original_op_point)
            incremented_op_point.__setattr__(
                derivative_denominator,
                original_op_point.__getattribute__(derivative_denominator) + finite_difference_amounts[
                    derivative_denominator]
            )

            aerobuildup_incremented = copy.copy(self)
            aerobuildup_incremented.op_point = incremented_op_point
            run_incremented = aerobuildup_incremented.run()

            for derivative_numerator in [
                "CL",
                "CD",
                "CY",
                "Cl",
                "Cm",
                "Cn",
            ]:
                derivative_name = derivative_numerator + abbreviations[derivative_denominator]  # Gives "CLa"
                run_base[derivative_name] = (
                        (  # Finite-difference out the derivatives
                                run_incremented[derivative_numerator] - run_base[
                            derivative_numerator]
                        ) / finite_difference_amounts[derivative_denominator]
                        * scaling_factors[derivative_denominator]
                )

        return run_base

    def wing_aerodynamics(self,
                          wing: Wing,
                          ) -> Dict[str, Any]:
        """
        Estimates the aerodynamic forces, moments, and derivatives on a wing in isolation.

        Moments are given with the reference at Wing [0, 0, 0].

        Args:

            wing: A Wing object that you wish to analyze.

            op_point: The OperatingPoint that you wish to analyze the fuselage at.

        Returns:

        """
        ##### Alias a few things for convenience
        op_point = self.op_point
        wing_options = self.get_options(wing)

        ##### Compute general wing properties
        sweep = wing.mean_sweep_angle()
        AR = wing.aspect_ratio()
        mach = op_point.mach()
        mach_normal = mach * np.cosd(sweep)
        CL_over_Cl = aerolib.CL_over_Cl(
            aspect_ratio=AR,
            mach=mach,
            sweep=sweep,
            Cl_is_compressible=True
        )
        oswalds_efficiency = aerolib.oswalds_efficiency(
            taper_ratio=wing.taper_ratio(),
            aspect_ratio=AR,
            sweep=sweep,
            fuselage_diameter_to_span_ratio=0  # an assumption
        )
        areas = wing.area(_sectional=True)
        aerodynamic_centers = wing.aerodynamic_center(_sectional=True)

        def compute_section_aerodynamics(
                sect_id: int,
                mirror_across_XZ=False
        ):
            """
            Computes the forces and moments about self.xyz_ref on a given wing section.
            Args:

                sect_id: Wing section id. An int that can be from 0 to len(wing.xsecs) - 2.

                mirror_across_XZ: Boolean. If true, computes the forces and moments for the section that is mirrored across the XZ plane.

            Returns: Forces and moments, in a `(F_g, M_g)` tuple, where `F_g` and `M_g` have the following formats:

                F_g: a [Fx, Fy, Fz] list, given in geometry (`_g`) axes.

                M_g: a [Mx, My, Mz] list, given in geometry (`_g`) axes. Moment reference is `AeroBuildup.xyz_ref`.

            """

            ##### Identify the wing cross sections adjacent to this wing section.
            xsec_a = wing.xsecs[sect_id]
            xsec_b = wing.xsecs[sect_id + 1]

            ##### When linearly interpolating, weight things by the relative chord.
            a_weight = xsec_a.chord / (xsec_a.chord + xsec_b.chord)
            b_weight = xsec_b.chord / (xsec_a.chord + xsec_b.chord)
            mean_chord = (xsec_a.chord + xsec_b.chord) / 2

            ##### Compute the local frame of this section.
            _, _, zg_local = wing._compute_frame_of_section(sect_id)
            zg_local = [zg_local[0], zg_local[1], zg_local[2]]
            if mirror_across_XZ:
                zg_local[1] *= -1

            ##### Compute the moment arm from the section AC
            sect_AC_raw = aerodynamic_centers[sect_id]
            if mirror_across_XZ:
                sect_AC_raw[1] *= -1

            sect_AC = [
                sect_AC_raw[i] - self.xyz_ref[i]
                for i in range(3)
            ]

            ##### Compute the generalized angle of attack, so the geometric alpha that the wing section "sees".
            vel_vector_g_from_freestream = op_point.convert_axes(
                x_from=-op_point.velocity, y_from=0, z_from=0,
                from_axes="wind",
                to_axes="geometry"
            )
            vel_vector_g_from_rotation = np.cross(
                sect_AC,
                op_point.convert_axes(
                    op_point.p, op_point.q, op_point.r,
                    from_axes="body",
                    to_axes="geometry"
                ),
                manual=True
            )
            vel_vector_g = [
                vel_vector_g_from_freestream[i] + vel_vector_g_from_rotation[i]
                for i in range(3)
            ]
            vel_mag_g = np.sqrt(sum([comp ** 2 for comp in vel_vector_g]))
            vel_dir_g = [
                vel_vector_g[i] / vel_mag_g
                for i in range(3)
            ]
            vel_dot_z = np.dot(vel_dir_g, zg_local, manual=True)

            alpha_generalized = 90 - np.arccosd(np.clip(vel_dot_z, -1, 1))

            # Compute the control surface deflection
            n_surfs = len(xsec_a.control_surfaces)
            if n_surfs == 0:
                deflection = 0.
            elif n_surfs == 1:
                surf = xsec_a.control_surfaces[0]
                deflection = surf.deflection
                if mirror_across_XZ:
                    if not surf.symmetric:
                        deflection *= -1
            else:
                raise NotImplementedError(
                    "AeroBuildup currently cannot handle multiple control surfaces attached to a given WingXSec.")

            # Compute Reynolds numbers
            Re_a = op_point.reynolds(xsec_a.chord)
            Re_b = op_point.reynolds(xsec_b.chord)

            ##### Compute sectional lift at cross sections using lookup functions. Merge them linearly to get section CL.
            xsec_a_args = dict(
                alpha=alpha_generalized,
                Re=Re_a,
                mach=mach_normal,
                deflection=deflection
            )
            xsec_b_args = dict(
                alpha=alpha_generalized,
                Re=Re_b,
                mach=mach_normal,
                deflection=deflection
            )

            xsec_a_Cl = xsec_a.airfoil.CL_function(**xsec_a_args)
            xsec_b_Cl = xsec_b.airfoil.CL_function(**xsec_b_args)
            sect_CL = (
                              xsec_a_Cl * a_weight +
                              xsec_b_Cl * b_weight
                      ) * CL_over_Cl

            ##### Compute sectional drag at cross sections using lookup functions. Merge them linearly to get section CD.
            xsec_a_Cdp = xsec_a.airfoil.CD_function(**xsec_a_args)
            xsec_b_Cdp = xsec_b.airfoil.CD_function(**xsec_b_args)
            sect_CDp = (
                    xsec_a_Cdp * a_weight +
                    xsec_b_Cdp * b_weight
            )

            ##### Compute sectional moment at cross sections using lookup functions. Merge them linearly to get section CM.
            xsec_a_Cm = xsec_a.airfoil.CM_function(**xsec_a_args)
            xsec_b_Cm = xsec_b.airfoil.CM_function(**xsec_b_args)
            sect_CM = (
                    xsec_a_Cm * a_weight +
                    xsec_b_Cm * b_weight
            )

            ##### Compute induced drag from local CL and full-wing properties (AR, e)
            sect_CDi = (
                    sect_CL ** 2 / (np.pi * AR * oswalds_efficiency)
            )

            ##### Total the drag.
            sect_CD = sect_CDp + sect_CDi

            ##### Go to dimensional quantities using the area.
            area = areas[sect_id]
            q_local = 0.5 * op_point.atmosphere.density() * vel_mag_g ** 2
            sect_L = q_local * area * sect_CL
            sect_D = q_local * area * sect_CD
            sect_M = q_local * area * sect_CM * mean_chord

            ##### Compute the direction of the lift by projecting the section's normal vector into the plane orthogonal to the local freestream.
            L_direction_g_unnormalized = [
                zg_local[i] - vel_dot_z * vel_dir_g[i]
                for i in range(3)
            ]
            L_direction_g_mag = np.sqrt(sum([comp ** 2 for comp in L_direction_g_unnormalized]))
            L_direction_g = [
                L_direction_g_unnormalized[i] / L_direction_g_mag
                for i in range(3)
            ]

            ##### Compute the direction of the drag by aligning the drag vector with the freestream vector.
            D_direction_g = vel_dir_g

            ##### Compute the force vector in geometry axes.
            sect_F_g = [
                sect_L * L_direction_g[i] + sect_D * D_direction_g[i]
                for i in range(3)
            ]

            ##### Compute the moment vector in geometry axes.
            M_g_lift = np.cross(
                sect_AC,
                sect_F_g,
                manual=True
            )
            M_direction_g = np.cross(L_direction_g, D_direction_g, manual=True)
            M_g_pitching_moment = [
                M_direction_g[i] * sect_M
                for i in range(3)
            ]
            sect_M_g = [
                M_g_lift[i] + M_g_pitching_moment[i]
                for i in range(3)
            ]

            return sect_F_g, sect_M_g

        ##### Iterate through all sections and add up all forces/moments.
        F_g = [0., 0., 0.]
        M_g = [0., 0., 0.]

        for sect_id in range(len(wing.xsecs) - 1):
            sect_F_g, sect_M_g = compute_section_aerodynamics(sect_id=sect_id)

            for i in range(3):
                F_g[i] += sect_F_g[i]
                M_g[i] += sect_M_g[i]

            if wing.symmetric:
                sect_F_g, sect_M_g = compute_section_aerodynamics(sect_id=sect_id, mirror_across_XZ=True)

                for i in range(3):
                    F_g[i] += sect_F_g[i]
                    M_g[i] += sect_M_g[i]

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
                              ) -> Dict[str, Any]:
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

        ##### Compute general fuselage properties
        q = op_point.dynamic_pressure()
        eta = jorgensen_eta(fuselage.fineness_ratio())

        def compute_section_aerodynamics(
                sect_id: int,
        ):
            ##### Identify the fuselage cross sections adjacent to this fuselage section.
            xsec_a = fuselage.xsecs[sect_id]
            xsec_b = fuselage.xsecs[sect_id + 1]

            ### Some metrics, like effective force location, are area-weighted. Here, we compute those weights.
            r_a = xsec_a.radius
            r_b = xsec_b.radius

            xyz_a = xsec_a.xyz_c
            xyz_b = xsec_b.xyz_c

            area_a = xsec_a.xsec_area()
            area_b = xsec_b.xsec_area()
            total_area = area_a + area_b

            a_weight = area_a / total_area
            b_weight = area_b / total_area

            mean_geometric_radius = (r_a + r_b) / 2
            mean_aerodynamic_radius = r_a * a_weight + r_b * b_weight

            ### Compute the key geometric properties of the centerline between the two sections.
            sect_length = np.sqrt(sum([(xyz_b[i] - xyz_a[i]) ** 2 for i in range(3)]))

            xg_local = [
                np.where(
                    sect_length != 0,
                    (xyz_b[i] - xyz_a[i]) / (sect_length + 1e-100),
                    1 if i == 0 else 0  # Default to [1, 0, 0]
                )
                for i in range(3)
            ]

            ##### Compute the moment arm from the section AC
            sect_AC = [
                (xyz_a[i] + xyz_b[i]) / 2 - self.xyz_ref[i]
                for i in range(3)
            ]

            ##### Compute the generalized angle of attack that the section sees
            vel_direction_g = op_point.convert_axes(-1, 0, 0, from_axes="wind", to_axes="geometry")
            vel_dot_x = np.dot(
                vel_direction_g,
                xg_local,
                manual=True
            )

            def soft_norm(xyz):
                return (
                               sum([comp ** 2 for comp in xyz])
                               + 1e-100  # Keeps the derivative from exploding
                       ) ** 0.5

            generalized_alpha = 2 * np.arctan2d(
                soft_norm([vel_direction_g[i] - xg_local[i] for i in range(3)]),
                soft_norm([vel_direction_g[i] + xg_local[i] for i in range(3)])
            )
            sin_generalized_alpha = np.sind(generalized_alpha)

            ##### Compute the normal-force and axial-force directions
            normal_direction_g_unnormalized = [
                vel_direction_g[i] - vel_dot_x * xg_local[i]
                for i in range(3)
            ]
            normal_direction_g_unnormalized[2] += 1e-16  # A hack that prevents NaN for 0-AoA case.
            normal_direction_g_mag = np.sqrt(sum([comp ** 2 for comp in normal_direction_g_unnormalized]))
            normal_direction_g = [
                normal_direction_g_unnormalized[i] / normal_direction_g_mag
                for i in range(3)
            ]

            axial_direction_g = xg_local

            ##### Inviscid Forces
            ### Jorgensen model
            ### Note the (N)ormal, (A)ligned coordinate system. (See Jorgensen for definitions.)
            force_potential_flow = q * (  # From Munk, via Jorgensen
                    np.sind(2 * generalized_alpha) *
                    (area_b - area_a)
            )  # Matches Drela, Flight Vehicle Aerodynamics Eqn. 6.75 in the small-alpha limit.
            # Note that no delta_x should be here; dA/dx * dx = dA.

            # Direction of force is midway between the normal to the axis of revolution of the body and the
            # normal to the free-stream velocity, according to:
            # Ward, via Jorgensen
            force_normal_potential_flow = force_potential_flow * np.cosd(generalized_alpha / 2)
            force_axial_potential_flow = -force_potential_flow * np.sind(generalized_alpha / 2)
            # Reminder: axial force is defined positive-aft

            ##### Viscous Forces
            ### Jorgensen model

            Re_n = sin_generalized_alpha * op_point.reynolds(reference_length=2 * mean_aerodynamic_radius)
            M_n = sin_generalized_alpha * op_point.mach()

            C_d_n = np.where(
                Re_n != 0,
                aerolib.Cd_cylinder(
                    Re_D=Re_n,
                    mach=M_n
                ),  # Replace with 1.20 from Jorgensen Table 1 if this isn't working well
                0,
            )

            force_viscous_crossflow = sect_length * q * (
                    2 * eta * C_d_n *
                    sin_generalized_alpha ** 2 *
                    mean_geometric_radius
            )

            ##### Viscous crossflow acts exactly normal to vehicle axis, definitionally.
            # (Axial viscous forces accounted for on a total-body basis)

            normal_force = force_normal_potential_flow + force_viscous_crossflow
            axial_force = force_axial_potential_flow

            ##### Compute the force vector in geometry axes
            sect_F_g = [
                normal_force * normal_direction_g[i] + axial_force * axial_direction_g[i]
                for i in range(3)
            ]

            ##### Compute the moment vector in geometry axes.
            sect_M_g = np.cross(
                sect_AC,
                sect_F_g,
                manual=True
            )

            return sect_F_g, sect_M_g

        ##### Iterate through all sections and add up all forces/moments.
        F_g = [0., 0., 0.]
        M_g = [0., 0., 0.]

        for sect_id in range(len(fuselage.xsecs) - 1):
            sect_F_g, sect_M_g = compute_section_aerodynamics(sect_id=sect_id)

            for i in range(3):
                F_g[i] += sect_F_g[i]
                M_g[i] += sect_M_g[i]

            if fuselage.symmetric:
                raise NotImplementedError()

        ##### Add in profile drag: viscous drag forces and wave drag forces
        ### Base Drag
        base_drag_coefficient = fuselage_base_drag_coefficient(mach=op_point.mach())
        D_base = base_drag_coefficient * fuselage.area_base() * q

        ### Skin friction drag
        form_factor = fuselage_form_factor(
            fineness_ratio=fuselage.fineness_ratio(),
            ratio_of_corner_radius_to_body_width=0.5
        )
        C_f = aerolib.Cf_flat_plate(Re_L=Re) * form_factor
        D_skin = C_f * fuselage.area_wetted() * q

        ### Wave drag
        S_ref = 1  # Does not matter here, just for accounting.

        if self.include_wave_drag:
            sears_haack_drag_area = transonic.sears_haack_drag_from_volume(
                volume=fuselage.volume(),
                length=fuselage.length()
            )  # Units of area
            sears_haack_C_D_wave = sears_haack_drag_area / S_ref

            C_D_wave = transonic.approximate_CD_wave(
                mach=op_point.mach(),
                mach_crit=critical_mach(
                    fineness_ratio_nose=fuse_options["nose_fineness_ratio"]
                ),
                CD_wave_at_fully_supersonic=fuse_options["E_wave_drag"] * sears_haack_C_D_wave,
            )
        else:
            C_D_wave = 0

        D_wave = C_D_wave * q * S_ref

        ### Sum up the profile drag
        D_profile = D_base + D_skin + D_wave

        D_profile_g = op_point.convert_axes(
            -D_profile, 0, 0,
            from_axes="wind",
            to_axes="geometry",
        )

        drag_moment_arm = [
            fuselage.xsecs[-1].xyz_c[i] - self.xyz_ref[i]  # TODO make this act at centroid
            for i in range(3)
        ]

        M_g_from_D_profile = np.cross(
            drag_moment_arm,
            D_profile_g,
            manual=True
        )

        for i in range(3):
            F_g[i] += D_profile_g[i]
            M_g[i] += M_g_from_D_profile[i]

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
