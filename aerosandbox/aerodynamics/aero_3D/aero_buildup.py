from aerosandbox import ExplicitAnalysis
from aerosandbox.geometry import *
from aerosandbox.performance import OperatingPoint
import aerosandbox.library.aerodynamics as aero


class AeroBuildup(ExplicitAnalysis):
    def __init__(self,
                 airplane,  # type: Airplane
                 op_point,  # type: OperatingPoint
                 ):
        ### Initialize
        self.airplane = airplane
        self.op_point = op_point

        ### Check assumptions
        assumptions = np.array([
            self.op_point.beta == 0,
            self.op_point.p == 0,
            self.op_point.q == 0,
            self.op_point.r == 0,
        ])
        if not assumptions.all():
            raise ValueError("The assumptions to use an aero buildup method are not met!")

        ### Fuselages
        for fuselage in self.airplane.fuselages:
            fuselage.Re = self.op_point.reynolds(fuselage.length())
            fuselage.CLA = 0
            fuselage.CDA = aero.Cf_flat_plate(fuselage.Re * fuselage.area_wetted()) * 1.2 # wetted area with form factor

            fuselage.lift_force = fuselage.CLA * self.op_point.dynamic_pressure()
            fuselage.drag_force = fuselage.CDA * self.op_point.dynamic_pressure()

        ### Wings
        for wing in self.airplane.wings:
            wing.Re = self.op_point.reynolds(wing.mean_aerodynamic_chord())
            wing.airfoil = wing.xsecs[0].airfoil
            wing.Cl_incompressible= wing.airfoil.Cl_function(
                alpha = wing.alpha,
                Re = wing.Re # TODO finish
            )

        self.wing_Res = [
            self.op_point.reynolds(wing.mean_geometric_chord())
            for i, wing in enumerate(self.airplane.wings)
        ]
        self.wing_airfoils = [
            wing.xsecs[0].airfoil  # type: asb.Airfoil
            for i, wing in enumerate(self.airplane.wings)
        ]

        self.wing_Cl_incs = [
            self.wing_airfoils[i].CL_function(self.op_point.alpha + wing.mean_twist_angle(), self.wing_Res[i], 0, 0)
            for i, wing in enumerate(self.airplane.wings)
        ]  # Incompressible 2D lift coefficient
        self.wing_CLs = [
            self.wing_Cl_incs[i] * aero.CL_over_Cl(wing.aspect_ratio(), mach=self.op_point.mach,
                                                   sweep=wing.mean_sweep_angle())
            for i, wing in enumerate(self.airplane.wings)
        ]  # Compressible 3D lift coefficient
        self.lift_wings = [
            self.wing_CLs[i] * self.op_point.dynamic_pressure() * wing.area()
            for i, wing in enumerate(self.airplane.wings)
        ]

        self.wing_Cd_profiles = [
            self.wing_airfoils[i].CDp_function(self.op_point.alpha + wing.mean_twist_angle(), self.wing_Res[i],
                                               self.op_point.mach, 0)
            for i, wing in enumerate(self.airplane.wings)
        ]
        self.drag_wing_profiles = [
            self.wing_Cd_profiles[i] * self.op_point.dynamic_pressure() * wing.area()
            for i, wing in enumerate(self.airplane.wings)
        ]

        self.wing_oswalds_efficiencies = [
            0.95  # TODO make this a function of taper ratio
            for i, wing in enumerate(self.airplane.wings)
        ]
        self.drag_wing_induceds = [
            self.lift_wings[i] ** 2 / (
                    self.op_point.dynamic_pressure() * np.pi * wing.span() ** 2 * self.wing_oswalds_efficiencies[i])
            for i, wing in enumerate(self.airplane.wings)
        ]

        self.drag_wings = [
            self.drag_wing_profiles[i] + self.drag_wing_induceds[i]
            for i, wing in enumerate(self.airplane.wings)
        ]

        self.wing_Cm_incs = [
            self.wing_airfoils[i].Cm_function(self.op_point.alpha + wing.mean_twist_angle(), self.wing_Res[i], 0, 0)
            for i, wing in enumerate(self.airplane.wings)
        ]  # Incompressible 2D moment coefficient
        self.wing_CMs = [
            self.wing_Cm_incs[i] * aero.CL_over_Cl(wing.aspect_ratio(), mach=self.op_point.mach,
                                                   sweep=wing.mean_sweep_angle())
            for i, wing in enumerate(self.airplane.wings)
        ]  # Compressible 3D moment coefficient
        self.local_moment_wings = [
            self.wing_CMs[i] * self.op_point.dynamic_pressure() * wing.area() * wing.mean_geometric_chord()
            for i, wing in enumerate(self.airplane.wings)
        ]
        self.body_moment_wings = [
            self.local_moment_wings[i] + wing.approximate_center_of_pressure()[0] * self.lift_wings[i]
            for i, wing in enumerate(self.airplane.wings)
        ]

        # Force totals
        lift_forces = self.lift_fuses + self.lift_wings
        drag_forces = self.drag_fuses + self.drag_wings
        self.lift_force = cas.sum1(cas.vertcat(*lift_forces))
        self.drag_force = cas.sum1(cas.vertcat(*drag_forces))
        self.side_force = 0

        # Moment totals
        self.pitching_moment = cas.sum1(cas.vertcat(*self.body_moment_wings))

        # Calculate nondimensional forces
        q = self.op_point.dynamic_pressure()
        s_ref = self.airplane.s_ref
        b_ref = self.airplane.b_ref
        c_ref = self.airplane.c_ref
        self.CL = self.lift_force / q / s_ref
        self.CD = self.drag_force / q / s_ref
        self.Cm = self.pitching_moment / q / s_ref / c_ref
