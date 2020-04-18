from aerosandbox.aerodynamics.aerodynamics import *
from aerosandbox.geometry import *
import aerosandbox.library.aerodynamics as aero


class Buildup(AeroProblem):
    def __init__(self,
                 airplane,  # type: Airplane
                 op_point,  # type: op_point
                 run_setup=True,
                 ):
        super().__init__(airplane, op_point)

        self.check()

        if run_setup:
            self.setup()

    def check(self):
        """
        Checks to see if this is a case where the assumptions are valid.
        :return: Throws an exception if this is the case.
        """
        assumptions = np.array([
            self.op_point.beta == 0,
            self.op_point.p == 0,
            self.op_point.q == 0,
            self.op_point.r == 0,
        ])
        if np.logical_not(assumptions).any():
            raise ValueError("The assumptions to use an aero buildup method are not met!")

    def setup(self,
              verbose=True,  # Choose whether or not you want verbose output
              run_symmetric_if_possible=True,
              # Choose whether or not you want to run a symmetric_problem analysis about XZ (~4x faster)
              ):
        # Runs a point analysis at the specified op-point.

        ### Fuselages
        self.fuse_Res = [
            self.op_point.compute_reynolds(fuse.length())
            for i, fuse in enumerate(self.airplane.fuselages)
        ]
        self.CLA_fuses = [
            0
            for i, fuse in enumerate(self.airplane.fuselages)
        ]
        self.CDA_fuses = [
            aero.Cf_flat_plate(self.fuse_Res[i]) * fuse.area_wetted() * 1.2  # wetted area with form factor
            for i, fuse in enumerate(self.airplane.fuselages)
        ]

        self.lift_fuses = [
            self.CLA_fuses[i] * self.op_point.dynamic_pressure()
            for i, fuse in enumerate(self.airplane.fuselages)
        ]
        self.drag_fuses = [
            self.CDA_fuses[i] * self.op_point.dynamic_pressure()
            for i, fuse in enumerate(self.airplane.fuselages)
        ]

        ### Wings
        self.wing_Res = [
            self.op_point.compute_reynolds(wing.mean_geometric_chord())
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
