from aerosandbox import ExplicitAnalysis
from aerosandbox.geometry import *
from aerosandbox.performance import OperatingPoint
import aerosandbox.library.aerodynamics as aero


class AeroBuildup(ExplicitAnalysis):
    """
    A workbook-style aerodynamics buildup
    """

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
            fuselage.CDA = aero.Cf_flat_plate(
                fuselage.Re * fuselage.area_wetted()) * 1.2  # wetted area with form factor

            fuselage.lift_force = fuselage.CLA * self.op_point.dynamic_pressure()
            fuselage.drag_force = fuselage.CDA * self.op_point.dynamic_pressure()
            fuselage.pitching_moment = 0

        ### Wings
        for wing in self.airplane.wings:
            wing.alpha = op_point.alpha + wing.mean_twist_angle()  # TODO add in allmoving deflections
            wing.Re = self.op_point.reynolds(wing.mean_aerodynamic_chord())
            wing.airfoil = wing.xsecs[0].airfoil

            ## Lift calculation
            wing.Cl_incompressible = wing.airfoil.CL_function(
                alpha=wing.alpha,
                Re=wing.Re,  # TODO finish
                mach=0,  # TODO revisit this - is this right?
                deflection=0
            )
            CL_over_Cl = aero.CL_over_Cl(
                aspect_ratio=wing.aspect_ratio(),
                mach=op_point.mach(),
                sweep=wing.mean_sweep_angle()
            )
            wing.CL = wing.Cl_incompressible * CL_over_Cl

            ## Drag calculation
            wing.CD_profile = wing.airfoil.CD_function(
                alpha=wing.alpha,
                Re=wing.Re,
                mach=op_point.mach(),
                deflection=0
            )

            wing.oswalds_efficiency = aero.oswalds_efficiency(
                taper_ratio=wing.taper_ratio(),
                aspect_ratio=wing.aspect_ratio(),
                sweep=wing.mean_sweep_angle(),
            )
            wing.CD_induced = wing.CL ** 2 / (pi * wing.oswalds_efficiency * wing.aspect_ratio())

            ## Moment calculation
            wing.Cm_incompressible = wing.airfoil.CM_function(
                alpha=wing.alpha,
                Re=wing.Re,
                mach=0,  # TODO revisit this - is this right?
                deflection=0,
            )
            wing.CM = wing.Cm_incompressible * CL_over_Cl

            ## Force and moment calculation
            qS = op_point.dynamic_pressure() * wing.area()
            wing.lift_force = wing.CL * qS
            wing.drag_force_profile = wing.CD_profile * qS
            wing.drag_force_induced = wing.CD_induced * qS
            wing.drag_force = wing.drag_force_profile + wing.drag_force_induced
            wing.pitching_moment = wing.CM * qS * wing.mean_aerodynamic_chord()

        ### Total the forces
        self.lift_force = 0
        self.drag_force = 0
        self.pitching_moment = 0

        for fuselage in self.airplane.fuselages:
            self.lift_force += fuselage.lift_force
            self.drag_force += fuselage.drag_force
            self.pitching_moment += fuselage.pitching_moment

        for wing in self.airplane.wings:
            if wing.symmetric:  # Only add lift force if the wing is symmetric; a surrogate for "horizontal".
                self.lift_force += wing.lift_force
            self.drag_force += wing.drag_force
            self.pitching_moment += wing.pitching_moment # Raw pitching moment
            self.pitching_moment += -wing.aerodynamic_center()[0] * wing.lift_force # Pitching moment due to lift

        ### Calculate nondimensional forces
        qS = op_point.dynamic_pressure() * self.airplane.s_ref

        self.CL = self.lift_force / qS
        self.CD = self.drag_force / qS
        self.CM = self.pitching_moment / qS / self.airplane.c_ref
