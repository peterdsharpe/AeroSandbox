from aerosandbox import ExplicitAnalysis
from aerosandbox.geometry import *
from aerosandbox.performance import OperatingPoint
import aerosandbox.library.aerodynamics as aero
import aerosandbox.numpy as np
from aerosandbox.aerodynamics.aero_3D.aero_buildup_submodels import *


class AeroBuildup(ExplicitAnalysis):
    """
    A workbook-style aerodynamics buildup
    """

    def __init__(self,
                 airplane: Airplane,
                 op_point: OperatingPoint,
                 ):
        ### Initialize
        self.airplane = airplane
        self.op_point = op_point

    def run(self):
        aero_components = []

        for wing in self.airplane.wings:
            aero = wing_aerodynamics(
                wing=wing,
                op_point=self.op_point
            )
            aero_components.append(aero)

        for fuselage in self.airplane.fuselages:
            aero = fuselage_aerodynamics(
                fuselage=fuselage,
                op_point=self.op_point
            )
            aero_components.append(aero)

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
        qS = self.op_point.dynamic_pressure() * self.airplane.s_ref

        aero_total["CL"] = aero_total["L"] / qS
        aero_total["CY"] = aero_total["Y"] / qS
        aero_total["CD"] = aero_total["D"] / qS
        aero_total["Cl"] = aero_total["l_b"] / qS / self.airplane.b_ref
        aero_total["Cm"] = aero_total["m_b"] / qS / self.airplane.c_ref
        aero_total["Cn"] = aero_total["n_b"] / qS / self.airplane.b_ref

        return aero_total


if __name__ == '__main__':
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.conventional import airplane

    analysis = AeroBuildup(
        airplane=airplane,
        op_point=OperatingPoint(alpha=0, beta=1),
    )
    out = analysis.run()
