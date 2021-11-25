from aerosandbox import ExplicitAnalysis
from aerosandbox.geometry import *
from aerosandbox.performance import OperatingPoint
import aerosandbox.library.aerodynamics as aero
import aerosandbox.numpy as np
from aerosandbox.aerodynamics.aero_3D.aero_buildup_submodels import *


class AeroBuildup(ExplicitAnalysis):
    """
    A workbook-style aerodynamics buildup.
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
                wing=wing.translate(-self.airplane.xyz_ref),
                op_point=self.op_point
            )
            aero_components.append(aero)

        for fuselage in self.airplane.fuselages:
            aero = fuselage_aerodynamics(
                fuselage=fuselage.translate(-self.airplane.xyz_ref),
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

        self.output = aero_total

        return aero_total


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
