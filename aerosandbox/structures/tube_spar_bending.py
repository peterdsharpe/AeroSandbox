import aerosandbox as asb
import aerosandbox.numpy as np
from typing import Callable, Union, Dict


class TubeSparBendingStructure(asb.ImplicitAnalysis):

    @asb.ImplicitAnalysis.initialize
    def __init__(self,
                 length: float,
                 diameter_function: Union[float, Callable[[np.ndarray], np.ndarray]] = None,
                 wall_thickness_function: Union[float, Callable[[np.ndarray], np.ndarray]] = None,
                 bending_point_forces: Dict[float, float] = None,
                 bending_distributed_force_function: Union[float, Callable[[np.ndarray], np.ndarray]] = 0.,
                 points_per_point_load: int = 20,
                 elastic_modulus_function: Union[float, Callable[[np.ndarray], np.ndarray]] = 175e9,  # Pa
                 EI_guess: float = None,
                 assume_thin_tube=True,
                 ):
        """
        A structural spar model that simulates bending of a cantilever tube spar based on beam theory (static,
        linear elasticity). This tube spar is assumed to have uniform wall thickness in the azimuthal direction,
        but not necessarily along its length. The diameter of the tube spar and elastic modulus may vary along its
        length.

        Governing equation is Euler-Bernoulli beam theory:

        (E * I * u(y)'')'' = q(y)

        where:
            * y is the distance along the spar, with a cantilever support at y=0 and a free tip at y=length.
            * E is the elastic modulus
            * I is the bending moment of inertia
            * u(y) is the local displacement at y.
            * q(y) is the force-per-unit-length at y. (In other words, a dirac delta is a point load.)
            * ()' is a derivative w.r.t. y.

        Any applicable constraints relating to stress, buckling, ovalization, gauge limits, displacement, etc. should
        be applied after initialization of this class.

        Example:

            >>> opti = asb.Opti()
            >>>
            >>> span = 34
            >>> half_span = span / 2
            >>> lift = 200 * 9.81
            >>>
            >>> beam = TubeSparBendingStructure(
            >>>     opti=opti,
            >>>     length=half_span,
            >>>     diameter_function=0.12,
            >>>     points_per_point_load=100,
            >>>     bending_distributed_force_function=lambda y: (lift / span) * (
            >>>             4 / np.pi * (1 - (y / half_span) ** 2) ** 0.5
            >>>     ),  # Elliptical
            >>>     # bending_distributed_force_function=lambda y: lift / span * np.ones_like(y) # Uniform
            >>> )
            >>> opti.subject_to([
            >>>     beam.stress_axial <= 500e6,  # Stress constraint
            >>>     beam.u[-1] <= 3,  # Tip displacement constraint
            >>>     beam.wall_thickness > 1e-3  # Gauge constraint
            >>> ])
            >>> mass = beam.volume() * 1600  # Density of carbon fiber [kg/m^3]
            >>>
            >>> opti.minimize(mass / 100)
            >>> sol = opti.solve()
            >>>
            >>> beam = sol(beam)
            >>>
            >>> print(f"{sol.value(mass)} kg")
            >>>
            >>> beam.draw()

        Args:

            length: Length of the spar [m]. Spar is assumed to go from y=0 (cantilever support) to y=length (free tip).

            diameter_function: The diameter of the tube as a function of the distance along the spar y. Refers to the
                nominal diameter (e.g., the arithmetic mean of the inner diameter and outer diameter of the tube; the
                "centerline" diameter). In terms of data types, this can be one of:

                * None, in which case it's interpreted as a design variable to optimize over. Assumes that the value
                can freely vary along the length of the spar.

                * a scalar optimization variable (see asb.ImplicitAnalysis documentation to see how to link an Opti
                instance to this analysis), in which case it's interpreted as a design variable to optimize over
                that's uniform along the length of the spar.

                * a float, in which case it's interpreted as a uniform value along the spar

                * a function (or other callable) in the form f(y), where y is the coordinate along the length of the
                spar. This function should be vectorized (e.g., a vector input of y values produces a vector output).

            wall_thickness_function: The wall thickness of the tube as a function of the distance along the spar y. In
                terms of data types, this can be one of:

                * None, in which case it's interpreted as a design variable to optimize over. Assumes that the value
                can freely vary along the length of the spar.

                * a scalar optimization variable (see asb.ImplicitAnalysis documentation to see how to link an Opti
                instance to this analysis), in which case it's interpreted as a design variable to optimize over
                that's uniform along the length of the spar.

                * a float, in which case it's interpreted as a uniform value along the spar

                * a function (or other callable) in the form f(y), where y is the coordinate along the length of the
                spar. This function should be vectorized (e.g., a vector input of y values produces a vector output).

            bending_point_forces: Not yet implemented; will allow for inclusion of point loads in the future.

            bending_distributed_force_function: The (distributed) load per unit span applied to the spar,
                as a function of the distance along the spar y. Should be in units of force per unit length. In terms of
                data types, this can be one of:

                * None, in which case it's interpreted as a design variable to optimize over. Assumes that the value
                can freely vary along the length of the spar.

                * a scalar optimization variable (see asb.ImplicitAnalysis documentation to see how to link an Opti
                instance to this analysis), in which case it's interpreted as a design variable to optimize over
                that's uniform along the length of the spar.

                * a float, in which case it's interpreted as a uniform value along the spar

                * a function (or other callable) in the form f(y), where y is the coordinate along the length of the
                spar. This function should be vectorized (e.g., a vector input of y values produces a vector output).

            points_per_point_load: Controls the discretization resolution of the beam. [int] When point load support
                is added, this will be the number of nodes between each individual point load.

            elastic_modulus_function: The elastic modulus [Pa] of the spar as a function of the distance along the
                spar y. In terms of data types, can be one of:

                * None, in which case it's interpreted as a design variable to optimize over. Assumes that the value
                can freely vary along the length of the spar.

                * a scalar optimization variable (see asb.ImplicitAnalysis documentation to see how to link an Opti
                instance to this analysis), in which case it's interpreted as a design variable to optimize over
                that's uniform along the length of the spar.

                * a float, in which case it's interpreted as a uniform value along the spar

                * a function (or other callable) in the form f(y), where y is the coordinate along the length of the
                spar. This function should be vectorized (e.g., a vector input of y values produces a vector output).

            EI_guess: Provides an initial guess for the bending stiffness EI, which is used in problems where spar
                diameter and thickness is not known at the outset. If not provided, a heuristic will be used to calculate this.

            assume_thin_tube: Makes assumptions that are applicable in the limit of a thin-walled (wall_thickness <<
                diameter) tube. This greatly increases numerical stability.

                Relative error of this assumption in the thin-walled limit is:

                    (wall_thickness / diameter) ^ 2

                So, for t/d = 0.1, the relative error is roughly 1%.

        """
        ### Parse the inputs
        self.length = length
        self.diameter_function = diameter_function
        self.wall_thickness_function = wall_thickness_function

        if bending_point_forces is not None:
            self.bending_point_forces = bending_point_forces
            raise NotImplementedError
        else:
            self.bending_point_forces = dict()

        self.bending_distributed_force_function = bending_distributed_force_function
        self.points_per_point_load = points_per_point_load
        self.elastic_modulus_function = elastic_modulus_function

        if EI_guess is None:
            try:
                diameter_guess = float(diameter_function)
            except (TypeError, RuntimeError):
                diameter_guess = 1

            try:
                wall_thickness_guess = float(wall_thickness_function)
            except (TypeError, RuntimeError):
                wall_thickness_guess = 0.01

            try:
                E_guess = float(elastic_modulus_function)
            except (TypeError, RuntimeError):
                E_guess = 175e9

            if assume_thin_tube:
                I_guess = np.pi / 8 * diameter_guess ** 3 * wall_thickness_guess
            else:
                I_guess = np.pi / 64 * (
                        (diameter_guess + wall_thickness_guess) ** 4 -
                        (diameter_guess - wall_thickness_guess) ** 4
                )
            EI_guess = E_guess * I_guess

            # EI_guess *= 1e0  # A very high EI guess is numerically stabilizing

        self.EI_guess = EI_guess

        self.assume_thin_tube = assume_thin_tube

        ### Discretize
        y = np.linspace(
            0,
            length,
            points_per_point_load
        )

        N = np.length(y)
        dy = np.diff(y)

        ### Evaluate the beam properties
        if isinstance(diameter_function, Callable):
            diameter = diameter_function(y)
        elif diameter_function is None:
            diameter = self.opti.variable(init_guess=1, n_vars=N, lower_bound=0.)
        else:
            diameter = diameter_function * np.ones_like(y)

        if isinstance(wall_thickness_function, Callable):
            wall_thickness = wall_thickness_function(y)
        elif wall_thickness_function is None:
            wall_thickness = self.opti.variable(init_guess=1e-2, n_vars=N, lower_bound=0, upper_bound=diameter)
        else:
            wall_thickness = wall_thickness_function * np.ones_like(y)

        if isinstance(bending_distributed_force_function, Callable):
            distributed_force = bending_distributed_force_function(y)
        else:
            distributed_force = bending_distributed_force_function * np.ones_like(y)

        if isinstance(elastic_modulus_function, Callable):
            elastic_modulus = elastic_modulus_function(y)
        else:
            elastic_modulus = elastic_modulus_function * np.ones_like(y)

        ### Evaluate the beam properties
        if assume_thin_tube:
            I = np.pi / 8 * diameter ** 3 * wall_thickness
        else:
            I = np.pi / 64 * (
                    (diameter + wall_thickness) ** 4 -
                    (diameter - wall_thickness) ** 4
            )
        EI = elastic_modulus * I

        ### Compute the initial guess
        u = self.opti.variable(
            init_guess=np.zeros_like(y),
            scale=np.sum(np.trapz(distributed_force) * dy) * length ** 4 / EI_guess
        )
        du = self.opti.derivative_of(
            u, with_respect_to=y,
            derivative_init_guess=np.zeros_like(y),
            derivative_scale=np.sum(np.trapz(distributed_force) * dy) * length ** 3 / EI_guess
        )
        ddu = self.opti.derivative_of(
            du, with_respect_to=y,
            derivative_init_guess=np.zeros_like(y),
            derivative_scale=np.sum(np.trapz(distributed_force) * dy) * length ** 2 / EI_guess
        )
        dEIddu = self.opti.derivative_of(
            EI * ddu, with_respect_to=y,
            derivative_init_guess=np.zeros_like(y),
            derivative_scale=np.sum(np.trapz(distributed_force) * dy) * length
        )
        self.opti.constrain_derivative(
            variable=dEIddu, with_respect_to=y,
            derivative=distributed_force
        )

        self.opti.subject_to([
            u[0] == 0,
            du[0] == 0,
            ddu[-1] == 0,
            dEIddu[-1] == 0
        ])

        bending_moment = -EI * ddu
        shear_force = -dEIddu

        stress_axial = elastic_modulus * ddu * (diameter + wall_thickness) / 2

        self.y = y
        self.diameter = diameter
        self.wall_thickness = wall_thickness
        self.distributed_force = distributed_force
        self.elastic_modulus = elastic_modulus
        self.I = I
        self.u = u
        self.du = du
        self.ddu = ddu
        self.dEIddu = dEIddu
        self.bending_moment = bending_moment
        self.shear_force = shear_force
        self.stress_axial = stress_axial

    def volume(self):
        if self.assume_thin_tube:
            return np.sum(
                np.pi * np.trapz(
                    self.diameter * self.wall_thickness
                ) * np.diff(self.y)
            )
        else:
            return np.sum(
                np.pi / 4 * np.trapz(
                    (self.diameter + self.wall_thickness) ** 2 -
                    (self.diameter - self.wall_thickness) ** 2
                ) * np.diff(self.y)
            )

    def total_force(self):
        if len(self.bending_point_forces) != 0:
            raise NotImplementedError

        return np.sum(
            np.trapz(
                self.distributed_force
            ) * np.diff(self.y)
        )

    def draw(self, show=True):
        import matplotlib.pyplot as plt
        import aerosandbox.tools.pretty_plots as p

        plot_quantities = {
            "Displacement [m]"              : self.u,
            # "Local Slope [deg]": np.arctan2d(self.du, 1),
            "Local Load [N/m]"              : self.distributed_force,
            "Axial Stress [MPa]"            : self.stress_axial / 1e6,
            "Bending $EI$ [N $\cdot$ m$^2$]": self.elastic_modulus * self.I,
            "Tube Diameter [m]"             : self.diameter,
            "Wall Thickness [m]"            : self.wall_thickness,
        }

        fig, ax = plt.subplots(2, 3, figsize=(8, 6), sharex='all')

        for i, (k, v) in enumerate(plot_quantities.items()):
            plt.sca(ax.flatten()[i])
            plt.plot(
                self.y,
                v,
                # ".-"
            )
            plt.ylabel(k)
            plt.xlim(
                np.min(self.y),
                np.max(self.y),
            )

        for a in ax[-1, :]:
            a.set_xlabel(r"$y$ [m]")

        if show:
            p.show_plot("Tube Spar Bending Structure")


if __name__ == '__main__':
    import aerosandbox.tools.units as u

    opti = asb.Opti()

    span = 112 * u.foot
    lift = 229 * u.lbm * 9.81
    half_span = span / 2

    beam = TubeSparBendingStructure(
        opti=opti,
        length=half_span,
        diameter_function=3.5 * u.inch,  # lambda y: (3.5 * u.inch) - (3.5 - 1.25) * u.inch * (y / half_span),
        points_per_point_load=100,
        bending_distributed_force_function=lambda y: (lift / span) * (
                4 / np.pi * (1 - (y / half_span) ** 2) ** 0.5
        ),  # Elliptical
        # bending_distributed_force_function=lambda y: lift / span * np.ones_like(y) # Uniform,
        elastic_modulus_function=228e9,
    )
    opti.subject_to([
        beam.stress_axial <= 500e6,  # Stress constraint
        beam.u[-1] <= 2,  # Tip displacement constraint
        beam.wall_thickness > 0.1e-3  # Gauge constraint
    ])
    mass = beam.volume() * 1600  # Density of carbon fiber [kg/m^3]

    opti.minimize(mass / (lift / 9.81))
    sol = opti.solve()

    beam = sol(beam)

    print(f"{sol.value(mass)} kg per half-wing")

    beam.draw()

    computed_spar_mass = 2 * sol.value(mass)

    vehicle_mass = lift / 9.81
    ultimate_load_factor = 2

    cruz_estimated_spar_mass = (
            (span * 1.17e-1 + span ** 2 * 1.10e-2) *
            (1 + (ultimate_load_factor * vehicle_mass / 100 - 2) / 4)
    )
