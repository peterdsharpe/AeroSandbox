import casadi as cas
import aerosandbox.numpy as np
from aerosandbox.geometry import *
from aerosandbox.common import AeroSandboxObject


class TubeBeam1(AeroSandboxObject):

    def __init__(self,
                 opti,  # type: cas.Opti
                 length,
                 points_per_point_load=100,
                 E=228e9,  # Pa
                 isotropic=True,
                 poisson_ratio=0.5,
                 diameter_guess=100,  # Make this larger for more computational stability, lower for a bit faster speed
                 thickness=0.14e-3 * 5,
                 max_allowable_stress=570e6 / 1.75,
                 density=1600,
                 G=None,
                 bending=True,  # Should we consider beam bending?
                 torsion=True,  # Should we consider beam torsion?
                 ):
        """
        A beam model (static, linear elasticity) that simulates both bending and torsion.

        Governing equation for bending:
        Euler-Bernoulli beam theory.

        (E * I * u(x)'')'' = q(x)

        where:
            * E is the elastic modulus
            * I is the bending moment of inertia
            * u(x) is the local displacement at x.
            * q(x) is the force-per-unit-length at x. (In other words, a dirac delta is a point load.)
            * ()' is a derivative w.r.t. x.

        Governing equation for torsion:
        phi(x)'' = -T / (G * J)

        where:
            * phi is the local twist angle
            * T is the local torque per unit length
            * G is the local shear modulus
            * J is the polar moment of inertia
            * ()' is a derivative w.r.t. x.

        :param opti: An optimization environment. # type: cas.Opti
        :param length: Length of the beam [m]
        :param points_per_point_load: Number of discretization points to use per point load
        :param E: Elastic modulus [Pa]
        :param isotropic: Is the material isotropic? If so, attempts to find shear modulus from poisson's ratio, or vice versa. [boolean]
        :param poisson_ratio: Poisson's ratio (if isotropic, can't set both poisson_ratio and shear modulus - one must be None)
        :param diameter_guess: Initial guess for the tube diameter [m]. Make this larger for more computational stability, lower for a bit faster speed.
        :param thickness: Tube wall thickness. This will often be set by shell buckling considerations. [m]
        :param max_allowable_stress: Maximum allowable stress in the material. [Pa]
        :param density: Density of the material [kg/m^3]
        :param G: Shear modulus (if isotropic, can't set both poisson_ratio and shear modulus - one must be None)
        :param bending: Should we consider bending? [boolean]
        :param torsion: Should we consider torsion? [boolean]
        """
        # Transfer inputs
        self.opti = opti
        self.length = length
        self.points_per_point_load = points_per_point_load
        self.E = E
        self.isotropic = isotropic
        self.poisson_ratio = poisson_ratio
        self.diameter_guess = diameter_guess
        self.thickness = thickness
        self.max_allowable_stress = max_allowable_stress
        self.density = density
        self.G = G
        self.bending = bending
        self.torsion = torsion

        # Calculate G
        if isotropic:
            if G is None:
                self.G = E / 2 / (1 + poisson_ratio)
            elif poisson_ratio is None:
                pass  # TODO find poisson?
            else:
                raise ValueError(
                    "You can't uniquely specify shear modulus and Poisson's ratio on an isotropic material!")

        # Create data structures to track loads
        self.point_loads = []
        self.distributed_loads = []

    def add_point_load(self,
                       location,
                       force=0,
                       bending_moment=0,
                       torsional_moment=0,
                       ):
        """
        Adds a point force and/or moment.
        :param location: Location of the point force along the beam [m]
        :param force: Force to add [N]
        :param bending_moment: Bending moment to add [N-m] # TODO make this work
        :param torsional_moment: Torsional moment to add [N-m] # TODO make this work
        :return: None (in-place)
        """
        self.point_loads.append(
            {
                "location"        : location,
                "force"           : force,
                "bending_moment"  : bending_moment,
                "torsional_moment": torsional_moment
            }
        )

    def add_uniform_load(self,
                         force=0,
                         bending_moment=0,
                         torsional_moment=0,
                         ):
        """
        Adds a uniformly distributed force and/or moment across the entire length of the beam.
        :param force: Total force applied to beam [N]
        :param bending_moment: Bending moment to add [N-m] # TODO make this work
        :param torsional_moment: Torsional moment to add [N-m] # TODO make this work
        :return: None (in-place)
        """
        self.distributed_loads.append(
            {
                "type"            : "uniform",
                "force"           : force,
                "bending_moment"  : bending_moment,
                "torsional_moment": torsional_moment
            }
        )

    def add_elliptical_load(self,
                            force=0,
                            bending_moment=0,
                            torsional_moment=0,
                            ):
        """
        Adds an elliptically distributed force and/or moment across the entire length of the beam.
        :param force: Total force applied to beam [N]
        :param bending_moment: Bending moment to add [N-m] # TODO make this work
        :param torsional_moment: Torsional moment to add [N-m] # TODO make this work
        :return: None (in-place)
        """
        self.distributed_loads.append(
            {
                "type"            : "elliptical",
                "force"           : force,
                "bending_moment"  : bending_moment,
                "torsional_moment": torsional_moment
            }
        )

    def setup(self,
              bending_BC_type="cantilevered"
              ):
        """
        Sets up the problem. Run this last.
        :return: None (in-place)
        """
        ### Discretize and assign loads

        # Discretize
        point_load_locations = [load["location"] for load in self.point_loads]
        point_load_locations.insert(0, 0)
        point_load_locations.append(self.length)
        self.x = cas.vertcat(*[
            cas.linspace(
                point_load_locations[i],
                point_load_locations[i + 1],
                self.points_per_point_load)
            for i in range(len(point_load_locations) - 1)
        ])

        # Post-process the discretization
        self.n = self.x.shape[0]
        dx = cas.diff(self.x)

        # Add point forces
        self.point_forces = cas.GenMX_zeros(self.n - 1)
        for i in range(len(self.point_loads)):
            load = self.point_loads[i]
            self.point_forces[self.points_per_point_load * (i + 1) - 1] = load["force"]

        # Add distributed loads
        self.force_per_unit_length = cas.GenMX_zeros(self.n)
        self.moment_per_unit_length = cas.GenMX_zeros(self.n)
        for load in self.distributed_loads:
            if load["type"] == "uniform":
                self.force_per_unit_length += load["force"] / self.length
            elif load["type"] == "elliptical":
                load_to_add = load["force"] / self.length * (
                        4 / cas.pi * cas.sqrt(1 - (self.x / self.length) ** 2)
                )
                self.force_per_unit_length += load_to_add
            else:
                raise ValueError("Bad value of \"type\" for a load within beam.distributed_loads!")

        # Initialize optimization variables
        log_nominal_diameter = self.opti.variable(self.n)
        self.opti.set_initial(log_nominal_diameter, cas.log(self.diameter_guess))
        self.nominal_diameter = cas.exp(log_nominal_diameter)

        self.opti.subject_to([
            log_nominal_diameter > cas.log(self.thickness)
        ])

        def trapz(x):
            out = (x[:-1] + x[1:]) / 2
            # out[0] += x[0] / 2
            # out[-1] += x[-1] / 2
            return out

        # Mass
        self.volume = cas.sum1(
            cas.pi / 4 * trapz(
                (self.nominal_diameter + self.thickness) ** 2 -
                (self.nominal_diameter - self.thickness) ** 2
            ) * dx
        )
        self.mass = self.volume * self.density

        # Mass proxy
        self.volume_proxy = cas.sum1(
            cas.pi * trapz(
                self.nominal_diameter
            ) * dx * self.thickness
        )
        self.mass_proxy = self.volume_proxy * self.density

        # Find moments of inertia
        self.I = cas.pi / 64 * (  # bending
                (self.nominal_diameter + self.thickness) ** 4 -
                (self.nominal_diameter - self.thickness) ** 4
        )
        self.J = cas.pi / 32 * (  # torsion
                (self.nominal_diameter + self.thickness) ** 4 -
                (self.nominal_diameter - self.thickness) ** 4
        )

        if self.bending:
            # Set up derivatives
            self.u = 1 * self.opti.variable(self.n)
            self.du = 0.1 * self.opti.variable(self.n)
            self.ddu = 0.01 * self.opti.variable(self.n)
            self.dEIddu = 1 * self.opti.variable(self.n)
            self.opti.set_initial(self.u, 0)
            self.opti.set_initial(self.du, 0)
            self.opti.set_initial(self.ddu, 0)
            self.opti.set_initial(self.dEIddu, 0)

            # Define derivatives
            self.opti.subject_to([
                cas.diff(self.u) == trapz(self.du) * dx,
                cas.diff(self.du) == trapz(self.ddu) * dx,
                cas.diff(self.E * self.I * self.ddu) == trapz(self.dEIddu) * dx,
                cas.diff(self.dEIddu) == trapz(self.force_per_unit_length) * dx + self.point_forces,
            ])

            # Add BCs
            if bending_BC_type == "cantilevered":
                self.opti.subject_to([
                    self.u[0] == 0,
                    self.du[0] == 0,
                    self.ddu[-1] == 0,  # No tip moment
                    self.dEIddu[-1] == 0,  # No tip higher order stuff
                ])
            else:
                raise ValueError("Bad value of bending_BC_type!")

            # Stress
            self.stress_axial = (self.nominal_diameter + self.thickness) / 2 * self.E * self.ddu

        if self.torsion:

            # Set up derivatives
            phi = 0.1 * self.opti.variable(self.n)
            dphi = 0.01 * self.opti.variable(self.n)

            # Add forcing term
            ddphi = -self.moment_per_unit_length / (self.G * self.J)

        self.stress = self.stress_axial
        self.opti.subject_to([
            self.stress / self.max_allowable_stress < 1,
            self.stress / self.max_allowable_stress > -1,
        ])

    def draw_bending(self,
                     show=True,
                     for_print=False,
                     equal_scale=True,
                     ):
        """
        Draws a figure that illustrates some bending properties. Must be called on a solved object (i.e. using the substitute_sol method).
        :param show: Whether or not to show the figure [boolean]
        :param for_print: Whether or not the figure should be shaped for printing in a paper [boolean]
        :param equal_scale: Whether or not to make the displacement plot have equal scale (i.e. true deformation only)
        :return:
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(font_scale=1)

        fig, ax = plt.subplots(
            2 if not for_print else 3,
            3 if not for_print else 2,
            figsize=(
                10 if not for_print else 6,
                6 if not for_print else 6
            ),
            dpi=200
        )

        plt.subplot(231) if not for_print else plt.subplot(321)
        plt.plot(self.x, self.u, '.-')
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$u$ [m]")
        plt.title("Displacement (Bending)")
        if equal_scale:
            plt.axis("equal")

        plt.subplot(232) if not for_print else plt.subplot(322)
        plt.plot(self.x, np.arctan(self.du) * 180 / np.pi, '.-')
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"Local Slope [deg]")
        plt.title("Slope")

        plt.subplot(233) if not for_print else plt.subplot(323)
        plt.plot(self.x, self.force_per_unit_length, '.-')
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$q$ [N/m]")
        plt.title("Local Load per Unit Span")

        plt.subplot(234) if not for_print else plt.subplot(324)
        plt.plot(self.x, self.stress_axial / 1e6, '.-')
        plt.xlabel(r"$x$ [m]")
        plt.ylabel("Axial Stress [MPa]")
        plt.title("Axial Stress")

        plt.subplot(235) if not for_print else plt.subplot(325)
        plt.plot(self.x, self.dEIddu, '.-')
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$F$ [N]")
        plt.title("Shear Force")

        plt.subplot(236) if not for_print else plt.subplot(326)
        plt.plot(self.x, self.nominal_diameter, '.-')
        plt.xlabel(r"$x$ [m]")
        plt.ylabel("Diameter [m]")
        plt.title("Optimal Spar Diameter")
        plt.tight_layout()

        plt.show() if show else None


if __name__ == '__main__':
    opti = cas.Opti()
    beam = TubeBeam1(
        opti=opti,
        length=60 / 2,
        points_per_point_load=50,
        diameter_guess=100,
        bending=True,
        torsion=False
    )
    lift_force = 9.81 * 103.873
    load_location = opti.variable()
    opti.set_initial(load_location, 15)
    opti.subject_to([
        load_location > 2,
        load_location < 60 / 2 - 2,
        load_location == 18,
    ])
    beam.add_point_load(load_location, -lift_force / 3)
    beam.add_uniform_load(force=lift_force / 2)
    beam.setup()

    # Tip deflection constraint
    opti.subject_to([
        # beam.u[-1] < 2,  # Source: http://web.mit.edu/drela/Public/web/hpa/hpa_structure.pdf
        # beam.u[-1] > -2  # Source: http://web.mit.edu/drela/Public/web/hpa/hpa_structure.pdf
        beam.du * 180 / cas.pi < 10,
        beam.du * 180 / cas.pi > -10
    ])
    opti.subject_to([
        cas.diff(cas.diff(beam.nominal_diameter)) < 0.001,
        cas.diff(cas.diff(beam.nominal_diameter)) > -0.001,
    ])

    # opti.minimize(cas.sqrt(beam.mass))
    opti.minimize(beam.mass)
    # opti.minimize(beam.mass ** 2)
    # opti.minimize(beam.mass_proxy)

    p_opts = {}
    s_opts = {}
    s_opts["max_iter"] = 1e6  # If you need to interrupt, just use ctrl+c
    # s_opts["bound_frac"] = 0.5
    # s_opts["bound_push"] = 0.5
    # s_opts["slack_bound_frac"] = 0.5
    # s_opts["slack_bound_push"] = 0.5
    # s_opts["mu_strategy"] = "adaptive"
    # s_opts["mu_oracle"] = "quality-function"
    # s_opts["quality_function_max_section_steps"] = 20
    # s_opts["fixed_mu_oracle"] = "quality-function"
    # s_opts["alpha_for_y"] = "min"
    # s_opts["alpha_for_y"] = "primal-and-full"
    # s_opts["watchdog_shortened_iter_trigger"] = 1
    # s_opts["expect_infeasible_problem"]="yes" # TODO remove all this
    # s_opts["start_with_resto"] = "yes"
    # s_opts["required_infeasibility_reduction"] = 0.001
    # s_opts["evaluate_orig_obj_at_resto_trial"] = "yes"
    opti.solver('ipopt', p_opts, s_opts)

    try:
        sol = opti.solve()
    except Exception:
        print("Failed!")
        sol = opti.debug

    import copy

    beam_sol = sol(beam)

    print(f"Beam mass: {beam_sol.mass} kg")
    beam_sol.draw_bending()

    bs = beam_sol
