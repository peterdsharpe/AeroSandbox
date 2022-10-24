import aerosandbox as asb
import aerosandbox.numpy as np
from typing import Callable, Union, Dict


class TubeSparBendingStructure(asb.ImplicitAnalysis):

    @asb.ImplicitAnalysis.initialize
    def __init__(self,
                 length: float,
                 diameter_function: Union[float, Callable[[np.ndarray], np.ndarray]] = None,
                 thickness_function: Union[float, Callable[[np.ndarray], np.ndarray]] = None,
                 bending_point_forces: Dict[float, float] = None,
                 bending_distributed_force_function: Union[float, Callable[[np.ndarray], np.ndarray]] = 0.,
                 points_per_point_load: int = 20,
                 elastic_modulus_function: Union[float, Callable[[np.ndarray], np.ndarray]] = 175e9,  # Pa
                 allowable_stress_function: Union[float, Callable[[np.ndarray], np.ndarray]] = 925e6 / 3,  # Pa
                 ignore_buckling: bool = True,
                 EI_guess: float = None,
                 assume_thin_tube=True,
                 ):
        ### Parse the inputs
        self.length = length
        self.diameter_function = diameter_function
        self.thickness_function = thickness_function

        if bending_point_forces is not None:
            self.bending_point_forces = bending_point_forces
            raise NotImplementedError
        else:
            self.bending_point_forces = dict()

        self.bending_distributed_force_function = bending_distributed_force_function
        self.points_per_point_load = points_per_point_load
        self.elastic_modulus_function = elastic_modulus_function
        self.allowable_stress_function = allowable_stress_function
        self.ignore_buckling = ignore_buckling

        if EI_guess is None:
            try:
                diameter_guess = float(diameter_function)
            except (TypeError, RuntimeError):
                diameter_guess = 1

            try:
                thickness_guess = float(thickness_function)
            except (TypeError, RuntimeError):
                thickness_guess = 0.01

            try:
                E_guess = float(elastic_modulus_function)
            except (TypeError, RuntimeError):
                E_guess = 175e9

            if assume_thin_tube:
                I_guess = np.pi / 8 * diameter_guess ** 3 * thickness_guess
            else:
                I_guess = np.pi / 64 * (
                        (diameter_guess + thickness_guess) ** 4 -
                        (diameter_guess - thickness_guess) ** 4
                )
            EI_guess = E_guess * I_guess

            # EI_guess *= 1e0  # A very high EI guess is numerically stabilizing

        self.EI_guess = EI_guess

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

        if isinstance(thickness_function, Callable):
            thickness = thickness_function(y)
        elif thickness_function is None:
            thickness = self.opti.variable(init_guess=1e-3, n_vars=N, lower_bound=1.001 * diameter)
        else:
            thickness = thickness_function * np.ones_like(y)

        if isinstance(bending_distributed_force_function, Callable):
            distributed_force = bending_distributed_force_function(y)
        else:
            distributed_force = bending_distributed_force_function * np.ones_like(y)

        if isinstance(elastic_modulus_function, Callable):
            elastic_modulus = elastic_modulus_function(y)
        else:
            elastic_modulus = elastic_modulus_function * np.ones_like(y)

        if isinstance(allowable_stress_function, Callable):
            allowable_stress = allowable_stress_function(y)
        else:
            allowable_stress = allowable_stress_function * np.ones_like(y)

        ### Evaluate the beam properties
        if assume_thin_tube:
            I = np.pi / 8 * diameter ** 3 * thickness
        else:
            I = np.pi / 64 * (
                    (diameter + thickness) ** 4 -
                    (diameter - thickness) ** 4
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

        stress_axial = elastic_modulus * ddu * (diameter + thickness) / 2

        self.y = y
        self.diameter = diameter
        self.thickness = thickness
        self.u = u
        self.du = du
        self.ddu = ddu
        self.dEIddu = dEIddu
        self.bending_moment = bending_moment
        self.shear_force = shear_force
        self.stress_axial = stress_axial

    def volume(self):
        return np.sum(
            np.pi / 4 * np.trapz(
                (self.diameter + self.thickness) ** 2 -
                (self.diameter - self.thickness) ** 2
            ) * np.diff(self.y)
        )


if __name__ == '__main__':
    opti = asb.Opti()

    beam = TubeSparBendingStructure(
        opti=opti,
        length=34,
        diameter_function=0.1,
        # thickness_function=1e-3,
        thickness_function=opti.variable(
            init_guess=1e-3,
            n_vars=20,
            lower_bound=0.
        ),
        bending_distributed_force_function=lambda y: 200 * 9.81 / 34 * np.ones_like(y),
    )
    opti.subject_to(beam.stress_axial <= 500e6)
    opti.minimize(beam.volume())

    try:
        sol = opti.solve()
    except RuntimeError:
        sol = opti.debug
    beam.substitute_solution(sol)
