import matplotlib.pyplot as plt
import aerosandbox.numpy as np
from aerosandbox.common import *
from aerosandbox.library.aerodynamics.unsteady import *


class TransverseGustPitchControl(ImplicitAnalysis):
    """
    An implicit analysis that calculates the optimal pitching maneuver
    through a specified transverse gust, with the goal of minimzing the
    deviation from a specified lift coefficient. It utilizes differentiable
    duhamel superposition integrals for Kussner's gust model and Wagner's
    pitching model, as well as any additional lift from the added mass.

    Args:
        reduced_time (np.ndarray) : Reduced time, equal to the number of semichords travelled. See function reduced_time in the unsteady aero library
        gust_profile (np.ndarray) : An array that specifies the gust velocity at each reduced time
        velocity (float) : The velocity of the aircraft

    """

    @ImplicitAnalysis.initialize
    def __init__(self,
                 reduced_time: np.ndarray,
                 gust_profile: np.ndarray,
                 velocity: float
                 ):
        self.reduced_time = reduced_time
        self.gust_profile = gust_profile
        self.timesteps = len(reduced_time)
        self.velocity = velocity

        self._setup_unknowns()
        self._enforce_governing_equations()

    def _setup_unknowns(self):
        self.angles_of_attack = self.opti.variable(init_guess=1, n_vars=self.timesteps)
        self.lift_coefficients = self.opti.variable(init_guess=1, n_vars=self.timesteps - 1)

    def _enforce_governing_equations(self):
        # Calculate unsteady lift due to pitching 
        wagner = wagners_function(self.reduced_time)
        ds = self.reduced_time[1:] - self.reduced_time[:-1]
        da_ds = (self.angles_of_attack[1:] - self.angles_of_attack[:-1]) / ds
        init_term = self.angles_of_attack[0] * wagner[:-1]
        for i in range(self.timesteps - 1):
            integral_term = np.sum(da_ds[j] * wagner[i - j] * ds[j] for j in range(i))
            self.lift_coefficients[i] = 2 * np.pi * (integral_term + init_term[i])

        # Calculate unsteady lift due to transverse gust 
        kussner = kussners_function(self.reduced_time)
        dw_ds = (self.gust_profile[1:] - self.gust_profile[:-1]) / ds
        init_term = self.gust_profile[0] * kussner
        for i in range(self.timesteps - 1):
            integral_term = 0
            for j in range(i):
                integral_term += dw_ds[j] * kussner[i - j] * ds[j]
            self.lift_coefficients[i] += 2 * np.pi / self.velocity * (init_term[i] + integral_term)

        # Calculate unsteady lift due to added mass
        self.lift_coefficients += np.pi / 2 * np.cos(self.angles_of_attack[:-1]) ** 2 * da_ds

        # Integral of lift to be minimized
        lift_squared_integral = np.sum(self.lift_coefficients ** 2)

        # Constraints and objective to minimize
        self.opti.subject_to(self.angles_of_attack[0] == 0)
        self.opti.minimize(lift_squared_integral)

    def calculate_transients(self):
        self.optimal_pitching_profile_rad = self.opti.value(self.angles_of_attack)
        self.optimal_pitching_profile_deg = np.rad2deg(self.optimal_pitching_profile_rad)
        self.optimal_lift_history = self.opti.value(self.lift_coefficients)

        self.pitching_lift = np.zeros(self.timesteps - 1)
        # Calculate unsteady lift due to pitching 
        wagner = wagners_function(self.reduced_time)
        ds = self.reduced_time[1:] - self.reduced_time[:-1]
        da_ds = (self.optimal_pitching_profile_rad[1:] - self.optimal_pitching_profile_rad[:-1]) / ds
        init_term = self.optimal_pitching_profile_rad[0] * wagner[:-1]
        for i in range(self.timesteps - 1):
            integral_term = np.sum(da_ds[j] * wagner[i - j] * ds[j] for j in range(i))
            self.pitching_lift[i] = 2 * np.pi * (integral_term + init_term[i])

        self.gust_lift = np.zeros(self.timesteps - 1)
        # Calculate unsteady lift due to transverse gust 
        kussner = kussners_function(self.reduced_time)
        dw_ds = (self.gust_profile[1:] - self.gust_profile[:-1]) / ds
        init_term = self.gust_profile[0] * kussner
        for i in range(self.timesteps - 1):
            integral_term = 0
            for j in range(i):
                integral_term += dw_ds[j] * kussner[i - j] * ds[j]
            self.gust_lift[i] += 2 * np.pi / self.velocity * (init_term[i] + integral_term)

        # Calculate unsteady lift due to added mass
        self.added_mass_lift = np.pi / 2 * np.cos(self.optimal_pitching_profile_rad[:-1]) ** 2 * da_ds


if __name__ == "__main__":
    N = 100  # Number of discrete spatial points
    time = np.linspace(0, 10, N)  # Time in seconds
    wing_velocity = 2  # Velocity of wing/aircraft in m/s
    chord = 2
    reduced_time = calculate_reduced_time(time, wing_velocity, chord)
    profile = np.array([top_hat_gust(s) for s in reduced_time])
    optimal = TransverseGustPitchControl(reduced_time, profile, wing_velocity)

    print("Calculating Transients...")
    optimal.calculate_transients()

    fig, ax1 = plt.subplots(dpi=300)
    ax2 = ax1.twinx()
    ax1.plot(reduced_time[:-1], optimal.optimal_lift_history, label="Total Lift", lw=2, c="k")
    ax1.plot(reduced_time[:-1], optimal.gust_lift, label="Gust Lift", lw=2)
    ax1.plot(reduced_time[:-1], optimal.pitching_lift, label="Pitching Lift", lw=2)
    ax1.plot(reduced_time[:-1], optimal.added_mass_lift, label="Added Mass Lift", lw=2)
    ax2.plot(reduced_time, optimal.optimal_pitching_profile_deg, label="Angle of attack", lw=2, ls="--")

    ax2.set_ylim([-40, 40])
    ax1.legend(loc="lower left")
    ax2.legend(loc="lower right")
    ax1.set_xlabel("Reduced time")
    ax1.set_ylabel("$C_\ell$")
    ax2.set_ylabel("Angle of attack, degrees")
    plt.title("Optimal Pitch Maneuver Through Top-Hat Gust")
    plt.show()
