import aerosandbox.numpy as np
from typing import Union, Callable
from scipy.integrate import quad


#         Welcome to the unsteady aerodynamics library!
# In here you will find analytical, time-domain models for the
# unsteady lift response of thin airfoils. Here is a quick overview 
# of what's been implemented so far:

# 1) Unsteady pitching (Wagner's problem)
# 2) Transverse wing-gust encounters (Kussner's problem)
# 3) Added mass 
# 4) Pitching maneuver through a gust (Combination of all 3 models above)

# The models usually take Callable objects as arguments which given the reduced time, return the quantity of 
# interest (Velocity profile, angle of attack etc.). For an explanation of reduced time see function calculate_reduced_time.


# In main() you will find some example gusts as well as example pitchig profiles.
# You can easily build your own and pass them to the appropriate functions  
# to instantly get the lift response! Although not yet implemented, it is possible to 
# calculate an optimal unsteady maneuver through any known disturbance.

# If you run this file as is, the lift history of a flat plate pitching through a 
# top hat gust will be computed.


def calculate_reduced_time(
        time: Union[float, np.ndarray],
        velocity: Union[float, np.ndarray],
        chord: float
) -> Union[float, np.ndarray]:
    """ 
    Calculates reduced time from time in seconds and velocity history in m/s. 
    For constant velocity it reduces to s = 2*U*t/c
    The reduced time is the number of semichords travelled by the airfoil/aircaft 
    i.e. 2 / chord * integral from t0 to t of velocity dt  
    
    
    Args:
        time (float,np.ndarray) : Time in seconds 
        velocity (float,np.ndarray): Either a constant velocity or array of velocities at corresponding reduced times
        chord (float) : The chord of the airfoil
        
    Returns:
        The reduced time as an ndarray or float similar to the input. The first element is 0. 
    """
    if type(velocity) == float or type(velocity) == int:
        return 2 * velocity * time / chord
    else:
        assert np.size(velocity) == np.size(time), "The velocity history and time must have the same length"
        reduced_time = np.zeros_like(time)
        for i in range(len(time) - 1):
            reduced_time[i + 1] = reduced_time[i] + (velocity[i + 1] + velocity[i]) / 2 * (time[i + 1] - time[i])
        return 2 / chord * reduced_time


def wagners_function(reduced_time: Union[float, np.ndarray]):
    """ 
    A commonly used approximation to Wagner's function 
    (Jones, R.T. The Unsteady Lift of a Finite Wing; Technical Report NACA TN-682; NACA: Washington, DC, USA, 1939)
    
    Args:
        reduced_time (float,np.ndarray) : Equal to the number of semichords travelled. See function calculate_reduced_time
    """
    wagner = (1 - 0.165 * np.exp(-0.0455 * reduced_time) -
              0.335 * np.exp(-0.3 * reduced_time)) * np.where(reduced_time >= 0, 1, 0)
    return wagner


def kussners_function(reduced_time: Union[float, np.ndarray]):
    """ 
    A commonly used approximation to Kussner's function (Sears and Sparks 1941)
    
    Args:
        reduced_time (float,np.ndarray) : This is equal to the number of semichords travelled. See function calculate_reduced_time
    """
    kussner = (1 - 0.5 * np.exp(-0.13 * reduced_time) -
               0.5 * np.exp(-reduced_time)) * np.where(reduced_time >= 0, 1, 0)
    return kussner


def indicial_pitch_response(
        reduced_time: Union[float, np.ndarray],
        angle_of_attack: float  # In degrees
):
    """
    Computes the evolution of the lift coefficient in Wagner's problem which can be interpreted as follows
    1) An impulsively started flat plate at constant angle of attack
    2) An impuslive change in the angle of attack of a flat plate at constant velocity
   
    The model predicts infinite added mass at the first instant due to the infinite acceleration
    The delta function term (and therefore added mass) has been ommited in this case.
    Reduced_time = 0 corresponds to the instance the airfoil pitches/accelerates
    
        Args:
        reduced_time (float,np.ndarray) : Reduced time, equal to the number of semichords travelled. See function reduced_time
        angle_of_attack (float) : The angle of attack, in degrees
    """
    return 2 * np.pi * np.deg2rad(angle_of_attack) * wagners_function(reduced_time)


def indicial_gust_response(
        reduced_time: Union[float, np.ndarray],
        gust_velocity: float,
        plate_velocity: float,
        angle_of_attack: float = 0,  # In degrees
        chord: float = 1
):
    """
    Computes the evolution of the lift coefficient of a flat plate entering a 
    an infinitely long, sharp step gust (Heaveside function) at a constant angle of attack. 
    Reduced_time = 0 corresponds to the instance the gust is entered
    
    
    (Leishman, Principles of Helicopter Aerodynamics, S8.10,S8.11)
    
    Args:
        reduced_time (float,np.ndarray) : Reduced time, equal to the number of semichords travelled. See function reduced_time
        gust_velocity (float) : velocity in m/s of the top hat gust
        velocity (float) : velocity of the thin airfoil entering the gust
        angle_of_attack (float) : The angle of attack, in degrees
        chord (float) : The chord of the plate in meters
    """
    angle_of_attack_radians = np.deg2rad(angle_of_attack)
    offset = chord / 2 * (1 - np.cos(angle_of_attack_radians))
    return (2 * np.pi * np.arctan(gust_velocity / plate_velocity) *
            np.cos(angle_of_attack_radians) *
            kussners_function(reduced_time - offset))


def calculate_lift_due_to_transverse_gust(
        reduced_time: np.ndarray,
        gust_velocity_profile: Callable[[float], float],
        plate_velocity: float,
        angle_of_attack: Union[float, Callable[[float], float]] = 0,  # In Degrees
        chord: float = 1
):
    """
    Calculates the lift (as a function of reduced time) caused by an arbitrary transverse gust profile
    by computing duhamel superposition integral of Kussner's problem at a constant angle of attack
    
    Args:
        reduced_time (float,np.ndarray) : Reduced time, equal to the number of semichords travelled. See function reduced_time
        gust_velocity_profile (Callable[[float],float]) : The transverse velocity profile that the flate plate experiences. Must be a function that takes reduced time and returns a velocity
        plate_velocity (float) :The velocity by which the flat plate enters the gust
        angle_of_attack (Union[float,Callable[[float],float]]) : The angle of attack, in degrees. Can either be a float for constant angle of attack or a Callable that takes reduced time and returns angle of attack
        chord (float) : The chord of the plate in meters
    Returns:
        lift_coefficient (np.ndarray) : The lift coefficient history of the flat plate 
    """
    assert type(angle_of_attack) != np.ndarray, "Please provide either a Callable or a float for the angle of attack"

    if isinstance(angle_of_attack, float) or isinstance(angle_of_attack, int):
        def AoA_function(reduced_time):
            return np.deg2rad(angle_of_attack)
    else:
        def AoA_function(reduced_time):
            return np.deg2rad(angle_of_attack(reduced_time))

    def dK_ds(reduced_time):
        return (0.065 * np.exp(-0.13 * reduced_time) +
                0.5 * np.exp(-reduced_time))

    def integrand(sigma, s, chord):
        offset = chord / 2 * (1 - np.cos(AoA_function(s - sigma)))
        return (dK_ds(sigma) *
                gust_velocity_profile(s - sigma - offset) *
                np.cos(AoA_function(s - sigma)))

    lift_coefficient = np.zeros_like(reduced_time)
    for i, s in enumerate(reduced_time):
        I = quad(integrand, 0, s, args=(s, chord))[0]
        lift_coefficient[i] = 2 * np.pi * I / plate_velocity

    return lift_coefficient


def calculate_lift_due_to_pitching_profile(
        reduced_time: np.ndarray,
        angle_of_attack: Union[Callable[[float], float], float]  # In degrees
):
    """
    Calculates the duhamel superposition integral of Wagner's problem. 
    Given some arbitrary pitching profile. The lift coefficient as a function 
    of reduced time of a flat plate can be computed using this function
    
    
    Args:
        reduced_time (float,np.ndarray) : Reduced time, equal to the number of semichords travelled. See function reduced_time
        angle_of_attack (Callable[[float],float]) : The angle of attack as a function of reduced time of the flat plate. Must be a Callable that takes reduced time and returns angle of attack
    Returns:
        lift_coefficient (np.ndarray) : The lift coefficient history of the flat plate 
    """

    assert (reduced_time >= 0).all(), "Please use positive time. Negative time not supported"

    if isinstance(angle_of_attack, float) or isinstance(angle_of_attack, int):
        def AoA_function(reduced_time):
            return np.deg2rad(angle_of_attack)
    else:
        def AoA_function(reduced_time):
            return np.deg2rad(angle_of_attack(reduced_time))

    def dW_ds(reduced_time):
        return (0.1005 * np.exp(-0.3 * reduced_time) +
                0.00750075 * np.exp(-0.0455 * reduced_time))

    def integrand(sigma, s):
        if dW_ds(sigma) < 0:
            dW_ds(sigma)
        return dW_ds(sigma) * AoA_function(s - sigma)

    lift_coefficient = np.zeros_like(reduced_time)

    for i, s in enumerate(reduced_time):

        I = quad(integrand, 0, s, args=s)[0]
        # print(I)
        lift_coefficient[i] = 2 * np.pi * (AoA_function(s) *
                                           wagners_function(0) +
                                           I)

    return lift_coefficient


def added_mass_due_to_pitching(
        reduced_time: np.ndarray,
        angle_of_attack: Callable[[float], float]  # In degrees
):
    """
    This function calculate the lift coefficient due to the added mass of a flat plate
    pitching about its midchord while moving at constant velocity. 
    
    Args:
        reduced_time (np.ndarray) : Reduced time, equal to the number of semichords travelled. See function reduced_time
        angle_of_attack (Callable[[float],float]) : The angle of attack as a function of reduced time of the flat plate
    Returns:
        lift_coefficient (np.ndarray) : The lift coefficient history of the flat plate 
    """

    AoA = np.array([np.deg2rad(angle_of_attack(s)) for s in reduced_time])
    da_ds = np.gradient(AoA, reduced_time)

    # TODO: generalize to all unsteady motion

    return np.pi / 2 * np.cos(AoA) ** 2 * da_ds


def pitching_through_transverse_gust(
        reduced_time: np.ndarray,
        gust_velocity_profile: Callable[[float], float],
        plate_velocity: float,
        angle_of_attack: Union[Callable[[float], float], float],  # In degrees
        chord: float = 1
):
    """
    This function calculates the lift as a function of time of a flat plate pitching
    about its midchord through an arbitrary transverse gust. It combines Kussner's gust response with
    wagners pitch response as well as added mass. 
    
    The following physics are accounted for
    1) Vorticity shed from the trailing edge due to gust profile
    2) Vorticity shed from the trailing edge due to pitching profile
    3) Added mass (non-circulatory force) due to pitching about midchord
    
    The following physics are NOT taken accounted for
    1) Any type of flow separation
    2) Leading edge vorticity shedding
    3) Deflected wake due to gust (flat wake assumption)
    
    
    Args:
        reduced_time (float,np.ndarray) : Reduced time, equal to the number of semichords travelled. See function reduced_time
        gust_velocity_profile (Callable[[float],float]) : The transverse velocity profile that the flate plate experiences. Must be a function that takes reduced time and returns a velocity
        plate_velocity (float) :The velocity by which the flat plate enters the gust
        angle_of_attack (Union[float,Callable[[float],float]]) : The angle of attack, in degrees. Can either be a float for constant angle of attack or a Callable that takes reduced time and returns angle of attack
        chord (float) : The chord of the plate in meters
    
    Returns:
        lift_coefficient (np.ndarray) : The lift coefficient history of the flat plate 
    """
    gust_lift = calculate_lift_due_to_transverse_gust(reduced_time, gust_velocity_profile, plate_velocity,
                                                      angle_of_attack, chord)
    pitch_lift = calculate_lift_due_to_pitching_profile(reduced_time, angle_of_attack)
    added_mass_lift = added_mass_due_to_pitching(reduced_time, angle_of_attack)

    return gust_lift + pitch_lift + added_mass_lift


def top_hat_gust(reduced_time: float) -> float:
    """
    A canonical example gust.
    Args:
        reduced_time (float)
    Returns:
        gust_velocity (float)
    """
    if 5 <= reduced_time <= 10:
        gust_velocity = 1
    else:
        gust_velocity = 0

    return gust_velocity


def sine_squared_gust(reduced_time: float) -> float:
    """
    A canonical gust of used by the FAA to show 'compliance with the 
    requirements of Title 14, Code of Federal Regulations (14 CFR) 25.341, 
    Gust and turbulence loads. Section 25.341 specifies the discrete gust 
    and continuous turbulence dynamic load conditions that apply to the 
    airplane and engines.'
    Args:
        reduced_time (float) 
    Returns:
        gust_velocity (float)
    """
    gust_strength = 1
    start = 5
    finish = 10
    gust_width_to_chord_ratio = 5
    if start <= reduced_time <= finish:
        gust_velocity = (gust_strength *
                         np.sin((np.pi * reduced_time) /
                                gust_width_to_chord_ratio) ** 2)
    else:
        gust_velocity = 0

    return gust_velocity


def gaussian_pitch(reduced_time: float) -> float:
    """
    A pitch maneuver resembling a guassian curve
    Args:
        reduced_time (float) 
    Returns:
        angle_of_attack (float) : in degrees
    """
    return -25 * np.exp(-((reduced_time - 7.5) / 3) ** 2)


def linear_ramp_pitch(reduced_time: float) -> float:
    """
    A pitch maneuver resembling a linear ramp
    Args:
        reduced_time (float) 
    Returns:
        angle_of_attack (float) : in degrees
    """
    if reduced_time < 7.5:
        angle_of_attack = -3.3 * reduced_time
    else:
        angle_of_attack = 2 * reduced_time - 40

    return angle_of_attack


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    time = np.linspace(0, 10, 100)  # Time in seconds
    wing_velocity = 2  # Wing horizontal velocity in m/s
    chord = 2
    reduced_time = calculate_reduced_time(time, wing_velocity, chord)  # Number of semi chords travelled

    # Visualize the gust profiles as well as the pitch maneuvers
    fig, ax1 = plt.subplots(dpi=300)
    ln1 = ax1.plot(reduced_time, np.array([top_hat_gust(s) for s in reduced_time]), label="Top-Hat Gust", lw=3)
    ln2 = ax1.plot(reduced_time, np.array([sine_squared_gust(s) for s in reduced_time]), label="Sine-Squared Gust",
                   lw=3)
    ax1.set_xlabel("Reduced time")
    ax1.set_ylabel("Velocity (m/s)")
    ax2 = ax1.twinx()
    ln3 = ax2.plot(reduced_time, np.array([gaussian_pitch(s) for s in reduced_time]), label="Guassian Pitch", c="red",
                   ls="--", lw=3)
    ax2.set_ylabel("Angle of Attack, degrees")
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc="lower right")
    plt.title("Gust and pitch example profiles")

    total_lift = pitching_through_transverse_gust(reduced_time, top_hat_gust, wing_velocity, gaussian_pitch)
    gust_lift = calculate_lift_due_to_transverse_gust(reduced_time, top_hat_gust, wing_velocity, gaussian_pitch)
    pitch_lift = calculate_lift_due_to_pitching_profile(reduced_time, gaussian_pitch)
    added_mass_lift = added_mass_due_to_pitching(reduced_time, gaussian_pitch)

    # Visualize the different sources of lift
    plt.figure(dpi=300)
    plt.plot(reduced_time, total_lift, label="Total Lift", lw=2)
    plt.plot(reduced_time, gust_lift, label="Gust Lift", lw=2)
    plt.plot(reduced_time, pitch_lift, label="Pitching Lift", lw=2)
    plt.plot(reduced_time, added_mass_lift, label="Added Mass Lift", lw=2)
    plt.legend()
    plt.xlabel("Reduced time")
    plt.ylabel("$C_\ell$")
    plt.title("Guassian Pitch Maneuver Through Top-Hat Gust")
    plt.show()