import aerosandbox.numpy as np


def propeller_shaft_power_from_thrust(
        thrust_force,
        area_propulsive,
        airspeed,
        rho,
        propeller_coefficient_of_performance=0.8,
):
    """
    Using dynamic disc actuator theory, gives the shaft power required to generate
    a certain amount of thrust.

    Source: https://web.mit.edu/16.unified/www/FALL/thermodynamics/notes/node86.html

    :param thrust_force: Thrust force [N]
    :param area_propulsive: Total disc area of all propulsive surfaces [m^2]
    :param airspeed: Airspeed [m/s]
    :param rho: Air density [kg/m^3]
    :param propeller_coefficient_of_performance: propeller coeff. of performance (due to viscous losses) [unitless]
    :return: Shaft power [W]
    """
    return 0.5 * thrust_force * airspeed * (
            np.sqrt(
                thrust_force / (area_propulsive * airspeed ** 2 * rho / 2) + 1
            ) + 1
    ) / propeller_coefficient_of_performance


def mass_hpa_propeller(
        diameter,
        max_power,
        include_variable_pitch_mechanism=False
):
    """
    Returns the estimated mass of a propeller assembly for low-disc-loading applications (human powered airplane, paramotor, etc.)

    :param diameter: diameter of the propeller [m]
    :param max_power: maximum power of the propeller [W]
    :param include_variable_pitch_mechanism: boolean, does this propeller have a variable pitch mechanism?
    :return: estimated weight [kg]
    """

    mass_propeller = (
            0.495 *
            (diameter / 1.25) ** 1.6 *
            np.softmax(0.6, max_power / 14914, hardness=5) ** 2
    )  # Baselining to a 125cm E-Props Top 80 Propeller for paramotor, with some sketchy scaling assumptions
    # Parameters on diameter exponent and min power were chosen such that Daedalus propeller is roughly on the curve.

    mass_variable_pitch_mech = 216.8 / 800 * mass_propeller
    # correlation to Daedalus data: http://journals.sfu.ca/ts/index.php/ts/article/viewFile/760/718
    if include_variable_pitch_mechanism:
        mass_propeller += mass_variable_pitch_mech

    return mass_propeller


def mass_gearbox(
        power,
        rpm_in,
        rpm_out,
):
    """
    Estimates the mass of a gearbox.

    Based on data from NASA/TM-2009-215680, available here:
        https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20090042817.pdf

    R^2 = 0.92 to the data.

    To quote this document:
        "The correlation was developed based on actual weight
        data from over fifty rotorcrafts, tiltrotors, and turboprop
        aircraft."

    Data fits in the NASA document were thrown out and refitted to extrapolate more sensibly; see:
        C:\Projects\GitHub\AeroSandbox\studies\GearboxMassFits
    :param power: Shaft power through the gearbox [W]
    :param rpm_in: RPM of the input to the gearbox [rpm]
    :param rpm_out: RPM of the output of the gearbox [rpm]
    :return: Estimated mass of the gearbox [kg]
    """
    power_hp = power / 745.7

    beta = (power_hp / rpm_out) ** 0.75 * (rpm_in / rpm_out) ** 0.15
    # Beta is a parametric value that tends to collapse the masses of gearboxes onto a line.
    # Data fit is considered tightly valid for gearboxes with 1 < beta < 100. Sensible extrapolations are made beyond that.

    p1 = 1.0445171124733774
    p2 = 2.0083615496306910

    mass_lb = 10 ** (p1 * np.log10(beta) + p2)

    mass = mass_lb / 2.20462262185

    return mass


if __name__ == '__main__':
    import matplotlib.style as style

    style.use("seaborn")

    # Daedalus propeller
    print(
        mass_hpa_propeller(
            diameter=3.4442,
            max_power=177.93 * 8.2,  # max thrust at cruise speed
            include_variable_pitch_mechanism=False
        )
    )  # Should weight ca. 800 grams

    print(mass_gearbox(
        power=3000,
        rpm_in=6000,
        rpm_out=600
    ))
