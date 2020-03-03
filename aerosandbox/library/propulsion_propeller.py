import casadi as cas


def propeller_shaft_power_from_thrust(
        thrust_force,
        area_propulsive,
        airspeed,
        rho,
        propeller_efficiency=0.8,
):
    """
    Using dynamic disc actuator theory, gives the shaft power required to generate
    a certain amount of thrust.
    Source: https://web.mit.edu/16.unified/www/FALL/thermodynamics/notes/node86.html
    :param thrust_force: Thrust force [N]
    :param area_propulsive: Total disc area of all propulsive surfaces [m^2]
    :param airspeed: Airspeed [m/s]
    :param rho: Air density [kg/m^3]
    :param propeller_efficiency: propeller efficiency [unitless]
    :return: Shaft power [W]
    """
    return 0.5 * thrust_force * airspeed * (
            cas.sqrt(
                thrust_force / (area_propulsive * airspeed ** 2 * rho / 2) + 1
            ) + 1
    ) / propeller_efficiency


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
    smoothmax = lambda value1, value2, hardness: cas.log(
        cas.exp(hardness * value1) + cas.exp(hardness * value2)) / hardness  # soft maximum

    mass_propeller = (
            0.495 *
            (diameter / 1.25) ** 1.6 *
            smoothmax(0.6, max_power / 14914, hardness=5) ** 2
    )  # Baselining to a 125cm E-Props Top 80 Propeller for paramotor, with some sketchy scaling assumptions
    # Parameters on diameter exponent and min power were chosen such that Daedalus propeller is roughly on the curve.

    mass_variable_pitch_mech = 216.8 / 800 * mass_propeller
    # correlation to Daedalus data: http://journals.sfu.ca/ts/index.php/ts/article/viewFile/760/718
    if include_variable_pitch_mechanism:
        mass_propeller += mass_variable_pitch_mech

    return mass_propeller


if __name__ == '__main__':
    import matplotlib.style as style

    style.use("seaborn")

    # Daedalus propeller
    print(
        mass_hpa_propeller(
            diameter=3.4442,
            max_power=177.93 * 8.2, # max thrust at cruise speed
            include_variable_pitch_mechanism=False
        )
    ) # Should weight ca. 800 grams