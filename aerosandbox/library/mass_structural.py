def mass_hpa_wing(
        span,
        chord,
        vehicle_mass,
        n_ribs,  # You should optimize on this, there's a trade between rib weight and LE sheeting weight!
        n_wing_sections=1,  # defaults to a single-section wing (be careful: can you disassemble/transport this?)
        ultimate_load_factor=1.75,  # default taken from Daedalus design
        type="cantilevered",  # "cantilevered", "one-wire", "multi-wire"
        t_over_c=0.128,  # default from DAE11
        include_spar=True, # Should we include the mass of the spar? Useful if you want to do your own primary structure calculations.
):
    """
    Finds the mass of the wing structure of a human powered aircraft (HPA), following Juan Cruz's correlations in
    http://journals.sfu.ca/ts/index.php/ts/article/viewFile/760/718
    :param span: wing span [m]
    :param chord: wing mean chord [m]
    :param vehicle_mass: aircraft gross weight [kg]
    :param n_ribs: number of ribs in the wing
    :param n_wing_sections: number of wing sections or panels (for disassembly?)
    :param ultimate_load_factor: ultimate load factor [unitless]
    :param type: Type of bracing: "cantilevered", "one-wire", "multi-wire"
    :param t_over_c: wing airfoil thickness-to-chord ratio
    :param include_spar: Should we include the mass of the spar? Useful if you want to do your own primary structure calculations.
    :return: Wing structure mass [kg]
    """
    ### Primary structure
    if include_spar:
        if type == "cantilevered":
            mass_primary_spar = (
                    (span * 1.17e-1 + span ** 2 * 1.10e-2) *
                    (1 + (ultimate_load_factor * vehicle_mass / 100 - 2) / 4)
            )
        elif type == "one-wire":
            mass_primary_spar = (
                    (span * 3.10e-2 + span ** 2 * 7.56e-3) *
                    (1 + (ultimate_load_factor * vehicle_mass / 100 - 2) / 4)
            )
        elif type == "multi-wire":
            mass_primary_spar = (
                    (span * 1.35e-1 + span ** 2 * 1.68e-3) *
                    (1 + (ultimate_load_factor * vehicle_mass / 100 - 2) / 4)
            )
        else:
            raise ValueError("Bad input for 'type'!")

        mass_primary = mass_primary_spar * (
                11382.3 / 9222.2)  # accounts for rear spar, struts, fittings, kevlar x-bracing, and wing-fuselage mounts
    else:
        mass_primary = 0


    ### Secondary structure
    ratio_of_rib_spacing_to_chord = (span / n_ribs) / chord
    n_end_ribs = 2 * n_wing_sections - 2
    area = span * chord

    # Rib mass
    W_wr = n_ribs * (chord ** 2 * t_over_c * 5.50e-2 + chord * 1.91e-3)

    # End rib mass
    W_wer = n_end_ribs * (chord ** 2 * t_over_c * 6.62e-1 + chord * 6.57e-3)

    # LE sheeting mass
    W_wLE = 0.456 * (span ** 2 * ratio_of_rib_spacing_to_chord ** (4 / 3) / span)

    # TE mass
    W_wTE = span * 2.77e-2

    # Covering
    W_wc = area * 3.08e-2

    mass_secondary = W_wr + W_wer + W_wLE + W_wTE + W_wc

    return mass_primary + mass_secondary


def mass_hpa_stabilizer(
        span,
        chord,
        dynamic_pressure_at_manuever_speed,
        n_ribs,  # You should optimize on this, there's a trade between rib weight and LE sheeting weight!
        t_over_c=0.128,  # default from DAE11
):
    """
    Finds the mass of a stabilizer structure of a human powered aircraft (HPA), following Juan Cruz's correlations in
    http://journals.sfu.ca/ts/index.php/ts/article/viewFile/760/718
    Note: apply this once to BOTH the rudder and elevator!!!
    :param span: stabilizer span [m]
    :param chord: stabilizer mean chord [m]
    :param dynamic_pressure_at_manuever_speed: dynamic pressure at maneuvering speed [Pa]
    :param n_ribs: number of ribs in the wing
    :param t_over_c: wing airfoil thickness-to-chord ratio
    :return: Stabilizer structure mass [kg]
    """
    ### Primary structure
    area = span * chord
    q = dynamic_pressure_at_manuever_speed
    W_tss = (
            (span * 4.15e-2 + span ** 2 * 3.91e-3) *
            (1 + ((q * area) / 78.5 - 1) / 2)
    )

    mass_primary = W_tss

    ### Secondary structure
    ratio_of_rib_spacing_to_chord = (span / n_ribs) / chord

    # Rib mass
    W_tsr = n_ribs * (chord ** 2 * t_over_c * 1.16e-1 + chord * 4.01e-3)

    # Leading edge sheeting
    W_tsLE = 0.174 * (area ** 2 * ratio_of_rib_spacing_to_chord ** (4 / 3) / span)

    # Covering
    W_tsc = area * 1.93e-2

    mass_secondary = W_tsr + W_tsLE + W_tsc

    ### Totaling
    correction_factor = ((537.8 / (537.8 - 23.7 - 15.1)) * (623.3 / (623.3 - 63.2 - 8.1))) ** 0.5
    # geometric mean of Daedalus elevator and rudder corrections from misc. weight

    return (mass_primary + mass_secondary) * correction_factor


def mass_hpa_tail_boom(
        length_tail_boom,
        dynamic_pressure_at_manuever_speed,
        mean_tail_surface_area,
):
    """
    Finds the mass of a tail boom structure of a human powered aircraft (HPA), following Juan Cruz's correlations in
    http://journals.sfu.ca/ts/index.php/ts/article/viewFile/760/718
    Assumes a tubular tail boom of high modules (E > 228 GPa) graphite/epoxy
    :param length_tail_boom: length of the tail boom [m]. Calculated as distance from the wing 1/4 chord to the furthest tail surface.
    :param dynamic_pressure_at_manuever_speed: dynamic pressure at maneuvering speed [Pa]
    :param mean_tail_surface_area: mean of the areas of the tail surfaces (elevator, rudder)
    :return: mass of the tail boom [m]
    """
    l = length_tail_boom
    q = dynamic_pressure_at_manuever_speed
    area = mean_tail_surface_area
    w_tb = (l * 1.14e-1 + l ** 2 * 1.96e-2) * (1 + ((q * area) / 78.5 - 1) / 2)

    return w_tb


def mass_surface_balsa_monokote_cf(
        chord,
        span,
        mean_t_over_c=0.08
):
    """
    Estimates the mass of a lifting surface constructed with balsa-monokote-carbon-fiber construction techniques.
    Warning: Not well validated; spar sizing is a guessed scaling and not based on structural analysis.
    :param chord: wing mean chord [m]
    :param span: wing span [m]
    :param mean_t_over_c: wing thickness-to-chord ratio [unitless]
    :return: estimated surface mass [kg]
    """
    mean_t = chord * mean_t_over_c
    ### Balsa wood + Monokote + a 1" dia CF tube spar.
    monokote_mass = 0.061 * chord * span * 2  # 0.2 oz/sqft

    rib_density = 200  # mass density, in kg/m^3
    rib_spacing = 0.1  # one rib every x meters
    rib_width = 0.003  # width of an individual rib
    ribs_mass = (
            (mean_t * chord * rib_width) *  # volume of a rib
            rib_density *  # density of a rib
            (span / rib_spacing)  # number of ribs
    )

    spar_mass_1_inch = 0.2113 * span * 1.5  # assuming 1.5x 1" CF tube spar
    spar_mass = spar_mass_1_inch * (
            mean_t / 0.0254) ** 2  # Rough GUESS for scaling, FIX THIS before using seriously!

    return (monokote_mass + ribs_mass + spar_mass) * 1.2  # for glue


def mass_surface_solid(
        chord,
        span,
        density = 2700, # kg/m^3, defaults to that of aluminum
        mean_t_over_c=0.08
):
    """
    Estimates the mass of a lifting surface constructed out of a solid piece of material.
    Warning: Not well validated; spar sizing is a guessed scaling and not based on structural analysis.
    :param chord: wing mean chord [m]
    :param span: wing span [m]
    :param mean_t_over_c: wing thickness-to-chord ratio [unitless]
    :return: estimated surface mass [kg]
    """
    mean_t = chord * mean_t_over_c
    volume = chord * span * mean_t
    return density * volume


if __name__ == "__main__":
    import casadi as cas
    import numpy as np
    import matplotlib.pyplot as plt

    # Daedalus wing mass validation
    print(
        "Daedalus wing, estimated mass: %f" %
        mass_hpa_wing(
            span=34,
            chord=0.902,
            vehicle_mass=104.1,
            n_ribs=100,
            n_wing_sections=5,
            type="one-wire"
        )
    )
    print(
        "Daedalus wing, actual mass: %f" % 18.9854
    )

    nr = np.linspace(1, 400, 401)
    m = mass_hpa_wing(
        span=34,
        chord=0.902,
        vehicle_mass=104.1,
        n_ribs=nr,
        n_wing_sections=5,
        type="one-wire"
    )
    plt.plot(nr, m)
    plt.ylim([15, 20])
    plt.grid(True)
    plt.xlabel("Number of ribs")
    plt.ylabel("Wing mass [kg]")
    plt.title("Daedalus Wing Rib Count Optimization Test")
    plt.show()

    # Test rib number optimization
    opti = cas.Opti()
    nr_opt = opti.variable()
    opti.set_initial(nr_opt, 100)
    opti.minimize(mass_hpa_wing(
        span=34,
        chord=0.902,
        vehicle_mass=104.1,
        n_ribs=nr_opt,
        n_wing_sections=5,
        type="one-wire"
    ))
    opti.solver('ipopt')
    sol = opti.solve()
    print("Optimal number of ribs: %f" % sol.value(nr_opt))

    print(
        "Daedalus elevator, estimated mass: %f" %
        mass_hpa_stabilizer(
            span=4.26,
            chord=0.6,
            dynamic_pressure_at_manuever_speed=1 / 2 * 1.225 * 7 ** 2,
            n_ribs=20,
        )
    )
