import numpy as np
from typing import List
import aerosandbox as asb
from aerosandbox.atmosphere import Atmosphere as atmo
import sympy as sym
from aerosandbox import cas
from numpy import pi

airspeed = 10 # meters per second
rpm = 10000
altitude = 0 # meters
# air_density = atmo.get_density_at_altitude(altitude)
# mu = atmo.get_viscosity_from_temperature(atmo.get_temperature_at_altitude(altitude))
# speed_of_sound = 343

air_density = 1.225
mu = 0.178E-04
speed_of_sound = 340

## Dongjoon Prop Specs from 16.82 spring 2020
# propeller_diameter = 1.65
# n_blades = 2
# radial_locations_m = np.array([0.275, 0.55, 0.825, 1.1, 1.375, 1.65])
# blade_chord_m = np.array([0.055, 0.132, 0.176, 0.121, 0.077, 0.033])
# blade_beta_deg = np.array([70, 55, 35, 25, 20, 18])
# dBeta_deg = 0

## Prop Specs from CAM 6X3 for QPROP Validation
n_blades = 2 # number of blades
# give value in inches for some number of radial locations from root to tip
# tip radial location is propeller radius
radial_locations_in = np.array([0.75, 1, 1.5, 2, 2.5, 2.875, 3])
radial_locations_m = np.array([0.01905, 0.0254, 0.0381, 0.0508, 0.0635, 0.073025, 0.0762])
divisions = 4
# # give value of blade chord in inches for each station
blade_chord_in = np.array([0.66, 0.69, 0.63, 0.55, 0.44, 0.30, 0.19])
blade_chord_m = np.array([0.016764, 0.017526, 0.016002, 0.01397, 0.011176, 0.00762, 0.004826])
# # give value of blade beta in degrees for each station
blade_beta_deg = np.array([27.5, 22, 15.2, 10.2, 6.5, 4.6, 4.2])
# # variable pitch angle
dBeta_deg = 0


##Qprop air Parameters
# air_density = 1.225
# mu = 1.78E-5
# speed_of_sound = 340

def annick_propulsion_model(
        rpm: float,
        airspeed: float,
        air_density: float,
        mu: float,
        n_blades: int,
        radial_locations_m: np.ndarray,
        blade_chord_m: np.ndarray,
        blade_beta_deg: np.ndarray,
        dBeta_deg: float,
        divisions: float,
) -> [float, float]:
    """
    Ideally:
        * Physics-based where possible
        * Where fitted correction factors need to be added, add them nondimensionally
        * Theory from Drela's QPROP Formulation document found here:
            http://web.mit.edu/drela/Public/web/qprop/qprop_theory.pdf
    :param rpm: prop speed in revolutions per minute
    :param airspeed: m/s
    :param air_density:
    :param mu:
    :param n_blades:
    :param blade_chord:
    :param blade_twist:
    :param dBeta:
    :param divisions:
    :return:
    """

    # ## original CL function
    # def airfoil_CL(alpha, Re, Ma):
    #     alpha_rad = alpha * pi / 180
    #     Cl = 2 * pi * alpha_rad
    #     return Cl

    # Interpolation function
    def interpolate(radial_locations, blade_chords, blade_betas, div):
        radial_locations_new = np.array([])
        blade_chords_new = np.array([])
        blade_betas_new = np.array([])
        for n in range(len(radial_locations) - 1):
            r1 = radial_locations_m[n]
            r2 = radial_locations_m[n + 1]
            c1 = blade_chords[n]
            c2 = blade_chords[n + 1]
            b1 = blade_betas[n]
            b2 = blade_betas[n + 1]
            for i in range(0, div):
                radial_loc = r1 + (r2 - r1) * i / div
                radial_locations_new = np.append(radial_locations_new, radial_loc)
                chord = c1 + (radial_loc - r1) * (c2 - c1) / (r2 - r1)
                blade_chords_new = np.append(blade_chords_new, chord)
                beta = b1 + (radial_loc - r1) * (b2 - b1) / (r2 - r1)
                blade_betas_new = np.append(blade_betas_new, beta)
        radial_locations_new = np.append(radial_locations_new, r2)
        blade_chords_new = np.append(blade_chords_new, c2)
        blade_betas_new = np.append(blade_betas_new, b2)

        return radial_locations_new, blade_chords_new, blade_betas_new

    #QPROP CL function
    def airfoil_CL(alpha, Re, Ma):
        alpha_rad = alpha * pi / 180
        beta = (1 - Ma ** 2) ** 0.5
        cl_0 = 0.5
        cl_alpha = 5.8
        cl_min = -0.3
        cl_max = 1.2
        cl = (alpha_rad * cl_alpha + cl_0) / beta
        Cl = cas.fmin(cas.fmax(cl, cl_min), cl_max)
        return Cl

    # ## Peter Sharpe's CDp model
    # def airfoil_CDp(alpha, Re, Ma, Cl):
    #     Re_exp = -0.5
    #     Re_ref = 1e6
    #     alpha_ref = 5
    #     cd_0 = 0.00540
    #     cd_a2 = 0.00848 - cd_0
    #     Cd = (
    #                  cd_a2 * (alpha / alpha_ref) ** 2 + cd_0
    #          ) * (Re / Re_ref) ** Re_exp
    #     return Cd

    ## QPROP CDp model

    def airfoil_CDp(alpha, Re, Ma, Cl):
        alpha_rad = alpha * pi / 180
        Re_exp = -0.7
        Re_ref = 70000
        cd_0 = 0.028
        cd_2 = 0.05
        #cd_2 = 0.05
        cl_cd_0 = 0.5
        cl_0 = 0.5
        cl_alpha = 5.8
        cl_min = -0.3
        cl_max = 1.2
        # cd = (cd_0 + cd_2 * (cl - cl_cd_0) ** 2) * (Re / Re_ref) ** Re_exp
        cd = (cd_0 + cd_2 * (Cl - cl_cd_0) ** 2) * (Re / Re_ref) ** Re_exp
        aCD0 = (cl_cd_0 - cl_0) / cl_alpha
        dcd_stall = 2 * (cas.sin(alpha - aCD0)) ** 2
        if cas.is_equal(Cl, cl_max):
             cd = dcd_stall + cd
        if cas.is_equal(Cl, cl_min):
             cd = dcd_stall + cd
        return cd

    radial_locations_m, blade_chord_m, blade_beta_deg = interpolate(radial_locations_m, blade_chord_m, blade_beta_deg, divisions)
    n_stations = len(radial_locations_m) - 1
    tip_radius = radial_locations_m[n_stations] #use tip radial location as prop radius
    omega = rpm * 2 * pi / 60  # radians per second
    blade_twist_deg = blade_beta_deg + dBeta_deg
    blade_twist_rad = blade_twist_deg * pi / 180
    torque = []
    thrust = []

    for station in range(n_stations): # TODO undo this
    # for station in [22]:
        # radial_loc = radial_locations_m[station]
        radial_loc = (radial_locations_m[station] + radial_locations_m[station + 1]) / 2
        blade_section = (radial_locations_m[station + 1] - radial_locations_m[station])
        chord_local = (blade_chord_m[station] + blade_chord_m[station + 1]) / 2
        twist_local_rad = (blade_twist_rad[station] + blade_twist_rad[station + 1]) / 2

        opti = asb.Opti()
        # v_a = opti.variable(init_guess=15)
        # v_t = opti.variable(init_guess=15)
        # u_a = opti.variable(init_guess=5)
        Psi = opti.variable(init_guess=pi/2)
        # h_ati = opti.variable(init_guess=0.01)
        # f = opti.variable(init_guess=300)
        # F = opti.variable(init_guess=1)
        # gamma = opti.variable(init_guess=1)

        ### Define velocity triangle components
        U_a = airspeed #+ u_a # Axial velocity w/o induced eff. assuming u_a = 0
        U_t = omega * radial_loc  # Tangential velocity w/o induced eff.
        U = (U_a ** 2 + U_t ** 2) ** 0.5  # Velocity magnitude
        W_a = 0.5 * U_a + 0.5 * U * cas.sin(Psi)  # Axial velocity w/ induced eff.
        W_t = 0.5 * U_t + 0.5 * U * cas.cos(Psi)  # Tangential velocity w/ induced eff.
        W = (W_a ** 2 + W_t ** 2) ** 0.5
        v_a = W_a - U_a  # Axial induced velocity
        v_t = U_t - W_t  # Tangential induced velocity
        v = (v_a ** 2 + v_t ** 2) ** 0.5

        # alt method (adapted from gp propeller model) less accurate for zero airspeed
        # W_a = airspeed + v_a
        # W_t = omega * radial_loc + v_t
        # W = (W_a ** 2 + W_t ** 2) ** 0.5
        # v = (v_a ** 2 + v_t ** 2) ** 0.5
        eta_i = (airspeed / omega * radial_loc) * (W_t / W_a)
        loc_wake_adv_ratio = (radial_loc / tip_radius) * (W_a / W_t)
        f = (n_blades / 2) * (1 - radial_loc / tip_radius) * 1 / loc_wake_adv_ratio
        # f = - (radial_loc / tip_radius) * n_blades / (2 * loc_wake_adv_ratio) + (n_blades / 2) * (1 / loc_wake_adv_ratio)
        F = 2 / pi * cas.arccos(cas.exp(-f))


        ## Compute local blade quantities
        phi_rad = cas.arctan2(W_a, W_t) #local flow angle
        phi_deg = phi_rad * 180 / pi
        alpha_rad = twist_local_rad - phi_rad
        alpha_deg = alpha_rad * 180 / pi
        #alpha_deg = -6.1
        #alpha_deg = 240
        Ma = W / speed_of_sound
        Re = air_density * W * chord_local / mu

        ### Compute sectional lift and drag
        cl = airfoil_CL(alpha_deg, Re, Ma)
        cd = airfoil_CDp(alpha_deg, Re, Ma, cl)
        gamma = 0.5 * W * chord_local * cl

        ### Add governing equations
        opti.subject_to([
            # 0.5 * v == 0.5 * U * cas.sin(Psi / 4),
            # v_a == v_t * W_t / W_a,
            # U ** 2 == v ** 2 + W ** 2,
            # gamma == -0.0145,
            #gamma ==  (4 * pi * radial_loc / n_blades) * F * (
                       # 1 + ((4 * loc_wake_adv_ratio * tip_radius) / (pi * n_blades * radial_loc)) ** 2) ** 0.5,

            gamma == v_t * (4 * pi * radial_loc / n_blades) * F * (1 + ((4 * loc_wake_adv_ratio * tip_radius) / (pi * n_blades * radial_loc)) ** 2) ** 0.5,
            # vt**2*F**2*(1.+(4.*lam_w*R/(pi*B*r))**2) >= (B*G/(4.*pi*r))**2,
            # f + (radial_loc / tip_radius) * n_blades / (2 * loc_wake_adv_ratio) <= (n_blades / 2) * (1 / loc_wake_adv_ratio),
            #blade_twist_deg * pi / 180 == alpha_rad + 1 / h_ati,
            #h_ati ** 1.83442 == 0.966692 * (W_a / W_t) ** -1.84391 + 0.596688 * (W_a / W_t) ** -0.0973781,
            #v_t ** 2 * F ** 2 * (1 + (4 * loc_wake_adv_ratio * tip_radius/(pi * n_blades * radial_loc)) ** 2) >= (n_blades * gamma /(4 * pi * radial_loc)) ** 2,
            # alpha_deg >= -45
            # v_a >= 0,
            # v_t >= 0
        ])

        ### Solve
        sol = opti.solve()

        ### Compute sectional quantities
        # dLift = sol.value(
        #     n_blades * 0.5 * air_density * (W ** 2) *
        #     cl * chord_local * blade_section
        # )
        # dDrag = sol.value(
        #     n_blades * 0.5 * air_density * (W ** 2) *
        #     cd * chord_local * blade_section
        # )
        dThrust = sol.value(
            air_density * n_blades * gamma * (
                    W_t - W_a * cd / cl
            ) * blade_section
        )
        # dThrust2 = sol.value(
        #     n_blades * 0.5 * air_density * (W ** 2) *
        #     (cl * cas.cos(phi_rad) - cd * cas.sin(phi_rad))
        #     * chord_local * blade_section
        # )
        # dThrust3 = sol.value(
        #     dLift * cas.cos(phi_rad) - dDrag * cas.sin(phi_rad)
        # )
        dTorque = sol.value(
            air_density * n_blades * gamma * (
                    W_a + W_t * cd / cl
            ) * radial_loc * blade_section
        )
        # dTorque2 = sol.value(
        #     n_blades * 0.5 * (W ** 2) *
        #     (cl * cas.sin(phi_rad) + cd * cas.cos(phi_rad))
        #     * chord_local * radial_loc * blade_section
        # )
        # dTorque3 = sol.value(
        #     (dLift * cas.sin(phi_rad) + dDrag * cas.cos(phi_rad)) * radial_loc
        # )
        # if sol.value(alpha_deg) <= 0:
        #     break

        thrust.append(dThrust)
        torque.append(dTorque)
        # thrust2.append(dThrust2)
        # torque2.append(dTorque2)
        # thrust3.append(dThrust3)
        # torque3.append(dTorque3)

    Thrust = sum(thrust)
    Torque = sum(torque)
    # Thrust2 = sum(thrust2)
    # Torque2 = sum(torque2)
    # Thrust3 = sum(thrust3)
    # Torque3 = sum(torque3)
    #efficiency = airspeed * Thrust / (omega * Torque)
    if len(thrust) > 1:
        print(f"Thrust: {thrust}")
        print(f"Torque: {torque}")
        print(f"Thrust Total: {Thrust}")
        print(f"Torque Total: {Torque}")
    else:
        print(f"radial location: {radial_loc}")
        print(f"alpha (deg): {opti.value(alpha_deg)}")
        print(f"phi (deg): {opti.value(phi_deg)}")
        print(f"cl: {opti.value(cl)}")
        print(f"cd: {opti.value(cd)}")
        print(f"Wa: {opti.value(W_a)}")
        print(f"Re: {opti.value(Re)}")
        print(f"Ma: {opti.value(Ma)}")
        print(f"local adv ratio: {opti.value(loc_wake_adv_ratio)}")
    # print(f"Thrust2 Total: {Thrust2}")
    # print(f"Torque2 Total: {Torque2}")
    # print(f"Thrust3 Total: {Thrust3}")
    # print(f"Torque3 Total: {Torque3}")

    return Torque, Thrust


Thrust, Torque = annick_propulsion_model(
    rpm,
    airspeed,
    air_density,
    mu,
    n_blades,
    radial_locations_m,
    blade_chord_m,
    blade_beta_deg,
    dBeta_deg,
    divisions,
)