import aerosandbox as asb
import aerosandbox.numpy as np
import time


def solve():
    ### Constants
    form_factor = 1.2  # form factor [-]
    oswalds_efficiency = 0.95  # Oswald efficiency factor [-]
    viscosity = 1.78e-5  # viscosity of air [kg/m/s]
    density = 1.23  # density of air [kg/m^3]
    airfoil_thickness_fraction = 0.12  # airfoil thickness to chord ratio [-]
    ultimate_load_factor = 3.8  # ultimate load factor [-]
    airspeed_takeoff = 22  # takeoff speed [m/s]
    CL_max = 1.5  # max CL with flaps down [-]
    wetted_area_ratio = 2.05  # wetted area ratio [-]
    W_W_coeff1 = 8.71e-5  # Wing Weight Coefficient 1 [1/m]
    W_W_coeff2 = 45.24  # Wing Weight Coefficient 2 [Pa]
    drag_area_fuselage = 0.031  # fuselage drag area [m^2]
    weight_fuselage = 4940.0  # aircraft weight excluding wing [N]

    opti = asb.Opti()  # initialize an optimization environment

    ### Variables
    aspect_ratio = opti.variable(init_guess=10)  # aspect ratio
    wing_area = opti.variable(init_guess=10)  # total wing area [m^2]
    airspeed = opti.variable(init_guess=100)  # cruising speed [m/s]
    weight = opti.variable(init_guess=10000)  # total aircraft weight [N]
    CL = opti.variable(init_guess=1)  # Lift coefficient of wing [-]

    ### Models
    # Aerodynamics model
    CD_fuselage = drag_area_fuselage / wing_area
    Re = (density / viscosity) * airspeed * (wing_area / aspect_ratio) ** 0.5
    Cf = 0.074 / Re**0.2
    CD_profile = form_factor * Cf * wetted_area_ratio
    CD_induced = CL**2 / (np.pi * aspect_ratio * oswalds_efficiency)
    CD = CD_fuselage + CD_profile + CD_induced
    dynamic_pressure = 0.5 * density * airspeed**2
    drag = dynamic_pressure * wing_area * CD
    lift_cruise = dynamic_pressure * wing_area * CL
    lift_takeoff = 0.5 * density * wing_area * CL_max * airspeed_takeoff**2

    # Wing weight model
    weight_wing_structural = (
        W_W_coeff1
        * (
            ultimate_load_factor
            * aspect_ratio**1.5
            * (weight_fuselage * weight * wing_area) ** 0.5
        )
        / airfoil_thickness_fraction
    )
    weight_wing_surface = W_W_coeff2 * wing_area
    weight_wing = weight_wing_surface + weight_wing_structural

    ### Constraints
    opti.subject_to(
        [
            weight <= lift_cruise,
            weight <= lift_takeoff,
            weight == weight_fuselage + weight_wing,
        ]
    )

    # Objective
    opti.minimize(drag)

    sol = opti.solve(max_iter=100, verbose=False)


def timeit():
    start = time.time()
    solve()
    end = time.time()
    return end - start


if __name__ == "__main__":
    times = np.array([timeit() for i in range(10)])
    print(np.mean(times))
