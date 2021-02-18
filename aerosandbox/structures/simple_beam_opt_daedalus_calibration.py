"""
Simple Beam

A simple 2D beam example, to be integrated later for full aerostructural modeling. TODO do that.

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

"""
import aerosandbox.numpy as np
import casadi as cas

if __name__ == '__main__':

    opti = cas.Opti()  # Initialize a SAND environment

    # Define Assumptions
    L = 34.1376 / 2
    n = 200
    x = cas.linspace(0, L, n)
    dx = cas.diff(x)
    E = 228e9  # Pa, modulus of CF
    G = E / 2 / (1 + 0.5)  # TODO fix this!!! CFRP is not isotropic!
    max_allowable_stress = 570e6 / 1.75

    log_nominal_diameter = opti.variable(n)
    opti.set_initial(log_nominal_diameter, cas.log(200e-3))
    nominal_diameter = cas.exp(log_nominal_diameter)

    thickness = 0.14e-3 * 5
    opti.subject_to([
        nominal_diameter > thickness,
    ])

    # Bending loads

    I = cas.pi / 64 * ((nominal_diameter + thickness) ** 4 - (nominal_diameter - thickness) ** 4)
    EI = E * I
    total_lift_force = 9.81 * 103.873 / 2
    lift_distribution = "elliptical"
    if lift_distribution == "rectangular":
        force_per_unit_length = total_lift_force * cas.GenDM_ones(n) / L
    elif lift_distribution == "elliptical":
        force_per_unit_length = total_lift_force * cas.sqrt(1 - (x / L) ** 2) * (4 / cas.pi) / L

    # Torsion loads
    J = cas.pi / 32 * ((nominal_diameter + thickness) ** 4 - (nominal_diameter - thickness) ** 4)

    airfoil_lift_coefficient = 1
    airfoil_moment_coefficient = -0.14
    airfoil_chord = 1  # meter
    moment_per_unit_length = force_per_unit_length * airfoil_moment_coefficient * airfoil_chord / airfoil_lift_coefficient
    # Derivation of above:
    #   CL = L / q c
    #   CM = M / q c**2
    #   M / L = (CM * c) / (CL)

    # Set up derivatives
    u = 1 * opti.variable(n)
    du = 0.1 * opti.variable(n)
    ddu = 0.01 * opti.variable(n)
    dEIddu = 100 * opti.variable(n)
    phi = 0.1 * opti.variable(n)
    dphi = 0.01 * opti.variable(n)
    # opti.set_initial(u, 2 * (x/L)**4)
    # opti.set_initial(du, 2 * 4/L * (x/L)**3)
    # opti.set_initial(ddu, 2 * 3/L * 2/L * (x/L))
    # opti.set_initial(dEIddu, 2 * 3/L * 2/L * 1/L * 1e3)

    # Add forcing term
    ddEIddu = force_per_unit_length
    ddphi = -moment_per_unit_length / (G * J)


    # Define derivatives
    def trapz(x):
        out = (x[:-1] + x[1:]) / 2
        out[0] += x[0] / 2
        out[-1] += x[-1] / 2
        return out


    opti.subject_to([
        cas.diff(u) == trapz(du) * dx,
        cas.diff(du) == trapz(ddu) * dx,
        cas.diff(EI * ddu) == trapz(dEIddu) * dx,
        cas.diff(dEIddu) == trapz(ddEIddu) * dx,
        cas.diff(phi) == trapz(dphi) * dx,
        cas.diff(dphi) == trapz(ddphi) * dx,
    ])

    # Add BCs
    opti.subject_to([
        u[0] == 0,
        du[0] == 0,
        ddu[-1] == 0,  # No tip moment
        dEIddu[-1] == 0,  # No tip higher order stuff
        phi[0] == 0,
        dphi[-1] == 0,
    ])

    # Failure criterion
    stress_axial = (nominal_diameter + thickness) / 2 * E * ddu
    stress_shear = dphi * G * (nominal_diameter + thickness) / 2
    # stress_axial = cas.fmax(0, stress_axial)
    # stress_shear = cas.fmax(0, stress_shear)
    stress_von_mises_squared = cas.sqrt(
        stress_axial ** 2 + 0 * stress_shear ** 2)  # Source: https://en.wikipedia.org/wiki/Von_Mises_yield_criterion
    stress = stress_axial
    opti.subject_to([
        stress / max_allowable_stress < 1
    ])

    # Mass
    volume = cas.sum1(
        cas.pi / 4 * trapz((nominal_diameter + thickness) ** 2 - (nominal_diameter - thickness) ** 2) * dx
    )
    mass = volume * 1600
    opti.minimize(mass)

    # Tip deflection constraint
    opti.subject_to([
        u[-1] < 2  # Source: http://web.mit.edu/drela/Public/web/hpa/hpa_structure.pdf
    ])

    # Twist
    opti.subject_to([
        phi[-1] * 180 / cas.pi > -3
    ])

    p_opts = {}
    s_opts = {}
    s_opts["max_iter"] = 500  # If you need to interrupt, just use ctrl+c
    # s_opts["mu_strategy"] = "adaptive"
    opti.solver('ipopt', p_opts, s_opts)

    try:
        sol = opti.solve()
    except:
        print("Failed!")
        sol = opti.debug

    import matplotlib.pyplot as plt
    import matplotlib.style as style
    import seaborn as sns

    sns.set(font_scale=1)

    fig, ax = plt.subplots(2, 3, figsize=(10, 6), dpi=200)

    plt.subplot(231)
    plt.plot(sol.value(x), sol.value(u), '.-')
    plt.xlabel("x [m]")
    plt.ylabel("u [m]")
    plt.title("Displacement (Bending)")
    plt.axis("equal")

    # plt.subplot(232)
    # plt.plot(sol.value(x), np.arctan(sol.value(du))*180/np.pi, '.-')
    # plt.xlabel("x [m]")
    # plt.ylabel(r"Local Slope [deg]")
    # plt.title("Slope")

    plt.subplot(232)
    plt.plot(sol.value(x), sol.value(phi) * 180 / np.pi, '.-')
    plt.xlabel("x [m]")
    plt.ylabel("Twist angle [deg]")
    plt.title("Twist Angle (Torsion)")

    plt.subplot(233)
    plt.plot(sol.value(x), sol.value(force_per_unit_length), '.-')
    plt.xlabel("x [m]")
    plt.ylabel(r"$F$ [N/m]")
    plt.title("Local Load per Unit Span")

    plt.subplot(234)
    plt.plot(sol.value(x), sol.value(stress / 1e6), '.-')
    plt.xlabel("x [m]")
    plt.ylabel("Stress [MPa]")
    plt.title("Peak Stress at Section")

    plt.subplot(235)
    plt.plot(sol.value(x), sol.value(dEIddu), '.-')
    plt.xlabel("x [m]")
    plt.ylabel("F [N]")
    plt.title("Shear Force")

    plt.subplot(236)
    plt.plot(sol.value(x), sol.value(nominal_diameter), '.-')
    plt.xlabel("x [m]")
    plt.ylabel("t [m]")
    plt.title("Optimal Spar Diameter")

    plt.suptitle("Beam Modeling (Total Spar Mass: %.2f kg)" % (2 * sol.value(mass)))

    plt.subplots_adjust(hspace=0.4)
    plt.savefig("C:/Users/User/Downloads/beam.png")

    # plt.tight_layout()
    # plt.legend()
    plt.show()

    print("Mass (half-wing) [kg]:", sol.value(mass))
    print("Mass (full-wing) [kg]:", 2 * sol.value(mass))
