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

import aerosandbox as asb
import aerosandbox.numpy as np

if __name__ == "__main__":
    opti = asb.Opti()  # Initialize a SAND environment

    # Define Assumptions
    L = 34.1376 / 2
    n = 200
    x = np.linspace(0, L, n)
    dx = np.diff(x)
    E = 228e9  # Pa, modulus of CF
    G = E / 2 / (1 + 0.5)  # TODO fix this!!! CFRP is not isotropic!
    max_allowable_stress = 570e6 / 1.75

    log_nominal_diameter = opti.variable(n_vars=n, init_guess=np.log(200e-3))
    nominal_diameter = np.exp(log_nominal_diameter)

    thickness = 0.14e-3 * 5
    opti.subject_to(
        [
            nominal_diameter > thickness,
        ]
    )

    # Bending loads

    moment_of_inertia = (
        np.pi
        / 64
        * ((nominal_diameter + thickness) ** 4 - (nominal_diameter - thickness) ** 4)
    )
    EI = E * moment_of_inertia
    total_lift_force = 9.81 * 103.873 / 2
    lift_distribution = "elliptical"
    if lift_distribution == "rectangular":
        force_per_unit_length = total_lift_force * np.ones(n) / L
    elif lift_distribution == "elliptical":
        force_per_unit_length = (
            total_lift_force * np.sqrt(1 - (x / L) ** 2) * (4 / np.pi) / L
        )

    # Torsion loads
    J = (
        np.pi
        / 32
        * ((nominal_diameter + thickness) ** 4 - (nominal_diameter - thickness) ** 4)
    )

    airfoil_lift_coefficient = 1
    airfoil_moment_coefficient = -0.14
    airfoil_chord = 1  # meter
    moment_per_unit_length = (
        force_per_unit_length
        * airfoil_moment_coefficient
        * airfoil_chord
        / airfoil_lift_coefficient
    )
    # Derivation of above:
    #   CL = L / q c
    #   CM = M / q c**2
    #   M / L = (CM * c) / (CL)

    # Set up derivatives
    u = opti.variable(n_vars=n, init_guess=0, scale=1)
    du = opti.variable(n_vars=n, init_guess=0, scale=0.1)
    ddu = opti.variable(n_vars=n, init_guess=0, scale=0.01)
    dEIddu = opti.variable(n_vars=n, init_guess=0, scale=100)
    phi = opti.variable(n_vars=n, init_guess=0, scale=0.1)
    dphi = opti.variable(n_vars=n, init_guess=0, scale=0.01)

    # Add forcing term
    ddEIddu = force_per_unit_length
    ddphi = -moment_per_unit_length / (G * J)

    # Define derivatives
    def trapz(x):
        # Note: Original had endpoint corrections but these don't work with CasADi in-place ops
        out = (x[:-1] + x[1:]) / 2
        return out

    opti.subject_to(
        [
            np.diff(u) == trapz(du) * dx,
            np.diff(du) == trapz(ddu) * dx,
            np.diff(EI * ddu) == trapz(dEIddu) * dx,
            np.diff(dEIddu) == trapz(ddEIddu) * dx,
            np.diff(phi) == trapz(dphi) * dx,
            np.diff(dphi) == trapz(ddphi) * dx,
        ]
    )

    # Add BCs
    opti.subject_to(
        [
            u[0] == 0,
            du[0] == 0,
            ddu[-1] == 0,  # No tip moment
            dEIddu[-1] == 0,  # No tip higher order stuff
            phi[0] == 0,
            dphi[-1] == 0,
        ]
    )

    # Failure criterion
    stress_axial = (nominal_diameter + thickness) / 2 * E * ddu
    stress_shear = dphi * G * (nominal_diameter + thickness) / 2
    # stress_axial = np.fmax(0, stress_axial)
    # stress_shear = np.fmax(0, stress_shear)
    stress_von_mises_squared = np.sqrt(
        stress_axial**2 + 0 * stress_shear**2
    )  # Source: https://en.wikipedia.org/wiki/Von_Mises_yield_criterion
    stress = stress_axial
    opti.subject_to([stress / max_allowable_stress < 1])

    # Mass
    volume = np.sum(
        np.pi
        / 4
        * trapz(
            (nominal_diameter + thickness) ** 2 - (nominal_diameter - thickness) ** 2
        )
        * dx,
        axis=0,
    )
    mass = volume * 1600
    opti.minimize(mass)

    # Tip deflection constraint
    opti.subject_to(
        [u[-1] < 2]  # Source: http://web.mit.edu/drela/Public/web/hpa/hpa_structure.pdf
    )

    # Twist
    opti.subject_to([phi[-1] * 180 / np.pi > -3])

    p_opts = {}
    s_opts = {}
    s_opts["max_iter"] = 500  # If you need to interrupt, just use ctrl+c
    # s_opts["mu_strategy"] = "adaptive"
    opti.solver("ipopt", p_opts, s_opts)

    try:
        sol = opti.solve()
    except Exception:
        print("Failed!")
        sol = opti.debug

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(font_scale=1)

    fig, ax = plt.subplots(2, 3, figsize=(10, 6), dpi=200)

    plt.subplot(231)
    plt.plot(sol(x), sol(u), ".-")
    plt.xlabel("x [m]")
    plt.ylabel("u [m]")
    plt.title("Displacement (Bending)")
    plt.axis("equal")

    # plt.subplot(232)
    # plt.plot(sol(x), np.arctan(sol(du))*180/np.pi, '.-')
    # plt.xlabel("x [m]")
    # plt.ylabel(r"Local Slope [deg]")
    # plt.title("Slope")

    plt.subplot(232)
    plt.plot(sol(x), sol(phi) * 180 / np.pi, ".-")
    plt.xlabel("x [m]")
    plt.ylabel("Twist angle [deg]")
    plt.title("Twist Angle (Torsion)")

    plt.subplot(233)
    plt.plot(sol(x), sol(force_per_unit_length), ".-")
    plt.xlabel("x [m]")
    plt.ylabel(r"$F$ [N/m]")
    plt.title("Local Load per Unit Span")

    plt.subplot(234)
    plt.plot(sol(x), sol(stress / 1e6), ".-")
    plt.xlabel("x [m]")
    plt.ylabel("Stress [MPa]")
    plt.title("Peak Stress at Section")

    plt.subplot(235)
    plt.plot(sol(x), sol(dEIddu), ".-")
    plt.xlabel("x [m]")
    plt.ylabel("F [N]")
    plt.title("Shear Force")

    plt.subplot(236)
    plt.plot(sol(x), sol(nominal_diameter), ".-")
    plt.xlabel("x [m]")
    plt.ylabel("t [m]")
    plt.title("Optimal Spar Diameter")

    plt.suptitle(f"Beam Modeling (Total Spar Mass: {2 * sol(mass):.2f} kg)")

    plt.subplots_adjust(hspace=0.4)

    import tempfile
    from pathlib import Path

    savefig_path = Path(tempfile.gettempdir()) / "beam.png"
    plt.savefig(savefig_path)
    print(f"Saved figure to: {savefig_path}")

    # plt.tight_layout()
    # plt.legend()
    plt.show()

    print("Mass (half-wing) [kg]:", sol(mass))
    print("Mass (full-wing) [kg]:", 2 * sol(mass))
