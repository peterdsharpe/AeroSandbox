import numpy as onp
import pytest

import aerosandbox as asb


def test_constrain_derivatives_bad_state_name_raises_valueerror():
    """
    A typo in `which` used to leak a raw KeyError instead of the intended
    friendly ValueError.
    """
    opti = asb.Opti()
    time = onp.linspace(0, 1, 11)
    dyn = asb.DynamicsPointMass1DVertical(
        mass_props=asb.MassProperties(mass=1),
        z_e=opti.variable(init_guess=onp.zeros(11)),
        w_e=opti.variable(init_guess=onp.zeros(11)),
    )

    with pytest.raises(ValueError, match="does not have a state named"):
        dyn.constrain_derivatives(opti, time, which=["bogus_name"])


def test_constrain_derivatives_valid_names_still_work():
    opti = asb.Opti()
    time = onp.linspace(0, 1, 11)
    dyn = asb.DynamicsPointMass1DVertical(
        mass_props=asb.MassProperties(mass=1),
        z_e=opti.variable(init_guess=onp.zeros(11)),
        w_e=opti.variable(init_guess=onp.zeros(11)),
    )
    dyn.add_gravity_force()
    dyn.constrain_derivatives(opti, time)  # "all"
    opti.subject_to(
        [
            dyn.z_e[0] == 0,
            dyn.w_e[0] == 0,
        ]
    )
    sol = opti.solve(verbose=False)

    # Free-fall from rest: z_e(t) = 0.5 * g * t^2 (z_e points down)
    assert sol(dyn.z_e)[-1] == pytest.approx(0.5 * 9.81 * 1**2, rel=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
