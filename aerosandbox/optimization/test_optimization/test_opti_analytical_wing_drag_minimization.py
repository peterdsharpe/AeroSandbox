import aerosandbox as asb
from numpy import pi
import pytest

density = 1.225
velocity = 1
viscosity = 1.81e-5
CL = 0.4

def test_normal_problem():
    """
    An engineering optimization problem to minimize drag on a rectangular wing using analytical relations.
    Lift is effectively fixed (fixed CL, flight speed, and density, with wing area constraint)
    """
    opti = asb.Opti()

    chord = opti.variable(init_guess=1)
    span = opti.variable(init_guess=2)
    AR = span / chord

    Re = density * velocity * chord / viscosity
    CD_p = 1.328 * Re ** -0.5
    CD_i = CL ** 2 / (pi * AR)

    opti.subject_to(chord * span == 1)
    opti.minimize(CD_p + CD_i)

    sol = opti.solve()

    print(f"Chord = {sol.value(chord)}")
    print(f"Span = {sol.value(span)}")

    assert sol.value(chord) == pytest.approx(0.2288630528244024)
    assert sol.value(span) == pytest.approx(4.369425242121806)

def test_log_transformed_problem():
    opti = asb.Opti()

    chord = opti.variable(init_guess=1, log_transform=True)
    span = opti.variable(init_guess=2, log_transform=True)
    AR = span / chord

    Re = density * velocity * chord / viscosity
    CD_p = 1.328 * Re ** -0.5
    CD_i = CL ** 2 / (pi * AR)

    opti.subject_to(chord * span == 1)
    opti.minimize(CD_p + CD_i)

    sol = opti.solve()

    print(f"Chord = {sol.value(chord)}")
    print(f"Span = {sol.value(span)}")

    assert sol.value(chord) == pytest.approx(0.2288630528244024)
    assert sol.value(span) == pytest.approx(4.369425242121806)

def test_log_transformed_negativity_error():
    opti = asb.Opti()

    with pytest.raises(ValueError):
        myvar = opti.variable(log_transform=True, init_guess=-1)

def test_fixed_variable():
    opti = asb.Opti()

    chord = opti.variable(init_guess=1, freeze=True)
    span = opti.variable(init_guess=2)
    AR = span / chord

    Re = density * velocity * chord / viscosity
    CD_p = 1.328 * Re ** -0.5
    CD_i = CL ** 2 / (pi * AR)

    opti.subject_to(chord * span == 1)
    opti.minimize(CD_p + CD_i)

    sol = opti.solve()

    print(f"Chord = {sol.value(chord)}")
    print(f"Span = {sol.value(span)}")

    assert sol.value(chord) == pytest.approx(1)
    assert sol.value(span) == pytest.approx(1)

def test_fully_fixed_problem():
    opti = asb.Opti()

    chord = opti.variable(init_guess=1, freeze=True)
    span = opti.variable(init_guess=1, freeze=True)
    AR = span / chord

    Re = density * velocity * chord / viscosity
    CD_p = 1.328 * Re ** -0.5
    CD_i = CL ** 2 / (pi * AR)

    opti.subject_to(chord * span == 1)
    opti.minimize(CD_p + CD_i)

    sol = opti.solve()

    print(f"Chord = {sol.value(chord)}")
    print(f"Span = {sol.value(span)}")

    assert sol.value(chord) == pytest.approx(1)
    assert sol.value(span) == pytest.approx(1)

def test_overconstrained_fully_fixed_problem():
    opti = asb.Opti()

    chord = opti.variable(init_guess=1, freeze=True)
    span = opti.variable(init_guess=2, freeze=True)
    AR = span / chord

    Re = density * velocity * chord / viscosity
    CD_p = 1.328 * Re ** -0.5
    CD_i = CL ** 2 / (pi * AR)

    with pytest.raises(RuntimeError):
        opti.subject_to(chord * span == 1)
    opti.minimize(CD_p + CD_i)

    sol = opti.solve()

    print(f"Chord = {sol.value(chord)}")
    print(f"Span = {sol.value(span)}")

    # assert sol.value(chord) == pytest.approx(1)
    # assert sol.value(span) == pytest.approx(1)

if __name__ == '__main__':
    pytest.main()