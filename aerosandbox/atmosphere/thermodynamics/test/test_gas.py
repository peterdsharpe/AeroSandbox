import pytest

from aerosandbox.atmosphere.thermodynamics.gas import PerfectGas


def make_air(pressure=101325.0, temperature=288.15) -> PerfectGas:
    return PerfectGas(pressure=pressure, temperature=temperature)


def test_standard_air_properties():
    air = make_air()

    assert air.density == pytest.approx(1.225, abs=0.005)  # ISA sea level
    assert air.speed_of_sound == pytest.approx(340.3, abs=0.5)  # ISA sea level
    assert air.ratio_of_specific_heats == pytest.approx(1.4, abs=0.01)
    assert air.specific_volume == pytest.approx(1 / air.density)


def test_enthalpy_and_internal_energy_changes():
    air = make_air()

    assert air.specific_enthalpy_change(300, 400) == pytest.approx(
        air.specific_heat_constant_pressure * 100
    )
    assert air.specific_internal_energy_change(300, 400) == pytest.approx(
        air.specific_heat_constant_volume * 100
    )


def test_isentropic_process_pressure_specified():
    air = make_air()
    gam = air.ratio_of_specific_heats

    new = air.process("isentropic", new_pressure=2 * air.pressure)

    ### P * v^gamma is invariant across an isentropic process
    assert new.pressure * new.specific_volume**gam == pytest.approx(
        air.pressure * air.specific_volume**gam, rel=1e-10
    )
    ### Textbook temperature-ratio relation: T2/T1 = (P2/P1)^((gamma-1)/gamma)
    assert new.temperature / air.temperature == pytest.approx(
        2 ** ((gam - 1) / gam), rel=1e-10
    )


def test_isentropic_process_temperature_specified():
    air = make_air()
    gam = air.ratio_of_specific_heats

    new = air.process("isentropic", new_temperature=2 * air.temperature)

    assert new.temperature == pytest.approx(2 * air.temperature, rel=1e-10)
    assert new.pressure / air.pressure == pytest.approx(
        2 ** (gam / (gam - 1)), rel=1e-10
    )
    assert new.pressure * new.specific_volume**gam == pytest.approx(
        air.pressure * air.specific_volume**gam, rel=1e-10
    )


def test_isentropic_process_density_specified():
    air = make_air()
    gam = air.ratio_of_specific_heats

    new = air.process("isentropic", new_density=2 * air.density)

    assert new.density == pytest.approx(2 * air.density, rel=1e-10)
    assert new.pressure / air.pressure == pytest.approx(2**gam, rel=1e-10)
    assert new.pressure * new.specific_volume**gam == pytest.approx(
        air.pressure * air.specific_volume**gam, rel=1e-10
    )


def test_isothermal_process():
    air = make_air()

    new = air.process("isothermal", new_pressure=3 * air.pressure)

    assert new.temperature == pytest.approx(air.temperature)
    ### Boyle's law: P * v is invariant across an isothermal process
    assert new.pressure * new.specific_volume == pytest.approx(
        air.pressure * air.specific_volume, rel=1e-10
    )

    new = air.process("isothermal", new_density=2 * air.density)
    assert new.temperature == pytest.approx(air.temperature)
    assert new.pressure == pytest.approx(2 * air.pressure, rel=1e-10)


def test_isobaric_process():
    air = make_air()

    new = air.process("isobaric", new_temperature=2 * air.temperature)

    assert new.pressure == pytest.approx(air.pressure)
    ### Charles's law: v / T is invariant across an isobaric process
    assert new.specific_volume / new.temperature == pytest.approx(
        air.specific_volume / air.temperature, rel=1e-10
    )

    new = air.process("isobaric", new_density=0.5 * air.density)
    assert new.pressure == pytest.approx(air.pressure)
    assert new.temperature == pytest.approx(2 * air.temperature, rel=1e-10)


def test_isochoric_process():
    air = make_air()

    new = air.process("isochoric", new_temperature=2 * air.temperature)

    assert new.density == pytest.approx(air.density, rel=1e-10)
    ### Gay-Lussac's law: P / T is invariant across an isochoric process
    assert new.pressure / new.temperature == pytest.approx(
        air.pressure / air.temperature, rel=1e-10
    )

    new = air.process("isochoric", new_pressure=2 * air.pressure)
    assert new.density == pytest.approx(air.density, rel=1e-10)
    assert new.temperature == pytest.approx(2 * air.temperature, rel=1e-10)


def test_polytropic_process():
    air = make_air()
    n = 1.25

    new = air.process("polytropic", new_pressure=2 * air.pressure, polytropic_n=n)

    ### P * v^n is invariant across a polytropic process
    assert new.pressure * new.specific_volume**n == pytest.approx(
        air.pressure * air.specific_volume**n, rel=1e-10
    )
    assert new.temperature / air.temperature == pytest.approx(
        2 ** ((n - 1) / n), rel=1e-10
    )


def test_polytropic_process_requires_n():
    air = make_air()
    with pytest.raises(ValueError):
        air.process("polytropic", new_pressure=2 * air.pressure)


def test_process_round_trip():
    """
    Runs a gas through a Carnot cycle; it should return to its initial state.
    """
    air = make_air(pressure=100e3, temperature=300)
    gam = air.ratio_of_specific_heats

    T_hot = 450
    ### Density ratio across an isentropic process between the two cycle temperatures:
    ### T * v^(gamma-1) = const., so rho2 / rho1 = (T2 / T1)^(1 / (gamma - 1))
    isentropic_density_ratio = (T_hot / air.temperature) ** (1 / (gam - 1))

    g = air.process("isothermal", new_density=air.density * 2)
    g = g.process("isentropic", new_temperature=T_hot)
    g = g.process("isothermal", new_density=air.density * isentropic_density_ratio)
    g = g.process("isentropic", new_temperature=air.temperature)

    assert g.pressure == pytest.approx(air.pressure, rel=1e-10)
    assert g.temperature == pytest.approx(air.temperature, rel=1e-10)
    assert g.density == pytest.approx(air.density, rel=1e-10)


def test_process_inplace():
    air = make_air()
    reference = air.process("isentropic", new_pressure=2 * air.pressure)

    result = air.process("isentropic", new_pressure=2 * air.pressure, inplace=True)

    assert result is air  # inplace=True mutates and returns the same object.
    assert air.pressure == pytest.approx(reference.pressure)
    assert air.temperature == pytest.approx(reference.temperature)


def test_isobaric_process_with_enthalpy_addition():
    air = make_air()
    dh = 10e3

    new = air.process("isobaric", enthalpy_addition_at_constant_pressure=dh)

    assert new.pressure == pytest.approx(air.pressure)
    assert new.temperature == pytest.approx(
        air.temperature + dh / air.specific_heat_constant_pressure
    )
    assert new.density > 0

    ### Should be identical to specifying the corresponding new temperature directly
    reference = air.process(
        "isobaric",
        new_temperature=air.temperature + dh / air.specific_heat_constant_pressure,
    )
    assert new.pressure == pytest.approx(reference.pressure)
    assert new.temperature == pytest.approx(reference.temperature)


def test_isochoric_process_with_enthalpy_addition():
    air = make_air()
    du = 10e3

    new = air.process("isochoric", enthalpy_addition_at_constant_volume=du)

    assert new.temperature == pytest.approx(
        air.temperature + du / air.specific_heat_constant_volume
    )
    ### Regression test: pressure used to come back as None here.
    assert new.pressure == pytest.approx(
        air.pressure * new.temperature / air.temperature, rel=1e-10
    )
    ### Isochoric process: density is unchanged.
    assert new.density == pytest.approx(air.density, rel=1e-10)


def test_isentropic_process_with_enthalpy_addition():
    air = make_air()
    gam = air.ratio_of_specific_heats
    dh = 10e3

    new = air.process("isentropic", enthalpy_addition_at_constant_pressure=dh)

    assert new.temperature == pytest.approx(
        air.temperature + dh / air.specific_heat_constant_pressure
    )
    ### Regression test: pressure used to come back as None here.
    assert new.pressure * new.specific_volume**gam == pytest.approx(
        air.pressure * air.specific_volume**gam, rel=1e-10
    )


def test_polytropic_process_with_enthalpy_addition():
    air = make_air()
    n = 1.25
    dh = 10e3

    new = air.process(
        "polytropic", enthalpy_addition_at_constant_pressure=dh, polytropic_n=n
    )

    assert new.temperature == pytest.approx(
        air.temperature + dh / air.specific_heat_constant_pressure
    )
    ### Regression test: pressure used to come back as None here.
    assert new.pressure * new.specific_volume**n == pytest.approx(
        air.pressure * air.specific_volume**n, rel=1e-10
    )


def test_isothermal_process_with_enthalpy_addition_raises():
    air = make_air()

    ### Regression test: this contradictory specification used to silently return
    ### a gas with `pressure=None` instead of raising.
    with pytest.raises(ValueError):
        air.process("isothermal", enthalpy_addition_at_constant_pressure=10e3)

    with pytest.raises(ValueError):
        air.process("isothermal", enthalpy_addition_at_constant_volume=10e3)


def test_process_with_enthalpy_addition_casadi():
    import casadi

    air = make_air()
    dh = casadi.MX.sym("dh")

    new = air.process("isentropic", enthalpy_addition_at_constant_pressure=dh)

    f = casadi.Function("f", [dh], [new.pressure, new.temperature])
    pressure, temperature = [float(v) for v in f(10e3)]

    reference = air.process("isentropic", enthalpy_addition_at_constant_pressure=10e3)
    assert pressure == pytest.approx(reference.pressure, rel=1e-10)
    assert temperature == pytest.approx(reference.temperature, rel=1e-10)


def test_process_invalid_inputs():
    air = make_air()

    with pytest.raises(ValueError):  # No conditions specified
        air.process("isentropic")

    with pytest.raises(ValueError):  # Multiple conditions specified
        air.process("isentropic", new_pressure=2e5, new_temperature=400)

    with pytest.raises(ValueError):  # Contradictory specification
        air.process("isothermal", new_temperature=400)

    with pytest.raises(ValueError):  # Contradictory specification
        air.process("isobaric", new_pressure=2e5)

    with pytest.raises(ValueError):  # Contradictory specification
        air.process("isochoric", new_density=2.0)

    with pytest.raises(NotImplementedError):
        air.process("isenthalpic", new_pressure=2e5)

    with pytest.raises(ValueError):  # Invalid process name
        air.process("not_a_process", new_pressure=2e5)


if __name__ == "__main__":
    pytest.main([__file__])
