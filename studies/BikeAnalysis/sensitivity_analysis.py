"""First-order sensitivity of human-powered-vehicle top speed to each design driver.

This reuses the steady-state power balance from ``power_balance.py``: at the maximum
sustainable speed, the power delivered to the wheel exactly matches the power dissipated
by aerodynamic drag, rolling resistance, and (on a grade) gravity:

    eta * P  =  1/2 * rho * (V + w)**2 * (Cd*A) * V    [aerodynamic]
              + Crr * m * g * cos(theta) * V           [rolling]
              + m * g * sin(theta) * V                 [gravity / climbing]

That equation defines the top speed ``V`` *implicitly* as a function of the design
drivers. We use SymPy to differentiate the implicit relation (via the implicit-function
theorem, ``dV/dx = -(d balance/dx) / (d balance/dV)``) and report, for each driver, how
many mph the top speed moves per +1% change in that driver. These are the "design
derivatives" that show where engineering effort buys the most speed.

Operating point
---------------
The drivers are evaluated at the Eta's steady-state top speed - the speed at which the
pilot's wheel power exactly balances the resistances (P_in == P_out). The drag area and
rolling resistance are the design values from Trefor Evans's U. Toronto thesis
(Cd*A = 0.0068 m^2 [pg 40], Crr = 0.0027), as used in ``power_balance.py``, and the pilot
power is a First-Class athlete's 5-minute output. The system mass is split into a fixed
pilot and the design-variable empty bike (25 kg), so the mass row reports the sensitivity
to *bike* mass - the only part a designer actually controls.

This steady-state top speed (~155 km/h / 96 mph) is deliberately *higher* than the 144.17
km/h record, and that is correct rather than a calibration error. The steady-state speed is
an *asymptote* that the vehicle approaches but never reaches in a finite run, so at the
200 m timing trap the bike is still accelerating toward it (as AeroVelo's speed-vs-time
data show); the record is a transient *below* the asymptote. One therefore must NOT back
out Cd*A / Crr by forcing the steady-state speed down onto the achieved record - that would
imply a speed the vehicle could only reach after infinite distance. Edit ``DRIVERS`` (or
the environment) to explore other vehicles, athletes, or assumptions.

The analytic sensitivities are cross-checked against central finite differences before the
table is printed, so a silent error in the symbolic derivation would raise immediately.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import sympy as sp

import aerosandbox as asb
import aerosandbox.library.power_human as ph
import aerosandbox.tools.units as u

### --- Symbolic power-balance model --------------------------------------------------
# Ground speed (the unknown we solve for) and the design drivers; all positive reals so
# SymPy can simplify square roots and the cubic cleanly.
V = sp.Symbol("V", positive=True)
P, eta, rho, CdA, Crr, g = sp.symbols("P eta rho C_{D}A C_rr g", positive=True)
# Environmental terms are carried symbolically for generality, but both are zero at this
# operating point (flat, windless), where a percentage perturbation is undefined; they are
# therefore omitted from the % table.
theta, headwind = sp.symbols("theta headwind", real=True)
# Split the system mass into the pilot (a fixed overhead) and the empty bike (the actual
# design variable). They reach the physics only through their sum ``m``, so dV/d(bike mass)
# equals dV/d(total mass) in absolute terms - but a +1% change in bike mass is a far smaller
# *absolute* change than +1% of the total, which is why bike mass is the weakest lever.
m_pilot, m_bike = sp.symbols("m_pilot m_bike", positive=True)
m = m_pilot + m_bike

# Power dissipated by each resistance at ground speed V. Aerodynamic drag acts on the
# air-relative speed (V + headwind) but spends power at the ground speed V.
power_aero = sp.Rational(1, 2) * rho * (V + headwind) ** 2 * CdA * V
power_roll = Crr * m * g * sp.cos(theta) * V
power_grav = m * g * sp.sin(theta) * V
power_resist = power_aero + power_roll + power_grav

# Available power at the wheel, and the residual that is zero at the top speed.
power_avail = eta * P
balance = power_avail - power_resist

b = balance.subs(theta, 0).subs(headwind, 0)

### --- Symbolic sensitivities (implicit-function theorem) ----------------------------
# balance(V, x) = 0 defines V(x) implicitly, so dV/dx = -(d balance/dx) / (d balance/dV).
# The denominator is the (negative) slope of the resistance curve, dP_resist/dV.
_dbalance_dV = sp.diff(balance, V)


def speed_sensitivity(driver: sp.Symbol) -> sp.Expr:
    """Symbolic dV/d(driver): change in top speed per unit change in one driver.

    Args:
        driver: One of the model symbols (e.g. ``P``, ``CdA``). Units are [m/s] per the
            driver's own unit.

    Returns:
        A SymPy expression in the model symbols for the partial derivative ``dV/d(driver)``,
        obtained by the implicit-function theorem applied to ``balance == 0``.
    """
    return -sp.diff(balance, driver) / _dbalance_dV


def evaluate_at(expr: sp.Expr, point: Mapping[sp.Symbol, float]) -> float:
    """Numerically evaluate a SymPy expression at a fully-specified operating point.

    Args:
        expr: Any scalar SymPy expression in the model symbols.
        point: A ``{symbol: value}`` mapping that grounds every free symbol in ``expr``.

    Returns:
        The expression's value as a Python ``float``.
    """
    # Substitute via the (old, new) pair list rather than the dict form: SymPy treats both
    # identically here (independent numeric substitutions), but the iterable-of-tuples
    # overload is covariant in its key, so it type-checks where ``subs(dict)`` does not.
    return float(expr.subs(list(point.items())))


### --- Design drivers and the operating point ----------------------------------------
@dataclass(frozen=True)
class DesignDriver:
    """One scalar input to the power balance, with the value it takes at the operating point."""

    label: str  # human-readable name for the table index
    symbol: sp.Symbol  # the corresponding SymPy symbol in ``balance``
    display: str  # short math symbol for the "Symbol" column
    value: float  # operating-point value, in ``unit``
    unit: str  # physical unit string (empty for dimensionless)


# Battle Mountain elevation -> air density via the ISA model (matches power_balance.py).
ALTITUDE = 1375.0  # m
RHO_VENUE = float(asb.Atmosphere(altitude=ALTITUDE).density())  # kg/m^3

# Sustained pilot power over the ~5-minute Battle Mountain effort.
PILOT_POWER = float(ph.power_human(duration=5 * 60, dataset="First-Class Athletes"))  # ~412 W

# Mass split: the Eta's empty weight is the design variable; the pilot is fixed overhead.
# Their sum (97 kg) matches the Battle Mountain case in power_balance.py. The bike-mass
# sensitivity is nearly independent of the pilot's mass anyway: it equals dV/dm * m_bike, and
# m_bike is fixed while a heavier pilot barely shifts dV/dm. Verified - 55, 72, and 73 kg
# pilots all give ~-0.028 mph per +1% bike mass.
M_BIKE = 25.0  # kg, Eta empty weight (AeroVelo)
M_PILOT = 72.0  # kg, Todd Reichert; total = 25 + 72 = 97 kg (power_balance.py Battle Mountain)

# Operating point = the Eta exactly as modeled in power_balance.py: the design-thesis
# aerodynamic and rolling values (Cd*A = 0.0068, Crr = 0.0027) at a First-Class athlete's
# 5-minute power. The resulting steady-state top speed (~155 km/h) is the asymptote that
# sits *above* the 144.17 km/h record (see module docstring), not a fit to it. The mass
# driver is the *bike* (empty weight), since the pilot's mass is not a design choice.
DRIVERS: list[DesignDriver] = [
    DesignDriver("Pilot power", P, "P", PILOT_POWER, "W"),
    DesignDriver("Drivetrain efficiency", eta, "eta", 0.97, ""),
    DesignDriver("Drag area (Cd*A)", CdA, "Cd*A", 0.0068, "m^2"),
    DesignDriver("Rolling resistance (Crr)", Crr, "Crr", 0.0027, ""),
    DesignDriver("Bike mass (empty weight)", m_bike, "m_bike", M_BIKE, "kg"),
    DesignDriver("Air density (venue/altitude)", rho, "rho", RHO_VENUE, "kg/m^3"),
]

# Fixed, non-driver symbols at this operating point: gravity, flat road, no wind, and the
# pilot's (fixed) mass.
ENVIRONMENT: dict[sp.Symbol, float] = {g: 9.81, theta: 0.0, headwind: 0.0, m_pilot: M_PILOT}


def solve_top_speed(params: dict[sp.Symbol, float], guess: float = 45.0) -> float:
    """Solve the single-variable power balance for the top speed [m/s].

    Args:
        params: Numeric values for every model symbol except ``V``.
        guess: Initial guess for the Newton solve [m/s].

    Returns:
        The positive real root of ``balance == 0`` (the maximum sustainable ground speed).
    """
    return float(sp.nsolve(balance.subs(params), V, guess))


def build_sensitivity_table() -> pd.DataFrame:
    """Compute the per-driver speed sensitivities at the operating point.

    Returns:
        A DataFrame indexed by driver label, with the operating value, the signed speed
        sensitivity in mph and km/h per +1% change, and the dimensionless elasticity
        ``d(ln V) / d(ln x)``. A positive sensitivity means *increasing* the driver makes
        the vehicle faster. Rows are sorted by descending sensitivity magnitude.
    """
    operating_params = ENVIRONMENT | {d.symbol: d.value for d in DRIVERS}
    speed = solve_top_speed(operating_params)
    point = operating_params | {V: speed}  # full substitution set, including the solved V

    rows: list[dict[str, object]] = []
    for d in DRIVERS:
        # dV/d(driver) [m/s per driver-unit], then scaled to a +1% change of the driver.
        dV_ddriver = evaluate_at(speed_sensitivity(d.symbol), point)
        dV_per_percent = dV_ddriver * d.value * 0.01  # [m/s] per +1%
        rows.append(
            {
                "Symbol": d.display,
                "Operating value": f"{d.value:.4g} {d.unit}".strip(),
                "Sensitivity [mph / +1%]": dV_per_percent / u.mph,
                "Sensitivity [km/h / +1%]": dV_per_percent / u.kph,
                "Elasticity d(lnV)/d(lnx)": dV_ddriver * d.value / speed,
            }
        )

    table = pd.DataFrame(rows, index=pd.Index([d.label for d in DRIVERS]))
    table.index.name = "Design driver"
    return table.sort_values(
        "Sensitivity [mph / +1%]", key=lambda col: col.abs(), ascending=False
    )


def verify_against_finite_differences(rel_step: float = 1e-6) -> None:
    """Assert each analytic sensitivity matches a central finite-difference estimate.

    Re-solves the power balance with each driver nudged by +/- ``rel_step`` (relative) and
    compares ``(V_up - V_dn) / (2 * rel_step * value)`` to the symbolic ``dV/d(driver)``.
    Raises ``AssertionError`` if the symbolic derivation is wrong.
    """
    operating_params = ENVIRONMENT | {d.symbol: d.value for d in DRIVERS}
    speed = solve_top_speed(operating_params)
    point = operating_params | {V: speed}

    for d in DRIVERS:
        analytic = evaluate_at(speed_sensitivity(d.symbol), point)
        step = rel_step * d.value
        v_up = solve_top_speed(operating_params | {d.symbol: d.value + step}, guess=speed)
        v_dn = solve_top_speed(operating_params | {d.symbol: d.value - step}, guess=speed)
        numeric = (v_up - v_dn) / (2 * step)
        assert np.isclose(analytic, numeric, rtol=1e-4, atol=1e-9), (
            f"Sensitivity mismatch for {d.label!r}: "
            f"analytic={analytic:.6g}, finite-difference={numeric:.6g}"
        )


def main() -> None:
    """Print the operating point, the validation result, and the sensitivity table."""
    operating_params = ENVIRONMENT | {d.symbol: d.value for d in DRIVERS}
    speed = solve_top_speed(operating_params)
    point = operating_params | {V: speed}

    ### Operating-point summary, including the power split that the speed is balancing.
    p_aero = evaluate_at(power_aero, point)
    p_roll = evaluate_at(power_roll, point)
    print("=" * 78)
    print("Operating point: AeroVelo Eta steady-state top speed (P_in == P_out)")
    print("=" * 78)
    print(f"  Top speed   : {speed:.3f} m/s  =  {speed / u.kph:.2f} km/h  =  {speed / u.mph:.2f} mph"
          "   <- steady-state asymptote")
    print("  Record set  : 40.05 m/s  =  144.17 km/h  =  89.59 mph"
          "   <- transient below the asymptote (still accelerating at the trap)")
    print(f"  Pilot power : {evaluate_at(eta * P, point):.1f} W at the wheel "
          f"({DRIVERS[0].value:.0f} W at the crank, eta = {evaluate_at(eta, point):.2f}; First-Class 5-min)")
    print(f"  Resistance  : aero {p_aero:.0f} W ({p_aero / (p_aero + p_roll) * 100:.0f}%)"
          f"  +  rolling {p_roll:.0f} W ({p_roll / (p_aero + p_roll) * 100:.0f}%)")

    ### Validate the symbolic calculus before trusting the numbers.
    verify_against_finite_differences()
    print("\n  [OK] analytic sensitivities match central finite differences (rtol < 1e-4)")

    ### One symbolic derivative, printed to show the SymPy result behind the table.
    print("\n  Symbolic dV/dP (implicit-function theorem):")
    print(f"    {sp.simplify(speed_sensitivity(P))}")

    ### The sensitivity table.
    table = build_sensitivity_table()
    print("\n" + "=" * 78)
    print("First-order speed sensitivity to each design driver (signed, per +1% change)")
    print("=" * 78)
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(table.round({
            "Sensitivity [mph / +1%]": 4,
            "Sensitivity [km/h / +1%]": 4,
            "Elasticity d(lnV)/d(lnx)": 4,
        }).to_string())

    ### Mass enters only through rolling resistance (flat grade), so dV/d(bike mass) equals
    ### dV/d(pilot mass) equals dV/d(total mass) in absolute terms; only the %-scaling
    ### differs. That is why *bike* mass - a fraction m_bike/m_total of the total - is the
    ### weakest lever, even though shedding a kilogram anywhere helps by the same amount.
    dV_dmass = evaluate_at(speed_sensitivity(m_bike), point)  # m/s per kg of any mass
    print("\n" + "=" * 78)
    print("Mass breakdown: bike (design variable) vs pilot (fixed overhead) vs total")
    print("=" * 78)
    for component, kilograms in [
        ("Bike (empty, design var)", M_BIKE),
        ("Pilot (fixed overhead)", M_PILOT),
        ("Total (pilot + bike)", M_BIKE + M_PILOT),
    ]:
        per_pct = dV_dmass * kilograms * 0.01  # m/s per +1% of this mass
        print(f"  {component:26s} {kilograms:5.1f} kg  ->  {per_pct / u.mph:+.4f} mph/+1%"
              f"   ({per_pct / u.kph:+.4f} km/h)")
    print(f"  => bike-mass sensitivity = total-mass sensitivity x (m_bike / m_total) = "
          f"{M_BIKE / (M_BIKE + M_PILOT):.3f}")

    print(
        "\nReading the table: a positive value means *increasing* that driver makes the\n"
        "vehicle faster (power, efficiency); a negative value means increasing it slows the\n"
        "vehicle (drag, rolling resistance, bike mass, air density). Magnitudes rank the\n"
        "levers. These are steady-state sensitivities of the asymptote; AeroVelo's full-sim\n"
        "*trap*-speed derivatives run smaller. Notably, AeroVelo's published mass derivative\n"
        "(0.03 km/h per 1%) is itself a *bike*-mass figure - the same order as the bike row\n"
        "here (0.045 km/h) and far below the ~0.17 km/h a *total*-mass reading gives. Ranking:\n"
        "P = eta > Cd*A = rho > Crr > bike mass."
    )


if __name__ == "__main__":
    main()
