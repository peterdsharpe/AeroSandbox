import aerosandbox as asb
import aerosandbox.numpy as np
import casadi as cas
import aerosandbox.library.power_human as ph
import aerosandbox.tools.units as u

#%%

duration = 4 * 60 * 60  # 4 hours
CDA = 0.3
rolling_resistance_coefficient = 0.005
rho = 1.225  # kg/m^3
mass = 80  # kg (rider + bike)
g = 9.81  # m/s^2

drivetrain_efficiency = 0.97  # fraction of crank power reaching the road (chain, derailleur, bearings)
grade = 0.0  # road slope as rise/run (e.g. 0.05 = 5% climb); positive is uphill
headwind = 0.0  # air speed relative to the ground, along the direction of travel [m/s]; positive opposes motion

### Road inclination angle from the grade. `cos` reduces the normal force (and thus rolling
### resistance) on a slope, while `sin` sets the fraction of weight that opposes climbing.
incline_angle = np.arctan(grade)

opti = asb.Opti()

V = opti.variable(init_guess=10, log_transform=True)

### Crank power the rider can sustain, less the drivetrain losses between crank and wheel.
power_in = ph.power_human(duration, dataset="World-Class Athletes")
power_at_wheel = drivetrain_efficiency * power_in

### Aerodynamic drag acts on the air-relative speed (V + headwind), but the power it absorbs is
### force times *ground* speed, since that is the distance actually traveled.
airspeed = V + headwind
drag_force = 0.5 * rho * airspeed**2 * CDA
drag_power = drag_force * V

### Rolling resistance scales with the normal force, which is reduced on a grade by cos(angle).
rolling_resistance_force = rolling_resistance_coefficient * mass * g * np.cos(incline_angle)
rolling_resistance_power = rolling_resistance_force * V

### Gravitational (climbing) power: the rate of potential-energy gain on a slope. Zero on the flat.
gravity_force = mass * g * np.sin(incline_angle)
gravity_power = gravity_force * V

power_out = drag_power + rolling_resistance_power + gravity_power

### Find the maximum sustainable speed: the fastest V at which the power delivered to the wheel
### still covers the power required to overcome drag, rolling resistance, and climbing.
opti.subject_to(power_out <= power_at_wheel)
opti.maximize(V)

sol = opti.solve(verbose=False)

### Print every numeric/symbolic quantity at the solution by scraping the namespace. Two
### guards keep this robust in a live/Jupyter session, where `locals()` is messy:
###   1. Type filter: only scalars and CasADi symbolics are considered, so `sol()` never tries
###      to recursively copy arbitrary objects (e.g. the IPython shell's traitlets, which would
###      overflow the recursion limit).
###   2. Per-item EAFP: a `cas.MX` may belong to a *different* `Opti` instance (a re-run cell or
###      a second problem in the session); `sol()` raises `RuntimeError` on those, so we skip
###      them. Plain scalars and this problem's own symbolics always evaluate cleanly.
for name, value in locals().copy().items():
    if name.startswith("_") or not isinstance(value, (int, float, cas.MX)):
        continue
    try:
        print(f"{name}: {sol(value):.4g}")
    except RuntimeError:
        pass  # `value` is a symbolic from a different Opti instance; not part of this solution.

print("-"*100)
print(f"Speed: {sol(V):.4g} m/s, {sol(V) / u.mph:.4g} mph")
print(f"Airspeed: {sol(airspeed):.4g} m/s")
print(f"Drag force: {sol(drag_force):.4g} N")
print(f"Drag power: {sol(drag_power):.4g} W")
print(f"Rolling resistance force: {sol(rolling_resistance_force):.4g} N")
print(f"Rolling resistance power: {sol(rolling_resistance_power):.4g} W")
print(f"Gravity force: {sol(gravity_force):.4g} N")
print(f"Gravity (climb) power: {sol(gravity_power):.4g} W")
print(f"Total output (power_out): {sol(power_out):.4g} W")
print(f"Available power at wheel: {sol(power_at_wheel):.4g} W")
print(f"Grade: {grade*100:.2f} % (angle {np.degrees(incline_angle):.2f}°)")
print(f"Drivetrain efficiency: {drivetrain_efficiency:.4g}")
print(f"Rider input power: {sol(power_in):.4g} W")
print(f"Duration: {duration:.4g} s ({duration/60:.2f} min)")
