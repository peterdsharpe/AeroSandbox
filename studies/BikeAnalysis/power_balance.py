import aerosandbox as asb
import aerosandbox.numpy as np
import aerosandbox.library.power_human as ph
import aerosandbox.tools.units as u

# %%

studies = [
    "Tour De France",
    "UCI Flying 200m Record",
    "Aerovelo Eta",
    "Theoretical Max @ BM",
    "Theoretical Max @ La Paz",
]

g = 9.81  # m/s^2

for study in studies:
    match study:
        case "Tour De France":
            duration = 4 * 60 * 60  # 4 hours
            CDA = 0.3
            rolling_resistance_coefficient = 0.005
            rho = asb.Atmosphere(altitude=500).density()  # kg/m^3
            mass = 80  # kg (rider + bike)
            drivetrain_efficiency = 0.97  # fraction of crank power reaching the road (chain, derailleur, bearings)
            grade = (
                0.0  # road slope as rise/run (e.g. 0.05 = 5% climb); positive is uphill
            )
            headwind = 0.0  # air speed relative to the ground, along the direction of travel [m/s]; positive opposes motion
            athlete_type = "World-Class Athletes"
        case "UCI Flying 200m Record":
            duration = 1 * 60  # 3 minutes
            CDA = 0.19
            rolling_resistance_coefficient = 0.0025
            rho = asb.Atmosphere(altitude=1000).density()  # kg/m^3
            mass = 80  # kg (rider + bike)
            drivetrain_efficiency = 0.97  # fraction of crank power reaching the road (chain, derailleur, bearings)
            grade = (
                -5/200  # road slope as rise/run (e.g. 0.05 = 5% climb); positive is uphill
            )
            headwind = 0.0  # air speed relative to the ground, along the direction of travel [m/s]; positive opposes motion
            athlete_type = "World-Class Athletes"

        case "Aerovelo Eta":  # Aerovelo Eta
            duration = 1 * 60
            CDA = (
                0.0068  # https://treforevans.github.io/files/undergrad_thesis.pdf pg 40
            )
            rolling_resistance_coefficient = 0.0028
            rho = asb.Atmosphere(
                altitude=1375
            ).density()  # kg/m^3, Battle Mountain, Nevada
            mass = 97  # kg (rider + bike)
            drivetrain_efficiency = 0.97  # fraction of crank power reaching the road (chain, derailleur, bearings)
            grade = (
                0.0  # road slope as rise/run (e.g. 0.05 = 5% climb); positive is uphill
            )
            headwind = 0.0  # air speed relative to the ground, along the direction of travel [m/s]; positive opposes motion
            # athlete_type = "Healthy Men"
            athlete_type = "First-Class Athletes"
        case "Theoretical Max @ BM":
            duration = 1 * 60
            CDA = (
                0.0068  # https://treforevans.github.io/files/undergrad_thesis.pdf pg 40
            )
            rolling_resistance_coefficient = 0.002
            rho = asb.Atmosphere(
                altitude=1375
            ).density()  # kg/m^3, Battle Mountain, Nevada
            mass = 97  # kg (rider + bike)
            drivetrain_efficiency = 0.97  # fraction of crank power reaching the road (chain, derailleur, bearings)
            grade = (
                0.0  # road slope as rise/run (e.g. 0.05 = 5% climb); positive is uphill
            )
            headwind = 0.0  # air speed relative to the ground, along the direction of travel [m/s]; positive opposes motion
            athlete_type = "World-Class Athletes"
        case "Theoretical Max @ La Paz":
            duration = 1 * 60
            CDA = 0.0068
            rolling_resistance_coefficient = 0.002
            rho = asb.Atmosphere(altitude=3658).density()
            mass = 80
            drivetrain_efficiency = 0.97
            grade = 0.0
            headwind = 0.0
            athlete_type = "World-Class Athletes"
        case _:
            raise ValueError(f"Invalid study: {study}")

    ### Road inclination angle from the grade. `cos` reduces the normal force (and thus rolling
    ### resistance) on a slope, while `sin` sets the fraction of weight that opposes climbing.
    incline_angle = np.arctan(grade)

    print(f"\n{study}\n" + "-" * 100)

    opti = asb.Opti()

    V = opti.variable(init_guess=10, lower_bound=0)

    ### Crank power the rider can sustain, less the drivetrain losses between crank and wheel.
    power_in = ph.power_human(duration, dataset=athlete_type)
    power_at_wheel = drivetrain_efficiency * power_in

    ### Aerodynamic drag acts on the air-relative speed (V + headwind), but the power it absorbs is
    ### force times *ground* speed, since that is the distance actually traveled.
    airspeed = V + headwind
    drag_force = 0.5 * rho * airspeed**2 * CDA
    drag_power = drag_force * V

    ### Rolling resistance scales with the normal force, which is reduced on a grade by cos(angle).
    rolling_resistance_force = (
        rolling_resistance_coefficient * mass * g * np.cos(incline_angle)
    )
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

    # ### Print every numeric/symbolic quantity at the solution by scraping the namespace. Two
    # ### guards keep this robust in a live/Jupyter session, where `locals()` is messy:
    # ###   1. Type filter: only scalars and CasADi symbolics are considered, so `sol()` never tries
    # ###      to recursively copy arbitrary objects (e.g. the IPython shell's traitlets, which would
    # ###      overflow the recursion limit).
    # ###   2. Per-item EAFP: a `cas.MX` may belong to a *different* `Opti` instance (a re-run cell or
    # ###      a second problem in the session); `sol()` raises `RuntimeError` on those, so we skip
    # ###      them. Plain scalars and this problem's own symbolics always evaluate cleanly.
    # for name, value in locals().copy().items():
    #     if name.startswith("_") or not isinstance(value, (int, float, cas.MX)):
    #         continue
    #     try:
    #         print(f"{name}: {sol(value):.4g}")
    #     except RuntimeError:
    #         pass  # `value` is a symbolic from a different Opti instance; not part of this solution.

    # print("-"*100)
    # print(f"Speed: {sol(V):.4g} m/s, {sol(V) / u.mph:.4g} mph")
    # print(f"Airspeed: {sol(airspeed):.4g} m/s")
    # print(f"Drag force: {sol(drag_force):.4g} N")
    # print(f"Drag power: {sol(drag_power):.4g} W")
    # print(f"Rolling resistance force: {sol(rolling_resistance_force):.4g} N")
    # print(f"Rolling resistance power: {sol(rolling_resistance_power):.4g} W")
    # print(f"Gravity force: {sol(gravity_force):.4g} N")
    # print(f"Gravity (climb) power: {sol(gravity_power):.4g} W")
    # print(f"Total output (power_out): {sol(power_out):.4g} W")
    # print(f"Available power at wheel: {sol(power_at_wheel):.4g} W")
    # print(f"Grade: {grade*100:.2f} % (angle {np.degrees(incline_angle):.2f}°)")
    # print(f"Drivetrain efficiency: {drivetrain_efficiency:.4g}")
    # print(f"Rider input power: {sol(power_in):.4g} W")
    # print(f"Duration: {duration:.4g} s ({duration/60:.2f} min)")

    import plotly.graph_objects as go
    import matplotlib.colors as mcolors

    ### Evaluate the power budget at the solution. The drivetrain loss is whatever the rider puts
    ### into the crank that never reaches the wheel; the wheel power is then spent on the resistances.
    P_crank = sol(power_in)
    P_wheel = sol(power_at_wheel)
    P_drivetrain_loss = P_crank - P_wheel
    P_drag = sol(drag_power)
    P_roll = sol(rolling_resistance_power)
    P_grav = sol(gravity_power)

    ### A muted, professional palette: a calm blue for the supply chain (crank -> wheel), warm tones
    ### for the dissipative sinks, and a neutral gray for the drivetrain loss.
    node_colors = {
        "Rider (crank)": "#4C72B0",
        "Wheel": "#4C72B0",
        "Drivetrain loss": "#8C8C8C",
        "Aerodynamic drag": "#C44E52",
        "Rolling resistance": "#DD8452",
        "Climbing": "#937860",
        "Descent (gravity)": "#55A868",  # downhill gravity assist: a green "gain" feeding the wheel
    }

    ### Each flow is `(source, target, power)`. Branches carrying negligible power are dropped so the
    ### diagram stays clean (climbing vanishes on the flat; the drivetrain loss vanishes at 100% eff.).
    THRESHOLD_W = 0.5  # [W]
    flows = [
        ("Rider (crank)", "Wheel", P_wheel),
        ("Rider (crank)", "Drivetrain loss", P_drivetrain_loss),
        ("Wheel", "Aerodynamic drag", P_drag),
        ("Wheel", "Rolling resistance", P_roll),
    ]

    ### Gravity is a *signed* contribution. Uphill (`P_grav > 0`) it is a sink that draws power out
    ### of the wheel to gain potential energy (climbing). Downhill (`P_grav < 0`) it reverses into a
    ### source that feeds power into the wheel (descent assist), so the rider needs less of their own
    ### power to hold speed. Either way we record a positive magnitude in the correct direction.
    if P_grav >= 0:
        flows.append(("Wheel", "Climbing", P_grav))
    else:
        flows.append(("Descent (gravity)", "Wheel", -P_grav))

    flows = [
        (source, target, power)
        for source, target, power in flows
        if power > THRESHOLD_W
    ]

    ### Assign each referenced node a stable integer index (Plotly addresses links by index), keeping
    ### only nodes that actually appear in a surviving flow.
    node_labels = list(
        dict.fromkeys(
            [node for source, target, _ in flows for node in (source, target)]
        )
    )
    node_index = {label: i for i, label in enumerate(node_labels)}

    def node_throughput(label: str) -> float:
        """Power passing through a node: its inflow, or (for the source node) its outflow [W]."""
        inflow = sum(power for _, target, power in flows if target == label)
        outflow = sum(power for source, _, power in flows if source == label)
        return inflow if inflow > 0 else outflow

    ### Bake each node's value and its share of crank power into the label, so the figure is fully
    ### self-contained on a static slide (no hover needed). `<br>`/`<b>` are Plotly label markup.
    node_display = []
    for label in node_labels:
        value = node_throughput(label)
        node_display.append(
            f"{label}<br><b>{value:.0f} W</b> ({value / P_crank * 100:.0f}%)"
        )

    def tinted_rgba(hex_color: str, alpha: float = 0.45) -> str:
        """Convert a hex color to a translucent Plotly `rgba(...)` string for ribbon fills."""
        r, g, b, _ = mcolors.to_rgba(hex_color)
        return f"rgba({r * 255:.0f}, {g * 255:.0f}, {b * 255:.0f}, {alpha})"

    def format_duration(seconds: float) -> str:
        """Render a duration [s] with the largest sensible unit, so e.g. 30 s reads as "30 s" rather
        than "0 hr". Uses 3 significant figures, trimming trailing zeros (4 hr, not 4.00 hr)."""
        if seconds >= u.hour:
            return f"{seconds / u.hour:.3g} hr"
        if seconds >= u.minute:
            return f"{seconds / u.minute:.3g} min"
        return f"{seconds / u.second:.3g} s"

    ### Color each ribbon as a translucent tint of its *target* node, so sinks read clearly.
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            valueformat=".0f",
            valuesuffix=" W",
            textfont=dict(color="#222222", size=11),
            node=dict(
                label=node_display,
                color=[node_colors[label] for label in node_labels],
                pad=18,
                thickness=18,
                line=dict(color="white", width=0.5),
            ),
            link=dict(
                source=[node_index[source] for source, _, _ in flows],
                target=[node_index[target] for _, target, _ in flows],
                value=[power for _, _, power in flows],
                color=[tinted_rgba(node_colors[target]) for _, target, _ in flows],
            ),
        )
    )
    fig.update_layout(
        title=dict(
            text=(
                f"Power Balance: {study}<br>"
                f"<sup><b>{sol(V):.1f} m/s ({sol(V) / u.mph:.1f} mph, {sol(V) / u.kph:.1f} kph)</b><br>"
                f"{athlete_type} sustained for {format_duration(duration)}</sup>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=15),
        ),
        font=dict(family="Arial, sans-serif", size=11, color="#333333"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=320,
        height=200 + 120 * sol(P_wheel) / 400,
   
        margin=dict(t=64, l=10, r=10, b=12),
    )
    fig.show()
    fig.write_image(f"power_balance_{study}.png", scale=6)
