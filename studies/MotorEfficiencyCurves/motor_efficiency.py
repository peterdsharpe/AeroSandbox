import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.library.propulsion_electric import motor_electric_performance

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p


def get_perf(kv):
    throttle = np.arange(0.1, 1.01, 0.02)
    battery_voltage = 3.7 * 4

    prop_torque = 0.1
    prop_rpm = 3300

    opti = asb.Opti()

    rpm = opti.variable(init_guess=1000 * np.ones_like(throttle))

    perf = motor_electric_performance(
        voltage=battery_voltage * throttle,
        rpm=rpm,
        kv=kv,
        no_load_current=1,
        resistance=0.025,
    )

    opti.subject_to([
        perf['torque'] / rpm ** 2 == prop_torque / prop_rpm ** 2
    ])

    sol = opti.solve()

    return sol(perf)

kvs = np.geomspace(400, 2000, 7)

colors = plt.cm.get_cmap("rainbow_r")(np.linspace(0, 1, len(kvs)))

fig, ax = plt.subplots()
for i, kv in enumerate(kvs):
    perf = get_perf(kv)
    plt.plot(
        perf['rpm'],
        perf['efficiency'],
        label=f"{kv:.0f} kv",
        color=p.adjust_lightness(colors[i],0.7),
        alpha=0.7
    )

plt.xlim(0, 10000)
plt.ylim(0, 1)
p.set_ticks(2000, 500, 0.2, 0.05)
p.show_plot(
    "Motor kv vs. Motor Efficiency Curve",
    "RPM",
    "Efficiency [-]"
)
