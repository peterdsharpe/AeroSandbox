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


fig, ax = plt.subplots()
for kv in np.linspace(500, 1000, 6):
    perf = get_perf(kv)
    plt.plot(
        perf['rpm'],
        perf['efficiency'],
        label=f"{kv:.0f} kv"
    )

plt.xlim(left=0)
p.show_plot(
    "",
    "RPM",
    "Efficiency [-]"
)
