import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools.webplotdigitizer_reader import read_webplotdigitizer_csv
from aerosandbox.tools import units as u

data = read_webplotdigitizer_csv("data.csv")['data']

power_hp = data[:, 0]
specific_power_hp_per_lb = data[:, 1]

power = power_hp * u.hp
specific_power = specific_power_hp_per_lb * (u.hp / u.lbm)

mass = power / specific_power

more_data = np.array([  # cols: power, mass
    [0.8 * 343 * 110000 * u.lbf, 21230 * u.lbm],  # GE9X
    [0.8 * 343 * 97300 * u.lbf, 21230 * u.lbm],  # GE90
    [0.6 * 343 * 250, 2.6],  # JetCat P250
])

power = np.append(power, more_data[:, 0])
mass = np.append(mass, more_data[:, 1])

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots()
plt.plot(
    power,
    mass,
    ".k",
)
plt.xscale('log')
plt.yscale('log')

fit = asb.FittedModel(
    # model = lambda x, p: p['a'] * x ** p['b'],
    model=lambda x, p: p['min'] + p['a'] * x ** p['b'],
    x_data=power,
    y_data=mass,
    parameter_guesses={
        'min': 1,
        'a'  : 1,
        'b'  : 1,
    },
    parameter_bounds={
        'min': (0, None),
        'a'  : (0, None),
        'b'  : (0, None),
    },
    put_residuals_in_logspace=True,
    # residual_norm_type="L1"
)
# plt.xlim(left=0)
# plt.ylim(bottom=0)

x_plot = np.geomspace(power.min(), power.max(), 500)
plt.plot(x_plot, fit(x_plot))

p.show_plot(
    "Turboshaft Data",
    "Power [W]",
    "Mass [kg]"
)
