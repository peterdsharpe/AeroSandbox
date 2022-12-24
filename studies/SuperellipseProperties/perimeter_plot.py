from pysr import PySRRegressor
from perimeter import s, arc_lengths
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

arc_lengths_model = (
        (-2.2341106 / (((s ** -2.268331) / 1.2180636) + ((s + 0.4997542) * 1.9967787))) + 2.002495
)

fig, ax = plt.subplots()
plt.plot(s, arc_lengths, "k.", zorder=4, markersize=2)
p.plot_smooth(s, arc_lengths_model, "-")
# plt.plot(s, arc_lengths_model/arc_lengths)
# plt.xlim(0, 5)
# p.equal()
p.show_plot()
