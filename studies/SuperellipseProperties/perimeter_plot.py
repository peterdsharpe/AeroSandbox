import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import aerosandbox.numpy as np
from perimeter import S as s, H as h, Arc_lengths


def model(s, h):
    m = """
h + (((((s-0.88487077) * h + 0.2588574 / h) ^ exp(s / -0.90069205)) + h) + 0.09919785) ^ (-1.4812293 / s)
    """
    m = m.replace("exp", "np.exp").replace("log", "np.log").replace("^", "**")

    return eval(m.strip())


fig, ax = plt.subplots()
p.contour(
    s,
    h,
    100 * (model(s, h) - Arc_lengths) / Arc_lengths,
    linelabels_format=lambda x: f"{x:+.1f}%",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    levels=np.arange(-10, 10, 0.1),
    colorbar_label="Model Error [%]",
)
plt.xscale("log")
plt.yscale("log")
# plt.ylim(1, 10)
p.show_plot("Model Error", "s", "h")
