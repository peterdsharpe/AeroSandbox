import aerosandbox as asb
import aerosandbox.numpy as np

# import matplotlib.pyplot as plt
# import aerosandbox.tools.pretty_plots as p

# p.mpl.use('WebAgg')

# fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150)

airfoil = asb.Airfoil("rae2822")
Re = 6.5e6
alpha = 1

machs = np.concatenate(
    [
        np.arange(0.1, 0.5, 0.05),
        np.arange(0.5, 0.6, 0.01),
        np.arange(0.6, 0.8, 0.003),
    ]
)
#
# ##### XFoil v6
# xfoil6 = {}
# for mach in tqdm(machs[machs < 1], desc="XFoil 6"):
#     xfoil6[mach] = asb.XFoil(
#         airfoil=airfoil,
#         Re=Re,
#         mach=mach,
#         # verbose=True,
#     ).alpha(alpha)
# xfoil6_Cds = {k: v['CD'] for k, v in xfoil6.items() if len(v['CD']) != 0}
#
# plt.plot(
#     np.array(list(xfoil6_Cds.keys()), dtype=float),
#     np.concatenate(list(xfoil6_Cds.values())),
#     ".-",
#     label="XFoil 6"
# )
#
# ##### XFoil v7
# xfoil7 = {}
# for mach in tqdm(machs[machs < 1], desc="XFoil 7"):
#     try:
#         xfoil7[mach] = asb.XFoil(
#             airfoil=airfoil,
#             Re=Re,
#             mach=mach,
#             xfoil_command="xfoil7",
#             # verbose=True
#         ).alpha(alpha)
#     except RuntimeError:
#         pass
# xfoil7_Cds = {k: v['CD'] for k, v in xfoil7.items() if len(v['CD']) != 0}
#
# plt.plot(
#     np.array(list(xfoil7_Cds.keys()), dtype=float),
#     np.concatenate(list(xfoil7_Cds.values())),
#     ".-",
#     label="XFoil 7"
# )

##### MSES
ms = asb.MSES(
    airfoil=airfoil,
    behavior_after_unconverged_run="reinitialize",
    mset_n=280,
    mset_io=40,
    verbosity=1,
)
mses = ms.run(
    alpha=alpha,
    Re=Re,
    mach=machs,
)
# plt.plot(
#     mses['mach'],
#     mses['CD'],
#     ".-",
#     label="MSES"
# )
#
# plt.ylim(bottom=0)
#
# from aerosandbox.tools.string_formatting import eng_string
#
# p.show_plot(
#     f"Drela 2D Viscous Airfoil Tools Comparison\n{airfoil.name} Airfoil, alpha = {alpha}°, Re = {eng_string(Re)}",
#     "Mach [-]",
#     "Drag Coefficient $C_D$",
#     savefig="comparison.png"
# )
