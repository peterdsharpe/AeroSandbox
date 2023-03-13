import aerosandbox as asb
import aerosandbox.numpy as np
import aerosandbox.tools.units as u


# From Torenbeek: "Synthesis of Subsonic Airplane Design", 1976, Delft University Press
# Chapter 8: "Airplane Weight and Balance"

def mass_wing(
        wing: asb.Wing,
        design_mass_TOGW: float,
        ultimate_load_factor: float,
        suspended_mass: float,
        main_gear_mounted_to_wing: bool = True,
) -> float:

    k_w = np.blend(
        (design_mass_TOGW - 5670) / 2000,
        6.67e-3,
        4.90e-3
    )

    span = wing.span() / np.cosd(wing.mean_sweep_angle(x_nondim=0.5))

    wing_root_thickness = wing.xsecs[0].airfoil.max_thickness() * wing.xsecs[0].chord

    return suspended_mass * (
        k_w *
        span ** 0.75 *
        (1 + (1.905 / span) ** 0.5) *
        ultimate_load_factor ** 0.55 *
        ((span / wing_root_thickness) / (suspended_mass / wing.area())) ** 0.30 *
        (1 if main_gear_mounted_to_wing else 0.95)
    )