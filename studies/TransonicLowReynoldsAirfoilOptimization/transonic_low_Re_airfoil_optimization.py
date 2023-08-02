import aerosandbox as asb
import aerosandbox.numpy as np
import neuralfoil as nf
from typing import Union, List, Dict
from aerosandbox.modeling.splines.hermite import linear_hermite_patch, cubic_hermite_patch
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates, get_kulfan_parameters
from pprint import pprint


def get_aero(
        kulfan_parameters,
        alpha: Union[float, np.ndarray],
        Re: Union[float, np.ndarray],
        mach: Union[float, np.ndarray] = 0.,
        model_size: str = "large",
        control_surfaces: List["ControlSurface"] = None,
        control_surface_strategy="polar_modification",
        transonic_buffet_lift_knockdown: float = 0.3,
        include_360_deg_effects: bool = True,
) -> Dict[str, Union[float, np.ndarray]]:
    if control_surfaces is None:
        control_surfaces = []

    alpha = np.mod(alpha + 180, 360) - 180  # Enforce periodicity of alpha

    ##### Evaluate the control surfaces of the airfoil

    effective_d_alpha = 0.
    effective_CD_multiplier_from_control_surfaces = 1.

    if control_surface_strategy == "polar_modification":

        for surf in control_surfaces:

            effectiveness = 1 - np.maximum(0, surf.hinge_point + 1e-16) ** 2.751428551177291
            # From XFoil-based study at `/AeroSandbox/studies/ControlSurfaceEffectiveness/`

            effective_d_alpha += surf.deflection * effectiveness

            effective_CD_multiplier_from_control_surfaces *= (
                    2 + (surf.deflection / 11.5) ** 2 - (1 + (surf.deflection / 11.5) ** 2) ** 0.5
            )
            # From fit to wind tunnel data from Hoerner, "Fluid Dynamic Drag", 1965. Page 13-13, Figure 32,
            # "Variation of section drag coefficient of a horizontal tail surface at constant C_L"

    else:
        raise NotImplementedError

    ##### Use NeuralFoil to evaluate the incompressible aerodynamics of the airfoil
    nf_aero = nf.get_aero_from_kulfan_parameters(
        kulfan_parameters=kulfan_parameters,
        alpha=alpha + effective_d_alpha,
        Re=Re,
        model_size=model_size
    )

    CL = nf_aero["CL"]
    CD = nf_aero["CD"] * effective_CD_multiplier_from_control_surfaces
    CM = nf_aero["CM"]
    Cpmin_0 = nf_aero["Cpmin"]
    Top_Xtr = nf_aero["Top_Xtr"]
    Bot_Xtr = nf_aero["Bot_Xtr"]

    ##### Extend aerodynamic data to 360 degrees (post-stall) using wind tunnel behavior here.
    if include_360_deg_effects:
        from aerosandbox.aerodynamics.aero_2D.airfoil_polar_functions import airfoil_coefficients_post_stall

        CL_if_separated, CD_if_separated, CM_if_separated = airfoil_coefficients_post_stall(
            airfoil=asb.Airfoil("naca0012"),
            alpha=alpha
        )
        import aerosandbox.library.aerodynamics as lib_aero

        # These values are so high because NeuralFoil extrapolates quite well past stall
        alpha_stall_positive = 20
        alpha_stall_negative = -20

        # This will be an input to a tanh() sigmoid blend via asb.numpy.blend(), so a value of 1 means the flow is
        # ~90% separated, and a value of -1 means the flow is ~90% attached.
        is_separated = np.softmax(
            alpha - alpha_stall_positive,
            alpha_stall_negative - alpha
        ) / 3

        CL = np.blend(
            is_separated,
            CL_if_separated,
            CL
        )
        CD = np.exp(np.blend(
            is_separated,
            np.log(CD_if_separated + lib_aero.Cf_flat_plate(Re_L=Re, method="turbulent")),
            np.log(CD)
        ))
        CM = np.blend(
            is_separated,
            CM_if_separated,
            CM
        )
        """

        Separated Cpmin_0 model is a very rough fit to Figure 3 of:

        Shademan & Naghib-Lahouti, "Effects of aspect ratio and inclination angle on aerodynamic loads of a flat 
        plate", Advances in Aerodynamics. 
        https://www.researchgate.net/publication/342316140_Effects_of_aspect_ratio_and_inclination_angle_on_aerodynamic_loads_of_a_flat_plate

        """
        Cpmin_0 = np.blend(
            is_separated,
            -1 - 0.5 * np.sind(alpha) ** 2,
            Cpmin_0
        )

        Top_Xtr = np.blend(
            is_separated,
            0.5 - 0.5 * np.tanh(10 * np.sind(alpha)),
            Top_Xtr
        )
        Bot_Xtr = np.blend(
            is_separated,
            0.5 + 0.5 * np.tanh(10 * np.sind(alpha)),
            Bot_Xtr
        )

    ###### Add compressibility effects

    ### Step 1: compute mach_crit, the critical Mach number
    """
    Below is a function that computes the critical Mach number from the incompressible Cp_min.

    It's based on a Laitone-rule compressibility correction (similar to Prandtl-Glauert or Karman-Tsien, 
    but higher order), together with the Cp_sonic relation. When the Laitone-rule Cp equals Cp_sonic, we have reached
    the critical Mach number.

    This approach does not admit explicit solution for the Cp0 -> M_crit relation, so we instead regress a 
    relationship out using symbolic regression. In effect, this is a curve fit to synthetic data.

    See fits at: /AeroSandbox/studies/MachFitting/CriticalMach/
    """
    Cpmin_0 = np.softmin(
        Cpmin_0,
        0,
        softness=0.001
    )

    mach_crit = (
                        1.011571026701678
                        - Cpmin_0
                        + 0.6582431351007195 * (-Cpmin_0) ** 0.6724789439840343
                ) ** -0.5504677038358711

    mach_dd = mach_crit + (0.1 / 80) ** (1 / 3)  # drag divergence Mach number
    # Relation taken from W.H. Mason's Korn Equation

    ### Step 2: adjust CL, CD, CM, Cpmin by compressibility effects
    gamma = 1.4  # Ratio of specific heats, 1.4 for air (mostly diatomic nitrogen and oxygen)
    beta_squared_ideal = 1 - mach ** 2
    beta = np.softmax(
        beta_squared_ideal,
        -beta_squared_ideal,
        softness=0.5  # Empirically tuned to data
    ) ** 0.5

    CL = CL / beta
    # CD = CD / beta
    CM = CM / beta

    # Prandtl-Glauert
    Cpmin = Cpmin_0 / beta

    # Karman-Tsien
    # Cpmin = Cpmin_0 / (
    #     beta
    #     + mach ** 2 / (1 + beta) * (Cpmin_0 / 2)
    # )

    # Laitone's rule
    # Cpmin = Cpmin_0 / (
    #         beta
    #         + (mach ** 2) * (1 + (gamma - 1) / 2 * mach ** 2) / (1 + beta) * (Cpmin_0 / 2)
    # )

    ### Step 3: modify CL based on buffet and supersonic considerations
    # Accounts approximately for the lift drop due to buffet.
    buffet_factor = np.blend(
        50 * (mach - (mach_dd + 0.04)),  # Tuned to RANS CFD data empirically
        np.blend(
            (mach - 1) / 0.1,
            1,
            0.5
        ),
        1,
    )

    # Accounts for the fact that theoretical CL_alpha goes from 2 * pi (subsonic) to 4 (supersonic),
    # following linearized supersonic flow on a thin airfoil.
    cla_supersonic_ratio_factor = np.blend(
        (mach - 1) / 0.1,
        4 / (2 * np.pi),
        1,
    )
    CL = CL * buffet_factor * cla_supersonic_ratio_factor

    # Step 4: Account for wave drag
    t_over_c = 0.10  # TODO make this a parameter

    CD_wave = np.where(
        mach < mach_crit,
        0,
        np.where(
            mach < mach_dd,
            20 * (mach - mach_crit) ** 4,
            np.where(
                mach < 0.97,
                cubic_hermite_patch(
                    mach,
                    x_a=mach_dd,
                    x_b=0.97,
                    f_a=20 * (0.1 / 80) ** (4 / 3),
                    f_b=0.8 * t_over_c,
                    dfdx_a=0.1,
                    dfdx_b=0.8 * t_over_c * 8
                ),
                np.where(
                    mach < 1.1,
                    cubic_hermite_patch(
                        mach,
                        x_a=0.97,
                        x_b=1.1,
                        f_a=0.8 * t_over_c,
                        f_b=0.8 * t_over_c,
                        dfdx_a=0.8 * t_over_c * 8,
                        dfdx_b=-0.8 * t_over_c * 8,
                    ),
                    np.blend(
                        8 * 2 * (mach - 1.1) / (1.2 - 0.8),
                        0.8 * 0.8 * t_over_c,
                        1.2 * 0.8 * t_over_c,
                    )
                )
            )
        )
    )

    CD = CD + CD_wave

    # Step 5: If beyond M_crit or if separated, move the airfoil aerodynamic center back to x/c = 0.5 (Mach tuck)
    has_aerodynamic_center_shift = (mach - (mach_dd + 0.06)) / 0.06

    if include_360_deg_effects:
        has_aerodynamic_center_shift = np.softmax(
            is_separated,
            has_aerodynamic_center_shift,
            softness=0.1
        )

    CM = CM + np.blend(
        has_aerodynamic_center_shift,
        -0.25 * np.cosd(alpha) * CL - 0.25 * np.sind(alpha) * CD,
        0,
    )

    return {
        "CL"       : CL,
        "CD"       : CD,
        "CM"       : CM,
        "Cpmin"    : Cpmin,
        "Top_Xtr"  : Top_Xtr,
        "Bot_Xtr"  : Bot_Xtr,
        "mach_crit": mach_crit,
        "mach_dd"  : mach_dd,
        "Cpmin_0"  : Cpmin_0,
    }


# initial_guess_airfoil = asb.Airfoil("tasopt-c090")
initial_guess_airfoil = asb.Airfoil("naca0080")
initial_guess_parameters = get_kulfan_parameters(initial_guess_airfoil.coordinates)
initial_aero = get_aero(
    kulfan_parameters=initial_guess_parameters,
    alpha=0,
    Re=5e5,
    mach=0.9
)
# pprint(initial_aero)

opti = asb.Opti()
kulfan_parameters = dict(
    lower_weights=opti.variable(
        init_guess=initial_guess_parameters["lower_weights"],
        lower_bound=-1, upper_bound=1
    ),
    upper_weights=opti.variable(
        init_guess=initial_guess_parameters["upper_weights"],
        lower_bound=-1, upper_bound=1
    ),
    leading_edge_weight=opti.variable(
        init_guess=initial_guess_parameters["leading_edge_weight"],
        lower_bound=-1, upper_bound=1
    ),
    TE_thickness=0
)

opti.subject_to(
    kulfan_parameters["upper_weights"] > kulfan_parameters["lower_weights"]
)

alpha = opti.variable(init_guess=0, lower_bound=-20, upper_bound=20)

alphas = alpha + np.linspace(-3, 3, 11)

aero = get_aero(
    kulfan_parameters=kulfan_parameters,
    alpha=alphas,
    Re=5e5,
    mach=0.8,
    model_size="xlarge"
)

opti.minimize(np.mean(-aero["CL"] / aero["CD"]))

airfoil_history = []
aero_history = []


def callback(i):
    airfoil_history.append(
        asb.Airfoil(
            name="in-progress",
            coordinates=get_kulfan_coordinates(**{
                k: opti.debug.value(v) for k, v in kulfan_parameters.items()
            }))
    )
    aero_history.append({
        k: opti.debug.value(v) for k, v in aero.items()
    })


sol = opti.solve(
    callback=callback
)
kulfan_parameters = sol(kulfan_parameters)
alpha = sol(alpha)
aero = sol(aero)

af = asb.Airfoil(
    name="optimized",
    coordinates=get_kulfan_coordinates(
        **kulfan_parameters
    )
)
# af.draw()
pprint(aero)

### Animate
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
from matplotlib.animation import ArtistAnimation

fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
p.show_plot(show=False)
ax[0].set_title("Airfoil Shape")
ax[0].set_xlabel("$x/c$")
ax[0].set_ylabel("$y/c$")

ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("Lift-to-Drag Ratio $C_L/C_D$ [-]")
plt.tight_layout()

from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list(
    "custom_cmap",
    colors=[
        p.adjust_lightness(c, 0.8) for c in
        ["orange", "darkseagreen", "dodgerblue"]
    ]
)

colors = cmap(np.linspace(0, 1, len(airfoil_history)))

ims = []
for i in range(len(airfoil_history)):
    plt.sca(ax[0])
    plt.plot(
        airfoil_history[i].x(),
        airfoil_history[i].y(),
        "-",
        color=colors[i],
        alpha=0.2,
    )
    plt.axis('equal')

    plt.sca(ax[1])
    if i > 0:
        p.plot_color_by_value(
            np.arange(i),
            np.array([
                np.mean(aero_history[j]["CL"] / aero_history[j]["CD"])
                for j in range(i)
            ]),
            ".-",
            c=np.arange(i),
            cmap=cmap,
            clim=(0, len(airfoil_history)),
            alpha=0.8
        )


    plt.suptitle(f"Optimization Progress")

    ims.append([
        *ax[0].get_children(),
        *ax[1].get_children(),
        *fig.get_children(),
    ])

ims.extend([ims[-1]] * 30)

ani = ArtistAnimation(fig, ims, interval=100)

# writer = animation.PillowWriter(fps=15,
#                                 metadata=dict(artist='Me'),
#                                 bitrate=1800)
# ani.save('scatter.gif', writer=writer)

writer = matplotlib.animation.PillowWriter(fps=10)

ani.save("airfoil_optimization.gif", writer=writer)

plt.show()
