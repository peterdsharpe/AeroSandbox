import aerosandbox as asb
import aerosandbox.numpy as np

gamma = 1.4

Cp_crit = lambda M: 2 / (gamma * M ** 2) * (
        (
                (1 + (gamma - 1) / 2 * M ** 2)
                /
                (1 + (gamma - 1) / 2)
        ) ** (gamma / (gamma - 1))
        - 1
)

# Prandtl-Glauert correction
Cp_PG = lambda Cp0, M: Cp0 / (1 - M ** 2) ** 0.5

# Karman-Tsien correction
Cp_KT = lambda Cp0, M: Cp0 / (
        (1 - M ** 2) ** 0.5
        + M ** 2 / (1 + (1 - M ** 2) ** 0.5) * (Cp0 / 2)
)

### Laitone's rule
Cp_L = lambda Cp0, M: Cp0 / (
        (1 - M ** 2) ** 0.5
        + (M ** 2) * (1 + (gamma - 1) / 2 * M ** 2) / (1 + (1 - M ** 2) ** 0.5) * (Cp0 / 2)
)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots()

    machs = np.linspace(1e-6, 1-1e-6, 100)

    Cp0 = -1.5
    plt.plot(
        machs,
        Cp_crit(machs),
        "k",
        label="Critical",
    )
    plt.plot(
        machs,
        Cp_PG(Cp0, machs),
    )
    plt.plot(
        machs,
        Cp_KT(Cp0, machs),
    )
    plt.plot(
        machs,
        Cp_L(Cp0, machs),
    )
    plt.ylim(-4, 1)
    plt.legend([
        "Critical", "Prandtl-Glauert", "Karman-Tsien", "Laitone"
    ]
    )

    p.show_plot(
        title="Comparison of Prandtl-Glauert, Karman-Tsien, and Laitone's rules",
        xlabel="Mach number $M$ [-]",
        ylabel="Pressure coefficient $C_p$ [-]",
        legend=False
    )
