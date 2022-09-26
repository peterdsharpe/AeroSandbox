import aerosandbox as asb
import aerosandbox.numpy as np

masses = np.load("masses.npy")
spans = np.load("spans.npy")
spar_masses = np.load("spar_masses.npy")


def spar_mass_model(
        x, p
):
    return p["c"] * (x["span"] / 40) ** p["span_exp"] * (x["mass"] / 300) ** p["mass_exp"]


fit = asb.FittedModel(
    model=spar_mass_model,
    x_data={
        "mass": masses.reshape(-1),
        "span": spans.reshape(-1),
    },
    y_data=spar_masses.reshape(-1),
    parameter_guesses={
        "c"       : 1,
        "span_exp": 1,
        "mass_exp": 1,
    },
    put_residuals_in_logspace=True,
    verbose=False
)
print("\nMODEL\n-----")
for k, v in fit.parameters.items():
    print(
        f"{k} = {v}"
    )
print(
    "spar_mass = c * (span / 40) ** span_exp * (mass_eff / 300) ** mass_exp"
)
