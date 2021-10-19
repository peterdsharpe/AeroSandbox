from aerosandbox.modeling.fitting import *
from aerosandbox.modeling.interpolation import *

masses = np.load("masses.npy")
spans = np.load("spans.npy")
spar_masses = np.load("spar_masses.npy")
spar_diameters = np.load('spar_diameters.npy')

def spar_mass_model(
        x, p
):
    return p["c"] * (x["span"]/40) ** p["span_exp"] * (x["mass"]/300) ** p["mass_exp"]

params = FittedModel(
    model = spar_mass_model,
    x_data = {
        "mass": masses.reshape(-1),
        "span": spans.reshape(-1),
    },
    y_data = spar_masses.reshape(-1),
    parameter_guesses={
        "c": 1,
        "span_exp": 1,
        "mass_exp": 1,
    },
    put_residuals_in_logspace=True
)
print("\nMODEL\n-----")
print(params.parameters)
print(
    "spar_mass = c * (span/40) ** span_exp * (mass_eff/300) ** mass_exp"
)

def spar_mass_model_solved(
        span, mass,
):
    return spar_mass_model(
        x={
            "mass": mass,
            "span": span,
        },
        p=params
    )

# def spar_diameter_model(
#         x, p
# ):
#     return p["c"] * (x["span"]/40) ** p["span_exp"] * (x["mass"]/300) ** p["mass_exp"]
# InterpolatedModel({"alpha": alpha_array, "reynolds": np.log(reynolds_array)},
# #                                               cl_grid, "bspline")
spar_diameter_model = InterpolatedModel({
        "mass": np.array([i[0] for i in masses]),
        "span": spans[0],},
    spar_diameters,"bspline")

spar_diameter_model({'mass': 104, 'span': 34})

    # parameter_guesses={
    #     "c": 1,
    #     "span_exp": 1,
    #     "mass_exp": 1,
    # },
    # put_residuals_in_logspace=True

# print("\nMODEL\n-----")
# print(params.parameters)
# print(
#     "spar_diameter = c * (span/40) ** span_exp * (mass_eff/300) ** mass_exp"
# )

# def spar_diameter_model_solved(
#         span, mass,
# ):
#     return spar_diameter_model(
#         x={
#             "mass": mass,
#             "span": span,
#         },
#         p=params
#     )
