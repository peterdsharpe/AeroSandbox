import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools import units as u
from typing import Tuple

### See also:
# https://www.risingup.com/fars/info/23-index.shtml

def limit_load_factors(
        design_mass_TOGW: float,
        category: str = "normal",
) -> Tuple[float, float]:
    """
    From FAR Part 23: "Airworthiness Standards: Normal, Utility, Acrobatic, and Commuter Category Airplanes"

    Section 23.337: "Limit maneuvering load factors"


    """
    ### Compute positive load factor
    if category == "normal" or category == "commuter":
        positive_load_factor = np.softmin(
            2.1 + (24000 / (design_mass_TOGW / u.lbm + 10000)),
            3.8,
            softness=0.01
        )
    elif category == "utility":
        positive_load_factor = 4.4

    elif category == "acrobatic":
        positive_load_factor = 6.0

    else:
        raise ValueError("Bad value of `category`. Valid values are 'normal', 'utility', 'acrobatic', and 'commuter'.")

    ### Compute negative load factor
    if category == "normal" or category == "commuter" or category == "utility":
        negative_load_factor = -0.4 * positive_load_factor
    elif category == "acrobatic":
        negative_load_factor = -0.5 * positive_load_factor
    else:
        raise ValueError()

    return positive_load_factor, negative_load_factor
