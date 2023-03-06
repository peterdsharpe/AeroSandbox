import aerosandbox.numpy as np


def column_buckling_critical_load(
        elastic_modulus: float,
        moment_of_inertia: float,
        length: float,
        boundary_condition_type: str = "pin-pin",
        use_recommended_design_values: bool = True,
):
    """
    Computes the critical load (in N) for a column or tube in compression to buckle via primary buckling. Uses Euler's classical critical
    load formula.

    Args:
        elastic_modulus: The elastic modulus of the material, in Pa.

        moment_of_inertia: The moment of inertia of the cross-section, in m^4.

        length: The length of the column, in m.

        boundary_condition_type: The boundary condition type. Options are:
            - "pin-pin"
            - "pin-clamp"
            - "clamp-clamp"
            - "clamp-pin"
            - "clamp-free"
            - "free-clamp"

        use_recommended_design_values: Whether to use the recommended design value of K for a given boundary condition (True)
        or to use the less-conservative theoretical value (False).

            * Recommended values are from Table C.1.8.1 in Steel Construction Manual, 8th edition, 2nd revised
            printing, American Institute of Steel Construction, 1987 via WikiMedia:
            https://commons.wikimedia.org/wiki/File:ColumnEffectiveLength.png

    Returns:
        The critical compressive load (in N) for the column or tube to buckle via primary buckling.

    """
    if boundary_condition_type == "pin-pin":
        K = 1.00 if use_recommended_design_values else 1.00
    elif boundary_condition_type == "pin-clamp" or boundary_condition_type == "clamp-pin":
        K = 0.80 if use_recommended_design_values else 0.70
    elif boundary_condition_type == "clamp-clamp":
        K = 0.65 if use_recommended_design_values else 0.50
    elif boundary_condition_type == "clamp-free" or boundary_condition_type == "free-clamp":
        K = 2.10 if use_recommended_design_values else 2.00
    else:
        raise ValueError("Invalid `boundary_condition_type`.")

    return (
            np.pi ** 2 * elastic_modulus * moment_of_inertia
            / (K * length) ** 2
    )


def thin_walled_tube_crippling_buckling_critical_load(
        elastic_modulus: float,
        wall_thickness: float,
        radius: float,
        use_recommended_design_values: bool = True,
):
    """
    Computes the critical load for a thin-walled tube in compression to fail in the crippling mode. (Note: you should also check for
    failure by primary buckling using the `column_buckling_critical_load()` function.)

    The crippling mode is a specific instability mode for tubes with thin walls when loaded in compression. It can be
    seen when you step on a soda can and it buckles inwards. The critical load for this mode is given by the
    following formula:

        stress_crippling = crippling_constant * (E * t / r)

    where:

        A recommended value of crippling_constant = 0.3 is given in Raymer: Aircraft Design: A Conceptual Approach,
        5th Edition, Eq. 14.33, pg. 554.

        A theoretically more accurate value of crippling_constant = 0.605 is given in the Air Force Stress Manual,
        Section 2.3.2.1, Eq. 2-20. This value assumes mu = 0.3, which is a good assumption for most metals.

        and E is the elastic modulus, t is the wall thickness, and r is the radius.

    Args:
        elastic_modulus: The elastic modulus of the material, in Pa.

        wall_thickness: The wall thickness of the tube, in m.

        radius: The radius of the tube, in m.

        use_recommended_design_values: Whether to use the recommended design value of crippling_constant (True)
        or to use the less-conservative theoretical value (False).

    Returns:
        The critical compressive load (in N) for the tube to buckle in the crippling mode.

    """
    if use_recommended_design_values:
        crippling_stress_constant = 0.3
        # Taken from Raymer: Aircraft Design: A Conceptual Approach, 5th Edition, Eq. 14.33, pg. 554.
        #
        # According to the Air Force Stress Manual, Figure 2-67, this value should drop as radius/wall_thickness
        # increases.
    else:
        crippling_stress_constant = 0.605
        # Theoretically, this should be (3 * (1 - mu^2))^(-0.5), where mu is the Poisson's ratio.
        # Following the Air Force Stress Manual, Section 2.3.2.1, Eq. 2-20.
        # The above value assumes mu = 0.3, which is a good assumption for most metals.

    crippling_stress = 0.3 * (elastic_modulus * wall_thickness / radius)

    tube_xsec_area = 2 * np.pi * radius * wall_thickness

    crippling_load = crippling_stress * tube_xsec_area

    return crippling_load
