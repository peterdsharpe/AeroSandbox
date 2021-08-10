import aerosandbox.numpy as np
from typing import Union


def calculate_induced_velocity_horseshoe(
        x_field: Union[float, np.ndarray],
        y_field: Union[float, np.ndarray],
        z_field: Union[float, np.ndarray],
        x_left: Union[float, np.ndarray],
        y_left: Union[float, np.ndarray],
        z_left: Union[float, np.ndarray],
        x_right: Union[float, np.ndarray],
        y_right: Union[float, np.ndarray],
        z_right: Union[float, np.ndarray],
        gamma: np.ndarray,
        trailing_vortex_direction: np.ndarray = np.array([1, 0, 0]),
        regularize_dot_product: bool = True,
) -> [Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Calculates the induced velocity at a point:
        [x_field, y_field, z_field]
    in a 3D potential-flow flowfield.

    In this flowfield, the following singularity elements are assumed:
        * A single horseshoe vortex consisting of a bound leg and two trailing legs



    Args:
        x_field:
        y_field:
        z_field:
        x_left:
        y_left:
        z_left:
        x_right:
        y_right:
        z_right:
        gamma:

    Returns:

    """
    np.assert_equal_shape({
        "x_field": x_field,
        "y_field": y_field,
        "z_field": z_field,
    })
    np.assert_equal_shape({
        "x_left" : x_left,
        "y_left" : y_left,
        "z_left" : z_left,
        "x_right": x_right,
        "y_right": y_right,
        "z_right": z_right,
    })

    a_x = x_field - x_left
    a_y = y_field - y_left
    a_z = z_field - z_left

    b_x = x_field - x_right
    b_y = y_field - y_right
    b_z = z_field - z_right

    u_x = trailing_vortex_direction[0]
    u_y = trailing_vortex_direction[1]
    u_z = trailing_vortex_direction[2]

    ### Do some useful arithmetic

    a_cross_b_x = a_y * b_z - a_z * b_y
    a_cross_b_y = a_z * b_x - a_x * b_z
    a_cross_b_z = a_x * b_y - a_y * b_x
    a_dot_b = a_x * b_x + a_y * b_y + a_z * b_z

    a_cross_u_x = a_y * u_z - a_z * u_y
    a_cross_u_y = a_z * u_x - a_x * u_z
    a_cross_u_z = a_x * u_y - a_y * u_x
    a_dot_u = a_x * u_x + a_y * u_y + a_z * u_z

    b_cross_u_x = b_y * u_z - b_z * u_y
    b_cross_u_y = b_z * u_x - b_x * u_z
    b_cross_u_z = b_x * u_y - b_y * u_x
    b_dot_u = b_x * u_x + b_y * u_y + b_z * u_z

    norm_a = (a_x ** 2 + a_y ** 2 + a_z ** 2) ** 0.5
    norm_b = (b_x ** 2 + b_y ** 2 + b_z ** 2) ** 0.5
    norm_a_inv = 1 / norm_a
    norm_b_inv = 1 / norm_b

    # Handle the special case where the field point is on the bound leg
    if regularize_dot_product:
        a_dot_b -= 1e-8

    ### Calculate Vij
    term1 = (norm_a_inv + norm_b_inv) / (norm_a * norm_b + a_dot_b)
    term2 = norm_a_inv / (norm_a - a_dot_u)
    term3 = norm_b_inv / (norm_b - b_dot_u)

    constant = gamma / (4 * np.pi)

    u = constant * (
            a_cross_b_x * term1 +
            a_cross_u_x * term2 -
            b_cross_u_x * term3
    )
    v = constant * (
            a_cross_b_y * term1 +
            a_cross_u_y * term2 -
            b_cross_u_y * term3
    )
    w = constant * (
            a_cross_b_z * term1 +
            a_cross_u_z * term2 -
            b_cross_u_z * term3
    )

    return u, v, w


if __name__ == '__main__':
    u, v, w = calculate_induced_velocity_horseshoe(
        x_field=0,
        y_field=0,
        z_field=0,
        x_left=-1,
        y_left=-1,
        z_left=0,
        x_right=-1,
        y_right=1,
        z_right=0,
        gamma=1,
        regularize_dot_product=False
    )
    print(u, v, w)
