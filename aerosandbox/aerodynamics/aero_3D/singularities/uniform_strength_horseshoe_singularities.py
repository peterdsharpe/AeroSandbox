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
        trailing_vortex_direction: np.ndarray = None,
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
    if trailing_vortex_direction is None:
        trailing_vortex_direction = np.array([1, 0, 0])

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

    # Handle the special case where the field point is on one of the legs
    def smoothed_inv(x):
        "Approximates 1/x with a function that sharply goes to 0 in the x -> 0 limit."
        return x / (x ** 2 + 1e-8)

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
    norm_a_inv = smoothed_inv(norm_a)
    norm_b_inv = smoothed_inv(norm_b)

    ### Calculate Vij

    term1 = (norm_a_inv + norm_b_inv) * smoothed_inv(norm_a * norm_b + a_dot_b)
    term2 = norm_a_inv * smoothed_inv(norm_a - a_dot_u)
    term3 = norm_b_inv * smoothed_inv(norm_b - b_dot_u)

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
    ##### Check single vortex
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
    )
    print(u, v, w)

    ##### Plot grid of single vortex
    args = (-2, 2, 30)
    x = np.linspace(*args)
    y = np.linspace(*args)
    z = np.linspace(*args)
    X, Y, Z = np.meshgrid(x, y, z)

    Xf = X.flatten()
    Yf = Y.flatten()
    Zf = Z.flatten()

    left = [0, -1, 0]
    right = [0, 1, 0]

    Uf, Vf, Wf = calculate_induced_velocity_horseshoe(
        x_field=Xf,
        y_field=Yf,
        z_field=Zf,
        x_left=left[0],
        y_left=left[1],
        z_left=left[2],
        x_right=right[0],
        y_right=right[1],
        z_right=right[2],
        gamma=1,
    )

    pos = np.stack((Xf, Yf, Zf)).T
    dir = np.stack((Uf, Vf, Wf)).T

    dir_norm = np.reshape(np.linalg.norm(dir, axis=1), (-1, 1))

    dir = dir / dir_norm * dir_norm ** 0.2

    import pyvista as pv

    pv.set_plot_theme('dark')
    plotter = pv.Plotter()
    plotter.add_arrows(
        cent=pos,
        direction=dir,
        mag=0.15
    )
    plotter.add_lines(
        lines=np.array([
            [Xf.max(), left[1], left[2]],
            left,
            right,
            [Xf.max(), right[1], right[2]]
        ])
    )
    plotter.show_grid()
    plotter.show()

    ##### Check multiple vortices
    args = (-2, 2, 30)
    x = np.linspace(*args)
    y = np.linspace(*args)
    z = np.linspace(*args)
    X, Y, Z = np.meshgrid(x, y, z)

    Xf = X.flatten()
    Yf = Y.flatten()
    Zf = Z.flatten()

    left = [0, -1, 0]
    center = [0, 0, 0]
    right = [0, 1, 0]

    lefts = np.array([left, center])
    rights = np.array([center, right])
    strengths = np.array([2, 1])


    def wide(array):
        return np.reshape(array, (1, -1))


    def tall(array):
        return np.reshape(array, (-1, 1))


    Uf_each, Vf_each, Wf_each = calculate_induced_velocity_horseshoe(
        x_field=wide(Xf),
        y_field=wide(Yf),
        z_field=wide(Zf),
        x_left=tall(lefts[:, 0]),
        y_left=tall(lefts[:, 1]),
        z_left=tall(lefts[:, 2]),
        x_right=tall(rights[:, 0]),
        y_right=tall(rights[:, 1]),
        z_right=tall(rights[:, 2]),
        gamma=tall(strengths),
    )

    Uf = np.sum(Uf_each, axis=0)
    Vf = np.sum(Vf_each, axis=0)
    Wf = np.sum(Wf_each, axis=0)

    pos = np.stack((Xf, Yf, Zf)).T
    dir = np.stack((Uf, Vf, Wf)).T

    dir_norm = np.reshape(np.linalg.norm(dir, axis=1), (-1, 1))

    dir = dir / dir_norm * dir_norm ** 0.2

    import pyvista as pv

    pv.set_plot_theme('dark')
    plotter = pv.Plotter()
    plotter.add_arrows(
        cent=pos,
        direction=dir,
        mag=0.15
    )
    plotter.add_lines(
        lines=np.array([
            [Xf.max(), left[1], left[2]],
            left,
            center,
            [Xf.max(), center[1], center[2]],
            center,
            right,
            [Xf.max(), right[1], right[2]]
        ])
    )
    plotter.show_grid()
    plotter.show()
