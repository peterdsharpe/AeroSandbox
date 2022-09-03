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
        gamma: Union[float, np.ndarray] = 1,
        trailing_vortex_direction: np.ndarray = None,
        vortex_core_radius: float = 0,
) -> [Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Calculates the induced velocity at a point:
        [x_field, y_field, z_field]
    in a 3D potential-flow flowfield.

    In this flowfield, the following singularity elements are assumed:
        * A single horseshoe vortex consisting of a bound leg and two trailing legs

    This function consists entirely of scalar, elementwise NumPy ufunc operations - so it can be vectorized as
    desired assuming input dimensions/broadcasting are compatible.

    Args:
        x_field: x-coordinate of the field point

        y_field: y-coordinate of the field point

        z_field: z-coordinate of the field point

        x_left: x-coordinate of the left vertex of the bound vortex

        y_left: y-coordinate of the left vertex of the bound vortex

        z_left: z-coordinate of the left vertex of the bound vortex

        x_right: x-coordinate of the right vertex of the bound vortex

        y_right: y-coordinate of the right vertex of the bound vortex

        z_right: z-coordinate of the right vertex of the bound vortex

        gamma: The strength of the horseshoe vortex filament.

        trailing_vortex_direction: The direction that the trailing legs of the horseshoe vortex extend. Usually,
        this is modeled as the direction of the freestream.

        vortex_core_radius: To prevent a vortex singularity, here we use a Kaufmann vortex model. This parameter
        governs the radius of this vortex model. It should be significantly smaller (e.g., at least an order of
        magnitude smaller) than the smallest bound leg in the analysis in question.

    Returns: u, v, and w:
        The x-, y-, and z-direction induced velocities.
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

    a_x = np.add(x_field, -x_left)
    a_y = np.add(y_field, -y_left)
    a_z = np.add(z_field, -z_left)

    b_x = np.add(x_field, -x_right)
    b_y = np.add(y_field, -y_right)
    b_z = np.add(z_field, -z_right)

    u_x = trailing_vortex_direction[0]
    u_y = trailing_vortex_direction[1]
    u_z = trailing_vortex_direction[2]

    # Handle the special case where the field point is on one of the legs (either bound or trailing)
    def smoothed_inv(x):
        "Approximates 1/x with a function that sharply goes to 0 in the x -> 0 limit."
        if not np.all(vortex_core_radius == 0):
            return x / (x ** 2 + vortex_core_radius ** 2)
        else:
            return 1 / x

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

    u = np.multiply(
        constant,
        (
                a_cross_b_x * term1 +
                a_cross_u_x * term2 -
                b_cross_u_x * term3
        )
    )
    v = np.multiply(
        constant,
        (
                a_cross_b_y * term1 +
                a_cross_u_y * term2 -
                b_cross_u_y * term3
        )
    )
    w = np.multiply(
        constant,
        (
                a_cross_b_z * term1 +
                a_cross_u_z * term2 -
                b_cross_u_z * term3
        )
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
