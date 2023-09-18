import aerosandbox.numpy as np
from typing import Union


def calculate_induced_velocity_point_source(
        x_field: Union[float, np.ndarray],
        y_field: Union[float, np.ndarray],
        z_field: Union[float, np.ndarray],
        x_source: Union[float, np.ndarray],
        y_source: Union[float, np.ndarray],
        z_source: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray] = 1,
        viscous_radius=0,
) -> [Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Calculates the induced velocity at a point:
        [x_field, y_field, z_field]
    in a 3D potential-flow flowfield.

    In this flowfield, the following singularity elements are assumed:
        * A single point source

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

        viscous_radius: To prevent a vortex singularity, here we use a Kaufmann vortex model. This parameter
        governs the radius of this vortex model. It should be significantly smaller (e.g., at least an order of
        magnitude smaller) than the smallest bound leg in the analysis in question.

    Returns: u, v, and w:
        The x-, y-, and z-direction induced velocities.
    """
    dx = np.add(x_field, -x_source)
    dy = np.add(y_field, -y_source)
    dz = np.add(z_field, -z_source)

    r_squared = (
            dx ** 2 +
            dy ** 2 +
            dz ** 2
    )

    def smoothed_x_15_inv(x):
        """
        Approximates x^(-1.5) with a function that sharply goes to 0 in the x -> 0 limit.
        """
        if not np.all(viscous_radius == 0):
            return x / (x ** 2.5 + viscous_radius ** 2.5)
        else:
            return x ** -1.5

    grad_phi_multiplier = np.multiply(sigma, smoothed_x_15_inv(r_squared)) / (4 * np.pi)

    u = np.multiply(grad_phi_multiplier, dx)
    v = np.multiply(grad_phi_multiplier, dy)
    w = np.multiply(grad_phi_multiplier, dz)

    return u, v, w


if __name__ == '__main__':
    args = (-2, 2, 30)
    x = np.linspace(*args)
    y = np.linspace(*args)
    z = np.linspace(*args)
    X, Y, Z = np.meshgrid(x, y, z)

    Xf = X.flatten()
    Yf = Y.flatten()
    Zf = Z.flatten()


    def wide(array):
        return np.reshape(array, (1, -1))


    def tall(array):
        return np.reshape(array, (-1, 1))


    Uf, Vf, Wf = calculate_induced_velocity_point_source(
        x_field=Xf,
        y_field=Yf,
        z_field=Zf,
        x_source=1,
        y_source=0,
        z_source=0,
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
    plotter.show_grid()
    plotter.show()
