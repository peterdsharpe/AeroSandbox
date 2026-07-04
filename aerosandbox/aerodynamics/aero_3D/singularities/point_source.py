import aerosandbox.numpy as np
from aerosandbox.numpy.typing import Vectorizable


def calculate_induced_velocity_point_source(
    x_field: Vectorizable,
    y_field: Vectorizable,
    z_field: Vectorizable,
    x_source: Vectorizable,
    y_source: Vectorizable,
    z_source: Vectorizable,
    sigma: Vectorizable = 1,
    viscous_radius: float = 0,
) -> tuple[Vectorizable, Vectorizable, Vectorizable]:
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

        x_source: x-coordinate of the point source

        y_source: y-coordinate of the point source

        z_source: z-coordinate of the point source

        sigma: The strength (volume flux) of the point source.

        viscous_radius: A regularization length used to desingularize the velocity field near the source. If
        nonzero, induced velocities are smoothly attenuated to zero as the field point approaches the source
        location, rather than blowing up. It should be significantly smaller (e.g., at least an order of
        magnitude smaller) than the smallest length scale of interest in the analysis in question.

    Returns: u, v, and w:
        The x-, y-, and z-direction induced velocities.
    """
    dx = np.add(x_field, -x_source)
    dy = np.add(y_field, -y_source)
    dz = np.add(z_field, -z_source)

    r_squared = dx**2 + dy**2 + dz**2

    def smoothed_x_15_inv(x):
        """
        Approximates x^(-1.5) with a function that sharply goes to 0 in the x -> 0 limit.
        """
        if not np.all(viscous_radius == 0):
            return x / (x**2.5 + viscous_radius**2.5)
        else:
            return x**-1.5

    grad_phi_multiplier = np.multiply(sigma, smoothed_x_15_inv(r_squared)) / (4 * np.pi)

    u = np.multiply(grad_phi_multiplier, dx)
    v = np.multiply(grad_phi_multiplier, dy)
    w = np.multiply(grad_phi_multiplier, dz)

    return u, v, w


if __name__ == "__main__":
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

    dir = dir / dir_norm * dir_norm**0.2

    import pyvista as pv

    pv.set_plot_theme("dark")
    plotter = pv.Plotter()
    plotter.add_arrows(cent=pos, direction=dir, mag=0.15)
    plotter.show_grid()
    plotter.show()
