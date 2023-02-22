import aerosandbox.numpy as np
from aerosandbox.weights.mass_properties import MassProperties

"""
Most of these relations are taken from:
https://en.wikipedia.org/wiki/List_of_moments_of_inertia
"""


def mass_properties_from_radius_of_gyration(
        mass: float,
        x_cg: float = 0,
        y_cg: float = 0,
        z_cg: float = 0,
        radius_of_gyration_x: float = 0,
        radius_of_gyration_y: float = 0,
        radius_of_gyration_z: float = 0,
) -> MassProperties:
    """
    Returns the mass properties of an object, given its radius of gyration.

    It's assumed that the principle axes of the inertia tensor are aligned with the coordinate axes.

    This is a shorthand convenience function for common usage of the MassProperties constructor. For more detailed
    use, use the MassProperties object directly.

    Args:
        mass: Mass [kg]
        x_cg: x-position of the center of gravity
        y_cg: y-position of the center of gravity
        z_cg: z-position of the center of gravity
        radius_of_gyration_x: Radius of gyration along the x-axis, about the center of gravity [m]
        radius_of_gyration_y: Radius of gyration along the y-axis, about the center of gravity [m]
        radius_of_gyration_z: Radius of gyration along the z-axis, about the center of gravity [m]

    Returns: MassProperties object.

    """
    return MassProperties(
        mass=mass,
        x_cg=x_cg,
        y_cg=y_cg,
        z_cg=z_cg,
        Ixx=mass * radius_of_gyration_x ** 2,
        Iyy=mass * radius_of_gyration_y ** 2,
        Izz=mass * radius_of_gyration_z ** 2,
        Ixy=0,
        Iyz=0,
        Ixz=0,
    )


def mass_properties_of_ellipsoid(
        mass: float,
        radius_x: float,
        radius_y: float,
        radius_z: float,
) -> MassProperties:
    """
    Returns the mass properties of an ellipsoid centered on the origin.

    Args:
        mass: Mass [kg]
        radius_x: Radius along the x-axis [m]
        radius_y: Radius along the y-axis [m]
        radius_z: Radius along the z-axis [m]

    Returns: MassProperties object.

    """
    return MassProperties(
        mass=mass,
        x_cg=0,
        y_cg=0,
        z_cg=0,
        Ixx=0.2 * mass * (radius_y ** 2 + radius_z ** 2),
        Iyy=0.2 * mass * (radius_z ** 2 + radius_x ** 2),
        Izz=0.2 * mass * (radius_x ** 2 + radius_y ** 2),
        Ixy=0,
        Iyz=0,
        Ixz=0,
    )


def mass_properties_of_sphere(
        mass: float,
        radius: float,
) -> MassProperties:
    """
    Returns the mass properties of a sphere centered on the origin.

    Args:
        mass: Mass [kg]
        radius: Radius [m]

    Returns: MassProperties object.

    """
    return mass_properties_of_ellipsoid(
        mass=mass,
        radius_x=radius,
        radius_y=radius,
        radius_z=radius
    )


def mass_properties_of_rectangular_prism(
        mass: float,
        length_x: float,
        length_y: float,
        length_z: float,
) -> MassProperties:
    """
    Returns the mass properties of a rectangular prism centered on the origin.

    Args:
        mass: Mass [kg]
        length_x: Side length along the x-axis [m]
        length_y: Side length along the y-axis [m]
        length_z: Side length along the z-axis [m]

    Returns: MassProperties object.

    """
    return MassProperties(
        mass=mass,
        x_cg=0,
        y_cg=0,
        z_cg=0,
        Ixx=1 / 12 * mass * (length_y ** 2 + length_z ** 2),
        Iyy=1 / 12 * mass * (length_z ** 2 + length_x ** 2),
        Izz=1 / 12 * mass * (length_x ** 2 + length_y ** 2),
        Ixy=0,
        Iyz=0,
        Ixz=0,
    )


def mass_properties_of_cube(
        mass: float,
        side_length: float,
) -> MassProperties:
    """
    Returns the mass properties of a cube centered on the origin.

    Args:
        mass: Mass [kg]
        side_length: Side length of the cube [m]

    Returns: MassProperties object.

    """
    return mass_properties_of_rectangular_prism(
        mass=mass,
        length_x=side_length,
        length_y=side_length,
        length_z=side_length,
    )
