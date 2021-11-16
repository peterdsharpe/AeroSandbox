import aerosandbox.numpy as np
from aerosandbox.dynamics.point_mass.common_point_mass import _DynamicsPointMassBaseClass
from abc import ABC, abstractmethod, abstractproperty
from typing import Union, Tuple
from aerosandbox import OperatingPoint, Atmosphere


class _DynamicsRigidBodyBaseClass(_DynamicsPointMassBaseClass, ABC):
    @abstractmethod
    def convert_axes(self,
                     x_from: float,
                     y_from: float,
                     z_from: float,
                     from_axes: str,
                     to_axes: str,
                     ) -> Tuple[float, float, float]:
        """
        Converts a vector [x_from, y_from, z_from], as given in the `from_axes` frame, to an equivalent vector [x_to,
        y_to, z_to], as given in the `to_axes` frame.

        Identical to OperatingPoint.convert_axes(), but adds in "earth" as a valid axis frame. For more documentation,
        see the docstring of OperatingPoint.convert_axes().

        Both `from_axes` and `to_axes` should be a string, one of:
                * "geometry"
                * "body"
                * "wind"
                * "stability"
                * "earth"

        Args:
                x_from: x-component of the vector, in `from_axes` frame.
                y_from: y-component of the vector, in `from_axes` frame.
                z_from: z-component of the vector, in `from_axes` frame.
                from_axes: The axes to convert from.
                to_axes: The axes to convert to.

        Returns: The x-, y-, and z-components of the vector, in `to_axes` frame. Given as a tuple.

        """
        pass

    @abstractmethod
    def add_force(self,
                  Fx: Union[np.ndarray, float] = 0,
                  Fy: Union[np.ndarray, float] = 0,
                  Fz: Union[np.ndarray, float] = 0,
                  axes="body",
                  ) -> None:
        """
        Adds a force (in whichever axis system you choose) to this dynamics instance.

        Args:
            Fx: Force in the x-direction in the axis system chosen. [N]
            Fy: Force in the y-direction in the axis system chosen. [N]
            Fz: Force in the z-direction in the axis system chosen. [N]
            axes: The axis system that the specified force is in. One of:
                * "geometry"
                * "body"
                * "wind"
                * "stability"
                * "earth"

        Returns: None (in-place)

        """
        pass

    @abstractmethod
    def add_moment(self,
                   Mx: Union[np.ndarray, float] = 0,
                   My: Union[np.ndarray, float] = 0,
                   Mz: Union[np.ndarray, float] = 0,
                   axes="body",
                   ) -> None:
        """
        Adds a force (in whichever axis system you choose) to this dynamics instance.

        Args:
            Fx: Moment about the x-axis in the axis system chosen. Assumed these moments are applied about the center of mass. [Nm]
            Fy: Moment about the y-axis in the axis system chosen. Assumed these moments are applied about the center of mass. [Nm]
            Fz: Moment about the z-axis in the axis system chosen. Assumed these moments are applied about the center of mass. [Nm]
            axes: The axis system that the specified force is in. One of:
                * "geometry"
                * "body"
                * "wind"
                * "stability"
                * "earth"

        Returns: None (in-place)

        """
        pass

    @abstractproperty
    def alpha(self):
        """The angle of attack, in degrees."""
        return np.arctan2d(
            self.w_b,
            self.u_b
        )

    @abstractproperty
    def beta(self):
        """The sideslip angle, in degrees."""
        return np.arctan2d(
            self.v_b,
            (
                    self.u_b ** 2 +
                    self.w_b ** 2
            ) ** 0.5
        )

    @property
    def op_point(self):
        return OperatingPoint(
            atmosphere=Atmosphere(altitude=self.altitude),
            velocity=self.speed,
            alpha=self.alpha,
            beta=self.beta,
            p=self.p,
            q=self.q,
            r=self.r,
        )

    @property
    def rotational_kinetic_energy(self):
        return 0.5 * (
                self.mass_props.Ixx * self.p ** 2 +
                self.mass_props.Iyy * self.q ** 2 +
                self.mass_props.Izz * self.r ** 2
        )

    @property
    def kinetic_energy(self):
        return self.translational_kinetic_energy + self.rotational_kinetic_energy
