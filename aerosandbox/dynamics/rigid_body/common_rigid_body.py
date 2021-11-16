import aerosandbox.numpy as np
from aerosandbox.dynamics.point_mass.common_point_mass import _DynamicsPointMassBaseClass
from abc import ABC, abstractmethod, abstractproperty
from typing import Union, Tuple
from aerosandbox import OperatingPoint, Atmosphere


class _DynamicsRigidBodyBaseClass(_DynamicsPointMassBaseClass, ABC):
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
