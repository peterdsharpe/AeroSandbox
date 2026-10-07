import aerosandbox.numpy as np
from aerosandbox.dynamics.point_mass.common_point_mass import (
    _DynamicsPointMassBaseClass,
)
from abc import ABC, abstractmethod
from aerosandbox import OperatingPoint, Atmosphere
from aerosandbox.numpy.typing import Vectorizable
from typing import Literal


class _DynamicsRigidBodyBaseClass(_DynamicsPointMassBaseClass, ABC):
    # TODO: add method for force at offset (i.e., add moment and force)

    @abstractmethod
    def add_moment(
        self,
        Mx: Vectorizable = 0,
        My: Vectorizable = 0,
        Mz: Vectorizable = 0,
        axes: Literal["geometry", "body", "wind", "stability", "earth"] = "body",
    ) -> None:
        """
        Add a moment (in whichever axis system you choose) to this Dynamics instance.

        Parameters
        ----------
        Mx : Vectorizable
            Moment about the x-axis in the axis system chosen. Assumed these moments are applied
            about the center of mass. [Nm]
        My : Vectorizable
            Moment about the y-axis in the axis system chosen. Assumed these moments are applied
            about the center of mass. [Nm]
        Mz : Vectorizable
            Moment about the z-axis in the axis system chosen. Assumed these moments are applied
            about the center of mass. [Nm]
        axes : Literal["geometry", "body", "wind", "stability", "earth"]
            The axis system that the specified moment is in. One of:

            * "geometry"
            * "body"
            * "wind"
            * "stability"
            * "earth"

        Returns
        -------
        None
            (Operates in-place.)
        """
        pass

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
    def alpha(self):
        """Return the angle of attack, in degrees."""
        return np.arctan2d(self.w_b, self.u_b)

    @property
    def beta(self):
        """Return the sideslip angle, in degrees."""
        return np.arctan2d(self.v_b, (self.u_b**2 + self.w_b**2) ** 0.5)

    @property
    def rotational_kinetic_energy(self):
        """
        Compute the kinetic energy [J] from rotational motion.

            KE = 0.5 * omega^T @ I @ omega

        where `omega = [p, q, r]` is the angular velocity vector and `I` is the inertia tensor
        (about the center of mass), including the products of inertia. Note that
        `MassProperties.Ixy`, `.Iyz`, and `.Ixz` are the inertia-tensor *elements* (i.e.,
        I12 = -sum(m * x * y), etc.); see the `MassProperties` docstring.

        Returns
        -------
        float
            Kinetic energy [J]
        """
        return 0.5 * (
            self.mass_props.Ixx * self.p**2
            + self.mass_props.Iyy * self.q**2
            + self.mass_props.Izz * self.r**2
        ) + (
            self.mass_props.Ixy * self.p * self.q
            + self.mass_props.Iyz * self.q * self.r
            + self.mass_props.Ixz * self.p * self.r
        )

    @property
    def kinetic_energy(self):
        return self.translational_kinetic_energy + self.rotational_kinetic_energy
