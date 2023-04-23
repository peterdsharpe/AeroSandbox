from aerosandbox import AeroSandboxObject
from aerosandbox.geometry.common import *
from typing import List, Dict, Any, Union, Tuple, Optional
import copy


class Propulsor(AeroSandboxObject):
    """
    Definition for a Propulsor, which could be a propeller, a rotor, or a jet engine.

    Assumes a disk- or cylinder-shaped propulsor.
    """

    def __init__(self,
                 name: Optional[str] = "Untitled",
                 xyz_c: Union[np.ndarray, List[float]] = None,
                 xyz_normal: Union[np.ndarray, List[float]] = None,
                 radius: float = 1.,
                 length: float = 0.,
                 color: Optional[Union[str, Tuple[float]]] = None,
                 analysis_specific_options: Optional[Dict[type, Dict[str, Any]]] = None,
                 ):
        """
        Defines a new propulsor object.

        TODO add docs
        """
        ### Set defaults
        if xyz_c is None:
            xyz_c = np.array([0., 0., 0.])
        if xyz_normal is None:
            xyz_normal = np.array([-1., 0., 0.])
        if analysis_specific_options is None:
            analysis_specific_options = {}

        self.name = name
        self.xyz_c = np.array(xyz_c)
        self.xyz_normal = np.array(xyz_normal)
        self.radius = radius
        self.length = length
        self.color = color
        self.analysis_specific_options = analysis_specific_options

    def __repr__(self) -> str:
        return f"Propulsor '{self.name}' (xyz_c: {self.xyz_c}, radius: {self.radius})"

    def xsec_area(self) -> float:
        """
        Returns the cross-sectional area of the propulsor, in m^2.
        """
        return np.pi * self.radius ** 2

    def xsec_perimeter(self) -> float:
        """
        Returns the cross-sectional perimeter of the propulsor, in m.
        """
        return 2 * np.pi * self.radius

    def volume(self) -> float:
        """
        Returns the volume of the propulsor, in m^3.
        """
        return self.xsec_area() * self.length

    def compute_frame(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the local coordinate frame of the propulsor, in aircraft geometry axes.

        xg_local is aligned with the propulsor's normal vector.

        zg_local is roughly aligned with the z-axis of the aircraft geometry axes, but projected onto the propulsor's plane.

        yg_local is the cross product of zg_local and xg_local.

        Returns: A tuple:
            xg_local: The x-axis of the local coordinate frame, in aircraft geometry axes.
            yg_local: The y-axis of the local coordinate frame, in aircraft geometry axes.
            zg_local: The z-axis of the local coordinate frame, in aircraft geometry axes.

        """
        xyz_normal = self.xyz_normal / np.linalg.norm(self.xyz_normal)

        xg_local = xyz_normal

        zg_local = np.array([0, 0, 1])
        zg_local = zg_local - np.dot(zg_local, xg_local) * xg_local

        yg_local = np.cross(zg_local, xg_local)

        return xg_local, yg_local, zg_local

    def get_disk_3D_coordinates(self,
                                theta: Union[float, np.ndarray] = None,
                                l_over_length: Union[float, np.ndarray] = None,
                                ) -> Tuple[Union[float, np.ndarray]]:
        ### Set defaults
        if theta is None:
            theta = np.linspace(
                0,
                2 * np.pi,
                60 + 1
            )[:-1]
        if l_over_length is None:
            if self.length == 0:
                l_over_length = 0
            else:
                l_over_length = np.linspace(
                    0,
                    1,
                    4
                ).reshape((1, -1))

                theta = np.array(theta).reshape((-1, 1))

        st = np.sin(np.mod(theta, 2 * np.pi))
        ct = np.cos(np.mod(theta, 2 * np.pi))

        x = l_over_length * self.length
        y = ct * self.radius
        z = st * self.radius

        xg_local, yg_local, zg_local = self.compute_frame()

        return (
            self.xyz_c[0] + x * xg_local[0] + y * yg_local[0] + z * zg_local[0],
            self.xyz_c[1] + x * xg_local[1] + y * yg_local[1] + z * zg_local[1],
            self.xyz_c[2] + x * xg_local[2] + y * yg_local[2] + z * zg_local[2],
        )

    def translate(self,
                  xyz: Union[np.ndarray, List[float]],
                  ) -> 'Propulsor':
        """
        Returns a copy of this propulsor that has been translated by `xyz`.

        Args:
            xyz: The amount to translate the propulsor, in meters. Given in aircraft geometry axes, as with everything else.

        Returns: A copy of this propulsor, translated by `xyz`.
        """
        new_propulsor = copy.deepcopy(self)
        new_propulsor.xyz_c = new_propulsor.xyz_c + np.array(xyz)
        return new_propulsor


if __name__ == '__main__':
    p_disk = Propulsor(radius=3)
    p_can = Propulsor(length=1)
