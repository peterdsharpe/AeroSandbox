import aerosandbox.numpy as np
from aerosandbox.common import AeroSandboxObject


class MassProperties(AeroSandboxObject):
    """
    Mass properties of a rigid 3D object.
    """

    def __init__(self,
                 mass: float = 0,
                 x_cg: float = 0,
                 y_cg: float = 0,
                 z_cg: float = 0,
                 Ixx: float = 0,
                 Iyy: float = 0,
                 Izz: float = 0,
                 Ixy: float = 0,
                 Iyz: float = 0,
                 Ixz: float = 0,
                 ):
        """
        Initializes a new MassProperties object.

        Axes can be given in any convenient axes system, as long as mass properties are not combined across different
        axis systems. For aircraft design, the most common axis system is typically geometry axes.

        Args:

            mass: Mass of the component [kg]

            x_cg: X-location of the center of gravity of the component [m]

            y_cg: Y-location of the center of gravity of the component [m]

            z_cg:  Z-location of the center of gravity of the component [m]

            Ixx: Respective component of the inertia tensor, as measured about the component's center of mass. 0 if
            this is a point mass.

            Iyy: Respective component of the inertia tensor, as measured about the component's center of mass. 0 if
            this is a point mass.

            Izz: Respective component of the inertia tensor, as measured about the component's center of mass. 0 if
            this is a point mass.

            Iyz: Respective component of the inertia tensor, as measured about the component's center of mass. 0 if
            this is symmetric about x.

            Ixz: Respective component of the inertia tensor, as measured about the component's center of mass. 0 if
            this is symmetric about y.

            Ixy: Respective component of the inertia tensor, as measured about the component's center of mass. 0 if
            this is symmetric about z.

        """
        self.mass = mass
        self.x_cg = x_cg
        self.y_cg = y_cg
        self.z_cg = z_cg
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.Ixy = Ixy
        self.Iyz = Iyz
        self.Ixz = Ixz

    def __repr__(self) -> str:
        params_to_show = [
            "mass",
            "x_cg",
            "y_cg",
            "z_cg",
        ]

        param_data = ", ".join([
            f"{param}: {self.__getattribute__(param)}"
            for param in params_to_show
        ])

        return f"MassProperties ({param_data})"

    def __getitem__(self, index):
        def get_item_of_attribute(a):
            try:
                return a[index]
            except TypeError:  # object is not subscriptable
                return a
            except IndexError as e:  # index out of range
                raise IndexError("A state variable could not be indexed, since the index is out of range!")

        inputs = {
            "mass": self.mass,
            "x_cg": self.x_cg,
            "y_cg": self.y_cg,
            "z_cg": self.z_cg,
            "Ixx" : self.Ixx,
            "Iyy" : self.Iyy,
            "Izz" : self.Izz,
            "Ixy" : self.Ixy,
            "Iyz" : self.Iyz,
            "Ixz" : self.Ixz,
        }

        return self.__class__(
            **{
                k: get_item_of_attribute(v)
                for k, v in inputs.items()
            }
        )

    def __add__(self, other: "MassProperties") -> "MassProperties":
        """
        Combines one MassProperties object with another.
        """
        total_mass = self.mass + other.mass
        total_x_cg = (self.mass * self.x_cg + other.mass * other.x_cg) / total_mass
        total_y_cg = (self.mass * self.y_cg + other.mass * other.y_cg) / total_mass
        total_z_cg = (self.mass * self.z_cg + other.mass * other.z_cg) / total_mass
        self_inertia_tensor_elements = self.get_inertia_tensor_about_point(
            x=total_x_cg,
            y=total_y_cg,
            z=total_z_cg,
            return_tensor=False
        )
        other_inertia_tensor_elements = other.get_inertia_tensor_about_point(
            x=total_x_cg,
            y=total_y_cg,
            z=total_z_cg,
            return_tensor=False
        )

        total_inertia_tensor_elements = [
            I__ + J__
            for I__, J__ in zip(
                self_inertia_tensor_elements,
                other_inertia_tensor_elements
            )
        ]

        return MassProperties(
            mass=total_mass,
            x_cg=total_x_cg,
            y_cg=total_y_cg,
            z_cg=total_z_cg,
            Ixx=total_inertia_tensor_elements[0],
            Iyy=total_inertia_tensor_elements[1],
            Izz=total_inertia_tensor_elements[2],
            Ixy=total_inertia_tensor_elements[3],
            Iyz=total_inertia_tensor_elements[4],
            Ixz=total_inertia_tensor_elements[5],
        )

    def __sub__(self, other: "MassProperties") -> "MassProperties":
        """
        Subtracts one MassProperties object from another. (opposite of __add__() )
        """
        new_mass = self.mass - other.mass
        new_x_cg = (self.x_cg * self.mass - other.x_cg * other.mass) / (self.mass - other.mass)
        new_y_cg = (self.y_cg * self.mass - other.y_cg * other.mass) / (self.mass - other.mass)
        new_z_cg = (self.z_cg * self.mass - other.z_cg * other.mass) / (self.mass - other.mass)
        self_inertia_tensor_elements = self.get_inertia_tensor_about_point(
            x=new_x_cg,
            y=new_y_cg,
            z=new_z_cg,
            return_tensor=False
        )
        other_inertia_tensor_elements = other.get_inertia_tensor_about_point(
            x=new_x_cg,
            y=new_y_cg,
            z=new_z_cg,
            return_tensor=False
        )

        new_inertia_tensor_elements = [
            I__ - J__
            for I__, J__ in zip(
                self_inertia_tensor_elements,
                other_inertia_tensor_elements
            )
        ]

        return MassProperties(
            mass=new_mass,
            x_cg=new_x_cg,
            y_cg=new_y_cg,
            z_cg=new_z_cg,
            Ixx=new_inertia_tensor_elements[0],
            Iyy=new_inertia_tensor_elements[1],
            Izz=new_inertia_tensor_elements[2],
            Ixy=new_inertia_tensor_elements[3],
            Iyz=new_inertia_tensor_elements[4],
            Ixz=new_inertia_tensor_elements[5],
        )

    def __mul__(self, other: float) -> "MassProperties":
        """
        Returns a new MassProperties object that is equivalent to if you had summed together N (with `other`
        interpreted as N) identical MassProperties objects.
        """
        return MassProperties(
            mass=self.mass * other,
            x_cg=self.x_cg,
            y_cg=self.y_cg,
            z_cg=self.z_cg,
            Ixx=self.Ixx * other,
            Iyy=self.Iyy * other,
            Izz=self.Izz * other,
            Ixy=self.Ixy * other,
            Iyz=self.Iyz * other,
            Ixz=self.Ixz * other,
        )

    def __eq__(self, other: "MassProperties") -> bool:
        return all([
            self.__getattribute__(attribute) == other.__getattribute__(attribute)
            for attribute in [
                "mass",
                "x_cg",
                "y_cg",
                "z_cg",
                "Ixx",
                "Iyy",
                "Izz",
                "Ixy",
                "Iyz",
                "Ixz",
            ]
        ])

    def __ne__(self, other: "MassProperties") -> bool:
        return not self.__eq__(other)

    @property
    def xyz_cg(self):
        return self.x_cg, self.y_cg, self.z_cg

    @property
    def inertia_tensor(self):
        # Returns the inertia tensor about the component's centroid.
        return np.array(
            [[self.Ixx, self.Ixy, self.Ixz],
             [self.Ixy, self.Iyy, self.Iyz],
             [self.Ixz, self.Iyz, self.Izz]]
        )

    def inv_inertia_tensor(self):
        """
        Computes the inverse of the inertia tensor, in a slightly more efficient way than raw inversion by exploiting its known structure.

        If you are effectively using this inertia tensor to solve a linear system, you should use a linear algebra
        solve() method (ideally via Cholseky decomposition) instead, for best speed.
        """
        iIxx, iIyy, iIzz, iIxy, iIyz, iIxz = np.linalg.inv_symmetric_3x3(
            m11=self.Ixx,
            m22=self.Iyy,
            m33=self.Izz,
            m12=self.Ixy,
            m23=self.Iyz,
            m13=self.Ixz,
        )
        return np.array(
            [[iIxx, iIxy, iIxz],
             [iIxy, iIyy, iIyz],
             [iIxz, iIyz, iIzz]]
        )

    def get_inertia_tensor_about_point(self,
                                       x: float = 0,
                                       y: float = 0,
                                       z: float = 0,
                                       return_tensor=True,
                                       ):
        """
        Returns the inertia tensor about an arbitrary point.
        Using https://en.wikipedia.org/wiki/Parallel_axis_theorem#Tensor_generalization

        Args:
            x: x-position of the new point, in the same axes as this MassProperties instance is specified in.
            y: y-position of the new point, in the same axes as this MassProperties instance is specified in.
            z: z-position of the new point, in the same axes as this MassProperties instance is specified in.
            return_tensor: A switch for the desired return type.

        Returns:

            If `return_tensor` is True:
                Returns the new inertia tensor, as a 2D numpy ndarray.
            If `return_tensor` is False:
                Returns the components of the new inertia tensor, as a tuple.
                If J is the new inertia tensor, the tuple returned is:
                (Jxx, Jyy, Jzz, Jxy, Jyz, Jxz)

        """

        R = [x - self.x_cg, y - self.y_cg, z - self.z_cg]
        RdotR = np.dot(R, R, manual=True)

        Jxx = self.Ixx + self.mass * (RdotR - R[0] ** 2)
        Jyy = self.Iyy + self.mass * (RdotR - R[1] ** 2)
        Jzz = self.Izz + self.mass * (RdotR - R[2] ** 2)
        Jxy = self.Ixy - self.mass * R[0] * R[1]
        Jyz = self.Iyz - self.mass * R[1] * R[2]
        Jxz = self.Ixz - self.mass * R[2] * R[0]

        if return_tensor:
            return np.array([
                [Jxx, Jxy, Jxz],
                [Jxy, Jyy, Jyz],
                [Jxz, Jyz, Jzz],
            ])
        else:
            return Jxx, Jyy, Jzz, Jxy, Jyz, Jxz


if __name__ == '__main__':
    mp1 = MassProperties(
        mass=1
    )
    mp2 = MassProperties(
        mass=1,
        x_cg=1
    )
    mps = mp1 + mp2
    assert mps.x_cg == 0.5

    assert mp1 + mp2 - mp2 == mp1
