import aerosandbox.numpy as np
from aerosandbox.common import AeroSandboxObject
from typing import Union, Any, List
from aerosandbox.tools.string_formatting import trim_string


class MassProperties(AeroSandboxObject):
    """
    Mass properties of a rigid 3D object.

    ## Notes on Inertia Tensor Definition

    This class uses the standard mathematical definition of the inertia tensor, which is different from the
    alternative definition used by some CAD and CAE applications (such as SolidWorks, NX, etc.). These differ by a
    sign flip in the products of inertia.

    Specifically, we define the inertia tensor using the standard convention:

        [ I11  I12  I13 ]   [ Ixx  Ixy  Ixz ]   [sum(m*(y^2+z^2))  -sum(m*x*y)      -sum(m*x*z)      ]
    I = [ I21  I22  I23 ] = [ Ixy  Iyy  Iyz ] = [-sum(m*x*y)       sum(m*(x^2+z^2)) -sum(m*y*z)      ]
        [ I31  I32  I33 ]   [ Ixz  Iyz  Izz ]   [-sum(m*x*z)       -sum(m*y*z)       sum(m*(x^2+y^2))]

    Whereas SolidWorks, NX, etc. define the inertia tensor as:

        [ I11  I12  I13 ]   [ Ixx -Ixy -Ixz ]   [sum(m*(y^2+z^2))  -sum(m*x*y)      -sum(m*x*z)      ]
    I = [ I21  I22  I23 ] = [-Ixy  Iyy -Iyz ] = [-sum(m*x*y)       sum(m*(x^2+z^2)) -sum(m*y*z)      ]
        [ I31  I32  I33 ]   [-Ixz -Iyz  Izz ]   [-sum(m*x*z)       -sum(m*y*z)       sum(m*(x^2+y^2))]

    See also: https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor

    """

    def __init__(self,
                 mass: Union[float, np.ndarray] = None,
                 x_cg: Union[float, np.ndarray] = 0.,
                 y_cg: Union[float, np.ndarray] = 0.,
                 z_cg: Union[float, np.ndarray] = 0.,
                 Ixx: Union[float, np.ndarray] = 0.,
                 Iyy: Union[float, np.ndarray] = 0.,
                 Izz: Union[float, np.ndarray] = 0.,
                 Ixy: Union[float, np.ndarray] = 0.,
                 Iyz: Union[float, np.ndarray] = 0.,
                 Ixz: Union[float, np.ndarray] = 0.,
                 ):
        """
        Initializes a new MassProperties object.

        Axes can be given in any convenient axes system, as long as mass properties are not combined across different
        axis systems. For aircraft design, the most common axis system is typically geometry axes (x-positive aft,
        y-positive out the right wingtip, z-positive upwards).

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

            Ixy: Respective component of the inertia tensor, as measured about the component's center of mass. 0 if
            this is symmetric about z.

            Iyz: Respective component of the inertia tensor, as measured about the component's center of mass. 0 if
            this is symmetric about x.

            Ixz: Respective component of the inertia tensor, as measured about the component's center of mass. 0 if
            this is symmetric about y.


        """
        if mass is None:
            import warnings
            warnings.warn(
                "Defining a MassProperties object with zero mass. This can cause problems (divide-by-zero) in dynamics calculations, if this is not intended.\nTo silence this warning, please explicitly set `mass=0` in the MassProperties constructor.",
                stacklevel=2
            )
            mass = 0

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

        def fmt(x: Union[float, Any], width=14) -> str:
            if isinstance(x, (float, int)):
                if x == 0:
                    x = "0"
                else:
                    return f"{x:.8g}".rjust(width)
            return trim_string(str(x).rjust(width), length=40)

        return "\n".join([
            "MassProperties instance:",
            f"                 Mass : {fmt(self.mass)}",
            f"    Center of Gravity : ({fmt(self.x_cg)}, {fmt(self.y_cg)}, {fmt(self.z_cg)})",
            f"       Inertia Tensor : ",
            f"            (about CG)  [{fmt(self.Ixx)}, {fmt(self.Ixy)}, {fmt(self.Ixz)}]",
            f"                        [{fmt(self.Ixy)}, {fmt(self.Iyy)}, {fmt(self.Iyz)}]",
            f"                        [{fmt(self.Ixz)}, {fmt(self.Iyz)}, {fmt(self.Izz)}]",
        ])

    def __getitem__(self, index) -> "MassProperties":
        """
        Indexes one item from each attribute of an MassProperties instance.
        Returns a new MassProperties instance.

        Args:
            index: The index that is being called; e.g.,:
                >>> first_mass_props = mass_props[0]

        Returns: A new MassProperties instance, where each attribute is subscripted at the given value, if possible.

        """
        l = len(self)
        if index >= l or index < -l:
            raise IndexError("Index is out of range!")

        def get_item_of_attribute(a):
            if np.isscalar(a):  # If NumPy says its a scalar, return it.
                return a
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

    def __len__(self):
        length = 1
        for v in [
            self.mass,
            self.x_cg,
            self.y_cg,
            self.z_cg,
            self.Ixx,
            self.Iyy,
            self.Izz,
            self.Ixy,
            self.Iyz,
            self.Ixz,
        ]:
            if np.length(v) == 1:
                try:
                    v[0]
                    length = 1
                except (TypeError, IndexError, KeyError) as e:
                    pass
            elif length == 0 or length == 1:
                length = np.length(v)
            elif length == np.length(v):
                pass
            else:
                raise ValueError("State variables are appear vectorized, but of different lengths!")
        return length

    def __array__(self, dtype="O"):
        """
        Allows NumPy array creation without infinite recursion in __len__ and __getitem__.
        """
        return np.fromiter([self], dtype=dtype).reshape(())

    def __neg__(self) -> "MassProperties":
        return -1 * self

    def __add__(self, other: "MassProperties") -> "MassProperties":
        """
        Combines one MassProperties object with another.
        """
        if not isinstance(other, MassProperties):
            raise TypeError("MassProperties objects can only be added to other MassProperties objects.")

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

    def __radd__(self, other: "MassProperties") -> "MassProperties":
        """
        Allows sum() to work with MassProperties objects.

        Basically, makes addition commutative.
        """
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self, other: "MassProperties") -> "MassProperties":
        """
        Subtracts one MassProperties object from another. (opposite of __add__() )
        """
        return self.__add__(-other)

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

    def __rmul__(self, other: float) -> "MassProperties":
        """
        Allows multiplication of a scalar by a MassProperties object. Makes multiplication commutative.
        """
        return self.__mul__(other)

    def __truediv__(self, other: float) -> "MassProperties":
        """
        Returns a new MassProperties object that is equivalent to if you had divided the mass of the current
        MassProperties object by a factor.
        """
        return self.__mul__(1 / other)

    def __eq__(self, other: "MassProperties") -> bool:
        """
        Returns True if all expected attributes of the two MassProperties objects are exactly equal.
        """
        if not isinstance(other, MassProperties):
            raise TypeError("MassProperties objects can only be compared to other MassProperties objects.")

        return all([
            getattr(self, attribute) == getattr(other, attribute)
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

    def allclose(self,
                 other: "MassProperties",
                 rtol=1e-5,
                 atol=1e-8,
                 equal_nan=False
                 ) -> bool:
        return all([
            np.allclose(
                getattr(self, attribute),
                getattr(other, attribute),
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan
            )
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
        solve() method (ideally via Cholesky decomposition) instead, for best speed.
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
                                       x: float = 0.,
                                       y: float = 0.,
                                       z: float = 0.,
                                       return_tensor: bool = True,
                                       ):
        """
        Returns the inertia tensor about an arbitrary point.
        Using https://en.wikipedia.org/wiki/Parallel_axis_theorem#Tensor_generalization

        Args:
            x: x-position of the new point, in the same axes as this MassProperties instance is specified in.

            y: y-position of the new point, in the same axes as this MassProperties instance is specified in.

            z: z-position of the new point, in the same axes as this MassProperties instance is specified in.

            return_tensor: A switch for the desired return type; see below for details. [boolean]

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

    def is_physically_possible(self) -> bool:
        """
        Checks whether it's possible for this MassProperties object to correspond to the mass properties of a real
        physical object.

        Assumes that all physically-possible objects have a positive mass (or density).

        Some special edge cases:

            - A MassProperties object with mass of 0 (i.e., null object) will return True. Note: this will return
            True even if the inertia tensor is not zero (which would basically be infinitesimal point masses at
            infinite distance).

            - A MassProperties object that is a point mass (i.e., inertia tensor is all zeros) will return True.

        Returns:
            True if the MassProperties object is physically possible, False otherwise.
        """

        ### This checks the basics
        impossible_conditions = [
            self.mass < 0,
            self.Ixx < 0,
            self.Iyy < 0,
            self.Izz < 0,
        ]

        eigs = np.linalg.eig(self.inertia_tensor)[0]

        # ## This checks that the inertia tensor is positive definite, which is a necessary but not sufficient
        # condition for an inertia tensor to be physically possible.
        impossible_conditions.extend([
            eigs[0] < 0,
            eigs[1] < 0,
            eigs[2] < 0,
        ])

        # ## This checks the triangle inequality, which is a necessary but not sufficient condition for an inertia
        # tensor to be physically possible.
        impossible_conditions.extend([
            eigs[0] + eigs[1] < eigs[2],
            eigs[0] + eigs[2] < eigs[1],
            eigs[1] + eigs[2] < eigs[0],
        ])

        return not any(impossible_conditions)

    def is_point_mass(self) -> bool:
        """
        Returns True if this MassProperties object corresponds to a point mass, False otherwise.
        """
        return np.allclose(self.inertia_tensor, 0)

    def generate_possible_set_of_point_masses(self,
                                              method="optimization",
                                              check_if_already_a_point_mass: bool = True,
                                              ) -> List["MassProperties"]:
        """
        Generates a set of point masses (represented as MassProperties objects with zero inertia tensors), that, when
        combined, would yield this MassProperties object.

        Note that there are an infinite number of possible sets of point masses that could yield this MassProperties
        object. This method returns one possible set of point masses, but there are many others.

        Example:
            >>> mp = MassProperties(mass=1, Ixx=1, Iyy=1, Izz=1, Ixy=0.1, Iyz=-0.1, Ixz=0.1)
            >>> point_masses = mp.generate_possible_set_of_point_masses()
            >>> mp.allclose(sum(point_masses))  # Asserts these are equal, within tolerance
            True

        Args:
            method: The method to use to generate the set of point masses. Currently, only "optimization" is supported.

        Returns:
            A list of MassProperties objects, each of which is a point mass (i.e., zero inertia tensor).
        """
        if check_if_already_a_point_mass:
            if self.is_point_mass():
                return [self]

        if method == "optimization":
            from aerosandbox.optimization import Opti

            opti = Opti()

            approximate_radius = (self.Ixx + self.Iyy + self.Izz) ** 0.5 / self.mass + 1e-16

            point_masses = [
                MassProperties(
                    mass=self.mass / 4,
                    x_cg=opti.variable(init_guess=self.x_cg - approximate_radius, scale=approximate_radius),
                    y_cg=opti.variable(init_guess=self.y_cg, scale=approximate_radius),
                    z_cg=opti.variable(init_guess=self.z_cg, scale=approximate_radius),
                ),
                MassProperties(
                    mass=self.mass / 4,
                    x_cg=opti.variable(init_guess=self.x_cg, scale=approximate_radius),
                    y_cg=opti.variable(init_guess=self.y_cg, scale=approximate_radius),
                    z_cg=opti.variable(init_guess=self.z_cg + approximate_radius, scale=approximate_radius),
                ),
                MassProperties(
                    mass=self.mass / 4,
                    x_cg=opti.variable(init_guess=self.x_cg, scale=approximate_radius),
                    y_cg=opti.variable(init_guess=self.y_cg, scale=approximate_radius),
                    z_cg=opti.variable(init_guess=self.z_cg - approximate_radius, scale=approximate_radius),
                ),
                MassProperties(
                    mass=self.mass / 4,
                    x_cg=opti.variable(init_guess=self.x_cg, scale=approximate_radius),
                    y_cg=opti.variable(init_guess=self.y_cg + approximate_radius, scale=approximate_radius),
                    z_cg=opti.variable(init_guess=self.z_cg, scale=approximate_radius),
                ),
            ]

            mass_props_reconstructed = sum(point_masses)

            # Add constraints
            opti.subject_to(mass_props_reconstructed.x_cg == self.x_cg)
            opti.subject_to(mass_props_reconstructed.y_cg == self.y_cg)
            opti.subject_to(mass_props_reconstructed.z_cg == self.z_cg)
            opti.subject_to(mass_props_reconstructed.Ixx == self.Ixx)
            opti.subject_to(mass_props_reconstructed.Iyy == self.Iyy)
            opti.subject_to(mass_props_reconstructed.Izz == self.Izz)
            opti.subject_to(mass_props_reconstructed.Ixy == self.Ixy)
            opti.subject_to(mass_props_reconstructed.Iyz == self.Iyz)
            opti.subject_to(mass_props_reconstructed.Ixz == self.Ixz)

            opti.subject_to(point_masses[0].y_cg == self.y_cg)
            opti.subject_to(point_masses[0].z_cg == self.z_cg)
            opti.subject_to(point_masses[1].y_cg == self.y_cg)

            opti.subject_to(point_masses[0].x_cg < point_masses[1].x_cg)

            return opti.solve(verbose=False)(point_masses)


        elif method == "barbell":
            raise NotImplementedError("Barbell method not yet implemented!")
            principle_inertias, principle_axes = np.linalg.eig(self.inertia_tensor)

        else:
            raise ValueError("Bad value of `method` argument!")

    def export_AVL_mass_file(self,
                             filename,
                             ) -> None:
        """
        Exports this MassProperties object to an AVL mass file.

        Note: AVL uses the SolidWorks convention for inertia tensors, which is different from the typical
        mathematical convention, and the convention used by this MassProperties class. In short, these differ by a
        sign flip in the products of inertia. More details available in the MassProperties docstring. See also:
        https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor

        Args:
            filename: The filename to export to.

        Returns: None

        """
        lines = [
            "Lunit = 1.0 m",
            "Munit = 1.0 kg",
            "Tunit = 1.0 s",
            "",
            "g     = 9.81",
            "rho   = 1.225",
            "",
        ]

        def fmt(x: float) -> str:
            return f"{x:.8g}".ljust(14)

        lines.extend([
            " ".join([
                s.ljust(14) for s in [
                    "#  mass",
                    "x_cg",
                    "y_cg",
                    "z_cg",
                    "Ixx",
                    "Iyy",
                    "Izz",
                    "Ixy",
                    "Ixz",
                    "Iyz",
                ]
            ]),
            " ".join([
                fmt(x) for x in [
                    self.mass,
                    self.x_cg,
                    self.y_cg,
                    self.z_cg,
                    self.Ixx,
                    self.Iyy,
                    self.Izz,
                    -self.Ixy,
                    -self.Ixz,
                    -self.Iyz,
                ]
            ])
        ])

        with open(filename, "w+") as f:
            f.write("\n".join(lines))


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

    r = lambda: np.random.randn()

    valid = False
    while not valid:
        mass_props = MassProperties(
            mass=r(),
            x_cg=r(), y_cg=r(), z_cg=r(),
            Ixx=r(), Iyy=r(), Izz=r(),
            Ixy=r(), Iyz=r(), Ixz=r(),
        )
        valid = mass_props.is_physically_possible()  # adds a bunch of checks
