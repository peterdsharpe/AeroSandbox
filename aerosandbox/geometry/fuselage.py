from aerosandbox import AeroSandboxObject
from aerosandbox.geometry.common import *
from typing import List, Dict, Any, Union, Tuple, Optional
from pathlib import Path
import aerosandbox.geometry.mesh_utilities as mesh_utils
import copy


class Fuselage(AeroSandboxObject):
    """
    Definition for a Fuselage or other slender body (pod, fuel tank, etc.).

    Anatomy of a Fuselage:

        A fuselage consists chiefly of a collection of cross sections, or "xsecs". A cross section is a 2D "slice" of
        a fuselage. These can be accessed with `Fuselage.xsecs`, which gives a list of xsecs in the Fuselage. Each
        xsec is a FuselageXSec object, a class that is defined separately.

        You may also see references to fuselage "sections", which are different than cross sections (xsecs)! Sections
        are the portions of the fuselage that are in between xsecs. In other words, a fuselage with N cross sections
        (xsecs, FuselageXSec objects) will always have N-1 sections. Sections are never explicitly defined,
        since you can get all needed information by lofting from the adjacent cross sections. For example,
        section 0 (the first one) is a loft between cross sections 0 and 1.

        Fuselages are lofted linearly between cross sections.

    """

    def __init__(self,
                 name: Optional[str] = "Untitled",
                 xsecs: List['FuselageXSec'] = None,
                 analysis_specific_options: Optional[Dict[type, Dict[str, Any]]] = None,
                 symmetric: bool = False,  # Deprecated
                 xyz_le: np.ndarray = None,  # Deprecated
                 ):
        """
        Defines a new fuselage.

        Args:

            name: Name of the fuselage [optional]. It can help when debugging to give each fuselage a sensible name.

            xsecs: A list of fuselage cross sections ("xsecs") in the form of FuselageXSec objects.

            analysis_specific_options: Analysis-specific options are additional constants or modeling assumptions
            that should be passed on to specific analyses and associated with this specific geometry object.

                This should be a dictionary where:

                    * Keys are specific analysis types (typically a subclass of asb.ExplicitAnalysis or
                    asb.ImplicitAnalysis), but if you decide to write your own analysis and want to make this key
                    something else (like a string), that's totally fine - it's just a unique identifier for the
                    specific analysis you're running.

                    * Values are a dictionary of key:value pairs, where:

                        * Keys are strings.

                        * Values are some value you want to assign.

                This is more easily demonstrated / understood with an example:

                >>> analysis_specific_options = {
                >>>     asb.AeroBuildup: dict(
                >>>         include_wave_drag=True,
                >>>     )
                >>> }
        """
        ### Set defaults
        if xsecs is None:
            xsecs: List['FuselageXSec'] = []
        if analysis_specific_options is None:
            analysis_specific_options = {}

        ### Initialize
        self.name = name
        self.xsecs = xsecs
        self.analysis_specific_options = analysis_specific_options

        ### Handle deprecated parameters
        if symmetric:
            import warnings
            warnings.warn(
                "The `symmetric` argument for Fuselage objects will be deprecated soon. Make your fuselages separate instead!",
                stacklevel=2
            )

        self.symmetric = symmetric

        if xyz_le is not None:
            import warnings
            warnings.warn(
                "The `xyz_le` input for Fuselage is DEPRECATED and will be removed in a future version. Use Fuselage().translate(xyz) instead.",
                stacklevel=2
            )
            self.xsecs = [
                xsec.translate(xyz_le)
                for xsec in self.xsecs
            ]

    def __repr__(self) -> str:
        n_xsecs = len(self.xsecs)
        return f"Fuselage '{self.name}' ({len(self.xsecs)} {'xsec' if n_xsecs == 1 else 'xsecs'})"

    def translate(self,
                  xyz: np.ndarray
                  ):
        """
        Translates the entire Fuselage by a certain amount.

        Args:
            xyz:

        Returns: self

        """
        new_fuse = copy.copy(self)
        new_fuse.xsecs = [
            xsec.translate(xyz)
            for xsec in new_fuse.xsecs
        ]
        return new_fuse

    def area_wetted(self) -> float:
        """
        Returns the wetted area of the fuselage.

        If the Fuselage is symmetric (i.e. two symmetric wingtip pods),
        returns the combined wetted area of both pods.
        :return:
        """
        area = 0

        perimeters = [xsec.xsec_perimeter() for xsec in self.xsecs]

        for i in range(len(self.xsecs) - 1):
            x_separation = self.xsecs[i + 1].xyz_c[0] - self.xsecs[i].xyz_c[0]
            area += (perimeters[i] + perimeters[i + 1]) / 2 * x_separation

        if self.symmetric:
            area *= 2
        return area

    def area_projected(self) -> float:
        """
        Returns the area of the fuselage as projected onto the XY plane (top-down view).

        If the Fuselage is symmetric (i.e. two symmetric wingtip pods),
        returns the combined projected area of both pods.
        :return:
        """
        area = 0
        for i in range(len(self.xsecs) - 1):
            r_a = self.xsecs[i].radius
            r_b = self.xsecs[i + 1].radius
            x_separation = self.xsecs[i + 1].xyz_c[0] - self.xsecs[i].xyz_c[0]
            area += (r_a + r_b) * x_separation

        if self.symmetric:
            area *= 2
        return area

    def area_base(self) -> float:
        """
        Returns the area of the base (i.e. "trailing edge") of the fuselage. Useful for certain types of drag
        calculation.

        Returns:
        """
        return self.xsecs[-1].xsec_area()

    def fineness_ratio(self) -> float:
        """
        Approximates the fineness ratio using the volume and length.

        Formula derived from a generalization of the relation from a cylindrical fuselage.

        For a cylindrical fuselage, FR = l/d, where l is the length and d is the diameter.

        Returns:

        """
        return np.sqrt(
            self.length() ** 3 / self.volume() * np.pi / 4
        )

    def length(self) -> float:
        """
        Returns the total front-to-back length of the fuselage. Measured as the difference between the x-coordinates
        of the leading and trailing cross sections.
        :return:
        """
        return np.fabs(self.xsecs[-1].xyz_c[0] - self.xsecs[0].xyz_c[0])

    def volume(self) -> float:
        """
        Gives the volume of the Fuselage.

        Returns:
            Fuselage volume.
        """
        volume = 0

        xsec_areas = [xsec.xsec_area() for xsec in self.xsecs]

        for i in range(len(self.xsecs) - 1):
            x_separation = self.xsecs[i + 1].xyz_c[0] - self.xsecs[i].xyz_c[0]
            area_a = xsec_areas[i]
            area_b = xsec_areas[i + 1]

            volume += x_separation / 3 * (
                    area_a + area_b + (area_a * area_b + 1e-100) ** 0.5
            )
        return volume

    def x_centroid_projected(self) -> float:
        """
        Returns the x_g coordinate of the centroid of the planform area.
        """

        total_x_area_product = 0
        total_area = 0

        for xsec_a, xsec_b in zip(self.xsecs, self.xsecs[1:]):
            x_a = xsec_a.xyz_c[0]
            x_b = xsec_b.xyz_c[0]
            r_a = xsec_a.radius
            r_b = xsec_b.radius

            dx = x_b - x_a

            x_c = x_a + (r_a + 2 * r_b) / (3 * (r_a + r_b)) * dx
            area = (xsec_a.radius + xsec_b.radius) / 2 * dx

            total_area += area
            total_x_area_product += x_c * area

        x_centroid = total_x_area_product / total_area
        return x_centroid

    def mesh_body(self,
                  method="quad",
                  chordwise_resolution: int = 1,
                  spanwise_resolution: int = 36,
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Meshes the fuselage as a solid (thickened) body.

        Uses the `(points, faces)` standard mesh format. For reference on this format, see the documentation in
        `aerosandbox.geometry.mesh_utilities`.

        Args:
            method: Allows choice between "tri" and "quad" meshing.
            chordwise_resolution: Controls the chordwise resolution of the meshing.
            spanwise_resolution: Controls the spanwise resolution of the meshing.
            TODO add mesh_trailing_edge argument.

        Returns: (points, faces) in standard mesh format.

        """

        t = np.linspace(0, 2 * np.pi, spanwise_resolution + 1)[:-1]

        xsec_shape_parameters = np.array([
            xsec.shape
            for xsec in self.xsecs
        ])

        chordwise_strips = []
        for ti in t:
            st = np.sin(ti)
            ct = np.cos(ti)

            chordwise_strips.append(
                self.mesh_line(
                    x_nondim=np.abs(ct) ** (2 / xsec_shape_parameters) * np.where(ct > 0, 1, -1),
                    y_nondim=np.abs(st) ** (2 / xsec_shape_parameters) * np.where(st > 0, 1, -1),
                    chordwise_resolution=chordwise_resolution
                )
            )

        points = np.concatenate(chordwise_strips)

        faces = []

        num_i = len(chordwise_strips)
        num_j = chordwise_resolution * (len(self.xsecs) - 1)

        def index_of(iloc, jloc):
            return jloc + (iloc % spanwise_resolution) * (num_j + 1)

        def add_face(*indices):
            entry = list(indices)
            if method == "quad":
                faces.append(entry)
            elif method == "tri":
                faces.append([entry[0], entry[1], entry[3]])
                faces.append([entry[1], entry[2], entry[3]])

        for i in range(num_i):
            for j in range(num_j):
                add_face(
                    index_of(i, j),
                    index_of(i, j + 1),
                    index_of(i + 1, j + 1),
                    index_of(i + 1, j),
                )

        faces = np.array(faces)

        if self.symmetric:
            flipped_points = np.array(points)
            flipped_points[:, 1] = flipped_points[:, 1] * -1

            points, faces = mesh_utils.stack_meshes(
                (points, faces),
                (flipped_points, faces)
            )

        return points, faces

    def mesh_line(self,
                  x_nondim: Union[float, List[float]] = 0,
                  y_nondim: Union[float, List[float]] = 0,
                  chordwise_resolution: int = 1,
                  ) -> np.ndarray:
        xsec_points = []

        try:
            if len(x_nondim) != len(self.xsecs):
                raise ValueError(
                    "If x_nondim is going to be an iterable, it needs to be the same length as Fuselage.xsecs."
                )
        except TypeError:
            pass

        try:
            if len(y_nondim) != len(self.xsecs):
                raise ValueError(
                    "If y_nondim is going to be an iterable, it needs to be the same length as Fuselage.xsecs."
                )
        except TypeError:
            pass

        for i, xsec in enumerate(self.xsecs):

            origin = xsec.xyz_c
            xg_local, yg_local, zg_local = self._compute_frame_of_FuselageXSec(i)

            try:
                xsec_x_nondim = x_nondim[i]
            except (TypeError, IndexError):
                xsec_x_nondim = x_nondim

            try:
                xsec_y_nondim = y_nondim[i]
            except (TypeError, IndexError):
                xsec_y_nondim = y_nondim

            xsec_point = origin + (
                    xsec_x_nondim * xsec.radius * yg_local +
                    xsec_y_nondim * xsec.radius * zg_local
            )
            xsec_points.append(xsec_point)

        mesh_sections = []
        for i in range(len(xsec_points) - 1):
            mesh_section = np.linspace(
                xsec_points[i],
                xsec_points[i + 1],
                chordwise_resolution + 1
            )
            if not i == len(xsec_points) - 2:
                mesh_section = mesh_section[:-1]

            mesh_sections.append(mesh_section)

        mesh = np.concatenate(mesh_sections)

        return mesh

    def draw(self, *args, **kwargs):
        """
        An alias to the more general Airplane.draw() method. See there for documentation.

        Args:
            *args: Arguments to pass through to Airplane.draw()
            **kwargs: Keyword arguments to pass through to Airplane.draw()

        Returns: Same return as Airplane.draw()

        """
        from aerosandbox.geometry.airplane import Airplane
        return Airplane(fuselages=[self]).draw(*args, **kwargs)

    def _compute_frame_of_FuselageXSec(self, index: int):

        if index == len(self.xsecs) - 1:
            index = len(self.xsecs) - 2  # The last FuselageXSec has the same frame as the last section.

        xyz_c_a = self.xsecs[index].xyz_c
        xyz_c_b = self.xsecs[index + 1].xyz_c
        vector_between = xyz_c_b - xyz_c_a
        xg_local_norm = np.linalg.norm(vector_between)
        if xg_local_norm != 0:
            xg_local = vector_between / xg_local_norm
        else:
            xg_local = np.array([1, 0, 0])

        zg_local = np.array([0, 0, 1])  # TODO

        yg_local = np.cross(zg_local, xg_local)

        return xg_local, yg_local, zg_local


class FuselageXSec(AeroSandboxObject):
    """
    Definition for a fuselage cross section ("X-section").
    """

    def __init__(self,
                 xyz_c: Union[np.ndarray, List] = np.array([0, 0, 0]),
                 radius: float = 0,
                 shape: float = 2.,
                 analysis_specific_options: Optional[Dict[type, Dict[str, Any]]] = None,
                 ):
        """
        Defines a new fuselage cross section.

        Args:

            xyz_c: An array-like that represents the xyz-coordinates of the center of this fuselage cross section,
            in geometry axes.

            radius: Radius of the fuselage cross section.

            shape: A parameter that determines what shape the cross section is. Should be in the range 1 < shape < infinity.

                In short, here's how to interpret this value:

                    * shape=2 is a circle.

                    * shape=1 is a diamond shape.

                    * A high value of, say, 10, will get you a square-ish shape.

                To be more precise:

                    * If the `shape` parameter is `s`, then the corresponding shape is the same as a level-set of a L^s norm in R^2.

                    * Defined another way, if the `shape` parameter is `s`, then the shape is the solution to the equation:

                        * x^s + y^s = 1 in the first quadrant (x>0, y>0); then mirrored for all four quadrants.

            analysis_specific_options: Analysis-specific options are additional constants or modeling assumptions
            that should be passed on to specific analyses and associated with this specific geometry object.

                This should be a dictionary where:

                    * Keys are specific analysis types (typically a subclass of asb.ExplicitAnalysis or
                    asb.ImplicitAnalysis), but if you decide to write your own analysis and want to make this key
                    something else (like a string), that's totally fine - it's just a unique identifier for the
                    specific analysis you're running.

                    * Values are a dictionary of key:value pairs, where:

                        * Keys are strings.

                        * Values are some value you want to assign.

                This is more easily demonstrated / understood with an example:

                >>> analysis_specific_options = {
                >>>     asb.AeroBuildup: dict(
                >>>         include_wave_drag=True,
                >>>     )
                >>> }
        """
        ### Set defaults
        if analysis_specific_options is None:
            analysis_specific_options = {}

        ### Initialize
        self.xyz_c = np.array(xyz_c)
        self.radius = radius
        self.shape = shape
        self.analysis_specific_options = analysis_specific_options

    def __repr__(self) -> str:
        return f"FuselageXSec (xyz_c: {self.xyz_c}, radius: {self.radius}, shape: {self.shape})"

    def xsec_area(self):
        """
        Computes the FuselageXSec's cross-sectional (xsec) area.

        The computation method is a closed-form approximation for the area of a superellipse. The exact equation for
        the area of a superellipse with shape parameter `s` is:

            area = 4 * r^2 * (gamma(1 + 1/n))^2 / gamma(1 + 2/n)

        where gamma() is the gamma function. The gamma function is (relatively computationally expensive to evaluate,
        so we replace this area calculation with a closed-form approximation:

            area = 4 * r^2 / (s^-1.8717618013591173 + 1)

        This approximation has the following properties:

            * It is numerically exact for the case of s = 1 (a diamond)

            * It is numerically exact for the case of s = 2 (a circle)

            * It is correct in the asymptotic limit where s -> infinity (a square)

            * In the range of sensible s values (1 < s < infinity), its error is less than 0.6%.

            * It always produces a positive area for any physically-meaningful value of s (s > 0). In the range of s
            values where s is physically-meaningful but not in a sensible range (0 < s < 1), this equation will
            over-predict area.

        The value of the constant seen in this expression (1.872...) is given by log(4/pi - 1) / log(2), and it is
        chosen as such so that the expression is exactly correct in the s=2 (circle) case.

        Returns:

        """
        pi_effective = 4 / (self.shape ** -1.8717618013591173 + 1)
        area = pi_effective * self.radius ** 2

        return area

    def xsec_perimeter(self):
        """
        Computes the FuselageXSec's perimeter. ("Circumference" in the case of a circular cross section.)

        The computation method is a closed-form approximation for the perimeter of a superellipse. The exact equation
        for the perimeter of a superellipse is quite long and is not repeated here for brevity; a Google search will
        bring it up.

        We replace this exact equation with the following closed-form approximation obtained from symbolic regression:

            perimeter_per_quadrant = 2.0022144 - 2.2341106 / ( s^-2.2698476 / 1.218528 + (s + 0.50136507) * 1.9967787 )
            perimeter = perimeter_per_quadrant * 4 * radius

        This approximation has the following properties:

            * For the s=1 case (diamond), the error is +0.2%.

            * For the s=2 case (circle), the error is -0.1%.

            * In the s -> infinity limit (square), the error is +0.1%.

        Returns:

        """
        s = self.shape
        perimeter_per_quadrant = (
                (-2.2341106 / (((s ** -2.2698476) / 1.218528) + ((s + 0.50136507) * 1.9967787))) + 2.0022144
        )
        perimeter = perimeter_per_quadrant * 4 * self.radius

        return perimeter

    def translate(self,
                  xyz: np.ndarray
                  ) -> "FuselageXSec":
        """
        Returns a copy of this FuselageXSec that has been translated by `xyz`.

        Args:
            xyz: The amount to translate the FuselageXSec. Given as a 3-element NumPy vector.

        Returns: A new FuselageXSec object.

        """
        new_xsec = copy.copy(self)
        new_xsec.xyz_c = new_xsec.xyz_c + np.array(xyz)
        return new_xsec


if __name__ == '__main__':
    fuse = Fuselage(
        xsecs=[
            FuselageXSec(
                xyz_c=[0, 0, 1],
                radius=0,
            ),
            FuselageXSec(
                xyz_c=[1, 0, 1],
                radius=0.3,
                shape=10
            ),
            FuselageXSec(
                xyz_c=[2, 0, 1],
                radius=0.2,
            )
        ]
    ).translate([0, 0, 2])
    fuse.draw()
