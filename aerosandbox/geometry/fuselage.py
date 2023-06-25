import numpy as np

from aerosandbox import AeroSandboxObject
from aerosandbox.geometry.common import *
from typing import List, Dict, Any, Union, Tuple, Optional, Callable
import copy


class Fuselage(AeroSandboxObject):
    """
    Definition for a Fuselage or other slender body (pod, fuel tank, etc.).

    Anatomy of a Fuselage:

        A fuselage consists chiefly of a collection of cross-sections, or "xsecs". A cross-section is a 2D "slice" of
        a fuselage. These can be accessed with `Fuselage.xsecs`, which gives a list of xsecs in the Fuselage. Each
        xsec is a FuselageXSec object, a class that is defined separately.

        You may also see references to fuselage "sections", which are different from cross-sections (xsecs)! Sections
        are the portions of the fuselage that are in between xsecs. In other words, a fuselage with N cross-sections
        (xsecs, FuselageXSec objects) will always have N-1 sections. Sections are never explicitly defined,
        since you can get all needed information by lofting from the adjacent cross-sections. For example,
        section 0 (the first one) is a loft between cross-sections 0 and 1.

        Fuselages are lofted linearly between cross-sections.

    """

    def __init__(self,
                 name: Optional[str] = "Untitled",
                 xsecs: List['FuselageXSec'] = None,
                 color: Optional[Union[str, Tuple[float]]] = None,
                 analysis_specific_options: Optional[Dict[type, Dict[str, Any]]] = None,
                 **kwargs,  # Only to allow for capturing of deprecated arguments, don't use this.
                 ):
        """
        Defines a new fuselage object.

        Args:

            name: Name of the fuselage [optional]. It can help when debugging to give each fuselage a sensible name.

            xsecs: A list of fuselage cross-sections ("xsecs") in the form of FuselageXSec objects.

            color: Determines what color to use for this component when drawing the airplane. Optional,
                and for visualization purposes only. If left as None, a default color will be chosen at the time of
                drawing (usually, black). Can be any color format recognized by MatPlotLib, namely:

                * A RGB or RGBA tuple of floats in the interval [0, 1], e.g., (0.1, 0.2, 0.5, 0.3)

                * Case-insensitive hex RGB or RGBA string, e.g., '#0f0f0f80'

                * String representation of float value in closed interval [0, 1] for grayscale values, e.g.,
                    '0.8' for light gray

                * Single character shorthand notation for basic colors, e.g., 'k' -> black, 'r' -> red

                See also: https://matplotlib.org/stable/tutorials/colors/colors.html

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
        self.color = color
        self.analysis_specific_options = analysis_specific_options

        ### Handle deprecated parameters
        if 'symmetric' in locals():
            raise DeprecationWarning(
                "The `symmetric` argument for Fuselage objects is deprecated. Make your fuselages separate instead!")

        if 'xyz_le' in locals():
            import warnings
            warnings.warn(
                "The `xyz_le` input for Fuselage is pending deprecation and will be removed in a future version. Use Fuselage().translate(xyz) instead.",
                stacklevel=2
            )
            self.xsecs = [
                xsec.translate(xyz_le)
                for xsec in self.xsecs
            ]

    def __repr__(self) -> str:
        n_xsecs = len(self.xsecs)
        return f"Fuselage '{self.name}' ({len(self.xsecs)} {'xsec' if n_xsecs == 1 else 'xsecs'})"

    def add_loft(self,
                 kind: str,
                 to_xsec: 'FuselageXSec',
                 from_xsec: 'FuselageXSec' = None,
                 n_points: int = 5,
                 spacing: Callable[[float, float, int], np.ndarray] = np.cosspace,
                 ) -> "Fuselage":
        raise NotImplementedError # Function under construction!
        ### Set defaults
        if from_xsec is None:
            if len(self.xsecs) == 0:
                from_xsec = FuselageXSec(
                    xyz_c=[0, 0, 0],
                    width=0,
                    height=0,
                    shape=2
                )
            else:
                from_xsec = self.xsecs[-1]

        ### Define a nondimensional coordinate
        t = spacing(0, 1, n_points)

        if kind == "linear":
            new_xsecs = [
                FuselageXSec(
                    xyz_c=from_xsec.xyz_c * (1 - ti) + to_xsec.xyz_c * ti,
                    width=from_xsec.width * (1 - ti) + to_xsec.width * ti,
                    height=from_xsec.height * (1 - ti) + to_xsec.height * ti,
                    shape=from_xsec.shape * (1 - ti) + to_xsec.shape * ti,
                    analysis_specific_options=from_xsec.analysis_specific_options,
                )
                for ti in t
            ]
        elif kind == "ellipsoid-nose":
            new_xsecs = [
                FuselageXSec(
                    xyz_c=from_xsec.xyz_c * (1 - ti) + to_xsec.xyz_c * ti,
                    width=from_xsec.width * (1 - ti) + to_xsec.width * ti,
                    height=from_xsec.height * (1 - ti) + to_xsec.height * ti,
                    shape=from_xsec.shape * (1 - ti) + to_xsec.shape * ti,
                    analysis_specific_options=from_xsec.analysis_specific_options,
                )
                for ti in t
            ]

        self.xsecs.extend(new_xsecs)

    def translate(self,
                  xyz: Union[np.ndarray, List[float]]
                  ) -> "Fuselage":
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

        :return:
        """
        area = 0

        perimeters = [xsec.xsec_perimeter() for xsec in self.xsecs]

        for i in range(len(self.xsecs) - 1):
            x_separation = self.xsecs[i + 1].xyz_c[0] - self.xsecs[i].xyz_c[0]
            area += (perimeters[i] + perimeters[i + 1]) / 2 * x_separation

        return area

    def area_projected(self,
                       type: str = "XY",
                       ) -> float:
        """
        Returns the area of the fuselage as projected onto one of the principal planes.

        Args:
            type: A string, which determines which principal plane to use for projection. One of:

                * "XY", in which case the projected area is onto the XY plane (i.e., top-down)

                * "XZ", in which case the projected area is onto the XZ plane (i.e., side-view)

        Returns: The projected area.
        """
        area = 0
        for i in range(len(self.xsecs) - 1):
            x_separation = self.xsecs[i + 1].xyz_c[0] - self.xsecs[i].xyz_c[0]

            if type == "XY":
                width_a = self.xsecs[i].width
                width_b = self.xsecs[i + 1].width
                area += (width_a + width_b) / 2 * x_separation
            elif type == "XZ":
                height_a = self.xsecs[i].height
                height_b = self.xsecs[i + 1].height
                area += (height_a + height_b) / 2 * x_separation
            else:
                raise ValueError("Bad value of `type`!")

        return area

    def area_base(self) -> float:
        """
        Returns the area of the base (i.e. "trailing edge") of the fuselage. Useful for certain types of drag
        calculation.

        Returns:
        """
        return self.xsecs[-1].xsec_area()

    def fineness_ratio(
            self,
            assumed_shape="cylinder",
    ) -> float:
        """
        Approximates the fineness ratio using the volume and length. The fineness ratio of a fuselage is defined as:

            FR = length / max_diameter

        Args:

            assumed_shape: A string, which determines the assumed shape of the fuselage for the approximation. One of:

                * "cylinder", in which case the fuselage is assumed to have a cylindrical shape.

                * "sears-haack", in which case the fuselage is assumed to have Sears-Haack fuselage shape.

        Returns: An approximate value of the fuselage's fineness ratio.

        """
        if assumed_shape == "cylinder":
            return np.sqrt(
                self.length() ** 3 / self.volume() * np.pi / 4
            )
        elif assumed_shape == "sears-haack":
            length = self.length()

            r_max = np.sqrt(
                self.volume() / length / (3 * np.pi ** 2 / 16)
            )
            return length / r_max

    def length(self) -> float:
        """
        Returns the total front-to-back length of the fuselage. Measured as the difference between the x-coordinates
        of the leading and trailing cross-sections.
        :return:
        """
        return np.fabs(self.xsecs[-1].xyz_c[0] - self.xsecs[0].xyz_c[0])

    def volume(self,
               _sectional: bool = False
               ) -> Union[float, List[float]]:
        """
        Computes the volume of the Fuselage.

        Args:

            _sectional: A boolean. If False, returns the total volume. If True, returns a list of volumes for each of
            the `n-1` lofted sections (between the `n` fuselage cross-sections in fuselage.xsec).

        Returns:

            The computed volume.
        """
        xsec_areas = [
            xsec.xsec_area()
            for xsec in self.xsecs
        ]

        separations = [
            xsec_b.xyz_c[0] - xsec_a.xyz_c[0]
            for xsec_a, xsec_b in zip(
                self.xsecs[:-1],
                self.xsecs[1:]
            )
        ]

        sectional_volumes = [
            separation / 3 * (area_a + area_b + (area_a * area_b + 1e-100) ** 0.5)
            for area_a, area_b, separation in zip(
                xsec_areas[1:],
                xsec_areas[:-1],
                separations
            )
        ]

        volume = sum(sectional_volumes)

        if _sectional:
            return sectional_volumes
        else:
            return volume

    def x_centroid_projected(self,
                             type: str = "XY",
                             ) -> float:
        """
        Returns the x_g coordinate of the centroid of the planform area.

        Args:
            type: A string, which determines which principal plane to use for projection. One of:

                * "XY", in which case the projected area is onto the XY plane (i.e., top-down)

                * "XZ", in which case the projected area is onto the XZ plane (i.e., side-view)

        Returns: The x_g coordinate of the centroid.
        """

        total_x_area_product = 0
        total_area = 0

        for xsec_a, xsec_b in zip(self.xsecs, self.xsecs[1:]):
            x_a = xsec_a.xyz_c[0]
            x_b = xsec_b.xyz_c[0]

            if type == "XY":
                r_a = xsec_a.width / 2
                r_b = xsec_b.width / 2
            elif type == "XZ":
                r_a = xsec_a.height / 2
                r_b = xsec_b.height / 2
            else:
                raise ValueError("Bad value of `type`!")

            dx = x_b - x_a

            x_c = x_a + (r_a + 2 * r_b) / (3 * (r_a + r_b)) * dx
            area = (r_a + r_b) / 2 * dx

            total_area += area
            total_x_area_product += x_c * area

        x_centroid = total_x_area_product / total_area
        return x_centroid

    def mesh_body(self,
                  method="quad",
                  tangential_resolution: int = 36,
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Meshes the fuselage as a solid (thickened) body.

        Uses the `(points, faces)` standard mesh format. For reference on this format, see the documentation in
        `aerosandbox.geometry.mesh_utilities`.

        Args:

            method: A string, which determines whether to mesh the fuselage as a series of quadrilaterals or triangles.

                * "quad" meshes the fuselage as a series of quadrilaterals.

                * "tri" meshes the fuselage as a series of triangles.

            tangential_resolution: An integer, which determines the number of points to use to mesh each cross-section.

        Returns: Standard unstructured mesh format: A tuple of`points` and `faces`, where:

            * `points` is a `n x 3` array of points, where `n` is the number of points in the mesh.

            * `faces` is a `m x 3` array of faces if `method` is "tri", or a `m x 4` array of faces if `method` is "quad".

                * Each row of `faces` is a list of indices into `points`, which specifies a face.

        """

        t = np.linspace(0, 2 * np.pi, tangential_resolution + 1)[:-1]

        points = np.concatenate([
            np.stack(
                xsec.get_3D_coordinates(theta=t),
                axis=1
            )
            for xsec in self.xsecs
        ],
            axis=0
        )

        faces = []

        num_i = len(self.xsecs)
        num_j = len(t)

        def index_of(iloc, jloc):
            return iloc * num_j + (jloc % num_j)

        def add_face(*indices):
            entry = list(indices)
            if method == "quad":
                faces.append(entry)
            elif method == "tri":
                faces.append([entry[0], entry[1], entry[3]])
                faces.append([entry[1], entry[2], entry[3]])

        for i in range(num_i - 1):
            for j in range(num_j):
                add_face(
                    index_of(i, j),
                    index_of(i, j + 1),
                    index_of(i + 1, j + 1),
                    index_of(i + 1, j),
                )

        faces = np.array(faces)

        return points, faces

    def mesh_line(self,
                  y_nondim: Union[float, List[float]] = 0.,
                  z_nondim: Union[float, List[float]] = 0.,
                  ) -> List[np.ndarray]:
        """
        Returns points along a line that goes through each of the FuselageXSec objects in this Fuselage.

        Args:

            y_nondim: The nondimensional (width-normalized) y-coordinate that the line should go through. Can either
            be a single value used at all cross-sections, or can be an iterable of values to be used at the
            respective cross-sections.

            z_nondim: The nondimensional (height-normalized) z-coordinate that the line should go through. Can either
            be a single value used at all cross-sections, or can be an iterable of values to be used at the
            respective cross-sections.

        Returns: A list of points, where each point is a 3-element array of the form `[x, y, z]`. Goes from the nose
        to the tail.

        """

        points_on_line: List[np.ndarray] = []

        try:
            if len(y_nondim) != len(self.xsecs):
                raise ValueError(
                    f"If `y_nondim` is an iterable, it should be the same length as `Fuselage.xsecs` ({len(self.xsecs)})."
                )
        except TypeError:
            pass

        try:
            if len(z_nondim) != len(self.xsecs):
                raise ValueError(
                    f"If `z_nondim` is an iterable, it should be the same length as `Fuselage.xsecs` ({len(self.xsecs)})."
                )
        except TypeError:
            pass

        for i, xsec in enumerate(self.xsecs):

            origin = xsec.xyz_c
            xg_local, yg_local, zg_local = xsec.compute_frame()

            try:
                xsec_y_nondim = y_nondim[i]
            except (TypeError, IndexError):
                xsec_y_nondim = y_nondim

            try:
                xsec_z_nondim = z_nondim[i]
            except (TypeError, IndexError):
                xsec_z_nondim = z_nondim

            xsec_point = origin + (
                    xsec_y_nondim * (xsec.width / 2) * yg_local +
                    xsec_z_nondim * (xsec.height / 2) * zg_local
            )
            points_on_line.append(xsec_point)

        return points_on_line

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

    def draw_wireframe(self, *args, **kwargs):
        """
        An alias to the more general Airplane.draw_wireframe() method. See there for documentation.

        Args:
            *args: Arguments to pass through to Airplane.draw_wireframe()
            **kwargs: Keyword arguments to pass through to Airplane.draw_wireframe()

        Returns: Same return as Airplane.draw_wireframe()

        """
        from aerosandbox.geometry.airplane import Airplane
        return Airplane(fuselages=[self]).draw_wireframe(*args, **kwargs)

    def draw_three_view(self, *args, **kwargs):
        """
        An alias to the more general Airplane.draw_three_view() method. See there for documentation.

        Args:
            *args: Arguments to pass through to Airplane.draw_three_view()
            **kwargs: Keyword arguments to pass through to Airplane.draw_three_view()

        Returns: Same return as Airplane.draw_three_view()

        """
        from aerosandbox.geometry.airplane import Airplane
        return Airplane(fuselages=[self]).draw_three_view(*args, **kwargs)

    def subdivide_sections(self,
                           ratio: int,
                           spacing_function: Callable[[float, float, float], np.ndarray] = np.linspace
                           ) -> "Fuselage":
        """
        Generates a new Fuselage that subdivides the existing sections of this Fuselage into several smaller ones. Splits
        each section into N=`ratio` smaller subsections by inserting new cross-sections (xsecs) as needed.

        This can allow for finer aerodynamic resolution of sectional properties in certain analyses.

        Args:

            ratio: The number of new sections to split each old section into.

            spacing_function: A function that takes in three arguments: the start, end, and number of points to generate.

                The default is `np.linspace`, which generates a linearly-spaced array of points.

                Other options include `np.cosspace`, which generates a cosine-spaced array of points.

        Returns: A new Fuselage object with subdivided sections.

        """
        if not (ratio >= 2 and isinstance(ratio, int)):
            raise ValueError("`ratio` must be an integer greater than or equal to 2.")

        new_xsecs = []
        length_fractions_along_section = spacing_function(0, 1, ratio + 1)[:-1]

        for xsec_a, xsec_b in zip(self.xsecs[:-1], self.xsecs[1:]):
            for s in length_fractions_along_section:
                a_weight = 1 - s
                b_weight = s

                new_xsecs.append(
                    FuselageXSec(
                        xyz_c=xsec_a.xyz_c * a_weight + xsec_b.xyz_c * b_weight,
                        width=xsec_a.width * a_weight + xsec_b.width * b_weight,
                        height=xsec_a.height * a_weight + xsec_b.height * b_weight,
                        shape=xsec_a.shape * a_weight + xsec_b.shape * b_weight,
                        analysis_specific_options=xsec_a.analysis_specific_options,
                    )
                )

        new_xsecs.append(self.xsecs[-1])

        return Fuselage(
            name=self.name,
            xsecs=new_xsecs,
            analysis_specific_options=self.analysis_specific_options
        )

    def _compute_frame_of_FuselageXSec(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the local frame of a FuselageXSec, given the index of the FuselageXSec in the Fuselage.xsecs list.

        Args:
            index: The index of the FuselageXSec in the Fuselage.xsecs list.

        Returns:  A tuple:
            xg_local: The x-axis of the local coordinate frame, in aircraft geometry axes.
            yg_local: The y-axis of the local coordinate frame, in aircraft geometry axes.
            zg_local: The z-axis of the local coordinate frame, in aircraft geometry axes.
        """
        import warnings
        warnings.warn(
            "Fuselage._compute_frame_of_FuselageXSec() is deprecated. "
            "Use FuselageXSec.compute_frame() instead.",
            DeprecationWarning
        )

        return self.xsecs[index].compute_frame()


class FuselageXSec(AeroSandboxObject):
    """
    Definition for a fuselage cross-section ("X-section").
    """

    def __init__(self,
                 xyz_c: Union[np.ndarray, List[float]] = None,
                 xyz_normal: Union[np.ndarray, List[float]] = None,
                 radius: float = None,
                 width: float = None,
                 height: float = None,
                 shape: float = 2.,
                 analysis_specific_options: Optional[Dict[type, Dict[str, Any]]] = None,
                 ):
        """
        Defines a new Fuselage cross-section.

        Fuselage cross-sections are essentially a sketch on a 2D plane.

            * This plane is defined by a center point (`xyz_c`) and a normal vector (`xyz_normal`).

            * The cross-section is a superellipse shape, which is a generalization of a circle and a square.

                It is mathematically defined by three parameters, using `y` and `z` as the two axes:

                    abs(y / width) ^ shape + abs(z / height) ^ shape = 1

                See also: https://en.wikipedia.org/wiki/Superellipse

                There are some notable special cases:

                    * A circle is a special case of a superellipse, where `shape = 2`.

                    * A square is a special case of a superellipse, where `shape = Inf` (in practice, set this to some
                    high value like 1000).

                    * A diamond is a special case of a superellipse, where `shape = 1`.

        Must specify either `radius` or both `width` and `height`. Cannot specify both.

        Args:

            xyz_c: An array-like that represents the xyz-coordinates of the center of this fuselage cross-section,
            in geometry axes.

            xyz_normal: An array-like that represents the xyz-coordinates of the normal vector of this fuselage
            cross-section, in geometry axes.

            radius: Radius of the fuselage cross-section.

            width: Width of the fuselage cross-section.

            height: Height of the fuselage cross-section.

            shape: A parameter that determines what shape the cross-section is. Should be in the range 1 < shape < infinity.

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
        if xyz_c is None:
            xyz_c = np.array([0., 0., 0.])
        if xyz_normal is None:
            xyz_normal = np.array([1., 0., 0.])  # points backwards
        if analysis_specific_options is None:
            analysis_specific_options = {}

        ### Set width and height
        radius_specified = (radius is not None)
        width_height_specified = [
            (width is not None),
            (height is not None)
        ]

        if radius_specified:
            if any(width_height_specified):
                raise ValueError(
                    "Cannot specify both `radius` and (`width`, `height`) parameters - must be one or the other."
                )

            self.width = 2 * radius
            self.height = 2 * radius

        else:
            if not all(width_height_specified):
                raise ValueError(
                    "Must specify either `radius` or both (`width`, `height`) parameters."
                )
            self.width = width
            self.height = height

        ### Initialize
        self.xyz_c = np.array(xyz_c)
        self.xyz_normal = np.array(xyz_normal)
        self.shape = shape
        self.analysis_specific_options = analysis_specific_options

    def __repr__(self) -> str:
        return f"FuselageXSec (xyz_c: {self.xyz_c}, width: {self.width}, height: {self.height}, shape: {self.shape})"

    def xsec_area(self):
        """
        Computes the FuselageXSec's cross-sectional (xsec) area.

        The computation method is a closed-form approximation for the area of a superellipse. The exact equation for
        the area of a superellipse with shape parameter `s` is:

            area = width * height * (gamma(1 + 1/n))^2 / gamma(1 + 2/n)

        where gamma() is the gamma function. The gamma function is (relatively) computationally expensive to evaluate
        and differentiate, so we replace this area calculation with a closed-form approximation (with essentially no
        loss in accuracy):

            area = width * height / (s^-1.8717618013591173 + 1)

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
        area = self.width * self.height / (self.shape ** -1.8717618013591173 + 1)

        return area

    def xsec_perimeter(self):
        """
        Computes the FuselageXSec's perimeter. ("Circumference" in the case of a circular cross-section.)

        The computation method is a closed-form approximation for the perimeter of a superellipse. The exact equation
        for the perimeter of a superellipse is quite long and is not repeated here for brevity; a Google search will
        bring it up. More importantly, this exact equation can only be represented as an infinite sum - not
        particularly useful for fast computation.

        We replace this exact equation with the following closed-form approximation obtained from symbolic regression:

            Imagine a superellipse centered on the origin of a 2D plane. Now, imagine that the superellipse is
            stretched such that the first quadrant (e.g., x>0, y>0) goes from (1, 0) to (0, h). Assume it has shape
            parameter s (where, as a reminder, s=1 is a diamond, s=2 is a circle, s=Inf is a square).

            Then, the perimeter of that single quadrant is:

            h + (((((s-0.88487077) * h + 0.2588574 / h) ^ exp(s / -0.90069205)) + h) + 0.09919785) ^ (-1.4812293 / s)

            See `AeroSandbox/studies/SuperellipseProperties` for details about how this was obtained.

        We can extrapolate from here to the general case of a superellipse, as shown in the code below.

        This approximation has the following properties:

            * For the s=1 case (diamond), the error is +0.2%.

            * For the s=2 case (circle), the error is -0.1%.

            * In the s -> infinity limit (square), the error is +0.1%.

        Returns:

        """
        try:
            if self.width == 0:
                return 2 * self.height
            elif self.height == 0:
                return 2 * self.width
        except RuntimeError:  # Will error if width and height are optimization variables, as truthiness is indeterminate
            pass

        s = self.shape
        h = np.maximum(
            (self.width + 1e-16) / (self.height + 1e-16),
            (self.height + 1e-16) / (self.width + 1e-16)
        )
        nondim_quadrant_perimeter = (
                h + (((((s - 0.88487077) * h + 0.2588574 / h) ** np.exp(s / -0.90069205)) + h) + 0.09919785) ** (
                -1.4812293 / s)
        )
        perimeter = 2 * nondim_quadrant_perimeter * np.minimum(self.width, self.height)

        return np.where(
            self.width == 0,
            2 * self.height,
            np.where(
                self.height == 0,
                2 * self.width,
                perimeter
            )
        )

    def compute_frame(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the local coordinate frame of the FuselageXSec, in aircraft geometry axes.

        xg_local is aligned with the FuselageXSec's normal vector.

        zg_local is roughly aligned with the z-axis of the aircraft geometry axes, but projected onto the FuselageXSec's plane.

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

    def get_3D_coordinates(self,
                           theta: Union[float, np.ndarray] = None
                           ) -> Tuple[Union[float, np.ndarray]]:
        """
        Samples points from the perimeter of this FuselageXSec.

        Args:

            theta: Coordinate in the tangential-ish direction to sample points at. Given in the 2D FuselageXSec
            coordinate system, where:

                * y_2D points along the (global) y_g
                * z_2D points along the (global) z_g

                In other words, a value of:

                    * theta=0     -> samples points from the right side of the FuselageXSec
                    * theta=pi/2  -> samples points from the top of the FuselageXSec
                    * theta=pi    -> samples points from the left side of the FuselageXSec
                    * theta=3pi/2 -> samples points from the bottom of the FuselageXSec

        Returns: Points sampled from the perimeter of the FuselageXSec, as a [x, y, z] tuple.

            If theta is a float, then each of x, y, and z will be floats.

            If theta is an array, then x, y, and z will also be arrays of the same size.

        """
        ### Set defaults
        if theta is None:
            theta = np.linspace(
                0,
                2 * np.pi,
                60 + 1
            )[:-1]

        st = np.sin(np.mod(theta, 2 * np.pi))
        ct = np.cos(np.mod(theta, 2 * np.pi))

        y = (self.width / 2) * np.abs(ct) ** (2 / self.shape) * np.where(ct > 0, 1, -1)
        z = (self.height / 2) * np.abs(st) ** (2 / self.shape) * np.where(st > 0, 1, -1)

        xg_local, yg_local, zg_local = self.compute_frame()

        return (
            self.xyz_c[0] + y * yg_local[0] + z * zg_local[0],
            self.xyz_c[1] + y * yg_local[1] + z * zg_local[1],
            self.xyz_c[2] + y * yg_local[2] + z * zg_local[2],
        )

    def equivalent_radius(self,
                          preserve="area"
                          ) -> float:
        """
        Computes an equivalent radius for non-circular cross-sections. This may be necessary when doing analysis that
        uses axisymmetric assumptions.

        Can either hold area or perimeter fixed, depending on whether cross-sectional area or wetted area is more
        important.

        Args:

            preserve: One of:

                * "area": holds the cross-sectional area constant

                * "perimeter": holds the cross-sectional perimeter (i.e., the wetted area of the Fuselage) constant

        Returns: An equivalent radius value.

        """
        if preserve == "area":
            return (self.xsec_area() / np.pi + 1e-16) ** 0.5
        elif preserve == "perimeter":
            return (self.xsec_perimeter() / (2 * np.pi))
        else:
            raise ValueError("Bad value of `preserve`!")

    def translate(self,
                  xyz: Union[np.ndarray, List[float]]
                  ) -> "FuselageXSec":
        """
        Returns a copy of this FuselageXSec that has been translated by `xyz`.

        Args:
            xyz: The amount to translate the FuselageXSec. Given as a 3-element NumPy vector.

        Returns: A copy of this FuselageXSec, translated by `xyz`.

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
                width=0.5,
                height=0.2,
                shape=5
            ),
            FuselageXSec(
                xyz_c=[2, 0, 1],
                radius=0.2,
            )
        ]
    ).translate([0, 0, 2])
    fuse.draw()
