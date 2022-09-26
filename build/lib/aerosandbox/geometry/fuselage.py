from aerosandbox import AeroSandboxObject
from aerosandbox.geometry.common import *
from typing import List, Dict, Any, Union, Tuple, Optional
from pathlib import Path
import aerosandbox.geometry.mesh_utilities as mesh_utils
import copy


class Fuselage(AeroSandboxObject):
    """
    Definition for a fuselage or other slender body (pod, etc.).

    For now, all fuselages are assumed to be circular and fairly closely aligned with the body x axis. (<10 deg or
    so) # TODO update if this changes

    """

    def __init__(self,
                 name: str = "Untitled",
                 xsecs: List['FuselageXSec'] = None,
                 analysis_specific_options: Optional[Dict[type, Dict[str, Any]]] = None,
                 symmetric: bool = False,  # Deprecated
                 xyz_le: np.ndarray = None,  # Deprecated
                 ):
        """
        Defines a new fuselage.

        Args:

            name: Name of the fuselage [optional]. It can help when debugging to give each fuselage a sensible name.

            xsecs: A list of fuselage cross ("X") sections in the form of FuselageXSec objects.

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
        for i in range(len(self.xsecs) - 1):
            this_radius = self.xsecs[i].radius
            next_radius = self.xsecs[i + 1].radius
            x_separation = self.xsecs[i + 1].xyz_c[0] - self.xsecs[i].xyz_c[0]
            area += np.pi * (this_radius + next_radius) * np.sqrt(
                (this_radius - next_radius) ** 2 + x_separation ** 2)
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
            this_radius = self.xsecs[i].radius
            next_radius = self.xsecs[i + 1].radius
            x_separation = self.xsecs[i + 1].xyz_c[0] - self.xsecs[i].xyz_c[0]
            area += (this_radius + next_radius) * x_separation
        if self.symmetric:
            area *= 2
        return area

    def area_base(self) -> float:
        """
        Returns the area of the base (i.e. "trailing edge") of the fuselage. Useful for certain types of drag
        calculation.

        Returns:
        """
        return np.pi * self.xsecs[-1].radius ** 2

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
        for xsec_a, xsec_b in zip(self.xsecs, self.xsecs[1:]):
            h = np.abs(xsec_b.xyz_c[0] - xsec_a.xyz_c[0])
            r_a = xsec_a.radius
            r_b = xsec_b.radius
            volume += np.pi * h / 3 * (
                    r_a ** 2 + r_a * r_b + r_b ** 2
            )
        return volume

    def x_centroid_projected(self) -> float:
        """
        Returns the x_g coordinate of the centroid of the planform area.
        """

        total_x_area_product = 0
        total_area = 0
        for xsec_a, xsec_b in zip(self.xsecs, self.xsecs[1:]):
            x = (xsec_a.xyz_c[0] + xsec_b.xyz_c[0]) / 2
            area = (xsec_a.radius + xsec_b.radius) / 2 * np.abs(xsec_b.xyz_c[0] - xsec_a.xyz_c[0])
            total_area += area
            total_x_area_product += x * area
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

        theta = np.linspace(
            0,
            2 * np.pi,
            spanwise_resolution + 1,
        )[:-1]

        x_nondim = np.sin(theta)
        y_nondim = np.cos(theta)

        chordwise_strips = []
        for x_n, y_n in zip(x_nondim, y_nondim):
            chordwise_strips.append(
                self.mesh_line(
                    x_nondim=x_n,
                    y_nondim=y_n,
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
                 analysis_specific_options: Optional[Dict[type, Dict[str, Any]]] = None,
                 ):
        """
        Defines a new fuselage cross section.

        Args:

            xyz_c: An array-like that represents the xyz-coordinates of the center of this fuselage cross section,
            in geometry axes.

            radius: Radius of the fuselage cross section.

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
        self.analysis_specific_options = analysis_specific_options

    def __repr__(self) -> str:
        return f"FuselageXSec (xyz_c: {self.xyz_c}, radius: {self.radius})"

    def xsec_area(self):
        """
        Returns the FuselageXSec's cross-sectional (xsec) area.
        :return:
        """
        return np.pi * self.radius ** 2

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
        xyz_le=[0, 0, 2],
        xsecs=[
            FuselageXSec(
                xyz_c=[0, 0, 1],
                radius=0,
            ),
            FuselageXSec(
                xyz_c=[1, 0, 1],
                radius=0.3,
            ),
            FuselageXSec(
                xyz_c=[2, 0, 1],
                radius=0.2,
            )
        ]
    )
    fuse.draw()
