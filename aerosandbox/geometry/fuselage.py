from aerosandbox import AeroSandboxObject
from aerosandbox.geometry.common import *
from typing import List, Union, Tuple
from pathlib import Path
import aerosandbox.geometry.mesh_utilities as mesh_utils


class Fuselage(AeroSandboxObject):
    """
    Definition for a fuselage or other slender body (pod, etc.).
    For now, all fuselages are assumed to be circular and fairly closely aligned with the body x axis. (<10 deg or so) # TODO update if this changes
    """

    def __init__(self,
                 name: str = "Untitled Fuselage",  # It can help when debugging to give each fuselage a sensible name.
                 xyz_le: np.ndarray = np.array([0, 0, 0]),
                 xsecs: List['FuselageXSec'] = [],  # This should be a list of FuselageXSec objects.
                 symmetric: bool = False,  # Is the fuselage symmetric across the XZ plane?
                 ):
        """
        Initialize a new fuselage.
        Args:
            name: Name of the fuselage [optional]. It can help when debugging to give each fuselage a sensible name.
            xyz_le: xyz-coordinates of the datum point (typically the nose) of the fuselage.
            xsecs: A list of fuselage cross ("X") sections in the form of FuselageXSec objects.
            symmetric: Is the fuselage to be mirrored across the XZ plane (e.g. for wing-mounted pods).
            circumferential_panels:
        """
        self.name = name
        self.xyz_le = np.array(xyz_le)
        self.xsecs = xsecs
        self.symmetric = symmetric

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

    #
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
        for i in range(len(self.xsecs) - 1):
            xsec_a, xsec_b = self.xsecs[i], self.xsecs[i + 1]
            h = np.abs(xsec_b.xyz_c[0] - xsec_a.xyz_c[0])
            radius_a = xsec_a.radius
            radius_b = xsec_b.radius
            volume += np.pi * h / 3 * (
                    radius_a ** 2 + radius_a * radius_b + radius_b ** 2
            )
        return volume

    def write_avl_bfile(self,
                        filepath: Union[Path, str] = None,
                        include_name: bool = True,
                        ) -> str:
        """
        Writes an AVL-compatible BFILE corresponding to this fuselage to a filepath.

        For use with the AVL vortex-lattice-method aerodynamics analysis tool by Mark Drela at MIT.
        AVL is available here: https://web.mit.edu/drela/Public/web/avl/

        Args:
            filepath: filepath (including the filename and .avl extension) [string]
                If None, this function returns the would-be file contents as a string.

            include_name: Should the name be included in the .dat file? (This should be True for use with AVL.)

        Returns:

        """
        filepath = Path(filepath)

        contents = []

        if include_name:
            contents += [self.name]

        contents += [
                        f"{xyz_c[0]} {xyz_c[2] + r}"
                        for xyz_c, r in zip(
                [xsec.xyz_c for xsec in self.xsecs][::-1],
                [xsec.radius for xsec in self.xsecs][::-1]
            )
                    ] + [
                        f"{xyz_c[0]} {xyz_c[2] - r}"
                        for xyz_c, r in zip(
                [xsec.xyz_c for xsec in self.xsecs][1:],
                [xsec.radius for xsec in self.xsecs][1:]
            )
                    ]

        string = "\n".join(contents)

        if filepath is not None:
            with open(filepath, "w+") as f:
                f.write(string)

        return string

    def mesh_body(self,
                  method="quad",
                  chordwise_resolution: int = 6,
                  spanwise_resolution: int = 36,
                  ) -> Tuple[np.ndarray, np.ndarray]:

        theta = np.linspace(
            0,
            2 * np.pi,
            spanwise_resolution + 1,
        )[:-1]

        shape_nondim_coordinates = np.array([
            np.stack((
                np.sin(theta),
                np.cos(theta),
            )).T
            for xsec in self.xsecs
        ])

        x_nondim = shape_nondim_coordinates[:, :, 0].T
        y_nondim = shape_nondim_coordinates[:, :, 1].T

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

            origin = self._compute_xyz_le_of_FuselageXSec(i)
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

    def _compute_xyz_le_of_FuselageXSec(self, index: int):
        return self.xyz_le + self.xsecs[index].xyz_c

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
                 xyz_c: np.ndarray = np.array([0, 0, 0]),
                 radius: float = 0,
                 ):
        self.xyz_c = np.array(xyz_c)
        self.radius = radius

    def xsec_area(self):
        """
        Returns the FuselageXSec's cross-sectional (xsec) area.
        :return:
        """
        return np.pi * self.radius ** 2
