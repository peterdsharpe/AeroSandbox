from aerosandbox import AeroSandboxObject
from aerosandbox.geometry.common import *
from typing import List, Dict, Any, Tuple, Union
from aerosandbox.geometry.airfoil import Airfoil
from numpy import pi
import aerosandbox.numpy as np
import aerosandbox.geometry.mesh_utilities as mesh_utils
import copy


class Wing(AeroSandboxObject):
    """
    Definition for a Wing.

    Anatomy of a Wing:

        A wing consists chiefly of a collection of cross sections, or "xsecs". These can be accessed with `Wing.xsecs`,
        which gives a list of xsecs in the Wing. Each xsec is a WingXSec object, a class that is defined separately.

        You may also see references to wing "sections", which are different than cross sections (xsecs)! Sections are
        the portions of the wing that are in between xsecs. In other words, a wing with N cross sections (xsecs,
        WingXSec objects) will always have N-1 sections. Sections are never explicitly defined, since you can get all
        needed information by lofting from the adjacent cross sections. For example, section 0 is a loft between
        cross sections 0 and 1.

        Wings are lofted linearly between cross sections.

    If the wing is symmetric across the XZ plane, just define the right half and supply "symmetric = True" in the
    constructor.

    If the wing is not symmetric across the XZ plane (e.g., a vertical stabilizer), just define the wing.
    """

    def __init__(self,
                 name: str = "Untitled",
                 xsecs: List['WingXSec'] = None,
                 symmetric: bool = False,
                 xyz_le: np.ndarray = None,  # Note: deprecated.
                 analysis_specific_options: Dict[type, Dict[str, Any]] = None
                 ):
        """
        Defines a new wing.

        Args:

            name: Name of the wing [optional]. It can help when debugging to give each wing a sensible name.

            xsecs: A list of wing cross ("X") sections in the form of WingXSec objects.

            symmetric: Is the wing symmetric across the XZ plane?

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
            xsecs: List['WingXSec'] = []
        if analysis_specific_options is None:
            analysis_specific_options = {}

        ### Initialize
        self.name = name
        self.xsecs = xsecs
        self.symmetric = symmetric
        self.analysis_specific_options = analysis_specific_options

        ### Handle deprecated parameters
        if xyz_le is not None:
            import warnings
            warnings.warn(
                "The `xyz_le` input for Wing is DEPRECATED and will be removed in a future version. Use Wing().translate(xyz) instead.",
                stacklevel=2
            )
            self.xsecs = [
                xsec.translate(xyz_le)
                for xsec in self.xsecs
            ]

    def __repr__(self) -> str:
        n_xsecs = len(self.xsecs)
        symmetry_description = "symmetric" if self.symmetric else "asymmetric"
        return f"Wing '{self.name}' ({len(self.xsecs)} {'xsec' if n_xsecs == 1 else 'xsecs'}, {symmetry_description})"

    def translate(self,
                  xyz: Union[np.ndarray, List]
                  ):
        """
        Translates the entire Wing by a certain amount.

        Args:
            xyz:

        Returns: self

        """
        new_wing = copy.copy(self)
        new_wing.xsecs = [
            xsec.translate(xyz)
            for xsec in new_wing.xsecs
        ]
        return new_wing

    def span(self,
             type: str = "wetted",
             _sectional: bool = False,
             ) -> float:
        """
        Returns the span, with options for various ways of measuring this.
         * wetted: Adds up YZ-distances of each section piece by piece
         * y: Adds up the Y-distances of each section piece by piece
         * z: Adds up the Z-distances of each section piece by piece
         * y-full: Y-distance between the XZ plane and the tip of the wing. (Can't be used with _sectional).
        If symmetric, this is doubled left/right to obtain the full span.

        Args:
            type: One of the above options, as a string.
            _sectional: A boolean. If False, returns the total span. If True, returns a list of spans for each of the
                `n-1` lofted sections (between the `n` wing cross sections in wing.xsec).
        """
        if type == "y-full":
            if _sectional:
                raise ValueError("Cannot use `_sectional` with the parameter type as `y-full`!")
            return self._compute_xyz_of_WingXSec(
                -1,
                x_nondim=0.25,
                y_nondim=0.25
            )[1]

        sectional_spans = []

        i_range = range(len(self.xsecs))

        quarter_chord_vectors = [
            self._compute_xyz_of_WingXSec(
                i,
                x_nondim=0.25,
                y_nondim=0,
            )
            for i in i_range
        ]

        for inner_i, outer_i in zip(i_range, i_range[1:]):
            quarter_chord_vector = (
                    quarter_chord_vectors[outer_i] -
                    quarter_chord_vectors[inner_i]
            )

            if type == "wetted":
                section_span = (
                                       quarter_chord_vector[1] ** 2 +
                                       quarter_chord_vector[2] ** 2
                               ) ** 0.5
            elif type == "y":
                section_span = (
                    np.fabs(quarter_chord_vector[1])
                )
            elif type == "z":
                section_span = (
                    np.fabs(quarter_chord_vector[2])
                )
            else:
                raise ValueError("Bad value of 'type'!")

            sectional_spans.append(section_span)

        span = sum(sectional_spans)

        if self.symmetric:
            span *= 2

        if _sectional:
            return sectional_spans
        else:
            return span

    def area(self,
             type: str = "wetted",
             _sectional: bool = False,
             ) -> float:
        """
        Returns the area, with options for various ways of measuring this.
         * wetted: wetted area
         * projected: area projected onto the XY plane (top-down view)
        If symmetric, this is doubled left/right to obtain the full wing area.

        Args:
            type: One of the above options, as a string.
            _sectional: A boolean. If False, returns the total area. If True, returns a list of areas for each of the
                `n-1` lofted sections (between the `n` wing cross sections in wing.xsec).

        """
        area = 0
        chords = [xsec.chord for xsec in self.xsecs]

        if type == "wetted":
            sectional_spans = self.span(_sectional=True)
        elif type == "projected":
            sectional_spans = self.span(type="y", _sectional=True)
        else:
            raise ValueError("Bad value of `type`!")
        sectional_chords = [
            (inner_chord + outer_chord) / 2
            for inner_chord, outer_chord in zip(
                chords[1:],
                chords[:-1]
            )
        ]
        sectional_areas = [
            span * chord
            for span, chord in zip(
                sectional_spans,
                sectional_chords
            )
        ]
        area = sum(sectional_areas)

        if _sectional:
            return sectional_areas

        if self.symmetric:
            area *= 2

        return area

    def aspect_ratio(self) -> float:
        # Returns the aspect ratio (b^2/S).
        # Uses the full span and the full area if symmetric.
        return self.span() ** 2 / self.area()

    def is_entirely_symmetric(self) -> bool:
        # Returns a boolean of whether the wing is totally symmetric (i.e.), every xsec has symmetric control surfaces.
        for xsec in self.xsecs: # To be symmetric, all
            for surf in xsec.control_surfaces:
                if not (surf.symmetric or surf.deflection == 0):
                    return False

        if not self.symmetric: # If the wing itself isn't mirrored (e.g., vertical stabilizer), check that it's symmetric
            for xsec in self.xsecs:
                if not xsec.xyz_le[1] == 0: # Surface has to be right on the centerline
                    return False
                if not xsec.twist == 0: # Surface has to be untwisted
                    return False
                if not xsec.airfoil.CL_function(0, 1e6, 0, 0) == 0: # Surface has to have a symmetric airfoil.
                    return False
                if not xsec.airfoil.CM_function(0, 1e6, 0, 0) == 0: # Surface has to have a symmetric airfoil.
                    return False

        return True

    def mean_geometric_chord(self) -> float:
        """
        Returns the mean geometric chord of the wing (S/b).
        :return:
        """
        return self.area() / self.span()

    def mean_aerodynamic_chord(self) -> float:
        """
        Computes the length of the mean aerodynamic chord of the wing.
        Uses the generalized methodology described here:
            https://core.ac.uk/download/pdf/79175663.pdf

        Returns: The length of the mean aerodynamic chord.
        """
        sectional_areas = self.area(_sectional=True)
        sectional_MAC_lengths = []

        for inner_xsec, outer_xsec in zip(self.xsecs[:-1], self.xsecs[1:]):

            section_taper_ratio = outer_xsec.chord / inner_xsec.chord
            section_MAC_length = (2 / 3) * inner_xsec.chord * (
                    (1 + section_taper_ratio + section_taper_ratio ** 2) /
                    (1 + section_taper_ratio)
            )

            sectional_MAC_lengths.append(section_MAC_length)

        sectional_MAC_length_area_products = [
            MAC * area
            for MAC, area in zip(
                sectional_MAC_lengths,
                sectional_areas,
            )
        ]

        MAC_length = sum(sectional_MAC_length_area_products) / sum(sectional_areas)

        return MAC_length

    def mean_twist_angle(self) -> float:
        r"""
        Returns the mean twist angle (in degrees) of the wing, weighted by area.
        :return: mean twist angle (in degrees)
        """

        sectional_twists = [
            (inner_xsec.twist + outer_xsec.twist) / 2
            for inner_xsec, outer_xsec in zip(
                self.xsecs[1:],
                self.xsecs[:-1]
            )
        ]
        sectional_areas = self.area(_sectional=True)

        sectional_twist_area_products = [
            twist * area
            for twist, area in zip(
                sectional_twists, sectional_areas
            )
        ]

        mean_twist = sum(sectional_twist_area_products) / sum(sectional_areas)

        return mean_twist

    def mean_sweep_angle(self, x_nondim=0.25) -> float:
        """
        Returns the mean sweep angle (in degrees) of the wing, relative to the x-axis.
        Positive sweep is backwards, negative sweep is forward.

        By changing `x_nondim`, you can change whether it's leading-edge sweep (0), quarter-chord sweep (0.25),
        trailing-edge sweep (1), or anything else.

        :return: The mean sweep angle, in degrees.
        """
        root_quarter_chord = self._compute_xyz_of_WingXSec(
            0,
            x_nondim=x_nondim,
            y_nondim=0
        )
        tip_quarter_chord = self._compute_xyz_of_WingXSec(
            -1,
            x_nondim=x_nondim,
            y_nondim=0
        )

        vec = tip_quarter_chord - root_quarter_chord
        vec_norm = vec / np.linalg.norm(vec)

        sin_sweep = vec_norm[0]  # from dot product with x_hat

        sweep_deg = np.arcsind(sin_sweep)

        return sweep_deg

    def aerodynamic_center(self, chord_fraction: float = 0.25, _sectional=False) -> np.ndarray:
        """
        Computes the location of the aerodynamic center of the wing.
        Uses the generalized methodology described here:
            https://core.ac.uk/downloattttd/pdf/79175663.pdf

        Args: chord_fraction: The position of the aerodynamic center along the MAC, as a fraction of MAC length.

            Typically, this value (denoted `h_0` in the literature) is 0.25 for a subsonic wing. However,
            wing-fuselage interactions can cause a forward shift to a value more like 0.1 or less. Citing Cook,
            Michael V., "Flight Dynamics Principles", 3rd Ed., Sect. 3.5.3 "Controls-fixed static stability". PDF:
            https://www.sciencedirect.com/science/article/pii/B9780080982427000031

        Returns: The (x, y, z) coordinates of the aerodynamic center of the wing.

        """
        sectional_areas = self.area(_sectional=True)
        sectional_ACs = []

        for inner_xsec, outer_xsec in zip(self.xsecs[:-1], self.xsecs[1:]):

            section_taper_ratio = outer_xsec.chord / inner_xsec.chord
            section_MAC_length = (2 / 3) * inner_xsec.chord * (
                    (1 + section_taper_ratio + section_taper_ratio ** 2) /
                    (1 + section_taper_ratio)
            )
            section_MAC_le = (
                    inner_xsec.xyz_le +
                    (outer_xsec.xyz_le - inner_xsec.xyz_le) *
                    (1 + 2 * section_taper_ratio) /
                    (3 + 3 * section_taper_ratio)
            )
            section_AC = section_MAC_le + np.array([  # TODO rotate this vector by the local twist angle
                chord_fraction * section_MAC_length,
                0,
                0
            ])

            sectional_ACs.append(section_AC)

        if _sectional:
            return sectional_ACs

        sectional_AC_area_products = [
            AC * area
            for AC, area in zip(
                sectional_ACs,
                sectional_areas,
            )
        ]

        aerodynamic_center = sum(sectional_AC_area_products) / sum(sectional_areas)

        if self.symmetric:
            aerodynamic_center[1] = 0

        return aerodynamic_center

    def taper_ratio(self) -> float:
        """
        Gives the taper ratio of the Wing. Strictly speaking, only valid for trapezoidal wings.

        Returns:
            Taper ratio of the Wing.

        """
        return self.xsecs[-1].chord / self.xsecs[0].chord

    def mesh_body(self,
                  method="quad",
                  chordwise_resolution: int = 36,
                  spanwise_resolution: int = 1,
                  spanwise_spacing: str = "uniform",
                  mesh_surface: bool = True,
                  mesh_tips: bool = True,
                  mesh_trailing_edge: bool = True,
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Meshes the wing as a solid (thickened) body.

        Uses the `(points, faces)` standard mesh format. For reference on this format, see the documentation in
        `aerosandbox.geometry.mesh_utilities`.

        Args:
            method: Allows choice between "tri" and "quad" meshing.
            chordwise_resolution: Controls the chordwise resolution of the meshing.
            spanwise_resolution: Controls the spanwise resolution of the meshing.
            spanwise_spacing: Controls the spanwise spacing of the meshing. Can be "uniform" or "cosine".
            mesh_surface: Controls whether the actual wing surface is meshed.
            mesh_tips: Control whether the wing tips (both outside and inside) are meshed.
            mesh_trailing_edge: Controls whether the wing trailing edge is meshed, in the case of open-TE airfoils.

        Returns: (points, faces) in standard mesh format.

        """

        airfoil_nondim_coordinates = np.array([
            xsec.airfoil
                .repanel(n_points_per_side=chordwise_resolution + 1)
                .coordinates
            for xsec in self.xsecs
        ])

        x_nondim = airfoil_nondim_coordinates[:, :, 0].T
        y_nondim = airfoil_nondim_coordinates[:, :, 1].T

        spanwise_strips = []
        for x_n, y_n in zip(x_nondim, y_nondim):
            spanwise_strips.append(
                self.mesh_line(
                    x_nondim=x_n,
                    y_nondim=y_n,
                    add_camber=False,
                    spanwise_resolution=spanwise_resolution,
                    spanwise_spacing=spanwise_spacing,
                )
            )

        points = np.concatenate(spanwise_strips)

        faces = []

        num_i = spanwise_resolution * (len(self.xsecs) - 1)
        num_j = len(spanwise_strips) - 1

        def index_of(iloc, jloc):
            return iloc + jloc * (num_i + 1)

        def add_face(*indices):
            entry = list(indices)
            if method == "quad":
                faces.append(entry)
            elif method == "tri":
                faces.append([entry[0], entry[1], entry[3]])
                faces.append([entry[1], entry[2], entry[3]])

        if mesh_surface:
            for i in range(num_i):
                for j in range(num_j):
                    add_face(
                        index_of(i, j),
                        index_of(i + 1, j),
                        index_of(i + 1, j + 1),
                        index_of(i, j + 1),
                    )

        if mesh_tips:
            for j in range(num_j // 2):
                add_face(  # Mesh the root face
                    index_of(0, num_j - j),
                    index_of(0, j),
                    index_of(0, j + 1),
                    index_of(0, num_j - j - 1),
                )
                add_face(  # Mesh the tip face
                    index_of(num_i, j),
                    index_of(num_i, j + 1),
                    index_of(num_i, num_j - j - 1),
                    index_of(num_i, num_j - j),
                )
        if mesh_trailing_edge:
            for i in range(num_i):
                add_face(
                    index_of(i + 1, 0),
                    index_of(i + 1, num_j),
                    index_of(i, num_j),
                    index_of(i, 0),
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

    def mesh_thin_surface(self,
                          method="tri",
                          chordwise_resolution: int = 36,
                          spanwise_resolution: int = 1,
                          chordwise_spacing: str = "cosine",
                          spanwise_spacing: str = "uniform",
                          add_camber: bool = True,
                          ) -> Tuple[np.ndarray, List[List[int]]]:
        """
        Meshes the mean camber line of the wing as a thin-sheet body.

        Uses the `(points, faces)` standard mesh format. For reference on this format, see the documentation in
        `aerosandbox.geometry.mesh_utilities`.

        Args:
            method: Allows choice between "tri" and "quad" meshing.
            chordwise_resolution: Controls the chordwise resolution of the meshing.
            spanwise_resolution: Controls the spanwise resolution of the meshing.
            chordwise_spacing: Controls the chordwise spacing of the meshing. Can be "uniform" or "cosine".
            spanwise_spacing: Controls the spanwise spacing of the meshing. Can be "uniform" or "cosine".
            add_camber: Controls whether to mesh the thin surface with camber (i.e., mean camber line), or just the flat planform.

        Returns: (points, faces) in standard mesh format.

        """
        if chordwise_spacing == "cosine":
            space = np.cosspace
        elif chordwise_spacing == "uniform":
            space = np.linspace
        else:
            raise ValueError("Bad value of 'chordwise_spacing'")

        x_nondim = space(
            0,
            1,
            chordwise_resolution + 1
        )

        spanwise_strips = []
        for x_n in x_nondim:
            spanwise_strips.append(
                self.mesh_line(
                    x_nondim=x_n,
                    y_nondim=0,
                    add_camber=add_camber,
                    spanwise_resolution=spanwise_resolution,
                    spanwise_spacing=spanwise_spacing
                )
            )

        points = np.concatenate(spanwise_strips)

        faces = []

        num_i = spanwise_resolution * (len(self.xsecs) - 1)
        num_j = len(spanwise_strips) - 1

        def index_of(iloc, jloc):
            return iloc + jloc * (num_i + 1)

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
                    index_of(i + 1, j),
                    index_of(i + 1, j + 1),
                    index_of(i, j + 1),
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
                  x_nondim: Union[float, List[float]] = 0.25,
                  y_nondim: Union[float, List[float]] = 0,
                  add_camber: bool = True,
                  spanwise_resolution: int = 1,
                  spanwise_spacing: str = "cosine"
                  ) -> np.ndarray:
        """
        Meshes a line that goes through each of the WingXSec objects in this wing.

        Args:

            x_nondim: The nondimensional (chord-normalized) x-coordinate that the line should go through. Can either
            be a single value used at all cross sections, or can be an iterable of values to be used at the
            respective cross sections.

            y_nondim: The nondimensional (chord-normalized) y-coordinate that the line should go through. Here,
            y-coordinate means the "vertical" component (think standard 2D airfoil axes). Can either be a single
            value used at all cross sections, or can be an iterable of values to be used at the respective cross
            sections.

            add_camber: Controls whether camber should be added to the line or not.

            spanwise_resolution: Controls the number of times each WingXSec is subdivided.

            spanwise_spacing: Controls the spanwise spacing. Either "cosine" or "uniform".

        Returns:

            points: a Nx3 np.ndarray that gives the coordinates of each point on the meshed line. Goes from the root to the tip.

        """

        if spanwise_spacing == "cosine":
            space = np.cosspace
        elif spanwise_spacing == "uniform":
            space = np.linspace
        else:
            raise ValueError("Bad value of 'spanwise_spacing'")

        xsec_points = []

        try:
            if len(x_nondim) != len(self.xsecs):
                raise ValueError(
                    "If x_nondim is going to be an iterable, it needs to be the same length as Airplane.xsecs."
                )
        except TypeError:
            pass

        try:
            if len(y_nondim) != len(self.xsecs):
                raise ValueError(
                    "If y_nondim is going to be an iterable, it needs to be the same length as Airplane.xsecs."
                )
        except TypeError:
            pass

        for i, xsec in enumerate(self.xsecs):

            try:
                xsec_x_nondim = x_nondim[i]
            except (TypeError, IndexError):
                xsec_x_nondim = x_nondim

            try:
                xsec_y_nondim = y_nondim[i]
            except (TypeError, IndexError):
                xsec_y_nondim = y_nondim

            if add_camber:
                xsec_y_nondim = xsec_y_nondim + xsec.airfoil.local_camber(x_over_c=x_nondim)

            xsec_point = self._compute_xyz_of_WingXSec(
                i,
                x_nondim=xsec_x_nondim,
                y_nondim=xsec_y_nondim,
            )
            xsec_points.append(xsec_point)

        points_sections = []
        for i in range(len(xsec_points) - 1):
            points_section = space(
                xsec_points[i],
                xsec_points[i + 1],
                spanwise_resolution + 1
            )
            if not i == len(xsec_points) - 2:
                points_section = points_section[:-1]

            points_sections.append(points_section)

        points = np.concatenate(points_sections)

        return points

    def draw(self, *args, **kwargs):
        """
        An alias to the more general Airplane.draw() method. See there for documentation.

        Args:
            *args: Arguments to pass through to Airplane.draw()
            **kwargs: Keyword arguments to pass through to Airplane.draw()

        Returns: Same return as Airplane.draw()

        """
        from aerosandbox.geometry.airplane import Airplane
        return Airplane(wings=[self]).draw(*args, **kwargs)

    def _compute_xyz_le_of_WingXSec(self, index: int):
        return self.xsecs[index].xyz_le

    def _compute_xyz_te_of_WingXSec(self, index: int):
        return self._compute_xyz_of_WingXSec(
            index,
            x_nondim=1,
            y_nondim=0,
        )

    def _compute_xyz_of_WingXSec(self,
                                 index,
                                 x_nondim,
                                 y_nondim,
                                 ):
        xg_local, yg_local, zg_local = self._compute_frame_of_WingXSec(index)
        origin = self.xsecs[index].xyz_le
        xsec = self.xsecs[index]
        return origin + (
                x_nondim * xsec.chord * xg_local +
                y_nondim * xsec.chord * zg_local
        )

    def _compute_frame_of_WingXSec(
            self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the local reference frame associated with a particular cross section (XSec) of this wing.

        Args:

            index: Which cross section (as indexed in Wing.xsecs) should we get the frame of?

        Returns:

            A tuple of (xg_local, yg_local, zg_local), where each entry refers to the respective (normalized) axis of
            the local reference frame of the WingXSec. Given in geometry axes.

        """
        ### Compute the untwisted reference frame
        xg_local = np.array([1, 0, 0])
        if index == 0:
            span_vector = self.xsecs[1].xyz_le - self.xsecs[0].xyz_le
            span_vector[0] = 0
            yg_local = span_vector / np.linalg.norm(span_vector)
            z_scale = 1
        elif index == len(self.xsecs) - 1 or index == -1:
            span_vector = self.xsecs[-1].xyz_le - self.xsecs[-2].xyz_le
            span_vector[0] = 0
            yg_local = span_vector / np.linalg.norm(span_vector)
            z_scale = 1
        else:
            vector_before = self.xsecs[index].xyz_le - self.xsecs[index - 1].xyz_le
            vector_after = self.xsecs[index + 1].xyz_le - self.xsecs[index].xyz_le
            vector_before[0] = 0  # Project onto YZ plane.
            vector_after[0] = 0  # Project onto YZ plane.
            vector_before = vector_before / np.linalg.norm(vector_before)
            vector_after = vector_after / np.linalg.norm(vector_after)
            span_vector = (vector_before + vector_after) / 2
            yg_local = span_vector / np.linalg.norm(span_vector)
            cos_vectors = np.linalg.inner(vector_before, vector_after)
            z_scale = np.sqrt(2 / (cos_vectors + 1))

        zg_local = np.cross(xg_local, yg_local) * z_scale

        ### Twist the reference frame by the WingXSec twist angle
        rot = np.rotation_matrix_3D(
            self.xsecs[index].twist * pi / 180,
            yg_local
        )
        xg_local = rot @ xg_local
        zg_local = rot @ zg_local

        return xg_local, yg_local, zg_local

    def _compute_frame_of_section(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the local reference frame associated with a particular section. (Note that sections and cross
        sections are different! Cross sections, or xsecs, are the vertices, and sections are the parts in between. In
        other words, a wing with N cross sections (xsecs) will always have N-1 sections.

        Args:

            index: Which section should we get the frame of? If given `i`, this retrieves the frame of the section
            between xsecs `i` and `i+1`.

        Returns:

            A tuple of (xg_local, yg_local, zg_local), where each entry refers to the respective (normalized) axis
            of the local reference frame of the section. Given in geometry axes.

        """
        in_front = self._compute_xyz_le_of_WingXSec(index)
        in_back = self._compute_xyz_te_of_WingXSec(index)
        out_front = self._compute_xyz_le_of_WingXSec(index + 1)
        out_back = self._compute_xyz_te_of_WingXSec(index + 1)

        diag1 = out_back - in_front
        diag2 = out_front - in_back
        cross = np.cross(diag1, diag2)

        zg_local = cross / np.linalg.norm(cross)

        quarter_chord_vector = (
                                       0.75 * out_front + 0.25 * out_back
                               ) - (
                                       0.75 * in_front + 0.25 * in_back
                               )
        quarter_chord_vector[0] = 0

        yg_local = quarter_chord_vector / np.linalg.norm(quarter_chord_vector)

        xg_local = np.cross(yg_local, zg_local)

        return xg_local, yg_local, zg_local


class WingXSec(AeroSandboxObject):
    """
    Definition for a wing cross section ("X-section").
    """

    def __init__(self,
                 xyz_le: Union[np.ndarray, List] = np.array([0, 0, 0]),
                 chord: float = 1.,
                 twist: float = 0,
                 airfoil: Airfoil = None,
                 control_surfaces: List['ControlSurface'] = None,
                 analysis_specific_options: Dict[type, Dict[str, Any]] = None,
                 control_surface_is_symmetric=None,  # Note: deprecated.
                 control_surface_hinge_point=None,  # Note: deprecated.
                 control_surface_deflection=None,  # Note: deprecated.
                 twist_angle=None,  # Note: deprecated.
                 ):
        """
        Defines a new wing cross section.

        Args:

            xyz_le: An array-like that represents the xyz-coordinates of the leading edge of the cross section, in
            geometry axes.

            chord: Chord of the wing at this cross section.

            twist: Twist angle, in degrees, as defined about the leading edge.

                The twist axis is computed with the following procedure:

                    * The quarter-chord point of this WingXSec and the following one are identified.

                    * A line is drawn connecting them, and it is normalized to a unit direction vector.

                    * That direction vector is projected onto the geometry Y-Z plane.

                    * That direction vector is now the twist axis.

            airfoil: Airfoil associated with this cross section. [aerosandbox.Airfoil]

            control_surfaces: A list of control surfaces in the form of ControlSurface objects.

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

            Note: Control surface definition through WingXSec properties (control_surface_is_symmetric, control_surface_hinge_point, control_surface_deflection)
            is deprecated. Control surfaces should be handled according to the following protocol:
            
                1. If control_surfaces is an empty list (default, user does not specify any control surfaces), use deprecated WingXSec control surface definition properties.
                This will result in 1 control surface at this xsec.
                
                Usage example:

                >>> xsecs = asb.WingXSec(
                >>>     chord = 2
                >>> )

                2. If control_surfaces is a list of ControlSurface instances, use ControlSurface properties to define control surfaces. This will result in as many
                control surfaces at this xsec as there are entries in the control_surfaces list (an arbitrary number >= 1).

                Usage example:

                >>>xsecs = asb.WingXSec(
                >>>    chord = 2,
                >>>    control_surfaces = [
                >>>        ControlSurface(
                >>>            trailing_edge = False
                >>>        )
                >>>    ]
                >>>)

                3. If control_surfaces is None, override deprecated control surface definition properties and do not define a control surface at this xsec. This will
                result in 0 control surfaces at this xsec.

                Usage example:

                >>>xsecs = asb.WingXSec(
                >>>    chord = 2,
                >>>    control_surfaces = None
                >>>)
            
            See avl.py for example of control_surface handling using this protocol.
        """
        ### Set defaults
        if airfoil is None:
            airfoil = Airfoil("naca0012")
        if control_surfaces is None:
            control_surfaces = []
        if analysis_specific_options is None:
            analysis_specific_options = {}

        self.xyz_le = np.array(xyz_le)
        self.chord = chord
        self.twist = twist
        self.airfoil = airfoil
        self.control_surfaces = control_surfaces
        self.analysis_specific_options = analysis_specific_options

        ### Handle deprecated arguments
        if twist_angle is not None:
            import warnings
            warnings.warn(
                "DEPRECATED: 'twist_angle' has been renamed 'twist', and will break in future versions.",
                stacklevel=2
            )
            self.twist = twist_angle
        if not (
                control_surface_is_symmetric is None and
                control_surface_hinge_point is None and
                control_surface_deflection is None
        ):
            import warnings
            warnings.warn(
                "DEPRECATED: Define control surfaces using the `control_surfaces` parameter, which takes in a list of asb.ControlSurface objects.",
                stacklevel=2
            )
            if control_surface_is_symmetric is None:
                control_surface_is_symmetric = True
            if control_surface_hinge_point is None:
                control_surface_hinge_point = 0.75
            if control_surface_deflection is None:
                control_surface_deflection = 0

            self.control_surfaces.append(
                ControlSurface(
                    hinge_point=control_surface_hinge_point,
                    symmetric=control_surface_is_symmetric,
                    deflection=control_surface_deflection,
                )
            )

    def __repr__(self) -> str:
        return f"WingXSec (Airfoil: {self.airfoil.name}, chord: {self.chord:.3f}, twist: {self.twist:.3f})"

    def translate(self,
                  xyz: Union[np.ndarray, List]
                  ) -> "WingXSec":
        """
        Returns a copy of this WingXSec that has been translated by `xyz`.

        Args:
            xyz: The amount to translate the WingXSec. Given as a 3-element NumPy vector.

        Returns: A new WingXSec object.

        """
        new_xsec = copy.copy(self)
        new_xsec.xyz_le = new_xsec.xyz_le + np.array(xyz)
        return new_xsec


class ControlSurface(AeroSandboxObject):
    """
    Definition for a control surface, which is attached to a particular WingXSec via WingXSec's `control_surfaces=[]` parameter.
    """

    def __init__(self,
                 name: str = "Untitled",
                 trailing_edge: bool = True,
                 hinge_point: float = 0.75,
                 symmetric: bool = True,
                 deflection: float = 0.0,
                 analysis_specific_options: Dict[type, Dict[str, Any]] = None,
                 ):
        """
        Define a new control surface.

        Args:

            name: Name of the control surface [optional]. It can help when debugging to give each control surface a
            sensible name.

            trailing_edge: Is the control surface on the trailing edge? If False, control surface is on the leading
            edge. (e.g., True for flaps, False for slats.)

            hinge_point: The location of the control surface hinge, as a fraction of chord.

            symmetric: Is the control surface symmetric? If False, control surface is anti-symmetric. (e.g.,
            True for flaps, False for ailerons.)

            deflection: Control deflection, in degrees. Downwards-positive.

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

        self.name = name
        self.trailing_edge = trailing_edge
        self.hinge_point = hinge_point
        self.symmetric = symmetric
        self.deflection = deflection
        self.analysis_specific_options = analysis_specific_options


if __name__ == '__main__':
    wing = Wing(
        xsecs=[
            WingXSec(
                xyz_le=[0, 0, 0],
                chord=1,
                airfoil=Airfoil("naca0012"),
                twist=0,
            ),
            WingXSec(
                xyz_le=[0.5, 1, 0],
                chord=0.5,
                airfoil=Airfoil("naca0012"),
                twist=0,
            ),
            WingXSec(
                xyz_le=[0.7, 1, 0.3],
                chord=0.3,
                airfoil=Airfoil("naca0012"),
                twist=0,
            )
        ]
    ).translate([1, 0, 0])
    wing.draw()
