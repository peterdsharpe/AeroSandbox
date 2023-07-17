from aerosandbox.common import AeroSandboxObject
from aerosandbox.geometry.common import *
from typing import List, Dict, Any, Tuple, Union, Optional, Callable
from aerosandbox.geometry.airfoil import Airfoil
from numpy import pi
import aerosandbox.numpy as np
import aerosandbox.geometry.mesh_utilities as mesh_utils
import copy


class Wing(AeroSandboxObject):
    """
    Definition for a Wing.

    Anatomy of a Wing:

        A wing consists chiefly of a collection of cross-sections, or "xsecs". A cross-section is a 2D "slice" of a
        wing. These can be accessed with `Wing.xsecs`, which gives a list of xsecs in the Wing. Each xsec is a
        WingXSec object, a class that is defined separately.

        You may also see references to wing "sections", which are different than cross-sections (xsecs)! Sections are
        the portions of the wing that are in between xsecs. In other words, a wing with N cross-sections (xsecs,
        WingXSec objects) will always have N-1 sections. Sections are never explicitly defined, since you can get all
        needed information by lofting from the adjacent cross-sections. For example, section 0 (the first one) is a
        loft between cross-sections 0 and 1.

        Wings are lofted linearly between cross-sections.

    If the wing is symmetric across the XZ plane, just define the right half and supply `symmetric=True` in the
    constructor.

    If the wing is not symmetric across the XZ plane (e.g., a single vertical stabilizer), just define the wing.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 xsecs: List['WingXSec'] = None,
                 symmetric: bool = False,
                 color: Optional[Union[str, Tuple[float]]] = None,
                 analysis_specific_options: Optional[Dict[type, Dict[str, Any]]] = None,
                 **kwargs,  # Only to allow for capturing of deprecated arguments, don't use this.
                 ):
        """
        Defines a new wing object.

        Args:

            name: Name of the wing [optional]. It can help when debugging to give each wing a sensible name.

            xsecs: A list of wing cross-sections ("xsecs") in the form of WingXSec objects.

            symmetric: Is the wing symmetric across the XZ plane?

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
        if name is None:
            name = "Untitled"
        if xsecs is None:
            xsecs: List['WingXSec'] = []
        if analysis_specific_options is None:
            analysis_specific_options = {}

        ### Initialize
        self.name = name
        self.xsecs = xsecs
        self.symmetric = symmetric
        self.color = color
        self.analysis_specific_options = analysis_specific_options

        ### Handle deprecated parameters
        if 'xyz_le' in locals():
            import warnings
            warnings.warn(
                "The `xyz_le` input for Wing is pending deprecation and will be removed in a future version. Use Wing().translate(xyz) instead.",
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
                  xyz: Union[np.ndarray, List[float]]
                  ) -> 'Wing':
        """
        Translates the entire Wing by a certain amount.

        Args:
            xyz:

        Returns: The new wing object.

        """
        new_wing = copy.copy(self)
        new_wing.xsecs = [
            xsec.translate(xyz)
            for xsec in new_wing.xsecs
        ]
        return new_wing

    def span(self,
             type: str = "yz",
             include_centerline_distance=False,
             _sectional: bool = False,
             ) -> Union[float, List[float]]:
        """
        Computes the span, with options for various ways of measuring this (see `type` argument).

        If the wing is symmetric, both left/right sides are included in order to obtain the full span. In the case
        where the root cross-section is not coincident with the center plane (e.g., XZ plane), this function's
        behavior depends on the `include_centerline_distance` argument.

        Args:

            type: One of the following options, as a string:

                * "xyz": First, computes the quarter-chord point of each WingXSec. Then, connects these with
                straight lines. Then, adds up the lengths of these lines.

                * "xy" or "top": Same as "xyz", except it projects each line segment onto the XY plane before adding up the
                lengths.

                * "yz" (default) or "front": Same as "xyz", except it projects each line segment onto the YZ plane (i.e., front view)
                before adding up the lengths.

                * "xz" or "side": Same as "xyz", except it projects each line segment onto the XZ plane before adding up the
                lengths. Rarely needed.

                * "x": Same as "xyz", except it only counts the x-components of each line segment when adding up the
                lengths.

                * "y": Same as "xyz", except it only counts the y-components of each line segment when adding up the
                lengths.

                * "z": Same as "xyz", except it only counts the z-components of each line segment when adding up the
                lengths.

            include_centerline_distance: A boolean flag that tells the function what to do if a wing's root is not
            coincident with the centerline plane (i.e., XZ plane).

                * If True, we first figure out which WingXSec has its quarter-chord point closest to the centerline
                plane (i.e., XZ plane). Then, we compute the distance from that quarter-chord point directly to the
                centerline plane (along Y). We then add that distance to the span calculation. In other words,
                the fictitious span connecting the left and right root cross-sections is included.

                * If False, this distance is ignored. In other words, the fictitious span connecting the left and
                right root cross-sections is not included. This is the default behavior.

                Note: For computation, either the root WingXSec (i.e., index=0) or the tip WingXSec (i.e., index=-1)
                is used, whichever is closer to the centerline plane. This will almost-always be the root WingXSec,
                but some weird edge cases (e.g., a half-wing defined on the left-hand-side of the airplane,
                rather than the conventional right-hand side) will result in the tip WingXSec being used.

            _sectional: A boolean. If False, returns the total span. If True, returns a list of spans for each of the
                `n-1` lofted sections (between the `n` wing cross-sections in wing.xsec).
        """
        # Check inputs
        if include_centerline_distance and _sectional:
            raise ValueError("Cannot use `_sectional` with `include_centerline_distance`!")

        # Handle overloaded names
        if type == "top":
            type = "xy"
        elif type == "front":
            type = "yz"
        elif type == "side":
            type = "xz"

        # Figure out where the quarter-chord points of each WingXSec are
        i_range = range(len(self.xsecs))

        quarter_chord_locations = [
            self._compute_xyz_of_WingXSec(
                i,
                x_nondim=0.25,
                z_nondim=0,
            )
            for i in i_range
        ]

        # Compute sectional spans
        sectional_spans = []

        for inner_i, outer_i in zip(i_range, i_range[1:]):
            quarter_chord_vector = (
                    quarter_chord_locations[outer_i] -
                    quarter_chord_locations[inner_i]
            )

            if type == "xyz":
                section_span = (
                                       quarter_chord_vector[0] ** 2 +
                                       quarter_chord_vector[1] ** 2 +
                                       quarter_chord_vector[2] ** 2
                               ) ** 0.5

            elif type == "xy":
                section_span = (
                                       quarter_chord_vector[0] ** 2 +
                                       quarter_chord_vector[1] ** 2
                               ) ** 0.5

            elif type == "yz":
                section_span = (
                                       quarter_chord_vector[1] ** 2 +
                                       quarter_chord_vector[2] ** 2
                               ) ** 0.5

            elif type == "xz":
                section_span = (
                                       quarter_chord_vector[0] ** 2 +
                                       quarter_chord_vector[2] ** 2
                               ) ** 0.5

            elif type == "x":
                section_span = quarter_chord_vector[0]

            elif type == "y":
                section_span = quarter_chord_vector[1]

            elif type == "z":
                section_span = quarter_chord_vector[2]

            else:
                raise ValueError("Bad value of 'type'!")

            sectional_spans.append(section_span)

        if _sectional:
            return sectional_spans

        half_span = sum(sectional_spans)

        if include_centerline_distance and len(self.xsecs) > 0:

            half_span_to_XZ_plane = np.Inf

            for i in i_range:
                half_span_to_XZ_plane = np.minimum(
                    half_span_to_XZ_plane,
                    np.abs(quarter_chord_locations[i][1])
                )

            half_span = half_span + half_span_to_XZ_plane

        if self.symmetric:
            span = 2 * half_span
        else:
            span = half_span

        return span

    def area(self,
             type: str = "planform",
             include_centerline_distance=False,
             _sectional: bool = False,
             ) -> Union[float, List[float]]:
        """
        Computes the wing area, with options for various ways of measuring this (see `type` argument):

        If the wing is symmetric, both left/right sides are included in order to obtain the full area. In the case
        where the root cross-section is not coincident with the center plane (e.g., XZ plane), this function's
        behavior depends on the `include_centerline_distance` argument.

        Args:

            type: One of the following options, as a string:

                * "planform" (default): First, lofts a quadrilateral mean camber surface between each WingXSec. Then,
                computes the area of each of these sectional surfaces. Then, sums up all the areas and returns it.
                When airplane designers refer to "wing area" (in the absence of any other qualifiers),
                this is typically what they mean.

                * "wetted": Computes the actual surface area of the wing that is in contact with the air. Will
                typically be a little more than double the "planform" area above; intuitively, this is because it
                adds both the "top" and "bottom" surface areas. Accounts for airfoil thickness/shape effects.

                * "xy" or "projected" or "top": Same as "planform", but each sectional surface is projected onto the XY plane
                (i.e., top-down view) before computing the areas. Note that if you try to use this method with a
                vertically-oriented wing, like most vertical stabilizers, you will get an area near zero.

                * "xz" or "side": Same as "planform", but each sectional surface is projected onto the XZ plane before
                computing the areas.

            include_centerline_distance: A boolean flag that tells the function what to do if a wing's root chord is
            not coincident with the centerline plane (i.e., XZ plane).

                * If True, we first figure out which WingXSec is closest to the centerline plane (i.e., XZ plane).
                Then, we imagine that this WingXSec is extruded along the Y axis to the centerline plane (assuming a
                straight extrusion to produce a rectangular mid-camber surface). In doing so, we use the wing
                geometric chord as the extrusion width. We then add the area of this fictitious surface to the area
                calculation.

                * If False, this function will simply ignore this fictitious wing area. This is the default behavior.

            _sectional: A boolean. If False, returns the total area. If True, returns a list of areas for each of the
                `n-1` lofted sections (between the `n` wing cross-sections in wing.xsec).

        """
        # Check inputs
        if include_centerline_distance and _sectional:
            raise ValueError("`include_centerline_distance` and `_sectional` cannot both be True!")

        # Handle overloaded names
        if type == "projected" or type == "top":
            type = "xy"

        elif type == "side":
            type = "xz"

        # Compute sectional areas. Each method must compute the sectional spans and the effective chords at each
        # cross-section to use.
        if type == "planform":
            sectional_spans = self.span(type="yz", _sectional=True)
            xsec_chords = [xsec.chord for xsec in self.xsecs]

        elif type == "wetted":
            sectional_spans = self.span(type="yz", _sectional=True)
            xsec_chords = [
                xsec.chord * xsec.airfoil.perimeter()
                for xsec in self.xsecs
            ]

        elif type == "xy":
            sectional_spans = self.span(type="y", _sectional=True)
            xsec_chords = [xsec.chord for xsec in self.xsecs]

        elif type == "yz":
            raise ValueError("Area of wing projected to the YZ plane is zero.")

            # sectional_spans = self.span(type="yz", _sectional=True)
            # xsec_chords = [xsec.chord for xsec in self.xsecs]

        elif type == "xz":
            sectional_spans = self.span(type="z", _sectional=True)
            xsec_chords = [xsec.chord for xsec in self.xsecs]

        else:
            raise ValueError("Bad value of `type`!")

        sectional_chords = [
            (inner_chord + outer_chord) / 2
            for inner_chord, outer_chord in zip(
                xsec_chords[1:],
                xsec_chords[:-1]
            )
        ]

        sectional_areas = [
            span * chord
            for span, chord in zip(
                sectional_spans,
                sectional_chords
            )
        ]
        if _sectional:
            return sectional_areas

        half_area = sum(sectional_areas)

        if include_centerline_distance and len(self.xsecs) > 0:

            half_span_to_centerline = np.Inf

            for i in range(len(self.xsecs)):
                quarter_chord_location = self._compute_xyz_of_WingXSec(
                    i,
                    x_nondim=0.25,
                    z_nondim=0,
                )

                half_span_to_centerline = np.minimum(
                    half_span_to_centerline,
                    np.abs(quarter_chord_location[1])
                )

            half_area = half_area + (
                    half_span_to_centerline * self.mean_geometric_chord()
            )

        if self.symmetric:  # Returns the total area of both the left and right wing halves on mirrored wings.
            area = 2 * half_area
        else:
            area = half_area

        return area

    def aspect_ratio(self,
                     type: str = "geometric",
                     ) -> float:
        """
        Computes the aspect ratio of the wing, with options for various ways of measuring this.

         * geometric: geometric aspect ratio, computed in the typical fashion (b^2 / S).

         * effective: Differs from the geometric aspect ratio only in the case of symmetric wings whose root
         cross-section is not on the centerline. In these cases, it includes the span and area of the fictitious wing
         center when computing aspect ratio.

        Args:
            type: One of the above options, as a string.

        """
        if type == "geometric":
            return self.span() ** 2 / self.area()

        elif type == "effective":
            return (
                    self.span(type="yz", include_centerline_distance=True) ** 2 /
                    self.area(type="planform", include_centerline_distance=True)
            )

        else:
            raise ValueError("Bad value of `type`!")

    def is_entirely_symmetric(self) -> bool:
        # Returns a boolean of whether the wing is totally symmetric (i.e.), every xsec has symmetric control surfaces.
        for xsec in self.xsecs:  # To be symmetric, all
            for surf in xsec.control_surfaces:
                if not (surf.symmetric or surf.deflection == 0):
                    return False

        if not self.symmetric:  # If the wing itself isn't mirrored (e.g., vertical stabilizer), check that it's symmetric
            for xsec in self.xsecs:
                if not xsec.xyz_le[1] == 0:  # Surface has to be right on the centerline
                    return False
                if not xsec.twist == 0:  # Surface has to be untwisted
                    return False
                if not np.allclose(xsec.airfoil.local_camber(), 0):  # Surface has to have a symmetric airfoil.
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

    def mean_sweep_angle(self,
                         x_nondim=0.25
                         ) -> float:
        """
        Returns the mean sweep angle (in degrees) of the wing, relative to the x-axis.
        Positive sweep is backwards, negative sweep is forward.

        This is purely measured from root to tip, with no consideration for the sweep of the individual
        cross-sections in between.

        Args:

            x_nondim: The nondimensional x-coordinate of the cross-section to use for sweep angle computation.

                * If you provide 0, it will use the leading edge of the cross-section.

                * If you provide 0.25, it will use the quarter-chord point of the cross-section.

                * If you provide 1, it will use the trailing edge of the cross-section.

        Returns:

            The mean sweep angle, in degrees.
        """
        root_quarter_chord = self._compute_xyz_of_WingXSec(
            0,
            x_nondim=x_nondim,
            z_nondim=0
        )
        tip_quarter_chord = self._compute_xyz_of_WingXSec(
            -1,
            x_nondim=x_nondim,
            z_nondim=0
        )

        vec = tip_quarter_chord - root_quarter_chord
        vec_norm = vec / np.linalg.norm(vec)

        sin_sweep = vec_norm[0]  # from dot product with x_hat

        sweep_deg = np.arcsind(sin_sweep)

        return sweep_deg

    def mean_dihedral_angle(self,
                            x_nondim=0.25
                            ) -> float:
        """
        Returns the mean dihedral angle (in degrees) of the wing, relative to the XY plane.
        Positive dihedral is bending up, negative dihedral is bending down.

        This is purely measured from root to tip, with no consideration for the dihedral of the individual
        cross-sections in between.

        Args:

            x_nondim: The nondimensional x-coordinate of the cross-section to use for sweep angle computation.

                * If you provide 0, it will use the leading edge of the cross-section.

                * If you provide 0.25, it will use the quarter-chord point of the cross-section.

                * If you provide 1, it will use the trailing edge of the cross-section.

        Returns:

            The mean dihedral angle, in degrees

        """
        root_quarter_chord = self._compute_xyz_of_WingXSec(
            0,
            x_nondim=x_nondim,
            z_nondim=0
        )
        tip_quarter_chord = self._compute_xyz_of_WingXSec(
            -1,
            x_nondim=x_nondim,
            z_nondim=0
        )

        vec = tip_quarter_chord - root_quarter_chord
        vec_norm = vec / np.linalg.norm(vec)

        return np.arctan2d(
            vec_norm[2],
            vec_norm[1],
        )

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

    def volume(self,
               _sectional: bool = False
               ) -> Union[float, List[float]]:
        """
        Computes the volume of the Wing.

        Args:

            _sectional: A boolean. If False, returns the total volume. If True, returns a list of volumes for each of
            the `n-1` lofted sections (between the `n` wing cross-sections in wing.xsec).

        Returns:

            The computed volume.
        """
        xsec_areas = [
            xsec.xsec_area()
            for xsec in self.xsecs
        ]
        separations = self.span(
            type="yz",
            _sectional=True
        )

        sectional_volumes = [
            separation / 3 * (area_a + area_b + (area_a * area_b + 1e-100) ** 0.5)
            for area_a, area_b, separation in zip(
                xsec_areas[1:],
                xsec_areas[:-1],
                separations
            )
        ]

        volume = sum(sectional_volumes)

        if self.symmetric:
            volume *= 2

        if _sectional:
            return sectional_volumes
        else:
            return volume

    def get_control_surface_names(self) -> List[str]:
        """
        Gets the names of all control surfaces on this wing.

        Returns:

            A list of control surface names.

        """
        control_surface_names = []
        for xsec in self.xsecs:
            for control_surface in xsec.control_surfaces:
                control_surface_names.append(control_surface.name)

        return control_surface_names

    def set_control_surface_deflections(self,
                                        control_surface_mappings: Dict[str, float],
                                        ) -> None:
        """
        Sets the deflection of all control surfaces on this wing, based on the provided mapping.

        Args:
            control_surface_mappings: A dictionary mapping control surface names to their deflection angles, in degrees.

                Note: control surface names are set in the asb.ControlSurface constructor.

        Returns:

            None. (in-place)
        """
        for xsec in self.xsecs:
            for control_surface in xsec.control_surfaces:
                if control_surface.name in control_surface_mappings.keys():
                    control_surface.deflection = control_surface_mappings[control_surface.name]

    def control_surface_area(self,
                             by_name: Optional[str] = None,
                             type: Optional[str] = "planform",
                             ) -> float:
        """
        Computes the total area of all control surfaces on this wing, optionally filtered by their name.

        Control surfaces are defined on a section-by-section basis, and are defined in the WingXSec constructor using
        its `control_surfaces` argument.

        Note: If redundant control surfaces are defined (e.g., elevons, as defined by separate ailerons + elevator),
        the area will be duplicated.

        If the wing is symmetric, control surfaces on both left/right sides are included in order to obtain the full area.

        Args:

            by_name: If not None, only control surfaces with this name will be included in the area calculation.

                Note: control surface names are set in the asb.ControlSurface constructor.

            type: One of the following options, as a string:

                * "planform" (default): First, lofts a quadrilateral mean camber surface between each WingXSec. Then,
                computes the area of each of these sectional surfaces. Then, computes what fraction of this area is
                control surface. Then, sums up all the areas and returns it. When airplane designers refer to
                "control surface area" (in the absence of any other qualifiers), this is typically what they mean.

                * "wetted": Computes the actual surface area of the control surface that is in contact with the air.
                Will typically be a little more than double the "planform" area above; intuitively, this is because
                it adds both the "top" and "bottom" surface areas. Accounts for airfoil thickness/shape effects.

                * "xy" or "projected" or "top": Same as "planform", but each sectional surface is projected onto the XY plane
                (i.e., top-down view) before computing the areas. Note that if you try to use this method with a
                vertically-oriented wing, like most vertical stabilizers, you will get an area near zero.

                * "xz" or "side": Same as "planform", but each sectional surface is projected onto the XZ plane before
                computing the areas.

        """
        sectional_areas = self.area(
            type=type,
            include_centerline_distance=False,
            _sectional=True
        )

        control_surface_area = 0.

        for xsec, sect_area in zip(self.xsecs[:-1], sectional_areas):
            for control_surface in xsec.control_surfaces:
                if (by_name is None) or (control_surface.name == by_name):

                    if control_surface.trailing_edge:
                        control_surface_chord_fraction = np.maximum(
                            1 - control_surface.hinge_point,
                            0
                        )
                    else:
                        control_surface_chord_fraction = np.maximum(
                            control_surface.hinge_point,
                            0
                        )

                    control_surface_area += control_surface_chord_fraction * sect_area

        if self.symmetric:
            control_surface_area *= 2

        return control_surface_area

    def mesh_body(self,
                  method="quad",
                  chordwise_resolution: int = 36,
                  chordwise_spacing_function_per_side: Callable[[float, float, float], np.ndarray] = np.cosspace,
                  mesh_surface: bool = True,
                  mesh_tips: bool = True,
                  mesh_trailing_edge: bool = True,
                  mesh_symmetric: bool = True,
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Meshes the outer mold line surface of the wing.

        Uses the `(points, faces)` standard mesh format. For reference on this format, see the documentation in
        `aerosandbox.geometry.mesh_utilities`.

        Order of faces:

            * On the right wing (or, if `Wing.symmetric` is `False`, just the wing itself):

                * If `mesh_surface` is `True`:

                    * First face is nearest the top-side trailing edge of the wing root.

                    * Proceeds chordwise, along the upper surface of the wing from back to front. Upon reaching the
                    leading edge, continues along the lower surface of the wing from front to back.

                    * Then, repeats this process for the next spanwise slice of the wing, and so on.

                * If `mesh_trailing_edge` is `True`:

                    * Continues by meshing the trailing edge of the wing. Meshes the inboard trailing edge first, then
                    proceeds spanwise to the outboard trailing edge.

                * If `mesh_tips` is `True`:

                    * Continues by meshing the wing tips. Meshes the inboard tip first, then meshes the outboard tip.

                    * Within each tip, meshes from the

        Args:

            method: One of the following options, as a string:

                * "tri": Triangular mesh.

                * "quad": Quadrilateral mesh.

            chordwise_resolution: Number of points to use per wing chord, per wing section.

            chordwise_spacing_function_per_side: A function that determines how to space points in the chordwise
            direction along the top and bottom surfaces. Common values would be `np.linspace` or `np.cosspace`,
            but it can be any function with the call signature `f(a, b, n)` that returns a spaced array of `n` points
            between `a` and `b`. [function]

            mesh_surface: If True, includes the actual wing surface in the mesh.

            mesh_tips: If True, includes the wing tips (both on the inboard-most section and on the outboard-most
            section) in the mesh.

            mesh_trailing_edge: If True, includes the wing trailing edge in the mesh, if the trailing-edge thickness
            is nonzero.

            mesh_symmetric: Has no effect if the wing is not symmetric. If the wing is symmetric this determines whether
            the generated mesh is also symmetric, or if if only one side of the wing (right side) is meshed.

        Returns: Standard unstructured mesh format: A tuple of `points` and `faces`, where:

            * `points` is a `n x 3` array of points, where `n` is the number of points in the mesh.

            * `faces` is a `m x 3` array of faces if `method` is "tri", or a `m x 4` array of faces if `method` is "quad".

                * Each row of `faces` is a list of indices into `points`, which specifies a face.

        """

        airfoil_nondim_coordinates = np.array([
            xsec.airfoil
            .repanel(
                n_points_per_side=chordwise_resolution + 1,
                spacing_function_per_side=chordwise_spacing_function_per_side,
            )
            .coordinates
            for xsec in self.xsecs
        ])

        x_nondim = airfoil_nondim_coordinates[:, :, 0].T
        y_nondim = airfoil_nondim_coordinates[:, :, 1].T

        spanwise_strips = []
        for x_n, y_n in zip(x_nondim, y_nondim):
            spanwise_strips.append(
                np.stack(
                    self.mesh_line(
                        x_nondim=x_n,
                        z_nondim=y_n,
                        add_camber=False,
                    ),
                    axis=0
                )
            )

        points = np.concatenate(spanwise_strips, axis=0)

        faces = []

        num_i = (len(self.xsecs) - 1)
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

        if mesh_symmetric and self.symmetric:
            flipped_points = np.multiply(
                points,
                np.array([
                    [1, -1, 1]
                ])
            )

            points, faces = mesh_utils.stack_meshes(
                (points, faces),
                (flipped_points, faces)
            )

        return points, faces

    def mesh_thin_surface(self,
                          method="tri",
                          chordwise_resolution: int = 36,
                          chordwise_spacing_function: Callable[[float, float, float], np.ndarray] = np.cosspace,
                          add_camber: bool = True,
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Meshes the mean camber line of the wing as a thin-sheet body.

        Uses the `(points, faces)` standard mesh format. For reference on this format, see the documentation in
        `aerosandbox.geometry.mesh_utilities`.

        Order of faces:
            * On the right wing (or, if `Wing.symmetric` is `False`, just the wing itself):
                * First face is the face nearest the leading edge of the wing root.
                * Proceeds along a chordwise strip to the trailing edge.
                * Then, goes to the subsequent spanwise location and does another chordwise strip, et cetera until
                  we get to the wing tip.
            * On the left wing (applicable only if `Wing.symmetric` is `True`):
                * Same order: Starts at the root leading edge, goes in chordwise strips.

        Order of vertices within each face:
            * On the right wing (or, if `Wing.symmetric` is `False`, just the wing itself):
                * Front-left
                * Back-left
                * Back-right
                * Front-right
            * On the left wing (applicable only if `Wing.symmetric` is `True`):
                * Front-left
                * Back-left
                * Back-right
                * Front-right

        Args:

            method: A string, which determines whether to mesh the fuselage as a series of quadrilaterals or triangles.

                * "quad" meshes the fuselage as a series of quadrilaterals.

                * "tri" meshes the fuselage as a series of triangles.

            chordwise_resolution: Determines the number of chordwise panels to use in the meshing. [int]

            chordwise_spacing_function: Determines how to space the chordwise panels. Can be `np.linspace` or
            `np.cosspace`, or any other function of the call signature `f(a, b, n)` that returns a spaced array of
            `n` points between `a` and `b`. [function]

            add_camber: Controls whether to mesh the thin surface with camber (i.e., mean camber line), or to just
            mesh the flat planform. [bool]

        Returns: Standard unstructured mesh format: A tuple of `points` and `faces`, where:

            * `points` is a `n x 3` array of points, where `n` is the number of points in the mesh.

            * `faces` is a `m x 3` array of faces if `method` is "tri", or a `m x 4` array of faces if `method` is "quad".

                * Each row of `faces` is a list of indices into `points`, which specifies a face.


        """
        x_nondim = chordwise_spacing_function(
            0,
            1,
            chordwise_resolution + 1
        )

        spanwise_strips = []
        for x_n in x_nondim:
            spanwise_strips.append(
                np.stack(
                    self.mesh_line(
                        x_nondim=x_n,
                        z_nondim=0,
                        add_camber=add_camber,
                    ),
                    axis=0
                )
            )

        points = np.concatenate(spanwise_strips)

        faces = []

        num_i = np.length(spanwise_strips[0])  # spanwise
        num_j = np.length(spanwise_strips)  # chordwise

        def index_of(iloc, jloc):
            return iloc + jloc * num_i

        def add_face(*indices):
            entry = list(indices)
            if method == "quad":
                faces.append(entry)
            elif method == "tri":
                faces.append([entry[0], entry[1], entry[3]])
                faces.append([entry[1], entry[2], entry[3]])

        for i in range(num_i - 1):
            for j in range(num_j - 1):
                add_face(  # On right wing:
                    index_of(i, j),  # Front-left
                    index_of(i, j + 1),  # Back-left
                    index_of(i + 1, j + 1),  # Back-right
                    index_of(i + 1, j),  # Front-right
                )

        if self.symmetric:
            index_offset = np.length(points)

            points = np.concatenate([
                points,
                np.multiply(points, np.array([[1, -1, 1]]))
            ])

            def index_of(iloc, jloc):
                return index_offset + iloc + jloc * num_i

            for i in range(num_i - 1):
                for j in range(num_j - 1):
                    add_face(  # On left wing:
                        index_of(i + 1, j),  # Front-left
                        index_of(i + 1, j + 1),  # Back-left
                        index_of(i, j + 1),  # Back-right
                        index_of(i, j),  # Front-right
                    )

        faces = np.array(faces)

        return points, faces

    def mesh_line(self,
                  x_nondim: Union[float, List[float]] = 0.25,
                  z_nondim: Union[float, List[float]] = 0,
                  add_camber: bool = True,
                  ) -> List[np.ndarray]:
        """
        Meshes a line that goes through each of the WingXSec objects in this wing.

        Args:

            x_nondim: The nondimensional (chord-normalized) x-coordinate that the line should go through. Can either
            be a single value used at all cross-sections, or can be an iterable of values to be used at the
            respective cross-sections.

            z_nondim: The nondimensional (chord-normalized) y-coordinate that the line should go through. Here,
            y-coordinate means the "vertical" component (think standard 2D airfoil axes). Can either be a single
            value used at all cross-sections, or can be an iterable of values to be used at the respective cross
            sections.

            add_camber: Controls whether the camber of each cross-section's airfoil should be added to the line or
            not. Essentially modifies `z_nondim` to be `z_nondim + camber`.

        Returns: A list of points, where each point is a 3-element array of the form `[x, y, z]`. Goes from the root
        to the tip. Ignores any wing symmetry (e.g., only gives one side).

        """
        points_on_line: List[np.ndarray] = []

        try:
            if len(x_nondim) != len(self.xsecs):
                raise ValueError(
                    f"If `x_nondim` is an iterable, it should be the same length as `Wing.xsecs` ({len(self.xsecs)})."
                )
        except TypeError:
            pass

        try:
            if len(z_nondim) != len(self.xsecs):
                raise ValueError(
                    f"If `z_nondim` is an iterable, it should be the same length as `Wing.xsecs` ({len(self.xsecs)})."
                )
        except TypeError:
            pass

        for i, xsec in enumerate(self.xsecs):

            try:
                xsec_x_nondim = x_nondim[i]
            except (TypeError, IndexError):
                xsec_x_nondim = x_nondim

            try:
                xsec_z_nondim = z_nondim[i]
            except (TypeError, IndexError):
                xsec_z_nondim = z_nondim

            if add_camber:
                xsec_z_nondim = xsec_z_nondim + xsec.airfoil.local_camber(x_over_c=x_nondim)

            points_on_line.append(
                self._compute_xyz_of_WingXSec(
                    i,
                    x_nondim=xsec_x_nondim,
                    z_nondim=xsec_z_nondim,
                )
            )

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
        return Airplane(wings=[self]).draw(*args, **kwargs)

    def draw_wireframe(self, *args, **kwargs):
        """
        An alias to the more general Airplane.draw_wireframe() method. See there for documentation.

        Args:
            *args: Arguments to pass through to Airplane.draw_wireframe()
            **kwargs: Keyword arguments to pass through to Airplane.draw_wireframe()

        Returns: Same return as Airplane.draw_wireframe()

        """
        from aerosandbox.geometry.airplane import Airplane
        return Airplane(wings=[self]).draw_wireframe(*args, **kwargs)

    def draw_three_view(self, *args, **kwargs):
        """
        An alias to the more general Airplane.draw_three_view() method. See there for documentation.

        Args:
            *args: Arguments to pass through to Airplane.draw_three_view()
            **kwargs: Keyword arguments to pass through to Airplane.draw_three_view()

        Returns: Same return as Airplane.draw_three_view()

        """
        from aerosandbox.geometry.airplane import Airplane
        return Airplane(wings=[self]).draw_three_view(*args, **kwargs)

    def subdivide_sections(self,
                           ratio: int,
                           spacing_function: Callable[[float, float, float], np.ndarray] = np.linspace
                           ) -> "Wing":
        """
        Generates a new Wing that subdivides the existing sections of this Wing into several smaller ones. Splits
        each section into N=`ratio` smaller sub-sections by inserting new cross-sections (xsecs) as needed.

        This can allow for finer aerodynamic resolution of sectional properties in certain analyses.

        Args:

            ratio: The number of new sections to split each old section into.

            spacing_function: A function that takes in three arguments: the start, end, and number of points to generate.

                The default is `np.linspace`, which generates a linearly-spaced array of points.

                Other options include `np.cosspace`, which generates a cosine-spaced array of points.

        Returns: A new Wing object with subdivided sections.

        """
        if not (ratio >= 2 and isinstance(ratio, int)):
            raise ValueError("`ratio` must be an integer greater than or equal to 2.")

        new_xsecs = []
        span_fractions_along_section = spacing_function(0, 1, ratio + 1)[:-1]

        for xsec_a, xsec_b in zip(self.xsecs[:-1], self.xsecs[1:]):
            for s in span_fractions_along_section:
                a_weight = 1 - s
                b_weight = s

                if xsec_a.airfoil == xsec_b.airfoil:
                    blended_airfoil = xsec_a.airfoil
                elif a_weight == 1:
                    blended_airfoil = xsec_a.airfoil
                elif b_weight == 1:
                    blended_airfoil = xsec_b.airfoil
                else:
                    blended_airfoil = xsec_a.airfoil.blend_with_another_airfoil(
                        airfoil=xsec_b.airfoil,
                        blend_fraction=b_weight
                    )

                new_xsecs.append(
                    WingXSec(
                        xyz_le=xsec_a.xyz_le * a_weight + xsec_b.xyz_le * b_weight,
                        chord=xsec_a.chord * a_weight + xsec_b.chord * b_weight,
                        twist=xsec_a.twist * a_weight + xsec_b.twist * b_weight,
                        airfoil=blended_airfoil,
                        control_surfaces=xsec_a.control_surfaces,
                        analysis_specific_options=xsec_a.analysis_specific_options,
                    )
                )

        new_xsecs.append(self.xsecs[-1])

        return Wing(
            name=self.name,
            xsecs=new_xsecs,
            symmetric=self.symmetric,
            analysis_specific_options=self.analysis_specific_options
        )

    def _compute_xyz_le_of_WingXSec(self, index: int):
        return self.xsecs[index].xyz_le

    def _compute_xyz_te_of_WingXSec(self, index: int):
        return self._compute_xyz_of_WingXSec(
            index,
            x_nondim=1,
            z_nondim=0,
        )

    def _compute_xyz_of_WingXSec(self,
                                 index,
                                 x_nondim,
                                 z_nondim,
                                 ):
        xg_local, yg_local, zg_local = self._compute_frame_of_WingXSec(index)
        origin = self.xsecs[index].xyz_le
        xsec = self.xsecs[index]
        return origin + (
                x_nondim * xsec.chord * xg_local +
                z_nondim * xsec.chord * zg_local
        )

    def _compute_frame_of_WingXSec(
            self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the local reference frame associated with a particular cross-section (XSec) of this wing.

        Args:

            index: Which cross-section (as indexed in Wing.xsecs) should we get the frame of?

        Returns:

            A tuple of (xg_local, yg_local, zg_local), where each entry refers to the respective (normalized) axis of
            the local reference frame of the WingXSec. Given in geometry axes.

        """

        def project_to_YZ_plane_and_normalize(vector):
            YZ_magnitude = (vector[1] ** 2 + vector[2] ** 2) ** 0.5
            return np.array([0, vector[1], vector[2]]) / YZ_magnitude

        ### Compute the untwisted reference frame
        xg_local = np.array([1, 0, 0])
        if index == 0:
            yg_local = project_to_YZ_plane_and_normalize(
                self.xsecs[1].xyz_le - self.xsecs[0].xyz_le
            )
            z_scale = 1
        elif index == len(self.xsecs) - 1 or index == -1:
            yg_local = project_to_YZ_plane_and_normalize(
                self.xsecs[-1].xyz_le - self.xsecs[-2].xyz_le
            )
            z_scale = 1
        else:
            vector_before = project_to_YZ_plane_and_normalize(
                self.xsecs[index].xyz_le - self.xsecs[index - 1].xyz_le
            )
            vector_after = project_to_YZ_plane_and_normalize(
                self.xsecs[index + 1].xyz_le - self.xsecs[index].xyz_le
            )
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
        sections are different! cross-sections, or xsecs, are the vertices, and sections are the parts in between. In
        other words, a wing with N cross-sections (xsecs) will always have N-1 sections.

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
    Definition for a wing cross-section ("X-section").
    """

    def __init__(self,
                 xyz_le: Union[np.ndarray, List] = None,
                 chord: float = 1.,
                 twist: float = 0.,
                 airfoil: Airfoil = None,
                 control_surfaces: Optional[List['ControlSurface']] = None,
                 analysis_specific_options: Optional[Dict[type, Dict[str, Any]]] = None,
                 **deprecated_kwargs,
                 ):
        """
        Defines a new wing cross-section.

        Args:

            xyz_le: An array-like that represents the xyz-coordinates of the leading edge of the cross-section, in
            geometry axes.

            chord: Chord of the wing at this cross-section.

            twist: Twist angle, in degrees, as defined about the leading edge.

                The twist axis is computed with the following procedure:

                    * The quarter-chord point of this WingXSec and the following one are identified.

                    * A line is drawn connecting them, and it is normalized to a unit direction vector.

                    * That direction vector is projected onto the geometry Y-Z plane.

                    * That direction vector is now the twist axis.

            airfoil: Airfoil associated with this cross-section. [aerosandbox.Airfoil]

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
        if xyz_le is None:
            xyz_le = np.array([0., 0., 0.])
        if airfoil is None:
            import warnings
            warnings.warn(
                "An airfoil is not specified for WingXSec. Defaulting to NACA 0012.",
                stacklevel=2
            )
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
        if 'twist_angle' in deprecated_kwargs.keys():
            import warnings
            warnings.warn(
                "DEPRECATED: 'twist_angle' has been renamed 'twist', and will break in future versions.",
                stacklevel=2
            )
            self.twist = deprecated_kwargs['twist_angle']
        if (
                'control_surface_is_symmetric' in locals() or
                'control_surface_hinge_point' in locals() or
                'control_surface_deflection' in locals()
        ):
            import warnings
            warnings.warn(
                "DEPRECATED: Define control surfaces using the `control_surfaces` parameter, which takes in a list of asb.ControlSurface objects.",
                stacklevel=2
            )
            if 'control_surface_is_symmetric' not in locals():
                control_surface_is_symmetric = True
            if 'control_surface_hinge_point' not in locals():
                control_surface_hinge_point = 0.75
            if 'control_surface_deflection' not in locals():
                control_surface_deflection = 0

            self.control_surfaces.append(
                ControlSurface(
                    hinge_point=control_surface_hinge_point,
                    symmetric=control_surface_is_symmetric,
                    deflection=control_surface_deflection,
                )
            )

    def __repr__(self) -> str:
        return f"WingXSec (Airfoil: {self.airfoil.name}, chord: {self.chord}, twist: {self.twist})"

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

    def xsec_area(self):
        """
        Computes the WingXSec's cross-sectional (xsec) area.

        Returns: The (dimensional) cross-sectional area of the WingXSec.
        """
        return self.airfoil.area() * self.chord ** 2


class ControlSurface(AeroSandboxObject):
    """
    Definition for a control surface, which is attached to a particular WingXSec via WingXSec's `control_surfaces=[]` parameter.
    """

    def __init__(self,
                 name: str = "Untitled",
                 symmetric: bool = True,
                 deflection: float = 0.0,
                 hinge_point: float = 0.75,
                 trailing_edge: bool = True,
                 analysis_specific_options: Optional[Dict[type, Dict[str, Any]]] = None,
                 ):
        """
        Define a new control surface.

        Args:

            name: Name of the control surface [optional]. It can help when debugging to give each control surface a
            sensible name.

            symmetric: Is the control surface symmetric? If False, control surface is anti-symmetric. (e.g.,
            True for flaps, False for ailerons.)

            hinge_point: The location of the control surface hinge, as a fraction of chord. A float in the range of 0 to 1.

            deflection: Control deflection, in degrees. Downwards-positive.

            trailing_edge: Is the control surface on the trailing edge? If False, control surface is on the leading
            edge. (e.g., True for flaps, False for slats.). Support is experimental for leading-edge control
            surfaces, be aware that not all modules may treat this correctly.

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
        self.symmetric = symmetric
        self.deflection = deflection
        self.hinge_point = hinge_point
        self.trailing_edge = trailing_edge
        self.analysis_specific_options = analysis_specific_options

    def __repr__(self) -> str:
        keys = [
            "name",
            "symmetric",
            "deflection",
            "hinge_point",
        ]
        if not self.trailing_edge:
            keys += ["trailing_edge"]

        info = ", ".join([
            f"{k}={self.__dict__[k]}"
            for k in keys
        ])

        return f"ControlSurface ({info})"


if __name__ == '__main__':
    wing = Wing(
        xsecs=[
            WingXSec(
                xyz_le=[0, 0, 0],
                chord=1,
                airfoil=Airfoil("naca4412"),
                twist=0,
                control_surfaces=[
                    ControlSurface(
                        name="Elevator",
                        trailing_edge=True,
                        hinge_point=0.75,
                        deflection=5
                    )
                ]
            ),
            WingXSec(
                xyz_le=[0.5, 1, 0],
                chord=0.5,
                airfoil=Airfoil("naca4412"),
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
    # wing.subdivide_sections(5).draw()
