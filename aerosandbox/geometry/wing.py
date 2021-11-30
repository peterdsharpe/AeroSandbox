from aerosandbox import AeroSandboxObject
from aerosandbox.geometry.common import *
from typing import List
from aerosandbox.geometry.airfoil import Airfoil
from numpy import pi
import aerosandbox.numpy as np


class Wing(AeroSandboxObject):
    """
    Definition for a wing.

    If the wing is symmetric across the XZ plane, just define the right half and supply "symmetric = True" in the
    constructor.

    If the wing is not symmetric across the XZ plane, just define the wing.
    """

    def __init__(self,
                 name: str = "Untitled Wing",
                 xyz_le: np.ndarray = np.array([0, 0, 0]),
                 xsecs: List['WingXSec'] = [],
                 symmetric: bool = False,
                 ):
        """
        Initialize a new wing.
        Args:
            name: Name of the wing [optional]. It can help when debugging to give each wing a sensible name.
            xyz_le: xyz-coordinates of the datum point (typically the root leading edge) of the wing.
            xsecs: A list of wing cross ("X") sections in the form of WingXSec objects.
            symmetric: Is the wing symmetric across the XZ plane?
        """
        self.name = name
        self.xyz_le = np.array(xyz_le)
        self.xsecs = xsecs
        self.symmetric = symmetric

    def __repr__(self) -> str:
        n_xsecs = len(self.xsecs)
        symmetry_description = "symmetric" if self.symmetric else "asymmetric"
        return f"Wing '{self.name}' ({len(self.xsecs)} {'xsec' if n_xsecs == 1 else 'xsecs'}, {symmetry_description})"

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
            return self.xsecs[-1].quarter_chord()[1]

        sectional_spans = []

        for inner_xsec, outer_xsec in zip(self.xsecs[:-1], self.xsecs[1:]):
            quarter_chord_vector = outer_xsec.quarter_chord() - inner_xsec.quarter_chord()

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
        if not self.symmetric:
            return False
        for xsec in self.xsecs:
            if not (xsec.control_surface_is_symmetric or xsec.control_surface_deflection == 0):
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
        WARNING: This function's output is only exact in the case where all of the cross sections have the same twist axis!
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

    def mean_sweep_angle(self) -> float:
        """
        Returns the mean quarter-chord sweep angle (in degrees) of the wing, relative to the x-axis.
        Positive sweep is backwards, negative sweep is forward.
        :return:
        """
        root_quarter_chord = self.xsecs[0].quarter_chord()
        tip_quarter_chord = self.xsecs[-1].quarter_chord()

        vec = tip_quarter_chord - root_quarter_chord
        vec_norm = vec / np.linalg.norm(vec)

        sin_sweep = vec_norm[0]  # from dot product with x_hat

        sweep_deg = np.arcsind(sin_sweep)

        return sweep_deg

    def aerodynamic_center(self, chord_fraction: float = 0.25) -> np.ndarray:
        """
        Computes the location of the aerodynamic center of the wing.
        Uses the generalized methodology described here:
            https://core.ac.uk/download/pdf/79175663.pdf

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
            section_AC = section_MAC_le + np.array([ # TODO rotate this vector by the local twist angle
                chord_fraction * section_MAC_length,
                0,
                0
            ])

            sectional_ACs.append(section_AC)

        sectional_AC_area_products = [
            AC * area
            for AC, area in zip(
                sectional_ACs,
                sectional_areas,
            )
        ]

        aerodynamic_center = sum(sectional_AC_area_products) / sum(sectional_areas)

        aerodynamic_center += self.xyz_le

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


class WingXSec(AeroSandboxObject):
    """
    Definition for a wing cross section ("X-section").
    """

    def __init__(self,
                 xyz_le: np.ndarray = np.array([0, 0, 0]),
                 chord: float = 1.,
                 twist_angle: float = 0,
                 twist_axis: np.ndarray = np.array([0, 1, 0]),
                 airfoil: Airfoil = Airfoil("naca0012"),
                 control_surface_is_symmetric: bool = True,
                 control_surface_hinge_point: float = 0.75,
                 control_surface_deflection: float = 0.,
                 ):
        """
        Initialize a new wing cross section.
        Args:
            xyz_le: xyz-coordinates of the leading edge of the cross section, relative to the wing's datum.
            chord: Chord of the wing at this cross section
            twist_angle: Twist angle, in degrees, as defined about the leading edge.
            twist_axis: The twist axis vector, used if twist_angle != 0.
            airfoil: Airfoil associated with this cross section. [aerosandbox.Airfoil]
            control_surface_is_symmetric: Is the control surface symmetric? (e.g. True for flaps, False for ailerons.)
            control_surface_hinge_point: The location of the control surface hinge, as a fraction of chord.
            control_surface_deflection: Control deflection, in degrees. Downwards-positive.
        """

        self.xyz_le = np.array(xyz_le)
        self.chord = chord
        self.twist = twist_angle
        self.twist_axis = np.array(twist_axis)
        self.airfoil = airfoil
        self.control_surface_is_symmetric = control_surface_is_symmetric
        self.control_surface_hinge_point = control_surface_hinge_point
        self.control_surface_deflection = control_surface_deflection

    def __repr__(self) -> str:
        return f"WingXSec (Airfoil: {self.airfoil.name}, chord: {self.chord:.3f}, twist: {self.twist:.3f})"

    def quarter_chord(self) -> np.ndarray:
        """
        Returns the (wing-relative) coordinates of the quarter chord of the cross section.
        """
        return 0.75 * self.xyz_le + 0.25 * self.xyz_te()

    def xyz_te(self) -> np.ndarray:
        """
        Returns the (wing-relative) coordinates of the trailing edge of the cross section.
        """
        rot = np.rotation_matrix_3D(self.twist * pi / 180, self.twist_axis)
        xyz_te = self.xyz_le + rot @ np.array([self.chord, 0, 0])
        return xyz_te
