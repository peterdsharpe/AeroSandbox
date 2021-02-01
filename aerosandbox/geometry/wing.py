from aerosandbox import AeroSandboxObject
from aerosandbox.geometry.common import *
from aerosandbox.optimization.math import *
from typing import List
from aerosandbox.geometry.airfoil import Airfoil
from numpy import pi


class Wing(AeroSandboxObject):
    """
    Definition for a wing.
    If the wing is symmetric across the XZ plane, just define the right half and supply "symmetric = True" in the constructor.
    If the wing is not symmetric across the XZ plane, just define the wing.
    """

    def __init__(self,
                 name: str = "Untitled Wing",
                 xyz_le: np.ndarray = array([0, 0, 0]),
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
        self.xyz_le = xyz_le
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
        sectional_spans = self.span(_sectional=True)
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

    def has_symmetric_control_surfaces(self) -> bool:
        # Returns a boolean of whether the wing is totally symmetric (i.e.), every xsec has symmetric control surfaces.
        if not self.symmetric:
            return False
        for xsec in self.xsecs:
            if not xsec.control_surface_is_symmetric:
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
        Returns the mean aerodynamic chord of the wing.
        """

        sectional_areas = self.area(_sectional=True)
        sectional_MACs = []

        for inner_xsec, outer_xsec in zip(self.xsecs[:-1], self.xsecs[1:]):

            section_taper_ratio = outer_xsec.chord / inner_xsec.chord
            section_MAC = (2 / 3) * inner_xsec.chord * (
                    (1 + section_taper_ratio + section_taper_ratio ** 2) /
                    (1 + section_taper_ratio)
            )

            sectional_MACs.append(section_MAC)

        sectional_MAC_area_products = [
            area * MAC
            for area, MAC in zip(
                sectional_areas,
                sectional_MACs
            )
        ]

        MAC = sum(sectional_MAC_area_products) / sum(sectional_areas)

        return MAC

    # def mean_aerodynamic_chord_location(self): # TODO verify and add
    #     """
    #     Returns the x and y LE position of the mean aerodynamic chord.
    #     Based on the same assumptions as wing.mean_aerodynamic_chord.
    #     """
    #     area = 0
    #     sum_dx_dA = 0
    #     sum_dy_dA = 0
    #     sum_dz_dA = 0
    #
    #     for i, xsec_a in enumerate(self.xsecs[:-1]):
    #         xsec_b = self.xsecs[i + 1]
    #
    #         c_r = xsec_a.chord
    #         c_t = xsec_b.chord
    #         x_r = xsec_a.x_le
    #         x_t = xsec_b.x_le
    #         y_r = xsec_a.y_le
    #         y_t = xsec_b.y_le
    #         z_r = xsec_a.z_le
    #         z_t = xsec_b.z_le
    #
    #         taper_ratio = c_t / c_r
    #         d_area = (c_r + c_t) * (y_t - y_r) / 2
    #         area = area + d_area
    #
    #         x_mac = x_r + (x_t - x_r) * (1 + 2 * taper_ratio) / (3 + 3 * taper_ratio)
    #         sum_dx_dA = d_area * x_mac + sum_dx_dA
    #         y_mac = y_r + (y_t - y_r) * (1 + 2 * taper_ratio) / (3 + 3 * taper_ratio)
    #         sum_dy_dA = d_area * y_mac + sum_dy_dA
    #         z_mac = z_r + (z_t - z_r) * (1 + 2 * taper_ratio) / (3 + 3 * taper_ratio)
    #         sum_dz_dA = d_area * z_mac + sum_dz_dA
    #
    #     x_mac = sum_dx_dA / area
    #     y_mac = sum_dy_dA / area
    #     z_mac = sum_dz_dA / area
    #
    #     return cas.vertcat(x_mac, y_mac, z_mac)

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
        vec_norm = vec / norm(vec)

        sin_sweep = vec_norm[0]  # from dot product with x_hat

        sweep_deg = np.arcsin(sin_sweep) * 180 / cas.pi

        return sweep_deg

    def aerodynamic_center(self) -> cas.DM:
        """
        Returns the approximate location of the aerodynamic center.
        Approximately computed as the area-weighted quarter chord of the wing.
        :return: [x, y, z] of the approximate center of pressure in global coordinates.
        """
        sectional_ACs = [
            (inner_xsec.quarter_chord() + outer_xsec.quarter_chord()) / 2
            for inner_xsec, outer_xsec in zip(
                self.xsecs[:-1],
                self.xsecs[1:]
            )
        ]
        sectional_areas = self.area(_sectional=True)

        sectional_AC_area_products = [
            AC * area
            for AC, area in zip(
                sectional_ACs, sectional_areas
            )
        ]

        mean_AC = sum(sectional_AC_area_products) / sum(sectional_areas)

        mean_AC += self.xyz_le

        if self.symmetric:
            mean_AC[1] = 0

        return mean_AC

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
                 xyz_le: np.ndarray = array([0, 0, 0]),
                 chord: float = 1.,
                 twist_angle: float = 0,
                 twist_axis: np.ndarray = array([0, 1, 0]),
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

        self.xyz_le = xyz_le
        self.chord = chord
        self.twist = twist_angle
        self.twist_axis = twist_axis
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
        rot = rotation_matrix_angle_axis(self.twist * pi / 180, self.twist_axis)
        xyz_te = self.xyz_le + rot @ array([self.chord, 0, 0])
        return xyz_te
