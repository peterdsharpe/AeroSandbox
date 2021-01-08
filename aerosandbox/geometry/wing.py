from aerosandbox.geometry.common import *
from typing import List


class Wing(AeroSandboxObject):
    """
    Definition for a wing.
    If the wing is symmetric across the XZ plane, just define the right half and supply "symmetric = True" in the constructor.
    If the wing is not symmetric across the XZ plane, just define the wing.
    """

    def __init__(self,
                 name: str = "Untitled Wing",  # It can help when debugging to give each wing a sensible name.
                 x_le: float = 0,  # Will translate all of the xsecs of the wing. Useful for moving the wing around.
                 y_le: float = 0,  # Will translate all of the xsecs of the wing. Useful for moving the wing around.
                 z_le: float = 0,  # Will translate all of the xsecs of the wing. Useful for moving the wing around.
                 xsecs: List['WingXSec'] = [],  # This should be a list of WingXSec objects.
                 symmetric: bool = False,  # Is the wing symmetric across the XZ plane?
                 chordwise_panels: int = 8,  # The number of chordwise panels to be used in VLM and panel analyses.
                 chordwise_spacing: str = "cosine",  # Can be 'cosine' or 'uniform'. Highly recommended to be cosine.
                 ):
        self.name = name
        self.xyz_le = cas.vertcat(x_le, y_le, z_le)
        self.xsecs = xsecs
        self.symmetric = symmetric
        self.chordwise_panels = chordwise_panels
        self.chordwise_spacing = chordwise_spacing

    def __repr__(self) -> str:
        return "Wing %s (%i xsecs, %s)" % (
            self.name,
            len(self.xsecs),
            "symmetric" if self.symmetric else "not symmetric"
        )

    def area(self,
             type: str = "wetted"
             ) -> float:
        """
        Returns the area, with options for various ways of measuring this.
         * wetted: wetted area
         * projected: area projected onto the XY plane (top-down view)
        :param type:
        :return:
        """
        area = 0
        for i in range(len(self.xsecs) - 1):
            chord_eff = (self.xsecs[i].chord
                         + self.xsecs[i + 1].chord) / 2
            this_xyz_te = self.xsecs[i].xyz_te()
            that_xyz_te = self.xsecs[i + 1].xyz_te()
            if type == "wetted":
                span_le_eff = cas.sqrt(
                    (self.xsecs[i].xyz_le[1] - self.xsecs[i + 1].xyz_le[1]) ** 2 +
                    (self.xsecs[i].xyz_le[2] - self.xsecs[i + 1].xyz_le[2]) ** 2
                )
                span_te_eff = cas.sqrt(
                    (this_xyz_te[1] - that_xyz_te[1]) ** 2 +
                    (this_xyz_te[2] - that_xyz_te[2]) ** 2
                )
            elif type == "projected":
                span_le_eff = cas.fabs(
                    self.xsecs[i].xyz_le[1] - self.xsecs[i + 1].xyz_le[1]
                )
                span_te_eff = cas.fabs(
                    this_xyz_te[1] - that_xyz_te[1]
                )
            else:
                raise ValueError("Bad value of 'type'!")

            span_eff = (span_le_eff + span_te_eff) / 2
            area += chord_eff * span_eff
        if self.symmetric:
            area *= 2
        return area

    def span(self,
             type: str = "wetted"
             ) -> float:
        """
        Returns the span, with options for various ways of measuring this.
         * wetted: Adds up YZ-distances of each section piece by piece
         * yz: YZ-distance between the root and tip of the wing
         * y: Y-distance between the root and tip of the wing
         * z: Z-distance between the root and tip of the wing
         * y-full: Y-distance between the centerline and the tip of the wing
        If symmetric, this is doubled to obtain the full span.
        :param type: One of the above options, as a string.
        :return: span
        """
        if type == "wetted":
            span = 0
            for i in range(len(self.xsecs) - 1):
                sect1_xyz_le = self.xsecs[i].xyz_le
                sect2_xyz_le = self.xsecs[i + 1].xyz_le
                sect1_xyz_te = self.xsecs[i].xyz_te()
                sect2_xyz_te = self.xsecs[i + 1].xyz_te()

                span_le = cas.sqrt(
                    (sect1_xyz_le[1] - sect2_xyz_le[1]) ** 2 +
                    (sect1_xyz_le[2] - sect2_xyz_le[2]) ** 2
                )
                span_te = cas.sqrt(
                    (sect1_xyz_te[1] - sect2_xyz_te[1]) ** 2 +
                    (sect1_xyz_te[2] - sect2_xyz_te[2]) ** 2
                )
                span_eff = (span_le + span_te) / 2
                span += span_eff

        elif type == "yz":
            root = self.xsecs[0]  # type: WingXSec
            tip = self.xsecs[-1]  # type: WingXSec
            span = cas.sqrt(
                (root.xyz_le[1] - tip.xyz_le[1]) ** 2 +
                (root.xyz_le[2] - tip.xyz_le[2]) ** 2
            )
        elif type == "y":
            root = self.xsecs[0]  # type: WingXSec
            tip = self.xsecs[-1]  # type: WingXSec
            span = cas.fabs(
                tip.xyz_le[1] - root.xyz_le[1]
            )
        elif type == "z":
            root = self.xsecs[0]  # type: WingXSec
            tip = self.xsecs[-1]  # type: WingXSec
            span = cas.fabs(
                tip.xyz_le[2] - root.xyz_le[2]
            )
        elif type == "y-full":
            tip = self.xsecs[-1]
            span = cas.fabs(
                tip.xyz_le[1] - 0
            )
        else:
            raise ValueError("Bad value of 'type'!")
        if self.symmetric:
            span *= 2
        return span

    def aspect_ratio(self) -> float:
        # Returns the aspect ratio (b^2/S).
        # Uses the full span and the full area if symmetric.
        return self.span() ** 2 / self.area()

    def has_symmetric_control_surfaces(self) -> bool:
        # Returns a boolean of whether the wing is totally symmetric (i.e.), every xsec has control_surface_type = "symmetric".
        for xsec in self.xsecs:
            if not xsec.control_surface_type == "symmetric":
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
        area = 0
        sum_dMAC_dA = 0

        for i, xsec_a in enumerate(self.xsecs[:-1]):
            xsec_b = self.xsecs[i + 1]

            section_span = cas.sqrt(
                cas.sumsqr(xsec_a.quarter_chord()[1:, :] - xsec_b.quarter_chord()[1:, :])
            )

            section_taper_ratio = xsec_b.chord / xsec_a.chord
            section_area = (xsec_a.chord + xsec_b.chord) / 2 * section_span
            section_MAC = (2 / 3) * xsec_a.chord * (
                    (1 + section_taper_ratio + section_taper_ratio ** 2) /
                    (1 + section_taper_ratio)
            )
            area = area + section_area
            sum_dMAC_dA = sum_dMAC_dA + section_area * section_MAC

        MAC = sum_dMAC_dA / area

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
        Returns the mean twist angle (in degrees) of the wing, weighted by span.
        You can think of it as \int_{b}(twist)db, where b is span.
        WARNING: This function's output is only exact in the case where all of the cross sections have the same twist axis!
        :return: mean twist angle (in degrees)
        """
        # First, find the spans
        span = []
        for i in range(len(self.xsecs) - 1):
            sect1_xyz_le = self.xsecs[i].xyz_le
            sect2_xyz_le = self.xsecs[i + 1].xyz_le
            sect1_xyz_te = self.xsecs[i].xyz_te()
            sect2_xyz_te = self.xsecs[i + 1].xyz_te()

            span_le = cas.sqrt(
                (sect1_xyz_le[1] - sect2_xyz_le[1]) ** 2 +
                (sect1_xyz_le[2] - sect2_xyz_le[2]) ** 2
            )
            span_te = cas.sqrt(
                (sect1_xyz_te[1] - sect2_xyz_te[1]) ** 2 +
                (sect1_xyz_te[2] - sect2_xyz_te[2]) ** 2
            )
            span_eff = (span_le + span_te) / 2
            span.append(span_eff)

        # Then, find the twist-span product
        twist_span_product = 0
        for i in range(len(self.xsecs)):
            xsec = self.xsecs[i]
            if i > 0:
                twist_span_product += xsec.twist * span[i - 1] / 2
            if i < len(self.xsecs) - 1:
                twist_span_product += xsec.twist * span[i] / 2

        # Then, divide
        mean_twist = twist_span_product / cas.sum1(cas.vertcat(*span))
        return mean_twist

    def mean_sweep_angle(self) -> float:
        """
        Returns the mean quarter-chord sweep angle (in degrees) of the wing, relative to the x-axis.
        Positive sweep is backwards, negative sweep is forward.
        :return:
        """
        root_quarter_chord = 0.75 * self.xsecs[0].xyz_le + 0.25 * self.xsecs[0].xyz_te()
        tip_quarter_chord = 0.75 * self.xsecs[-1].xyz_le + 0.25 * self.xsecs[-1].xyz_te()

        vec = tip_quarter_chord - root_quarter_chord
        vec_norm = vec / cas.norm_2(vec)

        sin_sweep = vec_norm[0]  # from dot product with x_hat

        sweep_deg = cas.asin(sin_sweep) * 180 / cas.pi

        return sweep_deg

    def approximate_center_of_pressure(self) -> cas.DM:
        """
        Returns the approximate location of the center of pressure. Given as the area-weighted quarter chord of the wing.
        :return: [x, y, z] of the approximate center of pressure
        """
        areas = []
        quarter_chord_centroids = []
        for i in range(len(self.xsecs) - 1):
            # Find areas
            chord_eff = (self.xsecs[i].chord
                         + self.xsecs[i + 1].chord) / 2
            this_xyz_te = self.xsecs[i].xyz_te()
            that_xyz_te = self.xsecs[i + 1].xyz_te()
            span_le_eff = cas.sqrt(
                (self.xsecs[i].xyz_le[1] - self.xsecs[i + 1].xyz_le[1]) ** 2 +
                (self.xsecs[i].xyz_le[2] - self.xsecs[i + 1].xyz_le[2]) ** 2
            )
            span_te_eff = cas.sqrt(
                (this_xyz_te[1] - that_xyz_te[1]) ** 2 +
                (this_xyz_te[2] - that_xyz_te[2]) ** 2
            )
            span_eff = (span_le_eff + span_te_eff) / 2

            areas.append(chord_eff * span_eff)

            # Find quarter-chord centroids of each section
            quarter_chord_centroids.append(
                (
                        0.75 * self.xsecs[i].xyz_le + 0.25 * self.xsecs[i].xyz_te() +
                        0.75 * self.xsecs[i + 1].xyz_le + 0.25 * self.xsecs[i + 1].xyz_te()
                ) / 2 + self.xyz_le
            )

        areas = cas.vertcat(*areas)
        quarter_chord_centroids = cas.transpose(cas.horzcat(*quarter_chord_centroids))

        total_area = cas.sum1(areas)
        approximate_cop = cas.sum1(areas / cas.sum1(areas) * quarter_chord_centroids)

        if self.symmetric:
            approximate_cop[:, 1] = 0

        return approximate_cop

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
                 x_le: float = 0,  # Coordinate of the leading edge of the cross section, relative to the wing's datum.
                 y_le: float = 0,  # Coordinate of the leading edge of the cross section, relative to the wing's datum.
                 z_le: float = 0,  # Coordinate of the leading edge of the cross section, relative to the wing's datum.
                 chord: float = 0,  # Chord of the wing at this cross section
                 twist: float = 0,  # Twist always occurs about the leading edge!
                 twist_axis: cas.DM = cas.DM([0, 1, 0]),  # By default, always twists about the Y-axis.
                 airfoil: 'Airfoil' = None,  # The airfoil to be used at this cross section.
                 control_surface_type: str = "symmetric",  # TODO change this to a boolean
                 # Can be "symmetric" or "asymmetric". Symmetric is like flaps, asymmetric is like an aileron.
                 control_surface_hinge_point: float = 0.75,
                 # The location of the control surface hinge, as a fraction of chord.
                 # Point at which the control surface is applied, as a fraction of chord.
                 control_surface_deflection: float = 0,  # Control deflection, in degrees. Downwards-positive.
                 spanwise_panels: int = 8,
                 # The number of spanwise panels to be used between this cross section and the next one.
                 spanwise_spacing: str = "cosine"  # Can be 'cosine' or 'uniform'. Highly recommended to be cosine.
                 ):
        if airfoil is None:
            raise ValueError("'airfoil' argument missing! (Needs an object of Airfoil type)")

        self.x_le = x_le
        self.y_le = y_le
        self.z_le = z_le

        self.chord = chord
        self.twist = twist
        self.twist_axis = twist_axis
        self.airfoil = airfoil
        self.control_surface_type = control_surface_type
        self.control_surface_hinge_point = control_surface_hinge_point
        self.control_surface_deflection = control_surface_deflection
        self.spanwise_panels = spanwise_panels
        self.spanwise_spacing = spanwise_spacing

        self.xyz_le = cas.vertcat(x_le, y_le, z_le)

    def __repr__(self) -> str:
        return "WingXSec (airfoil = %s, chord = %f, twist = %f)" % (
            self.airfoil.name,
            self.chord,
            self.twist
        )

    def quarter_chord(self) -> cas.DM:
        """
        Returns the (wing-relative) coordinates of the quarter chord of the cross section.
        """
        return 0.75 * self.xyz_le + 0.25 * self.xyz_te()

    def xyz_te(self) -> cas.DM:
        """
        Returns the (wing-relative) coordinates of the trailing edge of the cross section.
        """
        rot = angle_axis_rotation_matrix(self.twist * cas.pi / 180, self.twist_axis)
        xyz_te = self.xyz_le + rot @ cas.vertcat(self.chord, 0, 0)
        return xyz_te
