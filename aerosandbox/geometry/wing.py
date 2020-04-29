from aerosandbox.geometry.common import *


class Wing(AeroSandboxObject):
    """
    Definition for a wing.
    If the wing is symmetric across the XZ plane, just define the right half and supply "symmetric = True" in the constructor.
    If the wing is not symmetric across the XZ plane, just define the wing.
    """

    def __init__(self,
                 name="Untitled Wing",  # It can help when debugging to give each wing a sensible name.
                 x_le=0,  # Will translate all of the xsecs of the wing. Useful for moving the wing around.
                 y_le=0,  # Will translate all of the xsecs of the wing. Useful for moving the wing around.
                 z_le=0,  # Will translate all of the xsecs of the wing. Useful for moving the wing around.
                 xsecs=[],  # This should be a list of WingXSec objects.
                 symmetric=False,  # Is the wing symmetric across the XZ plane?
                 chordwise_panels=8,  # The number of chordwise panels to be used in VLM and panel analyses.
                 chordwise_spacing="cosine",  # Can be 'cosine' or 'uniform'. Highly recommended to be cosine.
                 ):
        self.name = name
        self.xyz_le = cas.vertcat(x_le, y_le, z_le)
        self.xsecs = xsecs
        self.symmetric = symmetric
        self.chordwise_panels = chordwise_panels
        self.chordwise_spacing = chordwise_spacing

    def __repr__(self):
        return "Wing %s (%i xsecs, %s)" % (
            self.name,
            len(self.xsecs),
            "symmetric" if self.symmetric else "not symmetric"
        )

    def area(self,
             type="wetted"
             ):
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
             type="wetted"
             ):
        """
        Returns the span, with options for various ways of measuring this.
         * wetted: Adds up YZ-distances of each section piece by piece
         * yz: YZ-distance between the root and tip of the wing
         * y: Y-distance between the root and tip of the wing
         * z: Z-distance between the root and tip of the wing
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
        else:
            raise ValueError("Bad value of 'type'!")
        if self.symmetric:
            span *= 2
        return span

    def aspect_ratio(self):
        # Returns the aspect ratio (b^2/S).
        # Uses the full span and the full area if symmetric.
        return self.span() ** 2 / self.area()

    def has_symmetric_control_surfaces(self):
        # Returns a boolean of whether the wing is totally symmetric (i.e.), every xsec has control_surface_type = "symmetric".
        for xsec in self.xsecs:
            if not xsec.control_surface_type == "symmetric":
                return False
        return True

    def mean_geometric_chord(self):
        """
        Returns the mean geometric chord of the wing (S/b).
        :return:
        """
        return self.area() / self.span()

    def mean_twist_angle(self):
        """
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

    def mean_sweep_angle(self):
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

    def approximate_center_of_pressure(self):
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


class WingXSec(AeroSandboxObject):
    """
    Definition for a wing cross section ("X-section").
    """

    def __init__(self,
                 x_le=0,  # Coordinate of the leading edge of the cross section, relative to the wing's datum.
                 y_le=0,  # Coordinate of the leading edge of the cross section, relative to the wing's datum.
                 z_le=0,  # Coordinate of the leading edge of the cross section, relative to the wing's datum.
                 chord=0,  # Chord of the wing at this cross section
                 twist=0,  # Twist always occurs about the leading edge!
                 twist_axis=cas.DM([0, 1, 0]),  # By default, always twists about the Y-axis.
                 airfoil=None,  # type: Airfoil # The airfoil to be used at this cross section.
                 control_surface_type="symmetric",
                 # Can be "symmetric" or "asymmetric". Symmetric is like flaps, asymmetric is like an aileron.
                 control_surface_hinge_point=0.75,  # The location of the control surface hinge, as a fraction of chord.
                 # Point at which the control surface is applied, as a fraction of chord.
                 control_surface_deflection=0,  # Control deflection, in degrees. Downwards-positive.
                 spanwise_panels=8,
                 # The number of spanwise panels to be used between this cross section and the next one.
                 spanwise_spacing="cosine"  # Can be 'cosine' or 'uniform'. Highly recommended to be cosine.
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

    def __repr__(self):
        return "WingXSec (airfoil = %s, chord = %f, twist = %f)" % (
            self.airfoil.name,
            self.chord,
            self.twist
        )

    def xyz_te(self):
        """
        Returns the (wing-relative) coordinates of the trailing edge of the cross section.
        """
        rot = angle_axis_rotation_matrix(self.twist * cas.pi / 180, self.twist_axis)
        xyz_te = self.xyz_le + rot @ cas.vertcat(self.chord, 0, 0)
        # xyz_te = self.xyz_le + self.chord * cas.vertcat(
        #     cas.cos(self.twist * cas.pi / 180),
        #     0,
        #     -cas.sin(self.twist * cas.pi / 180)
        # )
        return xyz_te
