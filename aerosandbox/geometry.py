import plotly.express as px
import plotly.graph_objects as go
# Set the rendering to happen in browser
import plotly.io as pio
try:
    from xfoil import XFoil
    from xfoil import model as xfoil_model
except ModuleNotFoundError:
    pass

from .casadi_helpers import *

pio.renderers.default = "browser"


class Airplane:
    """
    Definition for an airplane (or other vehicle/item to analyze).
    """

    def __init__(self,
                 name="Untitled",  # A sensible name for your airplane.
                 x_ref=0,  # Ref. point for moments; should be the center of gravity.
                 y_ref=0,  # Ref. point for moments; should be the center of gravity.
                 z_ref=0,  # Ref. point for moments; should be the center of gravity.
                 mass_props=None,  # An object of MassProps type; only needed for dynamic analysis
                 # If xyz_ref is not set, but mass_props is, the xyz_ref will be taken from the CG there.
                 wings=[],  # A list of Wing objects.
                 fuselages=[],  # A list of Fuselage objects.
                 s_ref=None,  # If not set, populates from first wing object.
                 c_ref=None,  # See above
                 b_ref=None,  # See above
                 ):
        self.name = name

        self.xyz_ref = cas.vertcat(x_ref, y_ref, z_ref)
        # if xyz_ref is None and mass_props is not None:
        #     self.xyz_ref = mass_props.get_cg()
        # else:
        #     self.xyz_ref = cas.MX(xyz_ref)
        # self.mass_props = mass_props
        self.wings = wings
        self.fuselages = fuselages

        if len(self.wings) > 0:  # If there is at least one wing
            self.set_ref_dims_from_wing(main_wing_index=0)
        if s_ref is not None: self.s_ref = s_ref
        if c_ref is not None: self.c_ref = c_ref
        if b_ref is not None: self.b_ref = b_ref

        # Check that everything was set right:
        assert self.name is not None
        assert self.xyz_ref is not None
        assert self.s_ref is not None
        assert self.c_ref is not None
        assert self.b_ref is not None

    def substitute_solution(self, sol):
        """
        Substitutes a solution from CasADi's solver.
        :param sol:
        :return:
        """
        for a in dir(self):
            attrib_orig = getattr(self, a)
            if isinstance(attrib_orig, bool) or isinstance(attrib_orig, int):
                continue
            try:
                setattr(self, a, sol.value(attrib_orig))
            except NotImplementedError:
                pass
            if isinstance(attrib_orig, list):
                try:
                    new_attrib_orig = []
                    for item in attrib_orig:
                        new_attrib_orig.append(item.substitute_solution(sol))
                    setattr(self, a, new_attrib_orig)
                except:
                    pass
        return self

    def set_ref_dims_from_wing(self,
                               main_wing_index=0
                               ):
        # Sets the reference dimensions of the airplane from measurements obtained from a specific wing.

        main_wing = self.wings[main_wing_index]

        self.s_ref = main_wing.area()
        self.b_ref = main_wing.span()
        self.c_ref = main_wing.mean_geometric_chord()

    def set_paneling_everywhere(self, n_chordwise_panels, n_spanwise_panels):
        # Sets the chordwise and spanwise paneling everywhere to a specified init_val.
        # Useful for quickly changing the fidelity of your simulation.

        for wing in self.wings:
            wing.chordwise_panels = n_chordwise_panels
            for xsec in wing.xsecs:
                xsec.spanwise_panels = n_spanwise_panels

    def set_spanwise_paneling_everywhere(self, n_spanwise_panels):
        # Sets the spanwise paneling everywhere to a specified value.
        # Useful for quickly changing the fidelity of your simulation.

        for wing in self.wings:
            for xsec in wing.xsecs:
                xsec.spanwise_panels = n_spanwise_panels

    def draw(self, show=True):
        fig = go.Figure()

        # x, y, and z give the vertices
        x = []
        y = []
        z = []
        # i, j and k give the connectivity of the vertices
        i = []
        j = []
        k = []
        intensity = []
        # xe, ye, and ze give the outline of each panel
        xe = []
        ye = []
        ze = []

        # Wings
        for wing_id in range(len(self.wings)):
            wing = self.wings[wing_id]  # type: Wing

            for xsec_id in range(len(wing.xsecs) - 1):
                xsec_1 = wing.xsecs[xsec_id]  # type: WingXSec
                xsec_2 = wing.xsecs[xsec_id + 1]  # type: WingXSec

                le_start = xsec_1.xyz_le + wing.xyz_le
                te_start = xsec_1.xyz_te() + wing.xyz_le
                le_end = xsec_2.xyz_le + wing.xyz_le
                te_end = xsec_2.xyz_te() + wing.xyz_le

                x.append(float(le_start[0]))
                x.append(float(le_end[0]))
                x.append(float(te_end[0]))
                x.append(float(te_start[0]))
                y.append(float(le_start[1]))
                y.append(float(le_end[1]))
                y.append(float(te_end[1]))
                y.append(float(te_start[1]))
                z.append(float(le_start[2]))
                z.append(float(le_end[2]))
                z.append(float(te_end[2]))
                z.append(float(te_start[2]))
                intensity.append(wing_id)
                intensity.append(wing_id)
                intensity.append(wing_id)
                intensity.append(wing_id)
                xe.append(float(le_start[0]))
                xe.append(float(le_end[0]))
                xe.append(float(te_end[0]))
                xe.append(float(te_start[0]))
                xe.append(float(le_start[0]))
                ye.append(float(le_start[1]))
                ye.append(float(le_end[1]))
                ye.append(float(te_end[1]))
                ye.append(float(te_start[1]))
                ye.append(float(le_start[1]))
                ze.append(float(le_start[2]))
                ze.append(float(le_end[2]))
                ze.append(float(te_end[2]))
                ze.append(float(te_start[2]))
                ze.append(float(le_start[2]))
                xe.append(None)
                ye.append(None)
                ze.append(None)

                indices_added = np.arange(len(x) - 4, len(x))
                # Add front_inner triangle
                i.append(indices_added[0])
                j.append(indices_added[1])
                k.append(indices_added[3])
                # Add back_outer triangle
                i.append(indices_added[2])
                j.append(indices_added[3])
                k.append(indices_added[1])

                if wing.symmetric:
                    x.append(float(le_start[0]))
                    x.append(float(le_end[0]))
                    x.append(float(te_end[0]))
                    x.append(float(te_start[0]))
                    y.append(float(-le_start[1]))
                    y.append(float(-le_end[1]))
                    y.append(float(-te_end[1]))
                    y.append(float(-te_start[1]))
                    z.append(float(le_start[2]))
                    z.append(float(le_end[2]))
                    z.append(float(te_end[2]))
                    z.append(float(te_start[2]))
                    intensity.append(wing_id)
                    intensity.append(wing_id)
                    intensity.append(wing_id)
                    intensity.append(wing_id)
                    xe.append(float(le_start[0]))
                    xe.append(float(le_end[0]))
                    xe.append(float(te_end[0]))
                    xe.append(float(te_start[0]))
                    xe.append(float(le_start[0]))
                    ye.append(float(-le_start[1]))
                    ye.append(float(-le_end[1]))
                    ye.append(float(-te_end[1]))
                    ye.append(float(-te_start[1]))
                    ye.append(float(-le_start[1]))
                    ze.append(float(le_start[2]))
                    ze.append(float(le_end[2]))
                    ze.append(float(te_end[2]))
                    ze.append(float(te_start[2]))
                    ze.append(float(le_start[2]))
                    xe.append(None)
                    ye.append(None)
                    ze.append(None)

                    indices_added = np.arange(len(x) - 4, len(x))
                    # Add front_inner triangle
                    i.append(indices_added[0])
                    j.append(indices_added[1])
                    k.append(indices_added[3])
                    # Add back_outer triangle
                    i.append(indices_added[2])
                    j.append(indices_added[3])
                    k.append(indices_added[1])
        # Fuselages
        for fuse_id in range(len(self.fuselages)):
            fuse = self.fuselages[fuse_id]  # type: Fuselage

            for xsec_id in range(len(fuse.xsecs) - 1):
                xsec_1 = fuse.xsecs[xsec_id]  # type: FuselageXSec
                xsec_2 = fuse.xsecs[xsec_id + 1]  # type: FuselageXSec

                r1 = xsec_1.radius
                r2 = xsec_2.radius
                points_1 = np.zeros((fuse.circumferential_panels, 3))
                points_2 = np.zeros((fuse.circumferential_panels, 3))
                for point_index in range(fuse.circumferential_panels):
                    rot = angle_axis_rotation_matrix(
                        2 * cas.pi * point_index / fuse.circumferential_panels,
                        [1, 0, 0],
                        True
                    ).toarray()
                    points_1[point_index, :] = rot @ np.array([0, 0, r1])
                    points_2[point_index, :] = rot @ np.array([0, 0, r2])
                points_1 = points_1 + np.array(fuse.xyz_le).reshape(-1) + np.array(xsec_1.xyz_c).reshape(-1)
                points_2 = points_2 + np.array(fuse.xyz_le).reshape(-1) + np.array(xsec_2.xyz_c).reshape(-1)

                for point_index in range(fuse.circumferential_panels):
                    x.append(float(points_1[(point_index) % fuse.circumferential_panels, 0]))
                    x.append(float(points_1[(point_index + 1) % fuse.circumferential_panels, 0]))
                    x.append(float(points_2[(point_index + 1) % fuse.circumferential_panels, 0]))
                    x.append(float(points_2[(point_index) % fuse.circumferential_panels, 0]))
                    y.append(float(points_1[(point_index) % fuse.circumferential_panels, 1]))
                    y.append(float(points_1[(point_index + 1) % fuse.circumferential_panels, 1]))
                    y.append(float(points_2[(point_index + 1) % fuse.circumferential_panels, 1]))
                    y.append(float(points_2[(point_index) % fuse.circumferential_panels, 1]))
                    z.append(float(points_1[(point_index) % fuse.circumferential_panels, 2]))
                    z.append(float(points_1[(point_index + 1) % fuse.circumferential_panels, 2]))
                    z.append(float(points_2[(point_index + 1) % fuse.circumferential_panels, 2]))
                    z.append(float(points_2[(point_index) % fuse.circumferential_panels, 2]))
                    intensity.append(fuse_id)
                    intensity.append(fuse_id)
                    intensity.append(fuse_id)
                    intensity.append(fuse_id)
                    xe.append(float(points_1[(point_index) % fuse.circumferential_panels, 0]))
                    xe.append(float(points_1[(point_index + 1) % fuse.circumferential_panels, 0]))
                    xe.append(float(points_2[(point_index + 1) % fuse.circumferential_panels, 0]))
                    xe.append(float(points_2[(point_index) % fuse.circumferential_panels, 0]))
                    xe.append(float(points_1[(point_index) % fuse.circumferential_panels, 0]))
                    ye.append(float(points_1[(point_index) % fuse.circumferential_panels, 1]))
                    ye.append(float(points_1[(point_index + 1) % fuse.circumferential_panels, 1]))
                    ye.append(float(points_2[(point_index + 1) % fuse.circumferential_panels, 1]))
                    ye.append(float(points_2[(point_index) % fuse.circumferential_panels, 1]))
                    ye.append(float(points_1[(point_index) % fuse.circumferential_panels, 1]))
                    ze.append(float(points_1[(point_index) % fuse.circumferential_panels, 2]))
                    ze.append(float(points_1[(point_index + 1) % fuse.circumferential_panels, 2]))
                    ze.append(float(points_2[(point_index + 1) % fuse.circumferential_panels, 2]))
                    ze.append(float(points_2[(point_index) % fuse.circumferential_panels, 2]))
                    ze.append(float(points_1[(point_index) % fuse.circumferential_panels, 2]))
                    xe.append(None)
                    ye.append(None)
                    ze.append(None)

                    indices_added = np.arange(len(x) - 4, len(x))
                    # Add front_inner triangle
                    i.append(indices_added[0])
                    j.append(indices_added[1])
                    k.append(indices_added[3])
                    # Add back_outer triangle
                    i.append(indices_added[2])
                    j.append(indices_added[3])
                    k.append(indices_added[1])

                    if fuse.symmetric:
                        x.append(float(points_1[(point_index) % fuse.circumferential_panels, 0]))
                        x.append(float(points_1[(point_index + 1) % fuse.circumferential_panels, 0]))
                        x.append(float(points_2[(point_index + 1) % fuse.circumferential_panels, 0]))
                        x.append(float(points_2[(point_index) % fuse.circumferential_panels, 0]))
                        y.append(float(-points_1[(point_index) % fuse.circumferential_panels, 1]))
                        y.append(float(-points_1[(point_index + 1) % fuse.circumferential_panels, 1]))
                        y.append(float(-points_2[(point_index + 1) % fuse.circumferential_panels, 1]))
                        y.append(float(-points_2[(point_index) % fuse.circumferential_panels, 1]))
                        z.append(float(points_1[(point_index) % fuse.circumferential_panels, 2]))
                        z.append(float(points_1[(point_index + 1) % fuse.circumferential_panels, 2]))
                        z.append(float(points_2[(point_index + 1) % fuse.circumferential_panels, 2]))
                        z.append(float(points_2[(point_index) % fuse.circumferential_panels, 2]))
                        intensity.append(fuse_id)
                        intensity.append(fuse_id)
                        intensity.append(fuse_id)
                        intensity.append(fuse_id)
                        xe.append(float(points_1[(point_index) % fuse.circumferential_panels, 0]))
                        xe.append(float(points_1[(point_index + 1) % fuse.circumferential_panels, 0]))
                        xe.append(float(points_2[(point_index + 1) % fuse.circumferential_panels, 0]))
                        xe.append(float(points_2[(point_index) % fuse.circumferential_panels, 0]))
                        xe.append(float(points_1[(point_index) % fuse.circumferential_panels, 0]))
                        ye.append(float(-points_1[(point_index) % fuse.circumferential_panels, 1]))
                        ye.append(float(-points_1[(point_index + 1) % fuse.circumferential_panels, 1]))
                        ye.append(float(-points_2[(point_index + 1) % fuse.circumferential_panels, 1]))
                        ye.append(float(-points_2[(point_index) % fuse.circumferential_panels, 1]))
                        ye.append(float(-points_1[(point_index) % fuse.circumferential_panels, 1]))
                        ze.append(float(points_1[(point_index) % fuse.circumferential_panels, 2]))
                        ze.append(float(points_1[(point_index + 1) % fuse.circumferential_panels, 2]))
                        ze.append(float(points_2[(point_index + 1) % fuse.circumferential_panels, 2]))
                        ze.append(float(points_2[(point_index) % fuse.circumferential_panels, 2]))
                        ze.append(float(points_1[(point_index) % fuse.circumferential_panels, 2]))
                        xe.append(None)
                        ye.append(None)
                        ze.append(None)

                        indices_added = np.arange(len(x) - 4, len(x))
                        # Add front_inner triangle
                        i.append(indices_added[0])
                        j.append(indices_added[1])
                        k.append(indices_added[3])
                        # Add back_outer triangle
                        i.append(indices_added[2])
                        j.append(indices_added[3])
                        k.append(indices_added[1])

        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                flatshading=True,
                intensity=intensity,
                colorscale="mint"
            )
        )

        # define the trace for triangle sides
        fig.add_trace(
            go.Scatter3d(
                x=xe,
                y=ye,
                z=ze,
                mode='lines',
                name='',
                line=dict(color='rgb(0,0,0)', width=3))
        )

        fig.update_layout(
            title="%s Airplane" % self.name,
            scene=dict(aspectmode='data')
        )

        if show:
            fig.show()
        else:
            return fig

    def is_symmetric(self):
        for wing in self.wings:
            for xsec in wing.xsecs:
                if not (xsec.control_surface_type == "symmetric" or xsec.control_surface_deflection == 0):
                    return False
                if not wing.symmetric:
                    if not xsec.xyz_le[1] == 0:
                        return False
                    if not xsec.twist == 0:
                        if not (xsec.twist_axis[0] == 0 and xsec.twist_axis[2] == 0):
                            return False
                    if not xsec.airfoil.CL_function(0, 1e6, 0, 0) == 0:
                        return False
                    if not xsec.airfoil.Cm_function(0, 1e6, 0, 0) == 0:
                        return False

        return True


class Wing:
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

    def substitute_solution(self, sol):
        """
        Substitutes a solution from CasADi's solver.
        :param sol:
        :return:
        """
        for a in dir(self):
            attrib_orig = getattr(self, a)
            if isinstance(attrib_orig, bool) or isinstance(attrib_orig, int):
                continue
            try:
                setattr(self, a, sol.value(attrib_orig))
            except NotImplementedError:
                pass
            if isinstance(attrib_orig, list):
                try:
                    new_attrib_orig = []
                    for item in attrib_orig:
                        new_attrib_orig.append(item.substitute_solution(sol))
                    setattr(self, a, new_attrib_orig)
                except:
                    pass
        return self

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


class WingXSec:
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

    def substitute_solution(self, sol):
        """
        Substitutes a solution from CasADi's solver.
        :param sol:
        :return:
        """
        for attrib_name in dir(self):
            attrib_orig = getattr(self, attrib_name)
            if isinstance(attrib_orig, bool) or isinstance(attrib_orig, int):
                continue
            try:
                setattr(self, attrib_name, sol.value(attrib_orig))
            except NotImplementedError:
                pass
            if isinstance(attrib_orig, list):
                try:
                    new_attrib_orig = []
                    for item in attrib_orig:
                        new_attrib_orig.append(item.substitute_solution(sol))
                    setattr(self, attrib_name, new_attrib_orig)
                except:
                    pass
        return self

    def xyz_te(self):
        rot = angle_axis_rotation_matrix(self.twist * cas.pi / 180, self.twist_axis)
        xyz_te = self.xyz_le + rot @ cas.vertcat(self.chord, 0, 0)
        # xyz_te = self.xyz_le + self.chord * cas.vertcat(
        #     cas.cos(self.twist * cas.pi / 180),
        #     0,
        #     -cas.sin(self.twist * cas.pi / 180)
        # )
        return xyz_te


class Airfoil:
    cached_airfoils = []

    def __init__(self,
                 name=None,  # Examples: 'naca0012', 'ag10', 's1223', or anything you want.
                 coordinates=None,  # Treat this as an immutable, don't edit directly after initialization.
                 LE_index=None,
                 # If you supply "coordinates", you can manually specify the index of the leading edge here.
                 use_cache=True,  # Look in the airfoil cache, based on the airfoil's name. # TODO make airfoil caching
                 repanel=True,  # Should we repanel the airfoil upon initialization?
                 find_mcl=True, # Should we attempt to find the mean camber line upon initialization?
                 n_points_per_side=400,  # Number of points to use when repaneling the airfoil (if repanel is True)
                 CL_function=lambda alpha, Re, mach, deflection,: (  # Lift coefficient function (alpha in deg)
                         (alpha * np.pi / 180) * (2 * np.pi)
                 ),  # type: callable # with exactly the arguments listed (no more, no fewer).
                 CDp_function=lambda alpha, Re, mach, deflection: (  # Profile drag coefficient function (alpha in deg)
                         (1 + (alpha / 5) ** 2) * 2 * (0.074 / Re ** 0.2)
                 ),  # type: callable # with exactly the arguments listed (no more, no fewer).
                 Cm_function=lambda alpha, Re, mach, deflection: (  # Moment coefficient function (about quarter-chord) (alpha in deg)
                         0
                 ),  # type: callable # with exactly the arguments listed (no more, no fewer).
                 ):
        if (name is not None) or (coordinates is not None):
            if name is not None:
                self.name = name
            else:
                self.name = "Untitled"
            if coordinates is not None:
                self.coordinates = coordinates
                if LE_index is None:
                    self.LE_index = np.argmin(self.coordinates[:, 0])
                else:
                    self.LE_index = LE_index
            else:
                self.populate_coordinates()  # populates self.coordinates
            assert hasattr(self, 'coordinates'), "Couldn't figure out the coordinates of this airfoil! You need to either \
            a) use a name corresponding to an airfoil in the UIUC Airfoil Database or \
            b) provide your own coordinates in the constructor, such as Airfoil(""MyFoilName"", <Nx2 array of coordinates>)."

            # self.normalize()
            if repanel:
                self.repanel_current_airfoil(
                    n_points_per_side=n_points_per_side)  # all airfoils are automatically repaneled to ensure consistent, good paneling.

            if find_mcl:
                self.populate_mcl_coordinates()

        self.CL_function = CL_function
        self.CDp_function = CDp_function
        self.Cm_function = Cm_function

    def populate_coordinates(self):
        # Populates a variable called self.coordinates with the coordinates of the airfoil.
        name = self.name.lower().strip()

        # If it's a NACA 4-series airfoil, try to generate it
        if "naca" in name:
            nacanumber = name.split("naca")[1]
            if nacanumber.isdigit():
                if len(nacanumber) == 4:

                    # Parse
                    max_camber = int(nacanumber[0]) * 0.01
                    camber_loc = int(nacanumber[1]) * 0.1
                    thickness = int(nacanumber[2:]) * 0.01

                    # Set number of points per side
                    n_points_per_side = 100

                    # Referencing https://en.wikipedia.org/wiki/NACA_airfoil#Equation_for_a_cambered_4-digit_NACA_airfoil
                    # from here on out

                    # Make uncambered coordinates
                    x_t = cosspace(n_points=n_points_per_side)  # Generate some cosine-spaced points
                    y_t = 5 * thickness * (
                            + 0.2969 * x_t ** 0.5
                            - 0.1260 * x_t
                            - 0.3516 * x_t ** 2
                            + 0.2843 * x_t ** 3
                            - 0.1015 * x_t ** 4  # 0.1015 is original, #0.1036 for sharp TE
                    )

                    if camber_loc == 0:
                        camber_loc = 0.5  # prevents divide by zero errors for things like naca0012's.

                    # Get camber
                    y_c = cas.if_else(
                        x_t <= camber_loc,
                        max_camber / camber_loc ** 2 * (2 * camber_loc * x_t - x_t ** 2),
                        max_camber / (1 - camber_loc) ** 2 * ((1 - 2 * camber_loc) + 2 * camber_loc * x_t - x_t ** 2)
                    )

                    # Get camber slope
                    dycdx = cas.if_else(
                        x_t <= camber_loc,
                        2 * max_camber / camber_loc ** 2 * (camber_loc - x_t),
                        2 * max_camber / (1 - camber_loc) ** 2 * (camber_loc - x_t)
                    )
                    theta = cas.atan(dycdx)

                    # Combine everything
                    x_U = x_t - y_t * cas.sin(theta)
                    x_L = x_t + y_t * cas.sin(theta)
                    y_U = y_c + y_t * cas.cos(theta)
                    y_L = y_c - y_t * cas.cos(theta)

                    # Flip upper surface so it's back to front
                    x_U, y_U = flipud(x_U), flipud(y_U)

                    # Trim 1 point from lower surface so there's no overlap
                    x_L, y_L = x_L[1:], y_L[1:]

                    x = cas.vertcat(x_U, x_L)
                    y = cas.vertcat(y_U, y_L)

                    coordinates = cas.horzcat(x, y)

                    self.coordinates = coordinates
                    self.LE_index = np.argmin(self.coordinates[:, 0])
                    return
                else:
                    print("Unfortunately, only 4-series NACA airfoils can be generated at this time.")

        # Try to read from airfoil database
        try:
            import importlib.resources
            from . import airfoils
            raw_text = importlib.resources.read_text(airfoils, name + '.dat')
            trimmed_text = raw_text[raw_text.find('\n'):]

            coordinates1D = np.fromstring(trimmed_text, sep='\n')  # returns the coordinates in a 1D array
            assert len(
                coordinates1D) % 2 == 0, 'File was found in airfoil database, but it could not be read correctly!'  # Should be even

            coordinates = np.reshape(coordinates1D, (-1, 2))
            self.coordinates = coordinates
            self.LE_index = np.argmin(self.coordinates[:, 0])
            return

        except FileNotFoundError:
            print("Could not find a file associated with your airfoil (%s.dat) in airfoil database!" % name)

    def populate_mcl_coordinates(self):
        # Populates self.mcl_coordinates, a Nx2 list of the airfoil's mean camber line coordinates.
        # Ordered from the leading edge to the trailing edge.
        #
        # Also populates self.upper_minus_mcl and self.lower_minus mcl, which are Nx2 lists of the vectors needed to
        # go from the mcl coordinates to the upper and lower surfaces, respectively. Both listed leading-edge to trailing-edge.
        #
        # Also populates self.thickness, a vector of the thicknesses at the mcl_coordinates p-points.

        upper = flipud(self.upper_coordinates())
        lower = self.lower_coordinates()

        assert upper.shape == lower.shape, "The upper and lower surfaces must have the same number of coordinates before you can call Airfoil.populate_mcl_coordinates(). You should use the Airfoil.repanel() method first."

        mcl_coordinates = (upper + lower) / 2
        self.mcl_coordinates = mcl_coordinates

        self.upper_minus_mcl = upper - self.mcl_coordinates
        # self.lower_minus_mcl = -self.upper_minus_mcl

        self.thickness = 2 * self.upper_minus_mcl

    # def normalize(self): # TODO make this return a new airfoil instead
    #     # Alters the airfoil's coordinates to exactly achieve several goals:
    #     #   # x_c == 0
    #     #   # y_c == 0
    #     #   # average( y_te_upper, y_te_lower ) == 0
    #     #   # max( x_te_upper, x_te_upper ) == 1
    #     # The first two goals are achieved by translating in p and y. The third goal is achieved by rotating about (0,0).
    #     # The fourth goal is achieved by uniform scaling.
    #
    #     # Goals 1 and 2
    #     LE_point_original = self.coordinates[self.LE_index, :]
    #     assert abs(LE_point_original[
    #                    0]) < 0.02, "The leading edge point x_coordinate looks like it's at a really weird location! \
    #                    Are you sure this isn't bad airfoil geometry?"
    #     assert abs(LE_point_original[
    #                    1]) < 0.02, "The leading edge point x_coordinate looks like it's at a really weird location! \
    #                    Are you sure this isn't bad airfoil geometry?"
    #     self.coordinates -= LE_point_original
    #
    #     # Goal 3
    #     TE_point_pre_rotation = (self.coordinates[0, :] + self.coordinates[-1, :]) / 2
    #     rotation_angle = -cas.arctan(TE_point_pre_rotation[1] / TE_point_pre_rotation[
    #         0])  # You need to rotate this many radians counterclockwise
    #     assert abs(cas.degrees(
    #         rotation_angle)) < 0.5, "The foil appears to be really weirdly rotated! \
    #         Are you sure this isn't bad airfoil geometry?"
    #     cos_theta = cas.cos(rotation_angle)
    #     sin_theta = cas.sin(rotation_angle)
    #     rotation_matrix = cas.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    #     self.coordinates = cas.transpose(rotation_matrix @ cas.transpose(self.coordinates))
    #
    #     # Goal 4
    #     x_max = cas.fmax(self.coordinates[:, 0])
    #     assert x_max <= 1.02 and x_max >= 0.98, "x_max is really weird! Are you sure this isn't bad airfoil geometry?"
    #     scale_factor = 1 / x_max
    #     self.coordinates *= scale_factor

    def draw(self, draw_mcl=True):
        x = np.array(self.coordinates[:, 0]).reshape(-1)
        y = np.array(self.coordinates[:, 1]).reshape(-1)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name="Airfoil"
            ),
        )
        if draw_mcl:
            try:
                x_mcl = np.array(self.mcl_coordinates[:, 0]).reshape(-1)
                y_mcl = np.array(self.mcl_coordinates[:, 1]).reshape(-1)
            except AttributeError:
                self.populate_mcl_coordinates()
                x_mcl = np.array(self.mcl_coordinates[:, 0]).reshape(-1)
                y_mcl = np.array(self.mcl_coordinates[:, 1]).reshape(-1)
            fig.add_trace(
                go.Scatter(
                    x=x_mcl,
                    y=y_mcl,
                    mode="lines+markers",
                    name="Mean Camber Line (MCL)"
                )
            )

        fig.update_layout(
            xaxis_title="x",
            yaxis_title="y",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            title="%s Airfoil" % self.name
        )
        fig.show()

    def LE_index(self):
        # Returns the index of the leading-edge point.
        return np.argmin(self.coordinates[:, 0])  # TODO comment out

    def lower_coordinates(self):
        # Returns a matrix (N by 2) of [p y] coordinates that describe the lower surface of the airfoil.
        # Order is from leading edge to trailing edge.
        # Includes the leading edge point; be careful about duplicates if using this method in conjunction with self.upper_coordinates().
        return self.coordinates[self.LE_index:, :]

    def upper_coordinates(self):
        # Returns a matrix (N by 2) of [p y] coordinates that describe the upper surface of the airfoil.
        # Order is from trailing edge to leading edge.
        # Includes the leading edge point; be careful about duplicates if using this method in conjunction with self.lower_coordinates().
        return self.coordinates[:self.LE_index + 1, :]

    # def get_thickness_at_chord_fraction(self, chord_fraction):
    #     thickness_func = sp_interp.interp1d(
    #         p=self.mcl_coordinates[:, 0],
    #         y=self.thickness,
    #         copy=False,
    #         fill_value='extrapolate'
    #     )
    #     return thickness_func(chord_fraction)

    def get_downsampled_mcl(self, mcl_fractions):
        # Returns the mean camber line in downsampled form

        mcl = self.mcl_coordinates
        # Find distances along mcl, assuming linear interpolation
        mcl_distances_between_points = cas.sqrt(
            (mcl[:-1, 0] - mcl[1:, 0]) ** 2 +
            (mcl[:-1, 1] - mcl[1:, 1]) ** 2
        )
        mcl_distances_cumulative = cas.vertcat(0, cas.cumsum(mcl_distances_between_points))
        mcl_distances_cumulative_normalized = mcl_distances_cumulative / mcl_distances_cumulative[-1]

        mcl_downsampled_x = cas.interp1d(
            np.array(mcl_distances_cumulative_normalized).reshape(-1).tolist(),
            mcl[:, 0],
            mcl_fractions
        )
        mcl_downsampled_y = cas.interp1d(
            np.array(mcl_distances_cumulative_normalized).reshape(-1).tolist(),
            mcl[:, 1],
            mcl_fractions
        )

        mcl_downsampled = cas.horzcat(mcl_downsampled_x, mcl_downsampled_y)

        return mcl_downsampled

    def get_camber_at_chord_fraction(self, chord_fraction):
        return cas.interp1d(
            np.array(self.mcl_coordinates[:, 0]).reshape(-1).tolist(),
            self.mcl_coordinates[:, 1],
            chord_fraction
        )

    # def get_camber_at_chord_fraction_legacy(self, chord_fraction):
    #     # Returns the (interpolated) camber at a given location(s). The location is specified by the chord fraction, as measured from the leading edge. Camber is nondimensionalized by chord (i.e. this function returns camber/c at a given p/c).
    #     chord = cas.fmax(self.coordinates[:, 0]) - cas.min(
    #         self.coordinates[:, 0])  # This should always be 1, but this is just coded for robustness.
    #
    #     p = chord_fraction * chord + min(self.coordinates[:, 0])
    #
    #     upperCoors = self.upper_coordinates()
    #     lowerCoors = self.lower_coordinates()
    #
    #     y_upper_func = sp_interp.interp1d(p=upperCoors[:, 0], y=upperCoors[:, 1], copy=False, fill_value='extrapolate')
    #     y_lower_func = sp_interp.interp1d(p=lowerCoors[:, 0], y=lowerCoors[:, 1], copy=False, fill_value='extrapolate')
    #
    #     y_upper = y_upper_func(p)
    #     y_lower = y_lower_func(p)
    #
    #     camber = (y_upper + y_lower) / 2
    #
    #     return camber

    # def get_mcl_normal_direction_at_chord_fraction(self, chord_fraction):
    #     # Returns the normal direction of the mean camber line at a specified chord fraction.
    #     # If you input a single init_val, returns a 1D numpy array with 2 elements (p,y).
    #     # If you input a vector of values, returns a 2D numpy array. First index is the point number, second index is (p,y)
    #
    #     # Right now, does it by finite differencing camber values :(
    #     # When I'm less lazy I'll make it do it in a proper, more efficient way
    #     # TODO make this not finite difference
    #     epsilon = cas.sqrt(cas.finfo(float).eps)
    #
    #     cambers = self.get_camber_at_chord_fraction(chord_fraction)
    #     cambers_incremented = self.get_camber_at_chord_fraction(chord_fraction + epsilon)
    #     dydx = (cambers_incremented - cambers) / epsilon
    #
    #     if dydx.shape == 1:  # single point
    #         normal = cas.hstack((-dydx, 1))
    #         normal /= cas.linalg.norm(normal)
    #         return normal
    #     else:  # multiple points vectorized
    #         normal = cas.column_stack((-dydx, cas.ones(dydx.shape)))
    #         normal /= cas.expand_dims(cas.linalg.norm(normal, axis=1), axis=1)  # normalize
    #         return normal

    def TE_thickness(self):
        # Returns the thickness of the trailing edge of the airfoil, in nondimensional (chord-normalized) units.
        return self.thickness[-1]

    def TE_angle(self):
        # Returns the trailing edge angle of the airfoil, in degrees
        upper_TE_vec = self.coordinates[0, :] - self.coordinates[1, :]
        lower_TE_vec = self.coordinates[-1, :] - self.coordinates[-2, :]

        return 180 / np.pi * (cas.atan2(
            upper_TE_vec[0] * lower_TE_vec[1] - upper_TE_vec[1] * lower_TE_vec[0],
            upper_TE_vec[0] * lower_TE_vec[0] + upper_TE_vec[1] * upper_TE_vec[1]
        ))

    # def area(self):
    #     # Returns the area of the airfoil, in nondimensional (normalized to chord^2) units.
    #     p = self.coordinates[:, 0]
    #     y = self.coordinates[:, 1]
    #     x_n = cas.roll(p, -1)  # x_next, or x_i+1
    #     y_n = cas.roll(y, -1)  # y_next, or y_i+1
    #
    #     a = p * y_n - x_n * y  # a is the area of the triangle bounded by a given point, the next point, and the origin.
    #
    #     A = 0.5 * cas.sum1(a)  # area
    #
    #     return A
    #
    # def centroid(self):
    #     # Returns the centroid of the airfoil, in nondimensional (chord-normalized) units.
    #     p = self.coordinates[:, 0]
    #     y = self.coordinates[:, 1]
    #     x_n = cas.roll(p, -1)  # x_next, or x_i+1
    #     y_n = cas.roll(y, -1)  # y_next, or y_i+1
    #
    #     a = p * y_n - x_n * y  # a is the area of the triangle bounded by a given point, the next point, and the origin.
    #
    #     A = 0.5 * cas.sum1(a)  # area
    #
    #     x_c = 1 / (6 * A) * cas.sum1(a * (p + x_n))
    #     y_c = 1 / (6 * A) * cas.sum1(a * (y + y_n))
    #     centroid = cas.array([x_c, y_c])
    #
    #     return centroid
    #
    # def Ixx(self):
    #     # Returns the nondimensionalized Ixx moment of inertia, taken about the centroid.
    #     p = self.coordinates[:, 0]
    #     y = self.coordinates[:, 1]
    #     x_n = cas.roll(p, -1)  # x_next, or x_i+1
    #     y_n = cas.roll(y, -1)  # y_next, or y_i+1
    #
    #     a = p * y_n - x_n * y  # a is the area of the triangle bounded by a given point, the next point, and the origin.
    #
    #     A = 0.5 * cas.sum1(a)  # area
    #
    #     x_c = 1 / (6 * A) * cas.sum1(a * (p + x_n))
    #     y_c = 1 / (6 * A) * cas.sum1(a * (y + y_n))
    #     centroid = cas.array([x_c, y_c])
    #
    #     Ixx = 1 / 12 * cas.sum1(a * (cas.power(y, 2) + y * y_n + cas.power(y_n, 2)))
    #
    #     Iuu = Ixx - A * centroid[1] ** 2
    #
    #     return Iuu
    #
    # def Iyy(self):
    #     # Returns the nondimensionalized Iyy moment of inertia, taken about the centroid.
    #     p = self.coordinates[:, 0]
    #     y = self.coordinates[:, 1]
    #     x_n = cas.roll(p, -1)  # x_next, or x_i+1
    #     y_n = cas.roll(y, -1)  # y_next, or y_i+1
    #
    #     a = p * y_n - x_n * y  # a is the area of the triangle bounded by a given point, the next point, and the origin.
    #
    #     A = 0.5 * cas.sum1(a)  # area
    #
    #     x_c = 1 / (6 * A) * cas.sum1(a * (p + x_n))
    #     y_c = 1 / (6 * A) * cas.sum1(a * (y + y_n))
    #     centroid = cas.array([x_c, y_c])
    #
    #     Iyy = 1 / 12 * cas.sum1(a * (cas.power(p, 2) + p * x_n + cas.power(x_n, 2)))
    #
    #     Ivv = Iyy - A * centroid[0] ** 2
    #
    #     return Ivv
    #
    # def Ixy(self):
    #     # Returns the nondimensionalized product of inertia, taken about the centroid.
    #     p = self.coordinates[:, 0]
    #     y = self.coordinates[:, 1]
    #     x_n = cas.roll(p, -1)  # x_next, or x_i+1
    #     y_n = cas.roll(y, -1)  # y_next, or y_i+1
    #
    #     a = p * y_n - x_n * y  # a is the area of the triangle bounded by a given point, the next point, and the origin.
    #
    #     A = 0.5 * cas.sum1(a)  # area
    #
    #     x_c = 1 / (6 * A) * cas.sum1(a * (p + x_n))
    #     y_c = 1 / (6 * A) * cas.sum1(a * (y + y_n))
    #     centroid = cas.array([x_c, y_c])
    #
    #     Ixy = 1 / 24 * cas.sum1(a * (p * y_n + 2 * p * y + 2 * x_n * y_n + x_n * y))
    #
    #     Iuv = Ixy - A * centroid[0] * centroid[1]
    #
    #     return Iuv
    #
    # def J(self):
    #     # Returns the nondimensionalized polar moment of inertia, taken about the centroid.
    #     p = self.coordinates[:, 0]
    #     y = self.coordinates[:, 1]
    #     x_n = cas.roll(p, -1)  # x_next, or x_i+1
    #     y_n = cas.roll(y, -1)  # y_next, or y_i+1
    #
    #     a = p * y_n - x_n * y  # a is the area of the triangle bounded by a given point, the next point, and the origin.
    #
    #     A = 0.5 * cas.sum1(a)  # area
    #
    #     x_c = 1 / (6 * A) * cas.sum1(a * (p + x_n))
    #     y_c = 1 / (6 * A) * cas.sum1(a * (y + y_n))
    #     centroid = cas.array([x_c, y_c])
    #
    #     Ixx = 1 / 12 * cas.sum1(a * (cas.power(y, 2) + y * y_n + cas.power(y_n, 2)))
    #
    #     Iyy = 1 / 12 * cas.sum1(a * (cas.power(p, 2) + p * x_n + cas.power(x_n, 2)))
    #
    #     J = Ixx + Iyy
    #
    #     return J

    def get_repaneled_airfoil(self, n_points_per_side=100):
        # Returns a repaneled version of the airfoil with cosine-spaced coordinates on the upper and lower surfaces.
        # Inputs:
        #   # n_points_per_side is the number of points PER SIDE (upper and lower) of the airfoil. 100 is a good number.
        # Notes: The number of points defining the final airfoil will be n_points_per_side*2-1,
        # since one point (the leading edge point) is shared by both the upper and lower surfaces.

        upper_original_coors = self.upper_coordinates()  # Note: includes leading edge point, be careful about duplicates
        lower_original_coors = self.lower_coordinates()  # Note: includes leading edge point, be careful about duplicates

        # Find distances between coordinates, assuming linear interpolation
        upper_distances_between_points = cas.sqrt(
            cas.power(upper_original_coors[:-1, 0] - upper_original_coors[1:, 0], 2) +
            cas.power(upper_original_coors[:-1, 1] - upper_original_coors[1:, 1], 2)
        )
        lower_distances_between_points = cas.sqrt(
            cas.power(lower_original_coors[:-1, 0] - lower_original_coors[1:, 0], 2) +
            cas.power(lower_original_coors[:-1, 1] - lower_original_coors[1:, 1], 2)
        )
        upper_distances_from_TE = cas.vertcat(0, cas.cumsum(upper_distances_between_points))
        lower_distances_from_LE = cas.vertcat(0, cas.cumsum(lower_distances_between_points))
        upper_distances_from_TE_normalized = upper_distances_from_TE / upper_distances_from_TE[-1]
        lower_distances_from_LE_normalized = lower_distances_from_LE / lower_distances_from_LE[-1]

        # Generate a cosine-spaced list of points from 0 to 1
        s = cosspace(n_points=n_points_per_side)

        x_upper = cas.interp1d(
            np.array(upper_distances_from_TE_normalized).reshape(-1).tolist(),
            upper_original_coors[:, 0],
            np.array(s).reshape(-1).tolist()
        )
        y_upper = cas.interp1d(
            np.array(upper_distances_from_TE_normalized).reshape(-1).tolist(),
            upper_original_coors[:, 1],
            np.array(s).reshape(-1).tolist()
        )
        x_lower = cas.interp1d(
            np.array(lower_distances_from_LE_normalized).reshape(-1).tolist(),
            lower_original_coors[:, 0],
            np.array(s).reshape(-1).tolist()
        )
        y_lower = cas.interp1d(
            np.array(lower_distances_from_LE_normalized).reshape(-1).tolist(),
            lower_original_coors[:, 1],
            np.array(s).reshape(-1).tolist()
        )

        x_coors = cas.vertcat(x_upper, x_lower[1:])
        y_coors = cas.vertcat(y_upper, y_lower[1:])

        coordinates = cas.horzcat(x_coors, y_coors)

        # Make a new airfoil with the coordinates
        name = "%s, repaneled to %i pts" % (self.name, n_points_per_side)
        new_airfoil = Airfoil(name=name, coordinates=coordinates, repanel=False, LE_index=n_points_per_side - 1)

        return new_airfoil

    def repanel_current_airfoil(self, n_points_per_side=100):
        # Returns a repaneled version of the airfoil with cosine-spaced coordinates on the upper and lower surfaces.
        # Inputs:
        #   # n_points_per_side is the number of points PER SIDE (upper and lower) of the airfoil. 100 is a good number.
        # Notes: The number of points defining the final airfoil will be n_points_per_side*2-1,
        # since one point (the leading edge point) is shared by both the upper and lower surfaces.

        upper_original_coors = self.upper_coordinates()  # Note: includes leading edge point, be careful about duplicates
        lower_original_coors = self.lower_coordinates()  # Note: includes leading edge point, be careful about duplicates

        # Find distances between coordinates, assuming linear interpolation
        upper_distances_between_points = cas.sqrt(
            cas.power(upper_original_coors[:-1, 0] - upper_original_coors[1:, 0], 2) +
            cas.power(upper_original_coors[:-1, 1] - upper_original_coors[1:, 1], 2)
        )
        lower_distances_between_points = cas.sqrt(
            cas.power(lower_original_coors[:-1, 0] - lower_original_coors[1:, 0], 2) +
            cas.power(lower_original_coors[:-1, 1] - lower_original_coors[1:, 1], 2)
        )
        upper_distances_from_TE = cas.vertcat(0, cas.cumsum(upper_distances_between_points))
        lower_distances_from_LE = cas.vertcat(0, cas.cumsum(lower_distances_between_points))
        upper_distances_from_TE_normalized = upper_distances_from_TE / upper_distances_from_TE[-1]
        lower_distances_from_LE_normalized = lower_distances_from_LE / lower_distances_from_LE[-1]

        # Generate a cosine-spaced list of points from 0 to 1
        s = cosspace(n_points=n_points_per_side)

        x_upper = cas.interp1d(
            np.array(upper_distances_from_TE_normalized).reshape(-1).tolist(),
            upper_original_coors[:, 0],
            np.array(s).reshape(-1).tolist()
        )
        y_upper = cas.interp1d(
            np.array(upper_distances_from_TE_normalized).reshape(-1).tolist(),
            upper_original_coors[:, 1],
            np.array(s).reshape(-1).tolist()
        )
        x_lower = cas.interp1d(
            np.array(lower_distances_from_LE_normalized).reshape(-1).tolist(),
            lower_original_coors[:, 0],
            np.array(s).reshape(-1).tolist()
        )
        y_lower = cas.interp1d(
            np.array(lower_distances_from_LE_normalized).reshape(-1).tolist(),
            lower_original_coors[:, 1],
            np.array(s).reshape(-1).tolist()
        )

        x_coors = cas.vertcat(x_upper, x_lower[1:])
        y_coors = cas.vertcat(y_upper, y_lower[1:])

        coordinates = cas.horzcat(x_coors, y_coors)

        self.coordinates = coordinates
        self.LE_index = n_points_per_side - 1

    # def get_sharp_TE_airfoil(self):
    #     # Returns a version of the airfoil with a sharp trailing edge.
    #
    #     upper_original_coors = self.upper_coordinates()  # Note: includes leading edge point, be careful about duplicates
    #     lower_original_coors = self.lower_coordinates()  # Note: includes leading edge point, be careful about duplicates
    #
    #     # Find data about the TE
    #
    #     # Get the scale factor
    #     x_mcl = self.mcl_coordinates[:, 0]
    #     x_max = cas.fmax(x_mcl)
    #     x_min = cas.min(x_mcl)
    #     scale_factor = (x_mcl - x_min) / (x_max - x_min)  # linear contraction
    #
    #     # Do the contraction
    #     upper_minus_mcl_adjusted = self.upper_minus_mcl - self.upper_minus_mcl[-1, :] * cas.expand_dims(scale_factor, 1)
    #
    #     # Recreate coordinates
    #     upper_coordinates_adjusted = cas.flipud(self.mcl_coordinates + upper_minus_mcl_adjusted)
    #     lower_coordinates_adjusted = self.mcl_coordinates - upper_minus_mcl_adjusted
    #
    #     coordinates = cas.vstack((
    #         upper_coordinates_adjusted[:-1, :],
    #         lower_coordinates_adjusted
    #     ))
    #
    #     # Make a new airfoil with the coordinates
    #     name = self.name + ", with sharp TE"
    #     new_airfoil = Airfoil(name=name, coordinates=coordinates, repanel=False)
    #
    #     return new_airfoil

    def get_airfoil_with_control_surface(self, deflection=0., hinge_point=0.75):
        # Returns a version of the airfoil with a control surface added at a given point.
        # Inputs:
        #   # deflection: the deflection angle, in degrees. Downwards-positive.
        #   # hinge_point: the location of the hinge, as a fraction of chord.

        # Make the rotation matrix for the given angle.
        sintheta = cas.sin(-cas.pi / 180 * deflection)
        costheta = cas.cos(-cas.pi / 180 * deflection)
        rotation_matrix = (
            cas.vertcat(
                cas.horzcat(costheta, -sintheta),
                cas.horzcat(sintheta, costheta),
            )
        )

        # Find the hinge point
        hinge_point = cas.vertcat(hinge_point,
                                  self.get_camber_at_chord_fraction(
                                      [hinge_point]))  # Make hinge_point a vector.

        # Split the airfoil into the sections before and after the hinge
        split_index = np.where(self.mcl_coordinates[:, 0] > hinge_point[0])[0][0]
        mcl_coordinates_before = self.mcl_coordinates[:split_index, :]
        mcl_coordinates_after = self.mcl_coordinates[split_index:, :]
        upper_minus_mcl_before = self.upper_minus_mcl[:split_index, :]
        upper_minus_mcl_after = self.upper_minus_mcl[split_index:, :]

        # Rotate the mean camber line (MCL) and "upper minus mcl"
        new_mcl_coordinates_after = cas.transpose(
            rotation_matrix @ (cas.transpose(mcl_coordinates_after) - hinge_point) + hinge_point)
        new_upper_minus_mcl_after = cas.transpose(rotation_matrix @ cas.transpose(upper_minus_mcl_after))

        # Do blending

        # Assemble airfoil
        new_mcl_coordinates = cas.vertcat(mcl_coordinates_before, new_mcl_coordinates_after)
        new_upper_minus_mcl = cas.vertcat(upper_minus_mcl_before, new_upper_minus_mcl_after)
        upper_coordinates = flipud(new_mcl_coordinates + new_upper_minus_mcl)
        lower_coordinates = new_mcl_coordinates - new_upper_minus_mcl
        coordinates = cas.vertcat(upper_coordinates, lower_coordinates[1:, :])

        new_airfoil = Airfoil(name="%s flapped" % self.name, coordinates=coordinates, repanel=False,
                              LE_index=self.LE_index)
        return new_airfoil  # TODO fix self-intersecting airfoils at high deflections

    def xfoil_a(self,
                alpha,
                Re=0,
                M=0,
                n_crit=9,
                xtr_bot=1,
                xtr_top=1,
                reset_bls=False,
                repanel=False,
                max_iter=100,
                ):
        """
        Interface to XFoil, provided through the open-source xfoil Python library by DARcorporation.
        Point analysis at a given alpha.
        :param alpha: angle of attack [deg]
        :param Re: Reynolds number
        :param M: Mach number
        :param n_crit: Critical Tollmien-Schlichting wave amplification factor
        :param xtr_bot: Bottom trip location [x/c]
        :param xtr_top: Top trip location [x/c]
        :param reset_bls: Reset boundary layer parameters upon initialization?
        :param repanel: Repanel airfoil within XFoil?
        :param max_iter: Maximum number of global Newton iterations
        :return: A tuple of (alpha, cl, cd, cm, cp)
        """
        try:
            xf = XFoil()
        except NameError:
            raise NameError("It appears that the XFoil-Python interface is not installed, so unfortunately you can't use this function!\n"
                            "To install it, run \"pip install xfoil\" in your terminal, or manually install it from: https://github.com/DARcorporation/xfoil-python .\n"
                            "Note: users on UNIX systems have reported errors with installing this (Windows seems fine).")
        xf.airfoil = xfoil_model.Airfoil(
            x=np.array(self.coordinates[:, 0]).reshape(-1)[::5],
            y=np.array(self.coordinates[:, 1]).reshape(-1)[::5],
        )
        xf.Re = Re
        xf.M = M
        xf.n_crit = n_crit
        xf.xtr = (xtr_top, xtr_bot)
        if reset_bls:
            xf.reset_bls()
        if repanel:
            xf.repanel()
        xf.max_iter = max_iter

        cl, cd, cm, cp = xf.a(alpha)
        a = alpha

        return a, cl, cd, cm, cp

    def xfoil_cl(self,
                 cl,
                 Re=0,
                 M=0,
                 n_crit=9,
                 xtr_bot=1,
                 xtr_top=1,
                 reset_bls=False,
                 repanel=False,
                 max_iter=100,
                 ):
        """
        Interface to XFoil, provided through the open-source xfoil Python library by DARcorporation.
        Point analysis at a given lift coefficient.
        :param cl: Lift coefficient
        :param Re: Reynolds number
        :param M: Mach number
        :param n_crit: Critical Tollmien-Schlichting wave amplification factor
        :param xtr_bot: Bottom trip location [x/c]
        :param xtr_top: Top trip location [x/c]
        :param reset_bls: Reset boundary layer parameters upon initialization?
        :param repanel: Repanel airfoil within XFoil?
        :param max_iter: Maximum number of global Newton iterations
        :return: A tuple of (alpha, cl, cd, cm, cp)
        """
        try:
            xf = XFoil()
        except NameError:
            raise NameError("It appears that the XFoil-Python interface is not installed, so unfortunately you can't use this function!\n"
                            "To install it, run \"pip install xfoil\" in your terminal, or manually install it from: https://github.com/DARcorporation/xfoil-python .\n"
                            "Note: users on UNIX systems have reported errors with installing this (Windows seems fine).")
        xf.airfoil = xfoil_model.Airfoil(
            x=np.array(self.coordinates[:, 0]).reshape(-1)[::5],
            y=np.array(self.coordinates[:, 1]).reshape(-1)[::5],
        )
        xf.Re = Re
        xf.M = M
        xf.n_crit = n_crit
        xf.xtr = (xtr_top, xtr_bot)
        if reset_bls:
            xf.reset_bls()
        if repanel:
            xf.repanel()
        xf.max_iter = max_iter

        a, cd, cm, cp = xf.cl(cl)
        cl = cl

        return a, cl, cd, cm, cp

    def xfoil_aseq(self,
                   a_start,
                   a_end,
                   a_step,
                   Re=0,
                   M=0,
                   n_crit=9,
                   xtr_bot=1,
                   xtr_top=1,
                   reset_bls=False,
                   repanel=False,
                   max_iter=100,
                   ):
        """
        Interface to XFoil, provided through the open-source xfoil Python library by DARcorporation.
        Alpha sweep analysis.
        :param a_start: First angle of attack [deg]
        :param a_end: Last angle of attack [deg]
        :param a_step: Amount to increment angle of attack by [deg]
        :param Re: Reynolds number
        :param M: Mach number
        :param n_crit: Critical Tollmien-Schlichting wave amplification factor
        :param xtr_bot: Bottom trip location [x/c]
        :param xtr_top: Top trip location [x/c]
        :param reset_bls: Reset boundary layer parameters upon initialization?
        :param repanel: Repanel airfoil within XFoil?
        :param max_iter: Maximum number of global Newton iterations
        :return: A tuple of (alphas, cls, cds, cms, cps)
        """
        try:
            xf = XFoil()
        except NameError:
            raise NameError("It appears that the XFoil-Python interface is not installed, so unfortunately you can't use this function!\n"
                            "To install it, run \"pip install xfoil\" in your terminal, or manually install it from: https://github.com/DARcorporation/xfoil-python .\n"
                            "Note: users on UNIX systems have reported errors with installing this (Windows seems fine).")
        xf.airfoil = xfoil_model.Airfoil(
            x=np.array(self.coordinates[:, 0]).reshape(-1)[::5],
            y=np.array(self.coordinates[:, 1]).reshape(-1)[::5],
        )
        xf.Re = Re
        xf.M = M
        xf.n_crit = n_crit
        xf.xtr = (xtr_top, xtr_bot)
        if reset_bls:
            xf.reset_bls()
        if repanel:
            xf.repanel()
        xf.max_iter = max_iter

        a, cl, cd, cm, cp = xf.aseq(a_start, a_end, a_step)

        return a, cl, cd, cm, cp

    def xfoil_cseq(self,
                   cl_start,
                   cl_end,
                   cl_step,
                   Re=0,
                   M=0,
                   n_crit=9,
                   xtr_bot=1,
                   xtr_top=1,
                   reset_bls=False,
                   repanel=False,
                   max_iter=100,
                   ):
        """
        Interface to XFoil, provided through the open-source xfoil Python library by DARcorporation.
        Lift coefficient sweep analysis.
        :param cl_start: First lift coefficient [unitless]
        :param cl_end: Last lift coefficient [unitless]
        :param cl_step: Amount to increment lift coefficient by [unitless]
        :param Re: Reynolds number
        :param M: Mach number
        :param n_crit: Critical Tollmien-Schlichting wave amplification factor
        :param xtr_bot: Bottom trip location [x/c]
        :param xtr_top: Top trip location [x/c]
        :param reset_bls: Reset boundary layer parameters upon initialization?
        :param repanel: Repanel airfoil within XFoil?
        :param max_iter: Maximum number of global Newton iterations
        :return: A tuple of (alphas, cls, cds, cms, cps)
        """
        try:
            xf = XFoil()
        except NameError:
            raise NameError("It appears that the XFoil-Python interface is not installed, so unfortunately you can't use this function!\n"
                            "To install it, run \"pip install xfoil\" in your terminal, or manually install it from: https://github.com/DARcorporation/xfoil-python .\n"
                            "Note: users on UNIX systems have reported errors with installing this (Windows seems fine).")
        xf.airfoil = xfoil_model.Airfoil(
            x=np.array(self.coordinates[:, 0]).reshape(-1)[::5],
            y=np.array(self.coordinates[:, 1]).reshape(-1)[::5],
        )
        xf.Re = Re
        xf.M = M
        xf.n_crit = n_crit
        xf.xtr = (xtr_top, xtr_bot)
        if reset_bls:
            xf.reset_bls()
        if repanel:
            xf.repanel()
        xf.max_iter = max_iter

        a, cl, cd, cm, cp = xf.cseq(cl_start, cl_end, cl_step)

        return a, cl, cd, cm, cp


class Fuselage:
    """
    Definition for a fuselage or other slender body (pod, etc.).
    For now, all fuselages are assumed to be fairly closely aligned with the body x axis. (<10 deg or so) # TODO update if this changes
    """

    def __init__(self,
                 name="Untitled Fuselage",  # It can help when debugging to give each fuselage a sensible name.
                 x_le=0,  # Will translate all of the xsecs of the fuselage. Useful for moving the fuselage around.
                 y_le=0,  # Will translate all of the xsecs of the fuselage. Useful for moving the fuselage around.
                 z_le=0,  # Will translate all of the xsecs of the fuselage. Useful for moving the fuselage around.
                 xsecs=[],  # This should be a list of FuselageXSec objects.
                 symmetric=False,  # Is the fuselage symmetric across the XZ plane?
                 circumferential_panels=24,
                 # Number of circumferential panels to use in VLM and Panel analysis. Should be even.
                 ):
        self.name = name
        self.xyz_le = cas.vertcat(x_le, y_le, z_le)
        self.xsecs = xsecs
        self.symmetric = symmetric
        assert circumferential_panels % 2 == 0
        self.circumferential_panels = circumferential_panels

    def substitute_solution(self, sol):
        """
        Substitutes a solution from CasADi's solver.
        :param sol:
        :return:
        """
        for a in dir(self):
            attrib_orig = getattr(self, a)
            if isinstance(attrib_orig, bool) or isinstance(attrib_orig, int):
                continue
            try:
                setattr(self, a, sol.value(attrib_orig))
            except NotImplementedError:
                pass
            if isinstance(attrib_orig, list):
                try:
                    new_attrib_orig = []
                    for item in attrib_orig:
                        new_attrib_orig.append(item.substitute_solution(sol))
                    setattr(self, a, new_attrib_orig)
                except:
                    pass
        return self

    def area_wetted(self):
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
            x_separation = self.xsecs[i + 1].x_c - self.xsecs[i].x_c
            area += cas.pi * (this_radius + next_radius) * cas.sqrt(
                (this_radius - next_radius) ** 2 + x_separation ** 2)
        if self.symmetric:
            area *= 2
        return area

    #
    def area_projected(self):
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
            x_separation = self.xsecs[i + 1].x_c - self.xsecs[i].x_c
            area += (this_radius + next_radius) * x_separation
        if self.symmetric:
            area *= 2
        return area

    def length(self):
        """
        Returns the total front-to-back length of the fuselage. Measured as the difference between the x-coordinates
        of the leading and trailing cross sections.
        :return:
        """
        return cas.fabs(self.xsecs[-1].x_c - self.xsecs[0].x_c)


class FuselageXSec:
    """
    Definition for a fuselage cross section ("X-section").
    """

    def __init__(self,
                 x_c=0,
                 y_c=0,
                 z_c=0,
                 radius=0,
                 ):
        self.x_c = x_c
        self.y_c = y_c
        self.z_c = z_c

        self.radius = radius

        self.xyz_c = cas.vertcat(x_c, y_c, z_c)

    def substitute_solution(self, sol):
        """
        Substitutes a solution from CasADi's solver.
        :param sol:
        :return:
        """
        for a in dir(self):
            attrib_orig = getattr(self, a)
            if isinstance(attrib_orig, bool) or isinstance(attrib_orig, int):
                continue
            try:
                setattr(self, a, sol.value(attrib_orig))
            except NotImplementedError:
                pass
            if isinstance(attrib_orig, list):
                try:
                    new_attrib_orig = []
                    for item in attrib_orig:
                        new_attrib_orig.append(item.substitute_solution(sol))
                    setattr(self, a, new_attrib_orig)
                except:
                    pass
        return self

    def xsec_area(self):
        """
        Returns the FuselageXSec's cross-sectional (xsec) area.
        :return:
        """
        return cas.pi * self.radius ** 2


def reflect_over_XZ_plane(input_vector):
    # Takes in a vector or an array and flips the y-coordinates.
    output_vector = input_vector
    shape = output_vector.shape
    if shape[1] == 1 and shape[0] == 3:  # Vector of 3 items
        output_vector = output_vector * cas.vertcat(1, -1, 1)
    elif shape[1] == 3:  # 2D Nx3 vector
        output_vector = cas.horzcat(output_vector[:, 0], -1 * output_vector[:, 1], output_vector[:, 2])
    # elif len(shape) == 3 and shape[2] == 3:  # 3D MxNx3 vector
    #     output_vector = output_vector * cas.array([1, -1, 1])
    else:
        raise Exception("Invalid input for reflect_over_XZ_plane!")

    return output_vector


def cosspace(min=0, max=1, n_points=50):
    mean = (max + min) / 2
    amp = (max - min) / 2
    return mean + amp * cas.cos(cas.linspace(cas.pi, 0, n_points))


def np_cosspace(min=0, max=1, n_points=50):
    mean = (max + min) / 2
    amp = (max - min) / 2
    return mean + amp * np.cos(np.linspace(np.pi, 0, n_points))


def angle_axis_rotation_matrix(angle, axis, axis_already_normalized=False):
    # Gives the rotation matrix from an angle and an axis.
    # An implmentation of https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    # Inputs:
    #   * angle: can be one angle or a vector (1d ndarray) of angles. Given in radians.
    #   * axis: a 1d numpy array of length 3 (p,y,z). Represents the angle.
    #   * axis_already_normalized: boolean, skips normalization for speed if you flag this true.
    # Outputs:
    #   * If angle is a scalar, returns a 3x3 rotation matrix.
    #   * If angle is a vector, returns a 3x3xN rotation matrix.
    if not axis_already_normalized:
        axis = axis / cas.norm_2(axis)

    sintheta = cas.sin(angle)
    costheta = cas.cos(angle)
    cpm = cas.vertcat(
        cas.horzcat(0, -axis[2], axis[1]),
        cas.horzcat(axis[2], 0, -axis[0]),
        cas.horzcat(-axis[1], axis[0], 0),
    )  # The cross product matrix of the rotation axis vector
    outer_axis = axis @ cas.transpose(axis)

    rot_matrix = costheta * cas.DM.eye(3) + sintheta * cpm + (1 - costheta) * outer_axis
    return rot_matrix


def linspace_3D(start, stop, n_points):
    # Given two points (a start and an end), returns an interpolated array of points on the line between the two.
    # Inputs:
    #   * start: 3D coordinates expressed as a 1D numpy array, shape==(3).
    #   * end: 3D coordinates expressed as a 1D numpy array, shape==(3).
    #   * n_points: Number of points to be interpolated (including endpoints), a scalar.
    # Outputs:
    #   * points: Array of 3D coordinates expressed as a 2D numpy array, shape==(N, 3)
    x = cas.linspace(start[0], stop[0], n_points)
    y = cas.linspace(start[1], stop[1], n_points)
    z = cas.linspace(start[2], stop[2], n_points)

    points = cas.horzcat(x, y, z)
    return points


def plot_point_cloud(p):
    """
    Plots an Nx3 point cloud
    :param p:
    :return:
    """
    p = np.array(p)
    px.scatter_3d(x=p[:, 0], y=p[:, 1], z=p[:, 2]).show()
