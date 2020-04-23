from aerosandbox.geometry.common import *


class Airplane(AeroSandboxObject):
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
        if s_ref is not None:
            self.s_ref = s_ref
        if c_ref is not None:
            self.c_ref = c_ref
        if b_ref is not None:
            self.b_ref = b_ref

        # Check that everything was set right:
        assert self.name is not None
        assert self.xyz_ref is not None
        assert self.s_ref is not None
        assert self.c_ref is not None
        assert self.b_ref is not None

    def __repr__(self):
        return "Airplane %s (%i wings, %i fuselages)" % (
            self.name,
            len(self.wings),
            len(self.fuselages)
        )

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

    def draw(self,
             show=True,  # type: bool
             colorscale="mint",  # type: str
             colorbar_title="Component ID",
             draw_quarter_chord=True,  # type: bool
             ):
        """
        Draws the airplane using a Plotly interface.
        :param show: Do you want to show the figure? [boolean]
        :param colorscale: Which colorscale do you want to use? ("viridis", "plasma", mint", etc.)
        :param draw_quarter_chord: Do you want to draw the quarter-chord? [boolean]
        :return: A plotly figure object [go.Figure]
        """
        fig = Figure3D()

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

                fig.add_quad(points=[
                    le_start,
                    le_end,
                    te_end,
                    te_start
                ],
                    intensity=wing_id,
                    mirror=wing.symmetric,
                )
                if draw_quarter_chord:
                    fig.add_line(  # draw the quarter-chord line
                        points=[
                            0.75 * le_start + 0.25 * te_start,
                            0.75 * le_end + 0.25 * te_end,
                        ],
                        mirror=wing.symmetric
                    )

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

                    fig.add_quad(points=[
                        points_1[(point_index) % fuse.circumferential_panels, :],
                        points_1[(point_index + 1) % fuse.circumferential_panels, :],
                        points_2[(point_index + 1) % fuse.circumferential_panels, :],
                        points_2[(point_index) % fuse.circumferential_panels, :],
                    ],
                        intensity=fuse_id,
                        mirror=fuse.symmetric,
                    )

        return fig.draw(
            show=show,
            colorscale=colorscale,
            colorbar_title=colorbar_title,
        )

    def is_symmetric(self):
        """
        Returns a boolean describing whether the airplane is geometrically entirely symmetric across the XZ-plane.
        :return: [boolean]
        """
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

    def write_aswing(self, filepath=None):
        """
        Contributed by Brent Avery, Edited by Peter Sharpe. Work in progress.
        Writes a geometry file compatible with Mark Drela's ASWing.
        :param filepath: Filepath to write to. Should include ".asw" extension [string]
        :return: None
        """
        if filepath is None:
            filepath = "%s.asw" % self.name
        with open(filepath, 'w+') as f:
            f.write('\n'.join([
                '#============',
                'Name',
                self.name,
                'End',
                '',
                '#============',
                'Units',
                'L 0.3048 m',
                'T 1.0  s',
                'F 4.450 N',
                'End',
                '',
                '#============',
                'Constant',
                '#  g     rho_0     a_0',
                '%f %f %f' % (9.81, 1.205, 343.3),
                'End',
                '', '#============',  # Reference values (change automatically with input file)
                'Reference',
                '#   Sref    Cref    Bref',
                '%f %f %f' % (self.s_ref, self.c_ref, self.b_ref),
                'End',
                # Ok so 'ground' is a point on the plane that is constrained from translation or rotation.
                # Based on the documentation this is usually the 'frontest' part of the aircraft (why do I suck at words).
                # There is definitely a much better way to do this and I'm working on it but right now I'm just assuming
                # that it is the front part of the main wing, and just make that a constraint in AeroSandBox
                '',
                '#============',
                'Ground',
                '#  Nbeam  t',
                '%i %i' % (1, 0),
                'End',
            ]))

            # The juicy stuff! This part of the code iterates over each wing and then subiterates (is that a word?) over
            # each wing's cross section. Along the way it collects information on chord length, angle, and coordinates of
            # all the leading edges. It then writes all this info in a way that ASWing likes
            for i, wing in enumerate(self.wings):
                xsecs = wing.xsecs
                chordalfa = []
                coords = []
                '''
                This part is hard to explain but basically I defined t (the beamwise axis)
                as the axis that the beam changes most along. This can be generalized
                but I'm not entirely sure how
                '''
                max_le = {abs(xsecs[-1].x_le - xsecs[0].x_le): 'sec.x_le', \
                          abs(xsecs[-1].y_le - xsecs[0].y_le): 'sec.y_le', \
                          abs(xsecs[-1].z_le - xsecs[0].z_le): 'sec.z_le'}
                for xsec in xsecs:
                    if max_le.get(max(max_le)) == 'sec.x_le':
                        t = xsec.x_le
                    elif max_le.get(max(max_le)) == 'sec.y_le':
                        t = xsec.y_le
                    elif max_le.get(max(max_le)) == 'sec.z_le':
                        t = xsec.z_le
                    chordalfa.append('    '.join([str(t), str(xsec.chord), str(xsec.twist)]))
                    coords.append('    '.join(
                        [str(t), str(xsec.x_le + wing.xyz_le[0]), str(xsec.y_le + wing.xyz_le[1]),
                         str(xsec.z_le + wing.xyz_le[2])]))
                f.write('\n'.join([
                    '',
                    '#============',
                    'Beam %i' % (i + 1),
                    wing.name,
                    't    chord    twist',
                    '\n'.join(chordalfa),
                    '#',
                    't    x    y    z',
                    '\n'.join(coords),
                    'End'
                ]))
