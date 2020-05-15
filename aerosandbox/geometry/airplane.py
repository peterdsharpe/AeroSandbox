from aerosandbox.geometry.common import *
from typing import List

class Airplane(AeroSandboxObject):
    """
    Definition for an airplane (or other vehicle/item to analyze).
    """

    def __init__(self,
                 name: str="Untitled",  # A sensible name for your airplane.
                 x_ref: float=0,  # Ref. point for moments; should be the center of gravity.
                 y_ref: float=0,  # Ref. point for moments; should be the center of gravity.
                 z_ref: float=0,  # Ref. point for moments; should be the center of gravity.
                 mass_props=None,  # An object of MassProps type; only needed for dynamic analysis
                 # If xyz_ref is not set, but mass_props is, the xyz_ref will be taken from the CG there.
                 wings: List['Wing']=[],  # A list of Wing objects.
                 fuselages: List['Fuselage']=[],  # A list of Fuselage objects.
                 s_ref: float=None,  # If not set, populates from first wing object.
                 c_ref: float=None,  # See above
                 b_ref: float=None,  # See above
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
        self.c_ref = main_wing.mean_aerodynamic_chord()

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
        with open(filepath, "w+") as f:
            f.write('\n'.join(['#============',  # Name of Plane
                               'Name',
                               self.name,
                               'End']))
            f.write('\n'.join(['', '#============',  # Units that the analysis would be in, usually metric
                               'Units',
                               'L 0.3048 m',
                               'T 1.0  s',
                               'F 4.450 N',
                               'End']))
            f.write('\n'.join(['', '#============',  # Value of constants (cant imagine these changing too much)
                               'Constant',
                               '#  g     rho_0     a_0',
                               '   '.join([str(9.81), str(1.205), str(343.3)]),
                               'End']))
            f.write('\n'.join(['', '#============',  # Reference values (change automatically with input file)
                               'Reference',
                               '#   Sref    Cref    Bref',
                               '   '.join([str(self.s_ref), str(self.c_ref), str(self.b_ref)]),
                               'End']))

            '''
            Ok so 'ground' is a point on the plane that is constrained from translation or rotation.   
            Based on the documentation this is usually the 'frontest' part of the aircraft (why do I suck at words).   
            There is definitely a much better way to do this and I'm working on it but right now I'm just assuming   
            that it is the front part of the main wing, and just make that a constraint in AeroSandBox
            '''
            f.write('\n'.join(['', '#============',
                               'Ground',
                               '#  Nbeam  t',
                               '    '.join([' ', str(1), str(0)]),
                               'End']))

            f.write('\n'.join(['', '#============',
                               'Joint',
                               '#   Nbeam1   Nbeam2    t1     t2']))
            for onewing in range(1, len(self.wings)):
                wing = self.wings[onewing]
                if wing.name == "Horizontal Stabilizer":
                    xsecs = []
                    for xsec in wing.xsecs:
                        xsecs.append(xsec)
                    t = xsecs[0].y_le + wing.xyz_le[1]
                    coords = '       '.join(['    1', str(onewing + 1), str(t), '0'])
                    f.write('\n'.join(['', coords]))
                if wing.name == "Vertical Stabilizer":
                    wing2 = self.wings[np.ceil(onewing / 2)]
                    wing3 = self.wings[0]
                    xsecs = []
                    for xsec in wing.xsecs:
                        xsecs.append(xsec)
                    xsecs2 = []
                    for xsec2 in wing2.xsecs:
                        xsecs2.append(xsec2)
                    xsecs3 = []
                    for xsec3 in wing3.xsecs:
                        xsecs3.append(xsec3)
                    t = xsecs2[0].y_le + wing.xyz_le[1]
                    t2 = 1 + (xsecs2[0].z_le + wing2.xyz_le[2]) - (xsecs[0].z_le + wing.xyz_le[2])
                    coords = '       '.join(['    1', str(onewing + 1), str(t), str(t2)])
                    f.write('\n'.join(['', coords]))

            for fuse in range(len(self.fuselages)):
                onefuse = self.fuselages[fuse]
                wing = self.wings[0]
                xsecs = []
                xsecs.append(onefuse.xsecs[0])
                xsecs.append(onefuse.xsecs[-1])
                xsecs2 = []
                for xsec2 in wing.xsecs:
                    xsecs2.append(xsec2)
                t = xsecs[0].y_c + onefuse.xyz_le[1]
                t2 = ((wing.xyz_le[0] + xsecs2[0].x_le) + (wing.xyz_le[0] + xsecs2[-1].x_le)) / 2
                coords = '      '.join(['    1', str(onewing + fuse + 2), str(t), str(t2)])
                f.write('\n'.join(['', coords]))

            corr_stab = {}

            for fuse in range(len(self.fuselages)):
                corr_stab.update({fuse + len(self.wings) + 1: [fuse + 1, fuse + 1 + np.floor(len(self.wings) / 2)]})

            for fuse in range(len(self.fuselages)):
                onefuse = self.fuselages[fuse]
                xsecs = []
                xsecs.append(onefuse.xsecs[0])
                xsecs.append(onefuse.xsecs[-1])
                horiz = self.wings[corr_stab[fuse + len(self.wings) + 1][0]]
                xsecs2 = []
                for xsec2 in horiz.xsecs:
                    xsecs2.append(xsec2)
                vert = self.wings[corr_stab[fuse + len(self.wings) + 1][1]]
                xsecs3 = []
                for xsec3 in vert.xsecs:
                    xsecs3.append(xsec3)
                t = xsecs2[0].x_le + horiz.xyz_le[0]
                t2 = 0
                t3 = xsecs3[0].x_le + vert.xyz_le[0]
                t4 = 1 + (xsecs2[0].z_le + horiz.xyz_le[2]) - (xsecs3[0].z_le + vert.xyz_le[2])
                coords = '     '.join(
                    ['', str(fuse + len(self.wings) + 1), str(corr_stab[fuse + len(self.wings) + 1][0] + 1), str(t),
                     str(t2)])
                coords2 = '     '.join(
                    ['', str(fuse + len(self.wings) + 1), str(corr_stab[fuse + len(self.wings) + 1][1] + 1), str(t3),
                     str(t4)])
                f.write('\n'.join(['', coords, coords2]))
            f.write('\n'.join(['', 'End']))
            '''
            The juicy stuff! This part of the code iterates over each wing and then subiterates (is that a word?) over 
            each wing's cross section. Along the way it collects information on chord length, angle, and coordinates of
            all the leading edges. It then writes all this info in a way that ASWing likes
            '''
            for onewing in range(len(self.wings)):
                wing = self.wings[onewing]
                if wing.name == "Main Wing":
                    xsecs = []
                    for xsec in wing.xsecs:
                        xsecs.append(xsec)
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
                    for sec in xsecs:
                        if max_le.get(max(max_le)) == 'sec.x_le':
                            t = sec.x_le
                        elif max_le.get(max(max_le)) == 'sec.y_le':
                            t = sec.y_le
                        elif max_le.get(max(max_le)) == 'sec.z_le':
                            t = sec.z_le
                        chordalfa.append('    '.join([str(t), str(sec.chord), str(sec.twist)]))
                        coords.append(
                            '    '.join([str(t), str(sec.x_le + wing.xyz_le[0]), str(sec.y_le + wing.xyz_le[1]),
                                         str(sec.z_le + wing.xyz_le[2])]))
                    f.write('\n'.join(['', '#============',
                                       ' '.join(['Beam', str(onewing + 1)]),
                                       wing.name,
                                       't    chord    twist',
                                       '\n'.join(chordalfa),
                                       '#',
                                       't    x    y    z',
                                       '\n'.join(coords),
                                       'End']))
                elif wing.name == "Horizontal Stabilizer":
                    xsecs = []
                    for xsec in wing.xsecs:
                        xsecs.append(xsec)
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
                    for sec in xsecs:
                        if max_le.get(max(max_le)) == 'sec.x_le':
                            t = sec.x_le
                        elif max_le.get(max(max_le)) == 'sec.y_le':
                            t = sec.y_le
                        elif max_le.get(max(max_le)) == 'sec.z_le':
                            t = sec.z_le
                        chordalfa.append('    '.join([str(t), str(sec.chord), str(sec.twist), str(0.07)]))
                        coords.append(
                            '    '.join([str(t), str(sec.x_le + wing.xyz_le[0]), str(sec.y_le + wing.xyz_le[1]),
                                         str(sec.z_le + wing.xyz_le[2])]))
                    f.write('\n'.join(['', '#============',
                                       ' '.join(['Beam', str(onewing + 1)]),
                                       wing.name,
                                       't    chord    twist dCLdF1',
                                       '\n'.join(chordalfa),
                                       '#',
                                       't    x    y    z',
                                       '\n'.join(coords),
                                       'End']))

                elif wing.name == "Vertical Stabilizer":
                    xsecs = []
                    for xsec in wing.xsecs:
                        xsecs.append(xsec)
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
                    for sec in xsecs:
                        if max_le.get(max(max_le)) == 'sec.x_le':
                            t = sec.x_le + 1
                        elif max_le.get(max(max_le)) == 'sec.y_le':
                            t = sec.y_le + 1
                        elif max_le.get(max(max_le)) == 'sec.z_le':
                            t = sec.z_le + 1
                        chordalfa.append('    '.join([str(t), str(sec.chord), str(sec.twist)]))
                        coords.append(
                            '    '.join([str(t), str(sec.x_le + wing.xyz_le[0]), str(sec.y_le + wing.xyz_le[1]),
                                         str(sec.z_le + wing.xyz_le[2])]))
                    f.write('\n'.join(['', '#============',
                                       ' '.join(['Beam', str(onewing + 1)]),
                                       wing.name,
                                       '  t    chord    twist',
                                       '\n'.join(chordalfa),
                                       '#',
                                       't    x    y    z',
                                       '\n'.join(coords),
                                       'End']))

            for fuse in range(len(self.fuselages)):
                onefuse = self.fuselages[fuse]
                xsecs = []
                xsecs.append(onefuse.xsecs[0])
                xsecs.append(onefuse.xsecs[-1])
                coords = []
                max_c = {abs(xsecs[1].x_c - xsecs[0].x_c): 'sec.x_c', \
                         abs(xsecs[1].y_c - xsecs[0].y_c): 'sec.y_c', \
                         abs(xsecs[1].z_c - xsecs[0].z_c): 'sec.z_c'}
                for sec in xsecs:
                    if max_c.get(max(max_c)) == 'sec.x_c':
                        t = sec.x_c
                    elif max_c.get(max(max_c)) == 'sec.y_c':
                        t = sec.y_c
                    elif max_c.get(max(max_c)) == 'sec.z_c':
                        t = sec.z_c
                    coords.append(
                        '    '.join([str(t), str(sec.x_c + onefuse.xyz_le[0]), str(sec.y_c + onefuse.xyz_le[1]),
                                     str(sec.z_c + onefuse.xyz_le[2])]))
                f.write('\n'.join(['', '#============',
                                   '  '.join(['Beam', str(onewing + fuse + 2)]),
                                   onefuse.name,
                                   't    x    y    z',
                                   '\n'.join(coords),
                                   'End']))

    def approximate_longitudinal_center_of_pressure(self):
        """
        Returns the approximate location of the center of pressure. Given as the area-weighted quarter chord of the wing.
        :return: [x, y, z] of the approximate center of pressure
        """
        areas = [wing.area(type="projected") for wing in self.wings]
        COPs = [wing.approximate_center_of_pressure() for wing in self.wings]

        total_area = cas.sum1(cas.vertcat(*areas))

        COP = cas.sum1(cas.vertcat(*[
            COPs[i]*areas[i]/total_area
            for i in range(len(self.wings))
        ]))

        return COP