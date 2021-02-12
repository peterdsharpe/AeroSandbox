def write_aswing(airplane, filepath=None):
    """
    Contributed by Brent Avery, Edited by Peter Sharpe. Work in progress.
    Writes a geometry file compatible with Mark Drela's ASWing.
    :param filepath: Filepath to write to. Should include ".asw" extension [string]
    :return: None
    """
    if filepath is None:
        filepath = "%s.asw" % airplane.name
    with open(filepath, "w+") as f:
        f.write('\n'.join(['#============',  # Name of Plane
                           'Name',
                           airplane.name,
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
                           '   '.join([str(airplane.s_ref), str(airplane.c_ref), str(airplane.b_ref)]),
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
        for onewing in range(1, len(airplane.wings)):
            wing = airplane.wings[onewing]
            if wing.name == "Horizontal Stabilizer":
                xsecs = []
                for xsec in wing.xsecs:
                    xsecs.append(xsec)
                t = xsecs[0].y_le + wing.xyz_le[1]
                coords = '       '.join(['    1', str(onewing + 1), str(t), '0'])
                f.write('\n'.join(['', coords]))
            if wing.name == "Vertical Stabilizer":
                wing2 = airplane.wings[int(np.ceil(onewing / 2))]
                wing3 = airplane.wings[0]
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

        for fuse in range(len(airplane.fuselages)):
            onefuse = airplane.fuselages[fuse]
            wing = airplane.wings[0]
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

        for fuse in range(len(airplane.fuselages)):
            corr_stab.update({fuse + len(airplane.wings) + 1: [fuse + 1, fuse + 1 + np.floor(len(airplane.wings) / 2)]})

        for fuse in range(len(airplane.fuselages)):
            onefuse = airplane.fuselages[fuse]
            xsecs = []
            xsecs.append(onefuse.xsecs[0])
            xsecs.append(onefuse.xsecs[-1])
            horiz = airplane.wings[corr_stab[fuse + len(airplane.wings) + 1][0]]
            xsecs2 = []
            for xsec2 in horiz.xsecs:
                xsecs2.append(xsec2)
            vert = airplane.wings[int(corr_stab[fuse + len(airplane.wings) + 1][1])]
            xsecs3 = []
            for xsec3 in vert.xsecs:
                xsecs3.append(xsec3)
            t = xsecs2[0].x_le + horiz.xyz_le[0]
            t2 = 0
            t3 = xsecs3[0].x_le + vert.xyz_le[0]
            t4 = 1 + (xsecs2[0].z_le + horiz.xyz_le[2]) - (xsecs3[0].z_le + vert.xyz_le[2])
            coords = '     '.join(
                ['', str(fuse + len(airplane.wings) + 1), str(corr_stab[fuse + len(airplane.wings) + 1][0] + 1), str(t),
                 str(t2)])
            coords2 = '     '.join(
                ['', str(fuse + len(airplane.wings) + 1), str(corr_stab[fuse + len(airplane.wings) + 1][1] + 1), str(t3),
                 str(t4)])
            f.write('\n'.join(['', coords, coords2]))
        f.write('\n'.join(['', 'End']))
        '''
        The juicy stuff! This part of the code iterates over each wing and then subiterates (is that a word?) over 
        each wing's cross section. Along the way it collects information on chord length, angle, and coordinates of
        all the leading edges. It then writes all this info in a way that ASWing likes
        '''
        for onewing in range(len(airplane.wings)):
            wing = airplane.wings[onewing]
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

        for fuse in range(len(airplane.fuselages)):
            onefuse = airplane.fuselages[fuse]
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
