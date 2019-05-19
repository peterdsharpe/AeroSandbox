import numpy as np
import matplotlib.pyplot as plt
import vpython as vp



class Airplane:
    def __init__(self,
                 name="Untitled",
                 xyz_ref=[0, 0, 0],
                 wings=[],
                 s_ref=1,
                 c_ref=1,
                 b_ref=1
                 ):
        self.name = name
        self.xyz_ref = np.array(xyz_ref)
        self.wings = wings
        self.s_ref = s_ref
        self.c_ref = c_ref
        self.b_ref = b_ref

    def plot_geometry(self,
                      new_figure=True
                      ):
        self.new_figure = new_figure

        # TODO format
        myscene = vp.canvas(
            title="Airplane Geometry (ctrl + mouse to rotate, shift + mouse to pan)",
            width=800,
            height=800
        )
        myscene.forward = vp.vec(1, 1, -1)
        myscene.up = vp.vec(0, 0, 1)
        myscene.center = vp.vec(
            self.xyz_ref[0],
            self.xyz_ref[1],
            self.xyz_ref[2],
        )

        # TODO plot bodies

        # TODO plot wings
        for wing in self.wings:
            print("Drawing Wing: " + wing.name)
            wing.plot_geometry(new_figure=False)

    def set_ref_dims_from_wing(self):
        pass
        # TODO set dims


class Wing:
    def __init__(self,
                 name="Untitled",
                 xyz_le=[0, 0, 0],
                 sections=[],
                 symmetric=True,
                 incidence_angle=0
                 ):
        self.name = name
        self.xyz_le = np.array(xyz_le)
        self.sections = sections
        self.symmetric = symmetric
        self.incidence_angle = incidence_angle

    def plot_geometry(self,
                      new_figure=False
                      ):
        self.new_figure = new_figure

        # TODO plot wing surfaces
        for i in range(len(self.sections) - 1):
            le_start = self.sections[i].xyz_le + self.xyz_le
            le_end = self.sections[i + 1].xyz_le + self.xyz_le
            te_start = self.sections[i].xyz_te() + self.xyz_le
            te_end = self.sections[i + 1].xyz_te() + self.xyz_le

            # format for VPython
            le_start_vec = vp.vec(
                le_start[0],
                le_start[1],
                le_start[2]
            )
            le_end_vec = vp.vec(
                le_end[0],
                le_end[1],
                le_end[2]
            )
            te_start_vec = vp.vec(
                te_start[0],
                te_start[1],
                te_start[2]
            )
            te_end_vec = vp.vec(
                te_end[0],
                te_end[1],
                te_end[2]
            )

            curve = vp.curve(le_start_vec, le_end_vec, te_end_vec, te_start_vec, le_start_vec)

            le_start_vert = vp.vertex(pos=le_start_vec)
            le_end_vert = vp.vertex(pos=le_end_vec)
            te_start_vert = vp.vertex(pos=te_start_vec)
            te_end_vert = vp.vertex(pos=te_end_vec)

            T1 = vp.triangle(vs=[
                le_start_vert,
                le_end_vert,
                te_start_vert,
            ])
            T2 = vp.triangle(vs=[
                te_start_vert,
                te_end_vert,
                le_end_vert,
            ])


            if self.symmetric:
                le_start_vec = vp.vec(
                    le_start[0],
                    -le_start[1],
                    le_start[2]
                )
                le_end_vec = vp.vec(
                    le_end[0],
                    -le_end[1],
                    le_end[2]
                )
                te_start_vec = vp.vec(
                    te_start[0],
                    -te_start[1],
                    te_start[2]
                )
                te_end_vec = vp.vec(
                    te_end[0],
                    -te_end[1],
                    te_end[2]
                )
                curve_sym = vp.curve(le_start_vec, le_end_vec, te_end_vec, te_start_vec, le_start_vec)

                le_start_vert = vp.vertex(pos=le_start_vec)
                le_end_vert = vp.vertex(pos=le_end_vec)
                te_start_vert = vp.vertex(pos=te_start_vec)
                te_end_vert = vp.vertex(pos=te_end_vec)

                T1 = vp.triangle(vs=[
                    le_start_vert,
                    le_end_vert,
                    te_start_vert,
                ])
                T2 = vp.triangle(vs=[
                    te_start_vert,
                    te_end_vert,
                    le_end_vert,
                ])

    def area(self):
        pass
        # TODO this

    def span(self):
        pass
        # TODO this

    def aspect_ratio(self):
        pass
        # TODO this


class WingSection:

    def __init__(self,
                 xyz_le=[0, 0, 0],
                 chord=0,
                 twist=0,
                 airfoil=[],
                 chordwise_panels=30,
                 chordwise_spacing="cosine",
                 spanwise_panels=30,
                 spanwise_spacing="cosine"
                 ):
        self.xyz_le = np.array(xyz_le)
        self.chord = chord
        self.twist = twist
        self.airfoil = airfoil
        self.chordwise_panels = chordwise_panels
        self.chordwise_spacing = chordwise_spacing
        self.spanwise_panels = spanwise_panels
        self.spanwise_spacing = spanwise_spacing

    def xyz_te(self):
        xyz_te = self.xyz_le + self.chord * np.array(
            [np.cos(np.radians(self.twist)),
             0,
             -np.sin(np.radians(self.twist))
             ])

        return xyz_te


class Airfoil:
    cached_airfoils = []

    def __init__(self,
                 name="naca0012"
                 ):
        self.name = name

        self.read_coordinates()
        self.get_2D_aero_data()

    def read_coordinates(self):
        pass
        # TODO do this

    def get_2D_aero_data(self):
        pass
        # TODO do this
