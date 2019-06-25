import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .Plotting import *
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv


class Airplane:
    def __init__(self,
                 name="Untitled Airplane",
                 xyz_ref=[0.0, 0.0, 0.0],
                 wings=[],
                 s_ref=1.0,
                 c_ref=1.0,
                 b_ref=1.0,
                 ):
        self.name = name
        self.xyz_ref = np.array(xyz_ref)
        self.wings = wings
        self.s_ref = s_ref
        self.c_ref = c_ref
        self.b_ref = b_ref

    def draw(self):
        # Using PyVista Polydata format
        vertices = np.empty((0, 3))
        faces = np.empty((0))

        for wing in self.wings:
            wing_vertices = np.empty((0, 3))
            wing_tri_faces = np.empty((0, 4))
            wing_quad_faces = np.empty((0, 5))
            for i in range(len(wing.sections) - 1):
                is_last_section = i == len(wing.sections) - 2

                le_start = wing.sections[i].xyz_le + wing.xyz_le
                te_start = wing.sections[i].xyz_te() + wing.xyz_le
                wing_vertices = np.vstack((wing_vertices, le_start, te_start))

                wing_quad_faces = np.vstack((
                    wing_quad_faces,
                    np.expand_dims(np.array([4, 2 * i + 0, 2 * i + 1, 2 * i + 3, 2 * i + 2]), 0)
                ))

                if is_last_section:
                    le_end = wing.sections[i + 1].xyz_le + wing.xyz_le
                    te_end = wing.sections[i + 1].xyz_te() + wing.xyz_le
                    wing_vertices = np.vstack((wing_vertices, le_end, te_end))

            vertices_starting_index = len(vertices)
            wing_quad_faces_reformatted = np.ndarray.copy(wing_quad_faces)
            wing_quad_faces_reformatted[:, 1:] = wing_quad_faces[:, 1:] + vertices_starting_index
            wing_quad_faces_reformatted = np.reshape(wing_quad_faces_reformatted, (-1), order='C')
            vertices = np.vstack((vertices, wing_vertices))
            faces = np.hstack((faces, wing_quad_faces_reformatted))

            if wing.symmetric:
                vertices_starting_index = len(vertices)
                reflect_over_XZ_plane(wing_vertices)
                wing_quad_faces_reformatted = np.ndarray.copy(wing_quad_faces)
                wing_quad_faces_reformatted[:, 1:] = wing_quad_faces[:, 1:] + vertices_starting_index
                wing_quad_faces_reformatted = np.reshape(wing_quad_faces_reformatted, (-1), order='C')
                vertices = np.vstack((vertices, wing_vertices))
                faces = np.hstack((faces, wing_quad_faces_reformatted))

        plotter = pv.Plotter()

        wing_surfaces = pv.PolyData(vertices, faces)
        plotter.add_mesh(wing_surfaces, color='#7EFC8F', show_edges=True, smooth_shading=True)

        xyz_ref = pv.PolyData(self.xyz_ref)
        plotter.add_points(xyz_ref, color='#50C7C7', point_size=10)

        plotter.show_grid(color='#444444')
        plotter.set_background(color="black")
        plotter.show(cpos=(-1, -1, 1), full_screen=False)

    def draw_legacy(self,
                    show=True,
                    fig_to_plot_on=None,
                    ax_to_plot_on=None
                    ):

        # Setup
        if fig_to_plot_on == None or ax_to_plot_on == None:
            fig, ax = fig3d()
            fig.set_size_inches(12, 9)
        else:
            fig = fig_to_plot_on
            ax = ax_to_plot_on

        # TODO plot bodies

        # Plot wings
        for wing in self.wings:

            for i in range(len(wing.sections) - 1):
                le_start = wing.sections[i].xyz_le + wing.xyz_le
                le_end = wing.sections[i + 1].xyz_le + wing.xyz_le
                te_start = wing.sections[i].xyz_te() + wing.xyz_le
                te_end = wing.sections[i + 1].xyz_te() + wing.xyz_le

                points = np.vstack((le_start, le_end, te_end, te_start, le_start))
                x = points[:, 0]
                y = points[:, 1]
                z = points[:, 2]

                ax.plot(x, y, z, color='#cc0039')

                if wing.symmetric:
                    ax.plot(x, -1 * y, z, color='#cc0039')

        # Plot reference point
        x = self.xyz_ref[0]
        y = self.xyz_ref[1]
        z = self.xyz_ref[2]
        ax.scatter(x, y, z)

        set_axes_equal(ax)
        plt.tight_layout()
        if show:
            plt.show()

    def set_ref_dims_from_wing(self,
                               main_wing_index=0
                               ):
        main_wing = self.wings[main_wing_index]

        self.s_ref = main_wing.area_wetted()
        self.b_ref = main_wing.span()
        self.c_ref = main_wing.area_wetted() / main_wing.span()

        pass
        # TODO set dims

    def set_paneling_everywhere(self, n_chordwise_panels, n_spanwise_panels):
        # Sets the chordwise and spanwise paneling everywhere to a specified value. Useful for quickly changing the fidelity of your simulation.
        for wing in self.wings:
            wing.chordwise_panels = n_chordwise_panels
            for wingsection in wing.sections:
                wingsection.spanwise_panels = n_spanwise_panels

    def get_bounding_cube(self):
        # Finds the axis-aligned cube that encloses the airplane with the smallest size.
        # Returns x, y, z, and s, where x, y, and z are the coordinates of the cube center, and s is half of the side length.

        # Get vertices to enclose
        vertices = None
        for wing in self.wings:
            for wingsection in wing.sections:
                if vertices is None:
                    vertices = wingsection.xyz_le + wing.xyz_le
                else:
                    vertices = np.vstack((
                        vertices,
                        wingsection.xyz_le + wing.xyz_le
                    ))
                vertices = np.vstack((
                    vertices,
                    wingsection.xyz_te() + wing.xyz_le
                ))

                if wing.symmetric:
                    vertices = np.vstack((
                        vertices,
                        reflect_over_XZ_plane(wingsection.xyz_le + wing.xyz_le)
                    ))
                    vertices = np.vstack((
                        vertices,
                        reflect_over_XZ_plane(wingsection.xyz_te() + wing.xyz_le)
                    ))

        # Enclose them
        x_max = np.max(vertices[:, 0])
        y_max = np.max(vertices[:, 1])
        z_max = np.max(vertices[:, 2])
        x_min = np.min(vertices[:, 0])
        y_min = np.min(vertices[:, 1])
        z_min = np.min(vertices[:, 2])

        x = np.mean((x_max, x_min))
        y = np.mean((y_max, y_min))
        z = np.mean((z_max, z_min))
        s = 0.5 * np.max((
            x_max - x_min,
            y_max - y_min,
            z_max - z_min
        ))

        return x, y, z, s


class Wing:
    def __init__(self,
                 name="Untitled Wing",
                 xyz_le=[0, 0, 0],
                 sections=[],
                 symmetric=True,
                 incidence_angle=0,
                 vlm_chordwise_panels=10,
                 vlm_chordwise_spacing="cosine",
                 ):
        self.name = name
        self.xyz_le = np.array(xyz_le)
        self.sections = sections
        self.symmetric = symmetric
        self.incidence_angle = incidence_angle
        self.chordwise_panels = vlm_chordwise_panels
        self.chordwise_spacing = vlm_chordwise_spacing

    def area_wetted(self):
        # Returns the wetted area of a wing.
        area = 0
        for i in range(len(self.sections) - 1):
            chord_eff = (self.sections[i].chord
                         + self.sections[i + 1].chord) / 2
            this_xyz_te = self.sections[i].xyz_te()
            that_xyz_te = self.sections[i + 1].xyz_te()
            span_le_eff = np.hypot(
                self.sections[i].xyz_le[1] - self.sections[i + 1].xyz_le[1],
                self.sections[i].xyz_le[2] - self.sections[i + 1].xyz_le[2]
            )
            span_te_eff = np.hypot(
                this_xyz_te[1] - that_xyz_te[1],
                this_xyz_te[2] - that_xyz_te[2]
            )
            span_eff = (span_le_eff + span_te_eff) / 2
            area += chord_eff * span_eff
        if self.symmetric:
            area *= 2
        return area

    def area_projected(self):
        # Returns the area of the wing as projected onto the XY plane.
        area = 0
        for i in range(len(self.sections) - 1):
            chord_eff = (self.sections[i].chord
                         + self.sections[i + 1].chord) / 2
            this_xyz_te = self.sections[i].xyz_te()
            that_xyz_te = self.sections[i + 1].xyz_te()
            span_le_eff = np.abs(
                self.sections[i].xyz_le[1] - self.sections[i + 1].xyz_le[1]
            )
            span_te_eff = np.abs(
                this_xyz_te[1] - that_xyz_te[1]
            )
            span_eff = (span_le_eff + span_te_eff) / 2
            area += chord_eff * span_eff
        if self.symmetric:
            area *= 2
        return area

    def span(self):
        # Returns the span (y-distance between the root of the wing and the tip). If symmetric, this is doubled to obtain the full span.
        spans = []
        for i in range(len(self.sections)):
            spans.append(np.abs(self.sections[i].xyz_le[1] - self.sections[0].xyz_le[1]))
        span = np.max(spans)
        if self.symmetric:
            span *= 2
        return span

    def aspect_ratio(self):
        return self.span() ** 2 / self.area_wetted()


class WingSection:

    def __init__(self,
                 xyz_le=[0, 0, 0],
                 chord=0,
                 twist=0,
                 airfoil=[],
                 vlm_spanwise_panels=10,
                 vlm_spanwise_spacing="cosine"
                 ):
        self.xyz_le = np.array(xyz_le)
        self.chord = chord
        self.twist = twist
        self.airfoil = airfoil
        self.spanwise_panels = vlm_spanwise_panels
        self.spanwise_spacing = vlm_spanwise_spacing

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
        self.get_coordinates()  # populates self.coordinates

    def get_coordinates(self):
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

                    # Referencing https://en.wikipedia.org/wiki/NACA_airfoil#Equation_for_a_cambered_4-digit_NACA_airfoil from here on out

                    # Make uncambered coordinates
                    x_t = 0.5 + 0.5 * np.cos(
                        np.linspace(np.pi, 0, n_points_per_side))  # Generate some cosine-spaced points
                    y_t = 5 * thickness * (
                            + 0.2969 * np.power(x_t, 0.5)
                            - 0.1260 * x_t
                            - 0.3516 * np.power(x_t, 2)
                            + 0.2843 * np.power(x_t, 3)
                            - 0.1015 * np.power(x_t, 4)
                    )

                    if camber_loc == 0:
                        camber_loc = 0.5  # prevents divide by zero errors for things like naca0012's.

                    # Get camber
                    y_c_piece1 = max_camber / camber_loc ** 2 * (
                            2 * camber_loc * x_t[x_t <= camber_loc]
                            - x_t[x_t <= camber_loc] ** 2
                    )
                    y_c_piece2 = max_camber / (1 - camber_loc) ** 2 * (
                            (1 - 2 * camber_loc) +
                            2 * camber_loc * x_t[x_t > camber_loc]
                            - x_t[x_t > camber_loc] ** 2
                    )
                    y_c = np.hstack((y_c_piece1, y_c_piece2))

                    # Get camber slope
                    dycdx_piece1 = 2 * max_camber / camber_loc ** 2 * (
                        camber_loc - x_t[x_t <=camber_loc]
                    )
                    dycdx_piece2 = 2 * max_camber / (1-camber_loc)**2 *(
                        camber_loc - x_t[x_t > camber_loc]
                    )
                    dycdx = np.hstack((dycdx_piece1,dycdx_piece2))
                    theta = np.arctan(dycdx)

                    # Combine everything
                    x_U = x_t - y_t*np.sin(theta)
                    x_L = x_t + y_t*np.sin(theta)
                    y_U = y_c + y_t*np.cos(theta)
                    y_L = y_c - y_t*np.cos(theta)

                    # Flip upper surface so it's back to front
                    x_U, y_U = np.flip(x_U), np.flip(y_U)

                    # Trim 1 point from lower surface so there's no overlap
                    x_L, y_L = x_L[1:], y_L[1:]

                    x = np.hstack((x_U, x_L))
                    y = np.hstack((y_U, y_L))

                    coordinates = np.column_stack((x,y))

                    self.coordinates = coordinates
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
            return

        except FileNotFoundError:
            print("File was not found in airfoil database!")

    def normalize(self):
        # Alters the airfoil's coordinates so that x_min is exactly 0 and x_max is exactly 1.
        pass # TODO do this function

    def get_mean_camber_line(self):
        # Populates self.mean_camber_line, an Nx2 array that contains the mean camber line coordinates ordered front to back
        n_points = 150

        x = 0.5 + 0.5 * np.cos(
            np.linspace(np.pi, 0, n_points))  # Generate some cosine-spaced points




    def draw(self):
        # Get coordinates if they don't already exist
        if not 'self.coordinates' in locals():
            print("You must call read_coordinates() on an Airfoil before drawing it. Automatically doing that...")
            self.get_coordinates()

        plt.plot(self.coordinates[:, 0], self.coordinates[:, 1])
        plt.xlim((-0.05, 1.05))
        plt.ylim((-0.5, 0.5))
        plt.axis('equal')

    def get_2D_aero_data(self):
        pass
        # TODO do this

    def compute_mean_camber_line(self):
        pass

    # TODO do this

    def get_point_on_chord_line(self, chordfraction):
        return np.array([chordfraction, 0])

    def get_point_on_camber_line(self, chordfraction):
        pass


def reflect_over_XZ_plane(input_vector):
    # Takes in a vector or an array and flips the y-coordinates.
    output_vector = input_vector
    shape = np.shape(output_vector)
    if len(shape) == 1 and shape[0] == 3:
        output_vector[1] *= -1
    elif len(shape) == 2 and shape[1] == 3:
        output_vector[:, 1] *= -1
    else:
        raise Exception("Invalid input for reflect_over_XZ_plane!")

    return output_vector
