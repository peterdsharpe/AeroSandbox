import numpy as np
import scipy.interpolate as sp_interp
import matplotlib as mpl
import matplotlib.pyplot as plt
from .Plotting import *
import pyvista as pv
import copy


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
        # Sets the chordwise and spanwise paneling everywhere to a specified value.
        # Useful for quickly changing the fidelity of your simulation.
        for wing in self.wings:
            wing.chordwise_panels = n_chordwise_panels
            for wingsection in wing.sections:
                wingsection.spanwise_panels = n_spanwise_panels

    def get_bounding_cube(self):
        # Finds the axis-aligned cube that encloses the airplane with the smallest size.
        # Returns x, y, z, and s, where x, y, and z are the coordinates of the cube center,
        # and s is half of the side length.

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
        # Returns the span (y-distance between the root of the wing and the tip).
        # If symmetric, this is doubled to obtain the full span.
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
                 name="naca0012",
                 coordinates=None,
                 ):

        self.name = name

        if coordinates is not None:
            self.coordinates = coordinates
        else:
            self.populate_coordinates()  # populates self.coordinates
        assert hasattr(self,'coordinates'), "Couldn't figure out the coordinates of this airfoil! You need to either \
        a) use a name corresponding to an airfoil in the UIUC Airfoil Database or \
        b) provide your own coordinates in the constructor, such as Airfoil(""MyFoilName"", <Nx2 array of coordinates>)."

        self.normalize()

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
                            camber_loc - x_t[x_t <= camber_loc]
                    )
                    dycdx_piece2 = 2 * max_camber / (1 - camber_loc) ** 2 * (
                            camber_loc - x_t[x_t > camber_loc]
                    )
                    dycdx = np.hstack((dycdx_piece1, dycdx_piece2))
                    theta = np.arctan(dycdx)

                    # Combine everything
                    x_U = x_t - y_t * np.sin(theta)
                    x_L = x_t + y_t * np.sin(theta)
                    y_U = y_c + y_t * np.cos(theta)
                    y_L = y_c - y_t * np.cos(theta)

                    # Flip upper surface so it's back to front
                    x_U, y_U = np.flip(x_U), np.flip(y_U)

                    # Trim 1 point from lower surface so there's no overlap
                    x_L, y_L = x_L[1:], y_L[1:]

                    x = np.hstack((x_U, x_L))
                    y = np.hstack((y_U, y_L))

                    coordinates = np.column_stack((x, y))

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
        # Alters the airfoil's coordinates to exactly achieve several goals:
        #   # x_le == 0
        #   # y_le == 0
        #   # average( y_te_upper, y_te_lower ) == 0
        #   # x_te == 1
        # The first two goals are achieved by translating in x and y. The third goal is achieved by rotating about (0,0).
        # The fourth goal is achieved by uniform scaling.

        # Goals 1 and 2
        LE_point_original = self.coordinates[self.LE_index(), :]
        assert abs(LE_point_original[
                       0]) < 0.02, "The leading edge point x_coordinate looks like it's at a really weird location! \
                       Are you sure this isn't bad airfoil geometry?"
        assert abs(LE_point_original[
                       1]) < 0.02, "The leading edge point x_coordinate looks like it's at a really weird location! \
                       Are you sure this isn't bad airfoil geometry?"
        self.coordinates -= LE_point_original

        # Goal 3
        TE_point_pre_rotation = (self.coordinates[0, :] + self.coordinates[-1, :]) / 2
        rotation_angle = -np.arctan(TE_point_pre_rotation[1] / TE_point_pre_rotation[
            0])  # You need to rotate this many radians counterclockwise
        assert abs(np.degrees(
            rotation_angle)) < 0.5, "The foil appears to be really weirdly rotated! \
            Are you sure this isn't bad airfoil geometry?"
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        self.coordinates = np.transpose(rotation_matrix @ np.transpose(self.coordinates))

        # Goal 4
        x_max = np.max(self.coordinates[:, 0])
        assert x_max <= 1.02 and x_max >= 0.98, "x_max is really weird! Are you sure this isn't bad airfoil geometry?"
        scale_factor = 1 / x_max
        self.coordinates *= scale_factor

    def draw(self, new_figure=True):
        # Get coordinates if they don't already exist
        if not hasattr(self, 'coordinates'):
            print("You must call read_coordinates() on an Airfoil before drawing it. Automatically doing that...")
            self.populate_coordinates()

        if new_figure:
            plt.figure()
            plt.title(self.name + " Coordinates")
        plt.plot(self.coordinates[:, 0], self.coordinates[:, 1])
        plt.xlim((-0.05, 1.05))
        plt.ylim((-0.5, 0.5))
        plt.axis('equal')

    def calculate_2D_aero_data(self):
        pass
        # TODO xfoil?

    def LE_index(self):
        # Returns the index of the leading-edge point.
        return np.argmin(self.coordinates[:, 0])

    def lower_coordinates(self):
        # Returns a matrix (N by 2) of [x y] coordinates that describe the lower surface of the airfoil.
        # Order is from leading edge to trailing edge.
        # Includes the leading edge point; be careful about duplicates if using this method in conjunction with self.upper_coordinates().
        return self.coordinates[self.LE_index():, :]

    def upper_coordinates(self):
        # Returns a matrix (N by 2) of [x y] coordinates that describe the upper surface of the airfoil.
        # Order is from trailing edge to leading edge.
        # Includes the leading edge point; be careful about duplicates if using this method in conjunction with self.lower_coordinates().
        return self.coordinates[:self.LE_index() + 1, :]

    def get_thickness_at_chord_fraction(self, chord_fraction):
        # Returns the (interpolated) camber at a given location(s). The location is specified by the chord fraction, as measured from the leading edge. Thickness is nondimensionalized by chord (i.e. this function returns t/c at a given x/c).
        chord = np.max(self.coordinates[:, 0]) - np.min(
            self.coordinates[:, 0])  # This should always be 1, but this is just coded for robustness.

        x = chord_fraction * chord + min(self.coordinates[:, 0])

        upperCoors = self.upper_coordinates()
        lowerCoors = self.lower_coordinates()

        y_upper_func = sp_interp.interp1d(x=upperCoors[:, 0], y=upperCoors[:, 1], copy=False, fill_value='extrapolate')
        y_lower_func = sp_interp.interp1d(x=lowerCoors[:, 0], y=lowerCoors[:, 1], copy=False, fill_value='extrapolate')

        y_upper = y_upper_func(x)
        y_lower = y_lower_func(x)

        thickness = np.maximum(y_upper - y_lower, 0)

        return thickness

    def get_camber_at_chord_fraction(self, chord_fraction):
        # Returns the (interpolated) camber at a given location(s). The location is specified by the chord fraction, as measured from the leading edge. Camber is nondimensionalized by chord (i.e. this function returns camber/c at a given x/c).
        chord = np.max(self.coordinates[:, 0]) - np.min(
            self.coordinates[:, 0])  # This should always be 1, but this is just coded for robustness.

        x = chord_fraction * chord + min(self.coordinates[:, 0])

        upperCoors = self.upper_coordinates()
        lowerCoors = self.lower_coordinates()

        y_upper_func = sp_interp.interp1d(x=upperCoors[:, 0], y=upperCoors[:, 1], copy=False, fill_value='extrapolate')
        y_lower_func = sp_interp.interp1d(x=lowerCoors[:, 0], y=lowerCoors[:, 1], copy=False, fill_value='extrapolate')

        y_upper = y_upper_func(x)
        y_lower = y_lower_func(x)

        camber = (y_upper + y_lower) / 2

        return camber

    def TE_thickness(self):
        # Returns the thickness of the trailing edge of the airfoil, in nondimensional (y/c) units.
        return np.abs(self.coordinates[0, 1] - self.coordinates[-1, 1])

    def TE_angle(self):
        # Returns the trailing edge angle of the airfoil, in degrees
        upper_TE_vec = self.coordinates[0, :] - self.coordinates[1, :]
        lower_TE_vec = self.coordinates[-1, :] - self.coordinates[-2, :]

        return np.degrees(np.arctan2(
            upper_TE_vec[0] * lower_TE_vec[1] - upper_TE_vec[1] * lower_TE_vec[0],
            upper_TE_vec[0] * lower_TE_vec[0] + upper_TE_vec[1] * upper_TE_vec[1]
        ))

    def area(self):
        # Returns the area of the airfoil, in nondimensional (chord-normalized) units.
        x = self.coordinates[:, 0]
        y = self.coordinates[:, 1]
        x_n = np.roll(x, -1)  # x_next, or x_i+1
        y_n = np.roll(y, -1)  # y_next, or y_i+1

        a = x * y_n - x_n * y  # a is the area of the triangle bounded by a given point, the next point, and the origin.

        A = 0.5 * np.sum(a)  # area

        return A

    def centroid(self):
        # Returns the centroid of the airfoil, in nondimensional (chord-normalized) units.
        x = self.coordinates[:, 0]
        y = self.coordinates[:, 1]
        x_n = np.roll(x, -1)  # x_next, or x_i+1
        y_n = np.roll(y, -1)  # y_next, or y_i+1

        a = x * y_n - x_n * y  # a is the area of the triangle bounded by a given point, the next point, and the origin.

        A = 0.5 * np.sum(a)  # area

        x_c = 1 / (6 * A) * np.sum(a * (x + x_n))
        y_c = 1 / (6 * A) * np.sum(a * (y + y_n))
        centroid = np.array([x_c, y_c])

        return centroid

    def Ixx(self):
        # Returns the nondimensionalized moment of inertia, taken about the centroid.
        x = self.coordinates[:, 0]
        y = self.coordinates[:, 1]
        x_n = np.roll(x, -1)  # x_next, or x_i+1
        y_n = np.roll(y, -1)  # y_next, or y_i+1

        a = x * y_n - x_n * y  # a is the area of the triangle bounded by a given point, the next point, and the origin.

        A = 0.5 * np.sum(a)  # area

        x_c = 1 / (6 * A) * np.sum(a * (x + x_n))
        y_c = 1 / (6 * A) * np.sum(a * (y + y_n))
        centroid = np.array([x_c, y_c])

        Ixx = 1 / 12 * np.sum(a * (np.power(y, 2) + y * y_n + np.power(y_n, 2)))

        Iuu = Ixx - A * centroid[1] ** 2

        return Iuu

    def Iyy(self):
        # Returns the nondimensionalized moment of inertia, taken about the centroid.
        x = self.coordinates[:, 0]
        y = self.coordinates[:, 1]
        x_n = np.roll(x, -1)  # x_next, or x_i+1
        y_n = np.roll(y, -1)  # y_next, or y_i+1

        a = x * y_n - x_n * y  # a is the area of the triangle bounded by a given point, the next point, and the origin.

        A = 0.5 * np.sum(a)  # area

        x_c = 1 / (6 * A) * np.sum(a * (x + x_n))
        y_c = 1 / (6 * A) * np.sum(a * (y + y_n))
        centroid = np.array([x_c, y_c])

        Iyy = 1 / 12 * np.sum(a * (np.power(x, 2) + x * x_n + np.power(x_n, 2)))

        Ivv = Iyy - A * centroid[0] ** 2

        return Ivv

    def Ixy(self):
        # Returns the nondimensionalized moment of inertia, taken about the centroid.
        x = self.coordinates[:, 0]
        y = self.coordinates[:, 1]
        x_n = np.roll(x, -1)  # x_next, or x_i+1
        y_n = np.roll(y, -1)  # y_next, or y_i+1

        a = x * y_n - x_n * y  # a is the area of the triangle bounded by a given point, the next point, and the origin.

        A = 0.5 * np.sum(a)  # area

        x_c = 1 / (6 * A) * np.sum(a * (x + x_n))
        y_c = 1 / (6 * A) * np.sum(a * (y + y_n))
        centroid = np.array([x_c, y_c])

        Ixy = 1 / 24 * np.sum(a * (x * y_n + 2 * x * y + 2 * x_n * y_n + x_n * y))

        Iuv = Ixy - A * centroid[0] * centroid[1]

        return Iuv

    def J(self):
        # Returns the nondimensionalized moment of inertia, taken about the centroid.
        x = self.coordinates[:, 0]
        y = self.coordinates[:, 1]
        x_n = np.roll(x, -1)  # x_next, or x_i+1
        y_n = np.roll(y, -1)  # y_next, or y_i+1

        a = x * y_n - x_n * y  # a is the area of the triangle bounded by a given point, the next point, and the origin.

        A = 0.5 * np.sum(a)  # area

        x_c = 1 / (6 * A) * np.sum(a * (x + x_n))
        y_c = 1 / (6 * A) * np.sum(a * (y + y_n))
        centroid = np.array([x_c, y_c])

        Ixx = 1 / 12 * np.sum(a * (np.power(y, 2) + y * y_n + np.power(y_n, 2)))

        Iyy = 1 / 12 * np.sum(a * (np.power(x, 2) + x * x_n + np.power(x_n, 2)))

        J = Ixx + Iyy

        return J

    def repanel(self, n_points_per_side=100):
        # Repanels an airfoil with cosine-spaced coordinates on the upper and lower surfaces.
        # Inputs:
        #   # n_points_per_side is the number of points PER SIDE (upper and lower) of the airfoil. 100 is a good number.
        # Notes: The number of points defining the final airfoil will be n_points_per_side*2-1,
        # since one point (the leading edge point) is shared by both the upper and lower surfaces.

        upper_original_coors = self.upper_coordinates()  # Note: includes leading edge point, be careful about duplicates
        lower_original_coors = self.lower_coordinates()  # Note: includes leading edge point, be careful about duplicates

        # Find distances between coordinates, assuming linear interpolation
        upper_distances_between_points = np.sqrt(
            np.power(upper_original_coors[:-1, 0] - upper_original_coors[1:, 0], 2) +
            np.power(upper_original_coors[:-1, 1] - upper_original_coors[1:, 1], 2)
        )
        lower_distances_between_points = np.sqrt(
            np.power(lower_original_coors[:-1, 0] - lower_original_coors[1:, 0], 2) +
            np.power(lower_original_coors[:-1, 1] - lower_original_coors[1:, 1], 2)
        )
        upper_distances_from_TE = np.hstack((0, np.cumsum(upper_distances_between_points)))
        lower_distances_from_LE = np.hstack((0, np.cumsum(lower_distances_between_points)))
        upper_distances_from_TE_normalized = upper_distances_from_TE / upper_distances_from_TE[-1]
        lower_distances_from_LE_normalized = lower_distances_from_LE / lower_distances_from_LE[-1]

        # Generate a cosine-spaced list of points from 0 to 1
        s = cosspace(n_points=n_points_per_side)

        x_upper_func = sp_interp.PchipInterpolator(upper_distances_from_TE_normalized, upper_original_coors[:, 0])
        y_upper_func = sp_interp.PchipInterpolator(upper_distances_from_TE_normalized, upper_original_coors[:, 1])
        x_lower_func = sp_interp.PchipInterpolator(lower_distances_from_LE_normalized, lower_original_coors[:, 0])
        y_lower_func = sp_interp.PchipInterpolator(lower_distances_from_LE_normalized, lower_original_coors[:, 1])

        x_coors = np.hstack((x_upper_func(s), x_lower_func(s)[1:]))
        y_coors = np.hstack((y_upper_func(s), y_lower_func(s)[1:]))

        coordinates = np.column_stack((x_coors, y_coors))

        self.coordinates = coordinates

        self.normalize()


def blend_airfoils(
        airfoil1,
        airfoil2,
        blend_fraction
):
    # Returns a new airfoil that is a blend of the two airfoils.
    # Inputs:
    #   # airfoil1: The first airfoil to use
    #   # airfoil2: The second airfoil to use
    #   # blend_fraction: a fraction (between 0 and 1) that specifies how much of the second airfoil to blend in.
    #   #   # 0 will give airfoil1, 1 will give airfoil2, 0.5 will give exactly in between.

    foil1 = copy.deepcopy(airfoil1)
    foil2 = copy.deepcopy(airfoil2)

    if blend_fraction==0:
        return foil1
    if blend_fraction==1:
        return foil2
    assert blend_fraction>=0 and blend_fraction<=1, "blend_fraction is out of the valid range of 0 to 1!"

    # Repanel to ensure the same number of points and the same point distribution on both airfoils.
    foil1.repanel(n_points_per_side=200)
    foil2.repanel(n_points_per_side=200)

    blended_coordinates = (1-blend_fraction)*foil1.coordinates + blend_fraction * foil2.coordinates

    new_airfoil = Airfoil(name="Blended Airfoils", coordinates = blended_coordinates)

    return new_airfoil


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


def cosspace(min=0, max=1, n_points=50):
    return 0.5 + 0.5 * np.cos(np.linspace(np.pi, 0, n_points))
