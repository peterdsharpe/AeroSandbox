import autograd.numpy as np
import scipy.interpolate as sp_interp
import matplotlib as mpl
import matplotlib.pyplot as plt
from .plotting import *
import pyvista as pv
import copy

import cProfile
import functools
import os


def profile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        try:
            profiler.enable()
            ret = func(*args, **kwargs)
            profiler.disable()
            return ret
        finally:
            filename = os.path.expanduser(
                os.path.join('~', func.__name__ + '.pstat')
            )
            profiler.dump_stats(filename)
            profiler.print_stats()

    return wrapper


class Airplane:
    # Definition for an airplane.
    def __init__(self,
                 name="Untitled Airplane",  # A sensible name for your airplane.

                 xyz_ref=None,  # Ref. point for moments; should be the center of gravity. Syntax: List of [x, y, z].
                 mass_props=None,  # An object of MassProps type; only needed for dynamic analysis
                 # If xyz_ref is not set, but mass_props is, the xyz_ref will be taken from the CG there.

                 wings=[],  # A list of Wing objects.
                 s_ref=None,  # If not set, populates from first wing object.
                 c_ref=None,  # See above
                 b_ref=None,  # See above

                 ):
        self.name = name

        if xyz_ref is None and mass_props is not None:
            self.xyz_ref = mass_props.get_cg()
        else:
            self.xyz_ref = np.array(xyz_ref)

        self.mass_props = mass_props
        self.wings = wings

        if len(self.wings) > 0:  # If there is at least one wing
            self.set_ref_dims_from_wing()
        if s_ref is not None: self.s_ref = s_ref
        if c_ref is not None: self.c_ref = c_ref
        if b_ref is not None: self.b_ref = b_ref

        # Check that everything was set right:
        assert self.name is not None
        assert self.xyz_ref is not None
        assert self.s_ref is not None
        assert self.c_ref is not None
        assert self.b_ref is not None

    def draw(self):
        # Draw the airplane in a new window.

        # Using PyVista Polydata format
        vertices = np.empty((0, 3))
        faces = np.empty((0))

        for wing in self.wings:
            wing_vertices = np.empty((0, 3))
            wing_tri_faces = np.empty((0, 4))
            wing_quad_faces = np.empty((0, 5))
            for i in range(len(wing.xsecs) - 1):
                is_last_section = i == len(wing.xsecs) - 2

                le_start = wing.xsecs[i].xyz_le + wing.xyz_le
                te_start = wing.xsecs[i].xyz_te() + wing.xyz_le
                wing_vertices = np.vstack((wing_vertices, le_start, te_start))

                wing_quad_faces = np.vstack((
                    wing_quad_faces,
                    np.expand_dims(np.array([4, 2 * i + 0, 2 * i + 1, 2 * i + 3, 2 * i + 2]), 0)
                ))

                if is_last_section:
                    le_end = wing.xsecs[i + 1].xyz_le + wing.xyz_le
                    te_end = wing.xsecs[i + 1].xyz_te() + wing.xyz_le
                    wing_vertices = np.vstack((wing_vertices, le_end, te_end))

            vertices_starting_index = len(vertices)
            wing_quad_faces_reformatted = np.ndarray.copy(wing_quad_faces)
            wing_quad_faces_reformatted[:, 1:] = wing_quad_faces[:, 1:] + vertices_starting_index
            wing_quad_faces_reformatted = np.reshape(wing_quad_faces_reformatted, (-1), order='C')
            vertices = np.vstack((vertices, wing_vertices))
            faces = np.hstack((faces, wing_quad_faces_reformatted))

            if wing.symmetric:
                vertices_starting_index = len(vertices)
                wing_vertices = reflect_over_XZ_plane(wing_vertices)
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
        # Draws the airplane using matplotlib.
        # This method is deprecated (superseded by draw() ) and will be removed in a future release.

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
        # Sets the reference dimensions of the airplane from measurements obtained from a specific wing.

        main_wing = self.wings[main_wing_index]

        self.s_ref = main_wing.area_wetted()
        self.b_ref = main_wing.span()
        self.c_ref = main_wing.area_wetted() / main_wing.span()

    def set_paneling_everywhere(self, n_chordwise_panels, n_spanwise_panels):
        # Sets the chordwise and spanwise paneling everywhere to a specified value.
        # Useful for quickly changing the fidelity of your simulation.

        for wing in self.wings:
            wing.chordwise_panels = n_chordwise_panels
            for xsec in wing.xsecs:
                xsec.spanwise_panels = n_spanwise_panels

    def get_bounding_cube(self):
        """ Finds the axis-aligned cube that encloses the airplane with the smallest size.
            Useful for plotting and getting a sense for the scale of a problem.
            
            Args:
                self.wings (iterable): All the wings included for analysis each containing their geometry in x,y,z notation using units of m
            Returns:
                tuple: Tuple of 4 floats x, y, z, and s, where x, y, and z are the coordinates of the cube center,
                and s is half of the side length.
        """

        # Get vertices to enclose
        vertices = None
        for wing in self.wings:
            for xsec in wing.xsecs:
                if vertices is None:
                    vertices = xsec.xyz_le + wing.xyz_le
                else:
                    vertices = np.vstack((
                        vertices,
                        xsec.xyz_le + wing.xyz_le
                    ))
                vertices = np.vstack((
                    vertices,
                    xsec.xyz_te() + wing.xyz_le
                ))

                if wing.symmetric:
                    vertices = np.vstack((
                        vertices,
                        reflect_over_XZ_plane(xsec.xyz_le + wing.xyz_le)
                    ))
                    vertices = np.vstack((
                        vertices,
                        reflect_over_XZ_plane(xsec.xyz_te() + wing.xyz_le)
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
    # Definition for a wing.
    # If the wing is symmetric across the XZ plane, just define the right half, and be sure to supply "symmetric = True" in the constructor.
    # If the wing is not symmetric across the XZ plane, just define the wing.
    def __init__(self,
                 name="Untitled Wing",  # It can help when debugging to give each wing a sensible name.
                 xyz_le=[0, 0, 0],  # Will translate all of the xsecs of the wing. Useful for moving the wing around.
                 xsecs=[],  # This should be a list of WingXSec objects.
                 symmetric=False,  # Is the wing symmetric across the XZ plane?
                 chordwise_panels=10,
                 # Number of chordwise panels used in VLM analysis. Turn this up if you have control surfaces or airfoils with high camberline curvature.
                 chordwise_spacing="cosine",  # Can be 'cosine' or 'uniform'. Highly recommended to be cosine.
                 panel_chordwise_panels=10,  # Number of chordwise panels used in 3D panel analysis.
                 ):
        self.name = name
        self.xyz_le = np.array(xyz_le)
        self.xsecs = xsecs
        self.symmetric = symmetric
        self.chordwise_panels = chordwise_panels
        self.chordwise_spacing = chordwise_spacing

    def area_wetted(self):
        # Returns the wetted area of a wing.
        area = 0
        for i in range(len(self.xsecs) - 1):
            chord_eff = (self.xsecs[i].chord
                         + self.xsecs[i + 1].chord) / 2
            this_xyz_te = self.xsecs[i].xyz_te()
            that_xyz_te = self.xsecs[i + 1].xyz_te()
            span_le_eff = np.sqrt(
                np.square(self.xsecs[i].xyz_le[1] - self.xsecs[i + 1].xyz_le[1]) +
                np.square(self.xsecs[i].xyz_le[2] - self.xsecs[i + 1].xyz_le[2])
            )
            span_te_eff = np.sqrt(
                np.square(this_xyz_te[1] - that_xyz_te[1]) +
                np.square(this_xyz_te[2] - that_xyz_te[2])
            )
            span_eff = (span_le_eff + span_te_eff) / 2
            area += chord_eff * span_eff
        if self.symmetric:
            area *= 2
        return area

    def area_projected(self):
        # Returns the area of the wing as projected onto the XY plane (top-down view).
        area = 0
        for i in range(len(self.xsecs) - 1):
            chord_eff = (self.xsecs[i].chord
                         + self.xsecs[i + 1].chord) / 2
            this_xyz_te = self.xsecs[i].xyz_te()
            that_xyz_te = self.xsecs[i + 1].xyz_te()
            span_le_eff = np.abs(
                self.xsecs[i].xyz_le[1] - self.xsecs[i + 1].xyz_le[1]
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
        for i in range(len(self.xsecs)):
            spans.append(np.abs(self.xsecs[i].xyz_le[1] - self.xsecs[0].xyz_le[1]))
        span = np.max(spans)
        if self.symmetric:
            span *= 2
        return span

    def aspect_ratio(self):
        # Returns the aspect ratio (b^2/S).
        # Uses the full span and the full area if symmetric.
        return self.span() ** 2 / self.area_wetted()

    def has_symmetric_control_surfaces(self):
        # Returns a boolean of whether the wing is totally symmetric (i.e.), every xsec has control_surface_type = "symmetric".
        for xsec in self.xsecs:
            if not xsec.control_surface_type == "symmetric":
                return False
        return True


class WingXSec:

    def __init__(self,
                 xyz_le=[0, 0, 0],
                 chord=0,
                 twist=0,  # Twist is defined as about the leading edge!
                 airfoil=None,
                 control_surface_type="symmetric",
                 # Can be "symmetric" or "asymmetric". Symmetric is like flaps, asymmetric is like an aileron.
                 control_surface_hinge_point=0.75,
                 # Point at which the control surface is applied, as a fraction of chord.
                 control_surface_deflection=0,  # Control deflection, in degrees. Downwards-positive.
                 spanwise_panels=10,
                 spanwise_spacing="cosine"
                 ):
        self.xyz_le = np.array(xyz_le)
        self.chord = chord
        self.twist = twist
        self.airfoil = airfoil
        self.control_surface_type = control_surface_type
        self.control_surface_hinge_point = control_surface_hinge_point
        self.control_surface_deflection = control_surface_deflection
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
                 name="Untitled Airfoil",  # Examples: 'naca0012', 'ag10', 's1223', or anything you want.
                 coordinates=None,  # Treat this as an immutable, don't edit directly after initialization.
                 use_cache=True,  # Look in the airfoil cache, based on the airfoil's name. # TODO make airfoil caching
                 repanel=True,  # Should we repanel the airfoil? Highly recommend to leave this as True.
                 n_points_per_side=400,  # Number of points to use when repaneling the airfoil (if repanel is True)
                 ):

        self.name = name

        if coordinates is not None:
            self.coordinates = coordinates
        else:
            self.populate_coordinates()  # populates self.coordinates
        assert hasattr(self, 'coordinates'), "Couldn't figure out the coordinates of this airfoil! You need to either \
        a) use a name corresponding to an airfoil in the UIUC Airfoil Database or \
        b) provide your own coordinates in the constructor, such as Airfoil(""MyFoilName"", <Nx2 array of coordinates>)."

        # self.normalize()
        if repanel:
            self.repanel_current_airfoil(
                n_points_per_side=n_points_per_side)  # all airfoils are automatically repaneled to ensure consistent, good paneling.

        self.populate_mcl_coordinates()

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
                            - 0.1015 * np.power(x_t, 4)  # 0.1015 is original, #0.1036 for sharp TE
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
                    x_U, y_U = np.flipud(x_U), np.flipud(y_U)

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

    def populate_mcl_coordinates(self):
        # Populates self.mcl_coordinates, a Nx2 list of the airfoil's mean camber line coordinates.
        # Ordered from the leading edge to the trailing edge.
        #
        # Also populates self.upper_minus_mcl and self.lower_minus mcl, which are Nx2 lists of the vectors needed to
        # go from the mcl coordinates to the upper and lower surfaces, respectively. Both listed leading-edge to trailing-edge.
        #
        # Also populates self.thickness, a Nx2 list of the thicknesses at the mcl_coordinates x-points.

        upper = np.flipud(self.upper_coordinates())
        lower = self.lower_coordinates()

        mcl_coordinates = (upper + lower) / 2
        self.mcl_coordinates = mcl_coordinates

        self.upper_minus_mcl = upper - self.mcl_coordinates
        # self.lower_minus_mcl = -self.upper_minus_mcl

        thickness = np.sqrt(
            np.sum(
                np.power(self.upper_minus_mcl, 2),
                axis=1
            )
        ) * 2
        self.thickness = np.column_stack((self.mcl_coordinates[:, 0], thickness))

    # def normalize(self): # TODO make this return a new airfoil instead
    #     # Alters the airfoil's coordinates to exactly achieve several goals:
    #     #   # x_le == 0
    #     #   # y_le == 0
    #     #   # average( y_te_upper, y_te_lower ) == 0
    #     #   # max( x_te_upper, x_te_upper ) == 1
    #     # The first two goals are achieved by translating in x and y. The third goal is achieved by rotating about (0,0).
    #     # The fourth goal is achieved by uniform scaling.
    #
    #     # Goals 1 and 2
    #     LE_point_original = self.coordinates[self.LE_index(), :]
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
    #     rotation_angle = -np.arctan(TE_point_pre_rotation[1] / TE_point_pre_rotation[
    #         0])  # You need to rotate this many radians counterclockwise
    #     assert abs(np.degrees(
    #         rotation_angle)) < 0.5, "The foil appears to be really weirdly rotated! \
    #         Are you sure this isn't bad airfoil geometry?"
    #     cos_theta = np.cos(rotation_angle)
    #     sin_theta = np.sin(rotation_angle)
    #     rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    #     self.coordinates = np.transpose(rotation_matrix @ np.transpose(self.coordinates))
    #
    #     # Goal 4
    #     x_max = np.max(self.coordinates[:, 0])
    #     assert x_max <= 1.02 and x_max >= 0.98, "x_max is really weird! Are you sure this isn't bad airfoil geometry?"
    #     scale_factor = 1 / x_max
    #     self.coordinates *= scale_factor

    def draw(self, new_figure=True, draw_vertices=True, draw_mcl=True):
        if new_figure:
            plt.figure()
            plt.title("Airfoil: " + self.name)
        if draw_vertices:
            plt.plot(self.coordinates[:, 0], self.coordinates[:, 1], '.-')
        else:
            plt.plot(self.coordinates[:, 0], self.coordinates[:, 1])
        if draw_mcl:
            plt.plot(self.mcl_coordinates[:, 0], self.mcl_coordinates[:, 1], '.-')

        plt.xlim((-0.05, 1.05))
        plt.ylim((-0.5, 0.5))
        plt.axis('equal')
        plt.show()

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
        thickness_func = sp_interp.interp1d(
            x=self.thickness[:, 0],
            y=self.thickness[:, 1],
            copy=False,
            fill_value='extrapolate'
        )
        return thickness_func(chord_fraction)

    def get_thickness_at_chord_fraction_legacy(self, chord_fraction):
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

    def get_downsampled_mcl(self, mcl_fractions):
        # Returns the mean camber line in downsampled form

        mcl = self.mcl_coordinates
        # Find distances along mcl, assuming linear interpolation
        mcl_distances_between_points = np.sqrt(
            np.power(mcl[:-1, 0] - mcl[1:, 0], 2) +
            np.power(mcl[:-1, 1] - mcl[1:, 1], 2)
        )
        mcl_distances_cumulative = np.hstack((0, np.cumsum(mcl_distances_between_points)))
        mcl_distances_cumulative_normalized = mcl_distances_cumulative / mcl_distances_cumulative[-1]

        mcl_downsampled_x = np.interp(
            x=mcl_fractions,
            xp=mcl_distances_cumulative_normalized,
            fp=mcl[:, 0]
        )
        mcl_downsampled_y = np.interp(
            x=mcl_fractions,
            xp=mcl_distances_cumulative_normalized,
            fp=mcl[:, 1]
        )

        mcl_downsampled = np.column_stack((mcl_downsampled_x, mcl_downsampled_y))

        return mcl_downsampled

    def get_camber_at_chord_fraction(self, chord_fraction):
        camber_func = sp_interp.interp1d(
            x=self.mcl_coordinates[:, 0],
            y=self.mcl_coordinates[:, 1],
            copy=False,
            fill_value='extrapolate'
        )
        return camber_func(chord_fraction)

    def get_camber_at_chord_fraction_legacy(self, chord_fraction):
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

    def get_mcl_normal_direction_at_chord_fraction(self, chord_fraction):
        # Returns the normal direction of the mean camber line at a specified chord fraction.
        # If you input a single value, returns a 1D numpy array with 2 elements (x,y).
        # If you input a vector of values, returns a 2D numpy array. First index is the point number, second index is (x,y)

        # Right now, does it by finite differencing camber values :(
        # When I'm less lazy I'll make it do it in a proper, more efficient way
        # TODO make this not finite difference
        epsilon = np.sqrt(np.finfo(float).eps)

        cambers = self.get_camber_at_chord_fraction(chord_fraction)
        cambers_incremented = self.get_camber_at_chord_fraction(chord_fraction + epsilon)
        dydx = (cambers_incremented - cambers) / epsilon

        if dydx.shape == 1:  # single point
            normal = np.hstack((-dydx, 1))
            normal /= np.linalg.norm(normal)
            return normal
        else:  # multiple points vectorized
            normal = np.column_stack((-dydx, np.ones(dydx.shape)))
            normal /= np.expand_dims(np.linalg.norm(normal, axis=1), axis=1)  # normalize
            return normal

    def TE_thickness(self):
        # Returns the thickness of the trailing edge of the airfoil, in nondimensional (chord-normalized) units.
        return self.thickness[-1, 1]
        # np.sqrt(
        #     (self.coordinates[0, 0] - self.coordinates[-1, 0]) ** 2 +
        #     (self.coordinates[0, 1] - self.coordinates[-1, 1]) ** 2
        # )

    def TE_angle(self):
        # Returns the trailing edge angle of the airfoil, in degrees
        upper_TE_vec = self.coordinates[0, :] - self.coordinates[1, :]
        lower_TE_vec = self.coordinates[-1, :] - self.coordinates[-2, :]

        return np.degrees(np.arctan2(
            upper_TE_vec[0] * lower_TE_vec[1] - upper_TE_vec[1] * lower_TE_vec[0],
            upper_TE_vec[0] * lower_TE_vec[0] + upper_TE_vec[1] * upper_TE_vec[1]
        ))

    def area(self):
        # Returns the area of the airfoil, in nondimensional (normalized to chord^2) units.
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
        # Returns the nondimensionalized Ixx moment of inertia, taken about the centroid.
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
        # Returns the nondimensionalized Iyy moment of inertia, taken about the centroid.
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
        # Returns the nondimensionalized product of inertia, taken about the centroid.
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
        # Returns the nondimensionalized polar moment of inertia, taken about the centroid.
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

    def get_repaneled_airfoil(self, n_points_per_side=100):
        # Returns a repaneled version of the airfoil with cosine-spaced coordinates on the upper and lower surfaces.
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

        x_upper_func = sp_interp.PchipInterpolator(x=upper_distances_from_TE_normalized, y=upper_original_coors[:, 0])
        y_upper_func = sp_interp.PchipInterpolator(x=upper_distances_from_TE_normalized, y=upper_original_coors[:, 1])
        x_lower_func = sp_interp.PchipInterpolator(x=lower_distances_from_LE_normalized, y=lower_original_coors[:, 0])
        y_lower_func = sp_interp.PchipInterpolator(x=lower_distances_from_LE_normalized, y=lower_original_coors[:, 1])

        x_coors = np.hstack((x_upper_func(s), x_lower_func(s)[1:]))
        y_coors = np.hstack((y_upper_func(s), y_lower_func(s)[1:]))

        coordinates = np.column_stack((x_coors, y_coors))

        # Make a new airfoil with the coordinates
        name = self.name + ", repaneled to " + str(n_points_per_side) + " pts"
        new_airfoil = Airfoil(name=name, coordinates=coordinates, repanel=False)

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

        x_upper_func = sp_interp.PchipInterpolator(x=upper_distances_from_TE_normalized, y=upper_original_coors[:, 0])
        y_upper_func = sp_interp.PchipInterpolator(x=upper_distances_from_TE_normalized, y=upper_original_coors[:, 1])
        x_lower_func = sp_interp.PchipInterpolator(x=lower_distances_from_LE_normalized, y=lower_original_coors[:, 0])
        y_lower_func = sp_interp.PchipInterpolator(x=lower_distances_from_LE_normalized, y=lower_original_coors[:, 1])

        x_coors = np.hstack((x_upper_func(s), x_lower_func(s)[1:]))
        y_coors = np.hstack((y_upper_func(s), y_lower_func(s)[1:]))

        coordinates = np.column_stack((x_coors, y_coors))
        self.coordinates = coordinates

    def get_sharp_TE_airfoil(self):
        # Returns a version of the airfoil with a sharp trailing edge.

        upper_original_coors = self.upper_coordinates()  # Note: includes leading edge point, be careful about duplicates
        lower_original_coors = self.lower_coordinates()  # Note: includes leading edge point, be careful about duplicates

        # Find data about the TE

        # Get the scale factor
        x_mcl = self.mcl_coordinates[:, 0]
        x_max = np.max(x_mcl)
        x_min = np.min(x_mcl)
        scale_factor = (x_mcl - x_min) / (x_max - x_min)  # linear contraction

        # Do the contraction
        upper_minus_mcl_adjusted = self.upper_minus_mcl - self.upper_minus_mcl[-1, :] * np.expand_dims(scale_factor, 1)

        # Recreate coordinates
        upper_coordinates_adjusted = np.flipud(self.mcl_coordinates + upper_minus_mcl_adjusted)
        lower_coordinates_adjusted = self.mcl_coordinates - upper_minus_mcl_adjusted

        coordinates = np.vstack((
            upper_coordinates_adjusted[:-1, :],
            lower_coordinates_adjusted
        ))

        # Make a new airfoil with the coordinates
        name = self.name + ", with sharp TE"
        new_airfoil = Airfoil(name=name, coordinates=coordinates, repanel=False)

        return new_airfoil

    def add_control_surface(self, deflection=0., hinge_point=0.75):
        # Returns a version of the airfoil with a control surface added at a given point.
        # Inputs:
        #   # deflection: the deflection angle, in degrees. Downwards-positive.
        #   # hinge_point: the location of the hinge, as a fraction of chord.

        # Make the rotation matrix for the given angle.
        sintheta = np.sin(np.radians(-deflection))
        costheta = np.cos(np.radians(-deflection))
        rotation_matrix = np.array(
            [[costheta, -sintheta],
             [sintheta, costheta]]
        )

        # Find the hinge point
        hinge_point = np.array(
            (hinge_point, self.get_camber_at_chord_fraction(hinge_point)))  # Make hinge_point a vector.

        # Split the airfoil into the sections before and after the hinge
        split_index = np.where(self.mcl_coordinates[:, 0] > hinge_point[0])[0][0]
        mcl_coordinates_before = self.mcl_coordinates[:split_index, :]
        mcl_coordinates_after = self.mcl_coordinates[split_index:, :]
        upper_minus_mcl_before = self.upper_minus_mcl[:split_index, :]
        upper_minus_mcl_after = self.upper_minus_mcl[split_index:, :]

        # Rotate the mean camber line (MCL) and "upper minus mcl"
        new_mcl_coordinates_after = np.transpose(
            rotation_matrix @ np.transpose(mcl_coordinates_after - hinge_point)) + hinge_point
        new_upper_minus_mcl_after = np.transpose(rotation_matrix @ np.transpose(upper_minus_mcl_after))

        # Do blending

        # Assemble airfoil
        new_mcl_coordinates = np.vstack((mcl_coordinates_before, new_mcl_coordinates_after))
        new_upper_minus_mcl = np.vstack((upper_minus_mcl_before, new_upper_minus_mcl_after))
        upper_coordinates = np.flipud(new_mcl_coordinates + new_upper_minus_mcl)
        lower_coordinates = new_mcl_coordinates - new_upper_minus_mcl
        coordinates = np.vstack((upper_coordinates, lower_coordinates[1:, :]))

        new_airfoil = Airfoil(name=self.name + " flapped", coordinates=coordinates, repanel=False)
        return new_airfoil  # TODO fix self-intersecting airfoils at high deflections


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

    if blend_fraction == 0:
        return foil1
    if blend_fraction == 1:
        return foil2
    assert blend_fraction >= 0 and blend_fraction <= 1, "blend_fraction is out of the valid range of 0 to 1!"

    # Repanel to ensure the same number of points and the same point distribution on both airfoils.
    foil1 = foil1.get_repaneled_airfoil(n_points_per_side=200)
    foil2 = foil2.get_repaneled_airfoil(n_points_per_side=200)

    blended_coordinates = (1 - blend_fraction) * foil1.coordinates + blend_fraction * foil2.coordinates

    new_airfoil = Airfoil(name="Blended Airfoils", coordinates=blended_coordinates)

    return new_airfoil


def reflect_over_XZ_plane(input_vector):
    # Takes in a vector or an array and flips the y-coordinates.
    output_vector = input_vector
    shape = np.shape(output_vector)
    if len(shape) == 1 and shape[0] == 3:  # Vector of 3 items
        output_vector = output_vector * np.array([1, -1, 1])
    elif len(shape) == 2 and shape[1] == 3:  # 2D Nx3 vector
        output_vector = output_vector * np.array([1, -1, 1])
    elif len(shape) == 3 and shape[2] == 3:  # 3D MxNx3 vector
        output_vector = output_vector * np.array([1, -1, 1])
    else:
        raise Exception("Invalid input for reflect_over_XZ_plane!")

    return output_vector


def cosspace(min=0, max=1, n_points=50):
    mean = (max + min) / 2
    amp = (max - min) / 2

    return mean + amp * np.cos(np.linspace(np.pi, 0, n_points))


def angle_axis_rotation_matrix(angle, axis, axis_already_normalized=False):
    # Gives the rotation matrix from an angle and an axis.
    # An implmentation of https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    # Inputs:
    #   * angle: can be one angle or a vector (1d ndarray) of angles. Given in radians.
    #   * axis: a 1d numpy array of length 3 (x,y,z). Represents the angle.
    #   * axis_already_normalized: boolean, skips normalization for speed if you flag this true.
    # Outputs:
    #   * If angle is a scalar, returns a 3x3 rotation matrix.
    #   * If angle is a vector, returns a 3x3xN rotation matrix.
    if not axis_already_normalized:
        axis = axis / np.linalg.norm(axis)

    sintheta = np.sin(angle)
    costheta = np.cos(angle)
    cpm = np.array(
        [[0, -axis[2], axis[1]],
         [axis[2], 0, -axis[0]],
         [-axis[1], axis[0], 0]]
    )  # The cross product matrix of the rotation axis vector
    outer_axis = np.outer(axis, axis)

    angle = np.array(angle)  # make sure angle is a ndarray
    if len(angle.shape) == 0:  # is a scalar
        rot_matrix = costheta * np.eye(3) + sintheta * cpm + (1 - costheta) * outer_axis
        return rot_matrix
    else:  # angle is assumed to be a 1d ndarray
        rot_matrix = costheta * np.expand_dims(np.eye(3), 2) + sintheta * np.expand_dims(cpm, 2) + (
                1 - costheta) * np.expand_dims(outer_axis, 2)
        return rot_matrix


def linspace_3D(start, stop, n_points):
    # Given two points (a start and an end), returns an interpolated array of points on the line between the two.
    # Inputs:
    #   * start: 3D coordinates expressed as a 1D numpy array, shape==(3).
    #   * end: 3D coordinates expressed as a 1D numpy array, shape==(3).
    #   * n_points: Number of points to be interpolated (including endpoints), a scalar.
    # Outputs:
    #   * points: Array of 3D coordinates expressed as a 2D numpy array, shape==(N, 3)
    x = np.linspace(start[0], stop[0], n_points)
    y = np.linspace(start[1], stop[1], n_points)
    z = np.linspace(start[2], stop[2], n_points)

    points = np.column_stack((x, y, z))
    return points
