from aerosandbox import AeroSandboxObject
from aerosandbox.geometry.common import *
from typing import List
from aerosandbox.visualization.plotly_Figure3D import Figure3D
from numpy import pi


class Airplane(AeroSandboxObject):
    """
    Definition for an airplane.
    """

    def __init__(self,
                 name: str = "Untitled",  # A sensible name for your airplane.
                 xyz_ref: np.ndarray = np.array([0, 0, 0]),  # Ref. point for moments; should be the center of gravity.
                 wings: List['Wing'] = None,  # A list of Wing objects.
                 fuselages: List['Fuselage'] = None,  # A list of Fuselage objects.
                 s_ref: float = None,  # If not set, populates from first wing object.
                 c_ref: float = None,  # See above
                 b_ref: float = None,  # See above
                 ):
        ### Initialize
        self.name = name

        self.xyz_ref = xyz_ref

        ## Add the wing objects
        if wings is not None:
            self.wings = wings
        else:
            self.wings = []

        ## Add the fuselage objects
        if fuselages is not None:
            self.fuselages = fuselages
        else:
            self.fuselages = []

        ## Assign reference values
        try:
            main_wing = self.wings[0]
            if s_ref is None:
                s_ref = main_wing.area()
            if c_ref is None:
                c_ref = main_wing.mean_aerodynamic_chord()
            if b_ref is None:
                b_ref = main_wing.span()
        except IndexError:
            pass
        self.s_ref = s_ref
        self.c_ref = c_ref
        self.b_ref = b_ref

    def __repr__(self):
        n_wings = len(self.wings)
        n_fuselages = len(self.fuselages)
        return f"Airplane '{self.name}' " \
               f"({n_wings} {'wing' if n_wings == 1 else 'wings'}, " \
               f"{n_fuselages} {'fuselage' if n_fuselages == 1 else 'fuselages'})"

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
        for wing_id, wing in enumerate(self.wings):
            for inner_xsec, outer_xsec in zip(wing.xsecs[:-1], wing.xsecs[1:]):

                le_inner = inner_xsec.xyz_le + wing.xyz_le
                te_inner = inner_xsec.xyz_te() + wing.xyz_le
                le_outer = outer_xsec.xyz_le + wing.xyz_le
                te_outer = outer_xsec.xyz_te() + wing.xyz_le

                fig.add_quad(points=[
                    le_inner,
                    le_outer,
                    te_outer,
                    te_inner
                ],
                    intensity=wing_id,
                    mirror=wing.symmetric,
                )
                if draw_quarter_chord:
                    fig.add_line(  # draw the quarter-chord line
                        points=[
                            0.75 * le_inner + 0.25 * te_inner,
                            0.75 * le_outer + 0.25 * te_outer,
                        ],
                        mirror=wing.symmetric
                    )

        # Fuselages
        for fuse_id, fuse in enumerate(self.fuselages):

            for front_xsec, back_xsec in zip(fuse.xsecs[:-1], fuse.xsecs[1:]):
                r_front = front_xsec.radius
                r_back = back_xsec.radius
                points_front = np.zeros((fuse.circumferential_panels, 3))
                points_rear = np.zeros((fuse.circumferential_panels, 3))
                for point_index in range(fuse.circumferential_panels):
                    rot = np.rotation_matrix_3D(
                        2 * pi * point_index / fuse.circumferential_panels,
                        [1, 0, 0],
                        _axis_already_normalized=True
                    )
                    points_front[point_index, :] = rot @ np.array([0, 0, r_front])
                    points_rear[point_index, :] = rot @ np.array([0, 0, r_back])
                points_front = points_front + np.array(fuse.xyz_le).reshape(-1) + np.array(xsec_1.xyz_c).reshape(-1)
                points_rear = points_rear + np.array(fuse.xyz_le).reshape(-1) + np.array(xsec_2.xyz_c).reshape(-1)

                for point_index in range(fuse.circumferential_panels):

                    fig.add_quad(points=[
                        points_front[(point_index) % fuse.circumferential_panels, :],
                        points_front[(point_index + 1) % fuse.circumferential_panels, :],
                        points_rear[(point_index + 1) % fuse.circumferential_panels, :],
                        points_rear[(point_index) % fuse.circumferential_panels, :],
                    ],
                        intensity=fuse_id,
                        mirror=fuse.symmetric,
                    )

        return fig.draw(
            show=show,
            colorscale=colorscale,
            colorbar_title=colorbar_title,
        )

    def is_entirely_symmetric(self):
        """
        Returns a boolean describing whether the airplane is geometrically entirely symmetric across the XZ-plane.
        :return: [boolean]
        """
        for wing in self.wings:
            if not wing.is_entirely_symmetric():
                return False
            for xsec in wing.xsecs:
                if not (xsec.control_surface_is_symmetric or xsec.control_surface_deflection == 0):
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

    def aerodynamic_center(self, chord_fraction: float = 0.25):
        """
        Computes the location of the aerodynamic center of the wing.
        Uses the generalized methodology described here:
            https://core.ac.uk/download/pdf/79175663.pdf

        Args:
            chord_fraction: The position of the aerodynamic center along the MAC, as a fraction of MAC length.
                Typically, this value (denoted `h_0` in the literature) is 0.25 for a subsonic wing.
                However, wing-fuselage interactions can cause a forward shift to a value more like 0.1 or less.
                Citing Cook, Michael V., "Flight Dynamics Principles", 3rd Ed., Sect. 3.5.3 "Controls-fixed static stability".
                PDF: https://www.sciencedirect.com/science/article/pii/B9780080982427000031

        Returns: The (x, y, z) coordinates of the aerodynamic center of the airplane.
        """
        wing_areas = [wing.area(type="projected") for wing in self.wings]
        ACs = [wing.aerodynamic_center() for wing in self.wings]

        wing_AC_area_products = [
            AC * area
            for AC, area in zip(
                ACs,
                wing_areas
            )
        ]

        aerodynamic_center = sum(wing_AC_area_products) / sum(wing_areas)

        return aerodynamic_center
