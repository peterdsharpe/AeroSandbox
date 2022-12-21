from aerosandbox import AeroSandboxObject
from aerosandbox.geometry.common import *
from typing import List, Dict, Any, Union, Optional
import aerosandbox.geometry.mesh_utilities as mesh_utils
from aerosandbox.geometry.wing import Wing
from aerosandbox.geometry.fuselage import Fuselage


class Airplane(AeroSandboxObject):
    """
    Definition for an airplane.

    Anatomy of an Airplane:

        An Airplane consists chiefly of a collection of wings and fuselages. These can be accessed with
        `Airplane.wings` and `Airplane.fuselages`, which gives a list of those respective components. Each wing is a
        Wing object, and each fuselage is a Fuselage object.

    """

    def __init__(self,
                 name: Optional[str] = None,
                 xyz_ref: Union[np.ndarray, List] = None,
                 wings: Optional[List[Wing]] = None,
                 fuselages: Optional[List[Fuselage]] = None,
                 s_ref: Optional[float] = None,
                 c_ref: Optional[float] = None,
                 b_ref: Optional[float] = None,
                 analysis_specific_options: Optional[Dict[type, Dict[str, Any]]] = None
                 ):
        """
        Defines a new airplane.

        Args:

            name: Name of the airplane [optional]. It can help when debugging to give the airplane a sensible name.

            xyz_ref: An array-like that gives the x-, y-, and z- reference point of the airplane, used when computing
            moments and stability derivatives. Generally, this should be the center of gravity.

                # In a future version, this will be deprecated and replaced with asb.MassProperties.

            wings: A list of Wing objects that are a part of the airplane.

            fuselages: A list of Fuselage objects that are a part of the airplane.

            s_ref: Reference area. If undefined, it's set from the area of the first Wing object. # Note: will be deprecated

            c_ref: Reference chord. If undefined, it's set from the mean aerodynamic chord of the first Wing object. # Note: will be deprecated

            b_ref: Reference span. If undefined, it's set from the span of the first Wing object. # Note: will be deprecated

            analysis_specific_options: Analysis-specific options are additional constants or modeling assumptions
            that should be passed on to specific analyses and associated with this specific geometry object.

                This should be a dictionary where:

                    * Keys are specific analysis types (typically a subclass of asb.ExplicitAnalysis or
                    asb.ImplicitAnalysis), but if you decide to write your own analysis and want to make this key
                    something else (like a string), that's totally fine - it's just a unique identifier for the
                    specific analysis you're running.

                    * Values are a dictionary of key:value pairs, where:

                        * Keys are strings.

                        * Values are some value you want to assign.

                This is more easily demonstrated / understood with an example:

                >>> analysis_specific_options = {
                >>>     asb.AeroBuildup: dict(
                >>>         include_wave_drag=True,
                >>>     )
                >>> }

        """
        ### Set defaults
        if name is None:
            name = "Untitled"
        if xyz_ref is None:
            xyz_ref = np.array([0., 0., 0.])
        if wings is None:
            wings: List[Wing] = []
        if fuselages is None:
            fuselages: List[Fuselage] = []
        if analysis_specific_options is None:
            analysis_specific_options = {}

        ### Initialize
        self.name = name
        self.xyz_ref = np.array(xyz_ref)
        self.wings = wings
        self.fuselages = fuselages
        self.analysis_specific_options = analysis_specific_options

        ### Assign reference values
        try:
            main_wing = self.wings[0]
            if s_ref is None:
                s_ref = main_wing.area()
            if c_ref is None:
                c_ref = main_wing.mean_aerodynamic_chord()
            if b_ref is None:
                b_ref = main_wing.span()
        except IndexError:
            s_ref = 1
            c_ref = 1
            b_ref = 1
        self.s_ref = s_ref
        self.c_ref = c_ref
        self.b_ref = b_ref

    def __repr__(self):
        n_wings = len(self.wings)
        n_fuselages = len(self.fuselages)
        return f"Airplane '{self.name}' " \
               f"({n_wings} {'wing' if n_wings == 1 else 'wings'}, " \
               f"{n_fuselages} {'fuselage' if n_fuselages == 1 else 'fuselages'})"

    # TODO def add_wing(wing: 'Wing') -> None

    def mesh_body(self,
                  method="quad",
                  thin_wings=False,
                  stack_meshes=True,
                  ):
        """
        Returns a surface mesh of the Airplane, in (points, faces) format. For reference on this format,
        see the documentation in `aerosandbox.geometry.mesh_utilities`.

        Args:

            method:

            thin_wings: Controls whether wings should be meshed as thin surfaces, rather than full 3D bodies.

            stack_meshes: Controls whether the meshes should be merged into a single mesh or not.

                * If True, returns a (points, faces) tuple in standard mesh format.

                * If False, returns a list of (points, faces) tuples in standard mesh format.

        Returns:

        """
        if thin_wings:
            wing_meshes = [
                wing.mesh_thin_surface(
                    method=method,
                )
                for wing in self.wings
            ]
        else:
            wing_meshes = [
                wing.mesh_body(
                    method=method,
                )
                for wing in self.wings
            ]

        fuse_meshes = [
            fuse.mesh_body(
                method=method
            )
            for fuse in self.fuselages
        ]

        meshes = wing_meshes + fuse_meshes

        if stack_meshes:
            points, faces = mesh_utils.stack_meshes(*meshes)
            return points, faces
        else:
            return meshes

    def draw(self,
             backend: str = "pyvista",
             thin_wings: bool = False,
             show: bool = True,
             show_kwargs: Dict = None,
             ):
        """

        Args:

            backend: One of:
                * "plotly" for a Plot.ly backend
                * "pyvista" for a PyVista backend
                * "trimesh" for a trimesh backend

            thin_wings: A boolean that determines whether to draw the full airplane (i.e. thickened, 3D bodies), or to use a
            thin-surface representation for any Wing objects.

            show: Should we show the visualization, or just return it?

        Returns: The plotted object, in its associated backend format. Also displays the object if `show` is True.

        """
        if show_kwargs is None:
            show_kwargs = {}

        points, faces = self.mesh_body(method="quad", thin_wings=thin_wings)

        if backend == "matplotlib":
            import matplotlib.pyplot as plt
            import aerosandbox.tools.pretty_plots as p
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            fig, ax = p.figure3d()

            ax.add_collection(
                Poly3DCollection(
                    points[faces],
                    edgecolors="k",
                    linewidths=0.2,
                ),
            )

            ax.set_xlim(points[:, 0].min(), points[:, 0].max())
            ax.set_ylim(points[:, 1].min(), points[:, 1].max())
            ax.set_zlim(points[:, 2].min(), points[:, 2].max())

            p.equal()

            ax.set_xlabel("$x_g$")
            ax.set_ylabel("$y_g$")
            ax.set_zlabel("$z_g$")

            if show:
                p.show_plot()

        elif backend == "plotly":

            from aerosandbox.visualization.plotly_Figure3D import Figure3D
            fig = Figure3D()
            for f in faces:
                fig.add_quad((
                    points[f[0]],
                    points[f[1]],
                    points[f[2]],
                    points[f[3]],
                ), outline=True)
                show_kwargs = {
                    "show": show,
                    **show_kwargs
                }
            return fig.draw(**show_kwargs)

        elif backend == "pyvista":

            import pyvista as pv
            fig = pv.PolyData(
                *mesh_utils.convert_mesh_to_polydata_format(points, faces)
            )
            show_kwargs = {
                "show_edges": True,
                "show_grid" : True,
                **show_kwargs,
            }
            if show:
                fig.plot(**show_kwargs)
            return fig

        elif backend == "trimesh":

            import trimesh as tri
            fig = tri.Trimesh(points, faces)
            if show:
                fig.show(**show_kwargs)
            return fig
        else:
            raise ValueError("Bad value of `backend`!")

    def draw_wireframe(self,
                       ax=None,
                       color="k",
                       thin_linewidth=0.2,
                       thick_linewidth=0.5,
                       fuselage_longeron_theta=None,
                       set_equal: bool = True,
                       show: bool = True,
                       ):
        import matplotlib.pyplot as plt
        import aerosandbox.tools.pretty_plots as p

        if ax is None:
            fig, ax = p.figure3d(figsize=(8, 8))
        else:
            if not p.ax_is_3d(ax):
                raise ValueError("`ax` must be a 3D axis.")

            plt.sca(ax)

        if fuselage_longeron_theta is None:
            fuselage_longeron_theta = np.linspace(0, 2 * np.pi, 4 + 1)[:-1]

        def plot_line(
                xyz,
                symmetric=False,
                fmt="-",
                color=color,
                linewidth=0.4,
                **kwargs
        ):
            if symmetric:
                xyz = np.vstack([
                    xyz,
                    [np.nan] * 3,
                    xyz * np.array([1, -1, 1])
                ])

            ax.plot(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                fmt,
                color=color,
                linewidth=linewidth,
                **kwargs
            )

        def reshape(x):
            return np.array(x).reshape((1, 3))

        ##### Wings
        for wing in self.wings:

            ### LE and TE lines
            for xy in [
                (0, 0),  # Leading Edge
                (1, 0),  # Trailing Edge
            ]:

                plot_line(
                    wing.mesh_line(x_nondim=xy[0], y_nondim=xy[1]),
                    symmetric=wing.symmetric,
                    linewidth=thick_linewidth,
                )

            ### Top and Bottom lines
            x = 0.4
            afs = [xsec.airfoil for xsec in wing.xsecs]
            thicknesses = np.array([af.local_thickness(x_over_c=x) for af in afs])

            plot_line(
                wing.mesh_line(x_nondim=x, y_nondim=thicknesses / 2, add_camber=True),
                symmetric=wing.symmetric,
                linewidth=thin_linewidth,
            )
            plot_line(
                wing.mesh_line(x_nondim=x, y_nondim=-thicknesses / 2, add_camber=True),
                symmetric=wing.symmetric,
                linewidth=thin_linewidth,
            )

            ### Airfoils
            for i, xsec in enumerate(wing.xsecs):
                xg_local, yg_local, zg_local = wing._compute_frame_of_WingXSec(i)
                xg_local = reshape(xg_local)
                yg_local = reshape(yg_local)
                zg_local = reshape(zg_local)
                origin = reshape(xsec.xyz_le)
                scale = xsec.chord

                line_upper = origin + (
                        xsec.airfoil.upper_coordinates()[:, 0].reshape((-1, 1)) * scale * xg_local +
                        xsec.airfoil.upper_coordinates()[:, 1].reshape((-1, 1)) * scale * zg_local
                )
                line_lower = origin + (
                        xsec.airfoil.lower_coordinates()[:, 0].reshape((-1, 1)) * scale * xg_local +
                        xsec.airfoil.lower_coordinates()[:, 1].reshape((-1, 1)) * scale * zg_local
                )

                for line in [line_upper, line_lower]:
                    plot_line(
                        line,
                        symmetric=wing.symmetric,
                        linewidth=thick_linewidth if i == 0 or i == len(wing.xsecs) - 1 else thin_linewidth
                    )

        ##### Fuselages
        for fuse in self.fuselages:

            xsec_shape_parameters = np.array([
                xsec.shape
                for xsec in fuse.xsecs
            ])

            ### Longerons
            for theta in fuselage_longeron_theta:
                st = np.sin(theta)
                ct = np.cos(theta)

                plot_line(
                    fuse.mesh_line(
                        x_nondim=np.abs(ct) ** (2 / xsec_shape_parameters) * np.where(ct > 0, 1, -1),
                        y_nondim=np.abs(st) ** (2 / xsec_shape_parameters) * np.where(st > 0, 1, -1)
                    ),
                    linewidth=thick_linewidth
                )

            ### Centerline
            plot_line(
                fuse.mesh_line(x_nondim=0, y_nondim=0),
                linewidth=thin_linewidth
            )

            ### Bulkheads
            for i, xsec in enumerate(fuse.xsecs):
                if xsec.radius < 1e-6:
                    continue

                xg_local, yg_local, zg_local = fuse._compute_frame_of_FuselageXSec(i)
                xg_local = reshape(xg_local)
                yg_local = reshape(yg_local)
                zg_local = reshape(zg_local)
                origin = reshape(xsec.xyz_c)
                scale = xsec.radius

                theta = np.linspace(0, 2 * np.pi, 121).reshape((-1, 1))
                st = np.sin(theta)
                ct = np.cos(theta)
                x_nondim = np.abs(ct) ** (2 / xsec.shape) * np.where(ct > 0, 1, -1)
                y_nondim = np.abs(st) ** (2 / xsec.shape) * np.where(st > 0, 1, -1)

                line = origin + (
                        x_nondim * scale * yg_local +
                        y_nondim * scale * zg_local
                )

                plot_line(
                    line,
                    linewidth=thick_linewidth if i == 0 or i == len(fuse.xsecs) - 1 else thin_linewidth
                )

        if set_equal:
            p.equal()

        if show:
            p.show_plot()

    def draw_three_view(self,
                        fig=None,
                        show=True,
                        ):
        import matplotlib.pyplot as plt
        import aerosandbox.tools.pretty_plots as p

        if fig is None:
            fig = plt.figure(figsize=(8, 8), dpi=400)

        preset_view_angles = np.array([
            ["XZ", "-YZ"],
            ["XY", "left_isometric"]
        ], dtype="O")

        axes = np.empty_like(preset_view_angles, dtype="O")

        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                ax = fig.add_subplot(
                    axes.shape[0],
                    axes.shape[1],
                    i * axes.shape[0] + j + 1,
                    projection='3d',
                    proj_type='ortho',
                    box_aspect=(1, 1, 1)
                )

                preset_view = preset_view_angles[i, j]

                if 'isometric' in preset_view:
                    ax.set_axis_off()
                if preset_view == 'XY' or preset_view == '-XY':
                    ax.set_zticks([])
                if preset_view == 'XZ' or preset_view == '-XZ':
                    ax.set_yticks([])
                if preset_view == 'YZ' or preset_view == '-YZ':
                    ax.set_xticks([])

                pane_color = ax.get_facecolor()
                ax.set_facecolor((0, 0, 0, 0))  # Set transparent
                #
                ax.xaxis.pane.set_facecolor(pane_color)
                ax.xaxis.pane.set_alpha(1)
                ax.yaxis.pane.set_facecolor(pane_color)
                ax.yaxis.pane.set_alpha(1)
                ax.zaxis.pane.set_facecolor(pane_color)
                ax.zaxis.pane.set_alpha(1)

                self.draw_wireframe(
                    ax=ax,
                    fuselage_longeron_theta=np.linspace(0, 2 * np.pi, 8 + 1)[:-1]
                    if 'isometric' in preset_view else None,
                    show=False
                )

                p.set_preset_3d_view_angle(
                    preset_view_angles[i, j]
                )

                axes[i, j] = ax

        axes[1, 0].set_xlabel("$x_g$ [m]")
        axes[1, 0].set_ylabel("$y_g$ [m]")
        axes[0, 0].set_zlabel("$z_g$ [m]")
        axes[0, 0].set_xticklabels([])
        axes[0, 1].set_yticklabels([])
        axes[0, 1].set_zticklabels([])

        plt.subplots_adjust(
            left=-0.08,
            right=1.08,
            bottom=-0.08,
            top=1.08,
            wspace=-0.38,
            hspace=-0.38,
        )

        if show:
            plt.show()

    def is_entirely_symmetric(self):
        """
        Returns a boolean describing whether the airplane is geometrically entirely symmetric across the XZ-plane.
        :return: [boolean]
        """
        for wing in self.wings:
            if not wing.is_entirely_symmetric():
                return False

        # TODO add in logic for fuselages

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


if __name__ == '__main__':
    import aerosandbox as asb
    import aerosandbox.numpy as np
    import aerosandbox.tools.units as u


    def ft(feet, inches=0):  # Converts feet (and inches) to meters
        return feet * u.foot + inches * u.inch


    naca2412 = asb.Airfoil("naca2412")
    naca0012 = asb.Airfoil("naca0012")

    airplane = asb.Airplane(
        name="Cessna 152",
        wings=[
            asb.Wing(
                name="Wing",
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[0, 0, 0],
                        chord=ft(5, 4),
                        airfoil=naca2412
                    ),
                    asb.WingXSec(
                        xyz_le=[0, ft(7), ft(7) * np.sind(1)],
                        chord=ft(5, 4),
                        airfoil=naca2412
                    ),
                    asb.WingXSec(
                        xyz_le=[
                            ft(4, 3 / 4) - ft(3, 8 + 1 / 2),
                            ft(33, 4) / 2,
                            ft(33, 4) / 2 * np.sind(1)
                        ],
                        chord=ft(3, 8 + 1 / 2),
                        airfoil=naca0012
                    )
                ],
                symmetric=True
            ),
            asb.Wing(
                name="Horizontal Stabilizer",
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[0, 0, 0],
                        chord=ft(3, 8),
                        airfoil=naca0012,
                        twist=-2
                    ),
                    asb.WingXSec(
                        xyz_le=[ft(1), ft(10) / 2, 0],
                        chord=ft(2, 4 + 3 / 8),
                        airfoil=naca0012,
                        twist=-2
                    )
                ],
                symmetric=True
            ).translate([ft(13, 3), 0, ft(-2)]),
            asb.Wing(
                name="Vertical Stabilizer",
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[ft(-5), 0, 0],
                        chord=ft(8, 8),
                        airfoil=naca0012,
                    ),
                    asb.WingXSec(
                        xyz_le=[ft(0), 0, ft(1)],
                        chord=ft(3, 8),
                        airfoil=naca0012,
                    ),
                    asb.WingXSec(
                        xyz_le=[ft(0, 8), 0, ft(5)],
                        chord=ft(2, 8),
                        airfoil=naca0012,
                    ),
                ]
            ).translate([ft(16, 11) - ft(3, 8), 0, ft(-2)])
        ],
        fuselages=[
            asb.Fuselage(
                xsecs=[
                    asb.FuselageXSec(
                        xyz_c=[0, 0, ft(-1)],
                        radius=0,
                    ),
                    asb.FuselageXSec(
                        xyz_c=[0, 0, ft(-1)],
                        radius=ft(1.5),
                        shape=3,
                    ),
                    asb.FuselageXSec(
                        xyz_c=[ft(3), 0, ft(-0.85)],
                        radius=ft(1.7),
                        shape=7,
                    ),
                    asb.FuselageXSec(
                        xyz_c=[ft(5), 0, ft(0)],
                        radius=ft(2.7),
                        shape=7,
                    ),
                    asb.FuselageXSec(
                        xyz_c=[ft(10, 4), 0, ft(0.3)],
                        radius=ft(2.3),
                        shape=7,
                    ),
                    asb.FuselageXSec(
                        xyz_c=[ft(21, 11), 0, ft(0.8)],
                        radius=ft(0.3),
                        shape=3,
                    ),
                ]
            ).translate([ft(-5), 0, ft(-3)])
        ]
    )

    airplane.draw_three_view()
