from aerosandbox import AeroSandboxObject
from aerosandbox.geometry.common import *
from typing import List, Dict, Any, Union, Optional, Tuple
import aerosandbox.geometry.mesh_utilities as mesh_utils
from aerosandbox.geometry.wing import Wing
from aerosandbox.geometry.fuselage import Fuselage
from aerosandbox.geometry.propulsor import Propulsor
from aerosandbox.weights.mass_properties import MassProperties
import copy
from pathlib import Path

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
                 propulsors: Optional[List[Propulsor]] = None,
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

            propulsors: A list of Propulsor objects that are a part of the airplane.

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
        if propulsors is None:
            propulsors: List[Propulsor] = []
        if analysis_specific_options is None:
            analysis_specific_options = {}

        ### Initialize
        self.name = name
        self.xyz_ref = np.array(xyz_ref)
        self.wings = wings
        self.fuselages = fuselages
        self.propulsors = propulsors
        self.analysis_specific_options = analysis_specific_options

        ### Assign reference values
        try:
            main_wing = self.wings[0]
        except IndexError:
            main_wing = None

        try:
            main_fuse = self.fuselages[0]
        except IndexError:
            main_fuse = None

        if s_ref is not None:
            self.s_ref = s_ref
        else:
            if main_wing is not None:
                self.s_ref = main_wing.area()
            else:
                if main_fuse is not None:
                    self.s_ref = main_fuse.area_projected()
                else:
                    raise ValueError(
                        "`s_ref` was not provided, and a value cannot be inferred automatically from wings or fuselages.\n"
                        "You must set this manually when instantiating your asb.Airplane object.")

        if c_ref is not None:
            self.c_ref = c_ref
        else:
            if main_wing is not None:
                self.c_ref = main_wing.mean_aerodynamic_chord()
            else:
                if main_fuse is not None:
                    self.c_ref = main_fuse.length()
                else:
                    raise ValueError(
                        "`c_ref` was not provided, and a value cannot be inferred automatically from wings or fuselages.\n"
                        "You must set this manually when instantiating your asb.Airplane object."
                    )

        if b_ref is not None:
            self.b_ref = b_ref
        else:
            if main_wing is not None:
                self.b_ref = main_wing.span(include_centerline_distance=True)
            else:
                if main_fuse is not None:
                    self.b_ref = main_fuse.area_projected() / main_fuse.length()
                else:
                    raise ValueError(
                        "`b_ref` was not provided, and a value cannot be inferred automatically from wings or fuselages.\n"
                        "You must set this manually when instantiating your asb.Airplane object."
                    )

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
             ax=None,
             use_preset_view_angle: str = None,
             set_background_pane_color: Union[str, Tuple[float, float, float]] = None,
             set_background_pane_alpha: float = None,
             set_lims: bool = True,
             set_equal: bool = True,
             set_axis_visibility: bool = None,
             show: bool = True,
             show_kwargs: Dict = None,
             ):
        """
        Produces an interactive 3D visualization of the airplane.

        Args:

            backend: The visualization backend to use. Options are:

                * "matplotlib" for a Matplotlib backend
                * "pyvista" for a PyVista backend
                * "plotly" for a Plot.ly backend
                * "trimesh" for a trimesh backend

            thin_wings: A boolean that determines whether to draw the full airplane (i.e. thickened, 3D bodies), or to use a
            thin-surface representation for any Wing objects.

            show: A boolean that determines whether to display the object after plotting it. If False, the object is
            returned but not displayed. If True, the object is displayed and returned.

        Returns: The plotted object, in its associated backend format. Also displays the object if `show` is True.

        """
        if show_kwargs is None:
            show_kwargs = {}

        points, faces = self.mesh_body(method="quad", thin_wings=thin_wings)

        if backend == "matplotlib":
            import matplotlib.pyplot as plt
            import aerosandbox.tools.pretty_plots as p
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            if ax is None:
                _, ax = p.figure3d(figsize=(8, 8), computed_zorder=False)

            else:
                if not p.ax_is_3d(ax):
                    raise ValueError("`ax` must be a 3D axis.")

            plt.sca(ax)

            ### Set the view angle
            if use_preset_view_angle is not None:
                p.set_preset_3d_view_angle(use_preset_view_angle)

            ### Set the background pane color
            if set_background_pane_color is not None:
                ax.xaxis.pane.set_facecolor(set_background_pane_color)
                ax.yaxis.pane.set_facecolor(set_background_pane_color)
                ax.zaxis.pane.set_facecolor(set_background_pane_color)

            ### Set the background pane alpha
            if set_background_pane_alpha is not None:
                ax.xaxis.pane.set_alpha(set_background_pane_alpha)
                ax.yaxis.pane.set_alpha(set_background_pane_alpha)
                ax.zaxis.pane.set_alpha(set_background_pane_alpha)

            ax.add_collection(
                Poly3DCollection(
                    points[faces], facecolors='lightgray', edgecolors=(0, 0, 0, 0.1),
                    linewidths=0.5, alpha=0.8, shade=True,
                ),
            )

            for prop in self.propulsors:

                ### Disk
                if prop.length == 0:
                    ax.add_collection(
                        Poly3DCollection(
                            np.stack([np.stack(
                                prop.get_disk_3D_coordinates(),
                                axis=1
                            )], axis=0),
                            facecolors='darkgray', edgecolors=(0, 0, 0, 0.2),
                            linewidths=0.5, alpha=0.35, shade=True, zorder=4,
                        )
                    )

            if set_lims:
                ax.set_xlim(points[:, 0].min(), points[:, 0].max())
                ax.set_ylim(points[:, 1].min(), points[:, 1].max())
                ax.set_zlim(points[:, 2].min(), points[:, 2].max())

            if set_equal:
                p.equal()

            if set_axis_visibility is not None:
                if set_axis_visibility:
                    ax.set_axis_on()
                else:
                    ax.set_axis_off()

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
                       use_preset_view_angle: str = None,
                       set_background_pane_color: Union[str, Tuple[float, float, float]] = None,
                       set_background_pane_alpha: float = None,
                       set_lims: bool = True,
                       set_equal: bool = True,
                       set_axis_visibility: bool = None,
                       show: bool = True,
                       ):
        """
        Draws a wireframe of the airplane on a Matplotlib 3D axis.

        Args:

            ax: The axis to draw on. Must be a 3D axis. If None, creates a new axis.

            color: The color of the wireframe.

            thin_linewidth: The linewidth of the thin lines.
        """
        ### Set defaults
        if fuselage_longeron_theta is None:
            fuselage_longeron_theta = np.linspace(0, 2 * np.pi, 8 + 1)[:-1]

        import matplotlib.pyplot as plt
        import aerosandbox.tools.pretty_plots as p

        if ax is None:
            _, ax = p.figure3d(figsize=(8, 8), computed_zorder=False)

        else:
            if not p.ax_is_3d(ax):
                raise ValueError("`ax` must be a 3D axis.")

        plt.sca(ax)

        ### Set the view angle
        if use_preset_view_angle is not None:
            p.set_preset_3d_view_angle(use_preset_view_angle)

        ### Set the background pane color
        if set_background_pane_color is not None:
            ax.xaxis.pane.set_facecolor(set_background_pane_color)
            ax.yaxis.pane.set_facecolor(set_background_pane_color)
            ax.zaxis.pane.set_facecolor(set_background_pane_color)

        ### Set the background pane alpha
        if set_background_pane_alpha is not None:
            ax.xaxis.pane.set_alpha(set_background_pane_alpha)
            ax.yaxis.pane.set_alpha(set_background_pane_alpha)
            ax.zaxis.pane.set_alpha(set_background_pane_alpha)

        def plot_line(
                xyz: np.ndarray,
                symmetric: bool = False,
                color=color,
                linewidth=0.4,
                **kwargs
        ):
            if symmetric:
                xyz = np.concatenate([
                    xyz,
                    np.array([[np.nan] * 3]),
                    xyz * np.array([[1, -1, 1]])
                ], axis=0)

            ax.plot(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                color=color,
                linewidth=linewidth,
                **kwargs
            )

        def reshape(x):
            return np.reshape(np.array(x), (1, 3))

        ##### Wings
        for wing in self.wings:
            try:
                if wing.color is not None:
                    color_to_use = wing.color
                else:
                    color_to_use = color
            except AttributeError:
                color_to_use = color

            ### LE and TE lines
            for xy in [
                (0, 0),  # Leading Edge
                (1, 0),  # Trailing Edge
            ]:

                plot_line(
                    np.stack(wing.mesh_line(x_nondim=xy[0], z_nondim=xy[1]), axis=0),
                    symmetric=wing.symmetric,
                    linewidth=thick_linewidth,
                    color=color_to_use
                )

            ### Top and Bottom lines
            x = 0.4
            afs = [xsec.airfoil for xsec in wing.xsecs]
            thicknesses = np.array([af.local_thickness(x_over_c=x) for af in afs])

            plot_line(
                np.stack(wing.mesh_line(x_nondim=x, z_nondim=thicknesses / 2, add_camber=True), axis=0),
                symmetric=wing.symmetric,
                linewidth=thin_linewidth,
                color=color_to_use
            )
            plot_line(
                np.stack(wing.mesh_line(x_nondim=x, z_nondim=-thicknesses / 2, add_camber=True), axis=0),
                symmetric=wing.symmetric,
                linewidth=thin_linewidth,
                color=color_to_use
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
                        linewidth=thick_linewidth if i == 0 or i == len(wing.xsecs) - 1 else thin_linewidth,
                        color=color_to_use
                    )

        ##### Fuselages
        for fuse in self.fuselages:
            try:
                if fuse.color is not None:
                    color_to_use = fuse.color
                else:
                    color_to_use = color
            except AttributeError:
                color_to_use = color

            ### Bulkheads
            perimeters_xyz = [
                xsec.get_3D_coordinates(theta=np.linspace(0, 2 * np.pi, 121))
                for xsec in fuse.xsecs
            ]
            for i, perim in enumerate(perimeters_xyz):
                plot_line(
                    np.stack(perim, axis=1),
                    linewidth=thick_linewidth if i == 0 or i == len(fuse.xsecs) - 1 else thin_linewidth,
                    color=color_to_use
                )

            ### Centerline
            plot_line(
                np.stack(
                    fuse.mesh_line(y_nondim=0, z_nondim=0),
                    axis=0,
                ),
                linewidth=thin_linewidth,
                color=color_to_use
            )

            ### Longerons
            for theta in fuselage_longeron_theta:
                plot_line(
                    np.stack([
                        np.array(xsec.get_3D_coordinates(theta=theta))
                        for xsec in fuse.xsecs
                    ], axis=0),
                    linewidth=thick_linewidth,
                    color=color_to_use
                )

        ##### Propulsors
        for prop in self.propulsors:
            try:
                if prop.color is not None:
                    color_to_use = prop.color
                else:
                    color_to_use = color
            except AttributeError:
                color_to_use = color

            ### Disk
            if prop.length == 0:
                plot_line(
                    np.stack(
                        prop.get_disk_3D_coordinates(),
                        axis=1
                    ),
                    color=color_to_use
                )

        if set_lims:
            points, _ = self.mesh_body()
            ax.set_xlim(points[:, 0].min(), points[:, 0].max())
            ax.set_ylim(points[:, 1].min(), points[:, 1].max())
            ax.set_zlim(points[:, 2].min(), points[:, 2].max())

        if set_equal:
            p.equal()

        if set_axis_visibility is not None:
            if set_axis_visibility:
                ax.set_axis_on()
            else:
                ax.set_axis_off()

        if show:
            p.show_plot()

    def draw_three_view(self,
                        style: str = "shaded",
                        show: bool = True,
                        ):
        """
        Draws a standard 4-panel three-view diagram of the airplane using Matplotlib backend. Creates a new figure.

        Args:

            style: Determines what drawing style to use for the three-view. A string, one of:

                * "shaded"
                * "wireframe"

            show: A boolean of whether to show the figure after creating it, or to hold it so   that the user can modify the figure further before showing.

        Returns:

        """
        import matplotlib.pyplot as plt
        import aerosandbox.tools.pretty_plots as p

        preset_view_angles = np.array([
            ["XZ", "-YZ"],
            ["XY", "left_isometric"]
        ], dtype="O")

        fig, axs = p.figure3d(
            nrows=preset_view_angles.shape[0],
            ncols=preset_view_angles.shape[1],
            figsize=(8, 8),
            computed_zorder=False,
        )

        for i in range(axs.shape[0]):
            for j in range(axs.shape[1]):
                ax = axs[i, j]
                preset_view = preset_view_angles[i, j]

                if style == "shaded":
                    self.draw(
                        backend="matplotlib",
                        ax=ax,
                        set_axis_visibility=False if 'isometric' in preset_view else None,
                        show=False
                    )
                elif style == "wireframe":
                    if preset_view == "XZ":
                        fuselage_longeron_theta = [np.pi / 2, 3 * np.pi / 2]
                    elif preset_view == "XY":
                        fuselage_longeron_theta = [0, np.pi]
                    else:
                        fuselage_longeron_theta = None

                    self.draw_wireframe(
                        ax=ax,
                        set_axis_visibility=False if 'isometric' in preset_view else None,
                        fuselage_longeron_theta=fuselage_longeron_theta,
                        show=False
                    )

                p.set_preset_3d_view_angle(preset_view)

                xres = np.diff(ax.get_xticks())[0]
                yres = np.diff(ax.get_yticks())[0]
                zres = np.diff(ax.get_zticks())[0]

                p.set_ticks(
                    xres, xres / 4,
                    yres, yres / 4,
                    zres, zres / 4,
                )

                ax.xaxis.set_tick_params(color='white', which='minor')
                ax.yaxis.set_tick_params(color='white', which='minor')
                ax.zaxis.set_tick_params(color='white', which='minor')

                if preset_view == 'XY' or preset_view == '-XY':
                    ax.set_zticks([])
                if preset_view == 'XZ' or preset_view == '-XZ':
                    ax.set_yticks([])
                if preset_view == 'YZ' or preset_view == '-YZ':
                    ax.set_xticks([])

        axs[1, 0].set_xlabel("$x_g$ [m]")
        axs[1, 0].set_ylabel("$y_g$ [m]")
        axs[0, 0].set_zlabel("$z_g$ [m]")
        axs[0, 0].set_xticklabels([])
        axs[0, 1].set_yticklabels([])
        axs[0, 1].set_zticklabels([])

        plt.subplots_adjust(
            left=-0.08,
            right=1.08,
            bottom=-0.08,
            top=1.08,
            wspace=-0.38,
            hspace=-0.38,
        )

        if show:
            p.show_plot(
                tight_layout=False,
            )

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
        Computes the approximate location of the aerodynamic center of the wing.
        Uses the generalized methodology described here:
            https://core.ac.uk/download/pdf/79175663.pdf

        Args:

            chord_fraction: The position of the aerodynamic center along the MAC, as a fraction of MAC length.
            Typically, this value (denoted `h_0` in the literature) is 0.25 for a subsonic wing. However,
            wing-fuselage interactions can cause a forward shift to a value more like 0.1 or less. Citing Cook,
            Michael V., "Flight Dynamics Principles", 3rd Ed., Sect. 3.5.3 "Controls-fixed static stability". PDF:
            https://www.sciencedirect.com/science/article/pii/B9780080982427000031

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

    def with_control_deflections(self,
                                 control_surface_deflection_mappings: Dict[str, float]
                                 ) -> "Airplane":
        """
        Returns a copy of the airplane with the specified control surface deflections applied.

        Args:
            control_surface_deflection_mappings: A dictionary mapping control surface names to deflections.

                * Keys: Control surface names.

                * Values: Deflections, in degrees. Downwards-positive, following typical convention.

        Returns: A copy of the airplane with the specified control surface deflections applied.

        """
        deflected_airplane = copy.deepcopy(self)

        for name, deflection in control_surface_deflection_mappings.items():

            for wi, wing in enumerate(deflected_airplane.wings):

                for xi, xsec in enumerate(wing.xsecs):

                    for csi, surf in enumerate(xsec.control_surfaces):

                        if surf.name == name:

                            surf.deflection = deflection

        return deflected_airplane

    def generate_cadquery_geometry(self,
                                   minimum_airfoil_TE_thickness: float = 0.001,
                                   fuselage_tol: float = 1e-4,
                                   ) -> "Workplane":
        """
        Uses the CADQuery library (OpenCASCADE backend) to generate a 3D CAD model of the airplane.

        Args:

            minimum_airfoil_TE_thickness: The minimum thickness of the trailing edge of the airfoils, as a fraction
            of each airfoil's chord. This will be enforced by thickening the trailing edge of the airfoils if
            necessary. This is useful for avoiding numerical issues in CAD software that can arise from extremely
            thin (i.e., <1e-6 meters) trailing edges.

            tol: The geometric tolerance (meters) to use when generating the CAD geometry. This is passed directly to the CADQuery

        Returns: A CADQuery Workplane object containing the CAD geometry of the airplane.

        """
        try:
            import cadquery as cq
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "The `cadquery` library is required to use this function. Please install it with `pip install cadquery`.")

        solids = []

        for wing in self.wings:

            xsec_wires = []

            for i, xsec in enumerate(wing.xsecs):
                csys = wing._compute_frame_of_WingXSec(i)

                af = xsec.airfoil
                if af.TE_thickness() < minimum_airfoil_TE_thickness:
                    af = af.set_TE_thickness(
                        thickness=minimum_airfoil_TE_thickness
                    )

                LE_index = af.LE_index()

                xsec_wires.append(
                    cq.Workplane(
                        inPlane=cq.Plane(
                            origin=tuple(xsec.xyz_le),
                            xDir=tuple(csys[0]),
                            normal=tuple(-csys[1])
                        )
                    ).spline(
                        listOfXYTuple=[
                            tuple(xy * xsec.chord)
                            for xy in af.coordinates[:LE_index, :]
                        ]
                    ).spline(
                        listOfXYTuple=[
                            tuple(xy * xsec.chord)
                            for xy in af.coordinates[LE_index:, :]
                        ]
                    ).close()
                )

            wire_collection = xsec_wires[0]
            for s in xsec_wires[1:]:
                wire_collection.ctx.pendingWires.extend(s.ctx.pendingWires)

            loft = wire_collection.loft(ruled=True, clean=False)

            solids.append(loft)

            if wing.symmetric:
                loft = loft.mirror(
                    mirrorPlane='XZ',
                    union=False
                )

                solids.append(loft)

        for fuse in self.fuselages:

            xsec_wires = []

            for i, xsec in enumerate(fuse.xsecs):

                if xsec.height < fuselage_tol or xsec.width < fuselage_tol:  # If the xsec is so small as to effectively be a point
                    xsec = copy.deepcopy(xsec)  # Modify the xsec to be big enough to not error out.
                    xsec.height = np.maximum(xsec.height, fuselage_tol)
                    xsec.width = np.maximum(xsec.width, fuselage_tol)

                xsec_wires.append(
                    cq.Workplane(
                        inPlane=cq.Plane(
                            origin=tuple(xsec.xyz_c),
                            xDir=(0, 1, 0),
                            normal=(-1, 0, 0)
                        )
                    ).spline(
                        listOfXYTuple=[
                            (y - xsec.xyz_c[1], z - xsec.xyz_c[2])
                            for x, y, z in zip(*xsec.get_3D_coordinates(
                                theta=np.linspace(
                                    np.pi / 2, np.pi / 2 + 2 * np.pi,
                                    181
                                )
                            ))
                        ]
                    ).close()
                )

            wire_collection = xsec_wires[0]
            for s in xsec_wires[1:]:
                wire_collection.ctx.pendingWires.extend(s.ctx.pendingWires)

            loft = wire_collection.loft(ruled=True, clean=False)

            solids.append(loft)

        solid = solids[0]
        for s in solids[1:]:
            solid.add(s)

        return solid.clean()

    def export_cadquery_geometry(self,
                                 filename: Union[Path, str],
                                 minimum_airfoil_TE_thickness: float = 0.001
                                 ) -> None:
        """
        Exports the airplane geometry to a STEP file.

        Args:
            filename: The filename to export to. Should include the ".step" extension.

            minimum_airfoil_TE_thickness: The minimum thickness of the trailing edge of the airfoils, as a fraction
            of each airfoil's chord. This will be enforced by thickening the trailing edge of the airfoils if
            necessary. This is useful for avoiding numerical issues in CAD software that can arise from extremely
            thin (i.e., <1e-6 meters) trailing edges.

        Returns: None, but exports the airplane geometry to a STEP file.
        """
        solid = self.generate_cadquery_geometry(
            minimum_airfoil_TE_thickness=minimum_airfoil_TE_thickness,
        )

        solid.objects = [
            o.scale(1000)
            for o in solid.objects
        ]

        from cadquery import exporters
        exporters.export(
            solid,
            fname=filename
        )

    def export_AVL(self,
                   filename,
                   include_fuselages: bool = True
                   ):
        # TODO include option for mass file export as well
        # Use MassProperties.export_AVL_mass...

        from aerosandbox.aerodynamics.aero_3D.avl import AVL
        avl = AVL(
            airplane=self,
            op_point=None,
            xyz_ref=self.xyz_ref
        )
        avl.write_avl(filepath=filename)

    def export_XFLR(self, *args, **kwargs) -> str:
        import warnings

        warnings.warn(
            "`Airplane.export_XFLR()` has been renamed to `Airplane.export_XFLR5_xml()`, to clarify\n"
            "that it exports to XFLR5's XML format, not to a XFL file.\n"
            "\n"
            "Please update your code to use `Airplane.export_XFLR5_xml()` instead.\n"
            "\n"
            "This function will be removed in a future version of AeroSandbox.",
            PendingDeprecationWarning
        )
        return self.export_XFLR5_xml(*args, **kwargs)

    def export_XFLR5_xml(self,
                         filename: Union[Path, str],
                         mass_props: MassProperties = None,
                         include_fuselages: bool = False,
                         mainwing: Wing = None,
                         elevator: Wing = None,
                         fin: Wing = None,
                         ) -> str:
        """
        Exports the airplane geometry to an XFLR5 `.xml` file. To import the `.xml` file into XFLR5, go to File ->
        Import -> Import from XML.

        Args:
            filename: The filename to export to. Should include the ".xml" extension.

            mass_props: The MassProperties object to use when exporting the airplane. If not specified, will default to
                a 1 kg point mass at the origin.

                - Note: XFLR5 does not natively support user-defined inertia tensors, so we have to synthesize an equivalent
                set of point masses to represent the inertia tensor.

            include_fuselages: Whether to include fuselages in the export.

            mainwing: The main wing of the airplane. If not specified, will default to the first wing in the airplane.

            elevator: The elevator of the airplane. If not specified, will default to the second wing in the airplane.

            fin: The fin of the airplane. If not specified, will default to the third wing in the airplane.

        Returns: None, but exports the airplane geometry to an XFLR5 `.xml` file.

            To import the `.xml` file into XFLR5, go to File -> Import -> Import from XML.
        """
        ### Handle default arguments
        if mass_props is None:
            mass_props = MassProperties(
                mass=1,
                x_cg=0,
                y_cg=0,
                z_cg=0,
            )

        ### Identify which wings are the main wing, elevator, and fin.
        wings_specified = [
            mainwing is not None,
            elevator is not None,
            fin is not None,
        ]
        if all(wings_specified):
            pass
        elif any(wings_specified):
            raise ValueError(
                "If any wings are specified (`mainwing`, `elevator`, `fin`), then all wings must be specified.")
        else:
            n_wings = len(self.wings)

            if n_wings == 0:
                pass
            else:
                import warnings
                warnings.warn(
                    "No wings were specified (`mainwing`, `elevator`, `fin`). Automatically assigning the first wing "
                    "to `mainwing`, the second wing to `elevator`, and the third wing to `fin`. If this is not "
                    "correct, manually specify these with (`mainwing`, `elevator`, and `fin`) arguments."
                )

                if n_wings == 1:
                    mainwing = self.wings[0]
                elif n_wings == 2:
                    mainwing = self.wings[0]
                    elevator = self.wings[1]
                elif n_wings == 3:
                    mainwing = self.wings[0]
                    elevator = self.wings[1]
                    fin = self.wings[2]
                else:
                    raise ValueError(
                        "Could not automatically parse which wings should be assigned to which XFLR5 lifting surfaces, "
                        "since there are too many. Manually assign these with (`mainwing`, `elevator`, and `fin`) "
                        "arguments."
                    )

        ### Determine where point masses should be in order to yield the specified mass properties.
        point_masses = mass_props.generate_possible_set_of_point_masses()

        ### Handle the fuselage
        if include_fuselages:
            raise NotImplementedError(
                "Fuselage export to XFLR5 is not yet implemented."
            )

        ### Write the XML file.
        import xml.etree.ElementTree as ET

        base_xml = f"""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE explane>
<explane version="1.0">
    <Units>
        <length_unit_to_meter>1</length_unit_to_meter>
        <mass_unit_to_kg>1</mass_unit_to_kg>
    </Units>
    <Plane>
        <Name>{self.name}</Name>
        <Description></Description>
        <Inertia>
        </Inertia>
        <has_body>false</has_body>
    </Plane>
</explane>
"""

        root = ET.fromstring(base_xml)
        plane = root.find("Plane")

        ### Add point masses
        inertia = plane.find("Inertia")
        for i, point_mass in enumerate(point_masses):
            point_mass_xml = ET.SubElement(inertia, "Point_Mass")

            for k, v in {
                "Tag"        : f"pm{i}",
                "Mass"       : point_mass.mass,
                "coordinates": ",".join([str(x) for x in point_mass.xyz_cg]),
            }.items():
                subelement = ET.SubElement(point_mass_xml, k)
                subelement.text = str(v)

        ### Add the wings
        if mainwing is not None:
            wing = mainwing
            wingxml = ET.SubElement(plane, "wing")

            xyz_le_root = wing._compute_xyz_of_WingXSec(index=0, x_nondim=0, z_nondim=0)

            for k, v in {
                "Name"      : wing.name,
                "Type"      : "MAINWING",
                "Position"  : ",".join([str(x) for x in xyz_le_root]),
                "Tilt_angle": 0.,
                "Symetric"  : wing.symmetric,  # This tag is a typo in XFLR...
                "isFin"     : "false",
                "isSymFin"  : "false",
            }.items():
                subelement = ET.SubElement(wingxml, k)
                subelement.text = str(v)

            sections = ET.SubElement(wingxml, "Sections")

            xyz_le_sects_rel = [
                wing._compute_xyz_of_WingXSec(index=i, x_nondim=0, z_nondim=0) - xyz_le_root
                for i in range(len(wing.xsecs))
            ]

            for i, xsec in enumerate(wing.xsecs):

                sect = ET.SubElement(sections, "Section")

                if i == len(wing.xsecs) - 1:
                    dihedral = 0
                else:
                    dihedral = np.arctan2d(
                        xyz_le_sects_rel[i + 1][2] - xyz_le_sects_rel[i][2],
                        xyz_le_sects_rel[i + 1][1] - xyz_le_sects_rel[i][1],
                    )

                for k, v in {
                    "y_position"         : xyz_le_sects_rel[i][1],
                    "Chord"              : xsec.chord,
                    "xOffset"            : xyz_le_sects_rel[i][0],
                    "Dihedral"           : dihedral,
                    "Twist"              : xsec.twist,
                    "Left_Side_FoilName" : xsec.airfoil.name,
                    "Right_Side_FoilName": xsec.airfoil.name,
                    "x_number_of_panels" : 8,
                    "y_number_of_panels" : 8,
                }.items():
                    subelement = ET.SubElement(sect, k)
                    subelement.text = str(v)

        if elevator is not None:
            wing = elevator
            wingxml = ET.SubElement(plane, "wing")

            xyz_le_root = wing._compute_xyz_of_WingXSec(index=0, x_nondim=0, z_nondim=0)

            for k, v in {
                "Name"      : wing.name,
                "Type"      : "ELEVATOR",
                "Position"  : ",".join([str(x) for x in xyz_le_root]),
                "Tilt_angle": 0.,
                "Symetric"  : wing.symmetric,  # This tag is a typo in XFLR...
                "isFin"     : "false",
                "isSymFin"  : "false",
            }.items():
                subelement = ET.SubElement(wingxml, k)
                subelement.text = str(v)

            sections = ET.SubElement(wingxml, "Sections")

            xyz_le_sects_rel = [
                wing._compute_xyz_of_WingXSec(index=i, x_nondim=0, z_nondim=0) - xyz_le_root
                for i in range(len(wing.xsecs))
            ]

            for i, xsec in enumerate(wing.xsecs):

                sect = ET.SubElement(sections, "Section")

                if i == len(wing.xsecs) - 1:
                    dihedral = 0
                else:
                    dihedral = np.arctan2d(
                        xyz_le_sects_rel[i + 1][2] - xyz_le_sects_rel[i][2],
                        xyz_le_sects_rel[i + 1][1] - xyz_le_sects_rel[i][1],
                    )

                for k, v in {
                    "y_position"         : xyz_le_sects_rel[i][1],
                    "Chord"              : xsec.chord,
                    "xOffset"            : xyz_le_sects_rel[i][0],
                    "Dihedral"           : dihedral,
                    "Twist"              : xsec.twist,
                    "Left_Side_FoilName" : xsec.airfoil.name,
                    "Right_Side_FoilName": xsec.airfoil.name,
                    "x_number_of_panels" : 8,
                    "y_number_of_panels" : 8,
                }.items():
                    subelement = ET.SubElement(sect, k)
                    subelement.text = str(v)

        if fin is not None:
            wing = fin
            wingxml = ET.SubElement(plane, "wing")

            xyz_le_root = wing._compute_xyz_of_WingXSec(index=0, x_nondim=0, z_nondim=0)

            for k, v in {
                "Name"      : wing.name,
                "Type"      : "FIN",
                "Position"  : ",".join([str(x) for x in xyz_le_root]),
                "Tilt_angle": 0.,
                "Symetric"  : "true",  # This tag is a typo in XFLR...
                "isFin"     : "true",
                "isSymFin"  : wing.symmetric,
            }.items():
                subelement = ET.SubElement(wingxml, k)
                subelement.text = str(v)

            sections = ET.SubElement(wingxml, "Sections")

            xyz_le_sects_rel = [
                wing._compute_xyz_of_WingXSec(index=i, x_nondim=0, z_nondim=0) - xyz_le_root
                for i in range(len(wing.xsecs))
            ]

            for i, xsec in enumerate(wing.xsecs):

                sect = ET.SubElement(sections, "Section")

                if i == len(wing.xsecs) - 1:
                    dihedral = 0
                else:
                    dihedral = np.arctan2d(
                        xyz_le_sects_rel[i + 1][1] - xyz_le_sects_rel[i][1],
                        xyz_le_sects_rel[i + 1][2] - xyz_le_sects_rel[i][2],
                    )

                for k, v in {
                    "y_position"         : xyz_le_sects_rel[i][2],
                    "Chord"              : xsec.chord,
                    "xOffset"            : xyz_le_sects_rel[i][0],
                    "Dihedral"           : dihedral,
                    "Twist"              : xsec.twist,
                    "Left_Side_FoilName" : xsec.airfoil.name,
                    "Right_Side_FoilName": xsec.airfoil.name,
                    "x_number_of_panels" : 8,
                    "y_number_of_panels" : 8,
                }.items():
                    subelement = ET.SubElement(sect, k)
                    subelement.text = str(v)

        ### Indents the XML file properly
        def indent(elem, level=0):
            i = "\n" + level * "  "
            if len(elem):
                if not elem.text or not elem.text.strip():
                    elem.text = i + "  "
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
                for elem in elem:
                    indent(elem, level + 1)
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
            else:
                if level and (not elem.tail or not elem.tail.strip()):
                    elem.tail = i

        indent(root)

        xml_string = ET.tostring(
            root,
            encoding="UTF-8",
            xml_declaration=True
        ).decode()

        with open(filename, "w+") as f:
            f.write(xml_string)

        return xml_string

    def export_OpenVSP_vspscript(self,
                                 filename: Union[Path, str],
                                 ) -> str:
        """
        Exports the airplane geometry to a `*.vspscript` file compatible with OpenVSP. To import the `.vspscript`
        file into OpenVSP:

        Open OpenVSP, then File -> Run Script -> Select the `.vspscript` file.

        Args:
            filename: The filename to export to, given as a string or Path. Should include the ".vspscript" extension.

        Returns: A string of the file contents, and also saves the file to the specified filename
        """
        from aerosandbox.geometry.openvsp_io.asb_to_openvsp.airplane_vspscript_generator import generate_airplane

        vspscript_code = generate_airplane(self)

        with open(filename, "w+") as f:
            f.write(vspscript_code)

        return vspscript_code


if __name__ == '__main__':
    import aerosandbox as asb
    # import aerosandbox.numpy as np
    import aerosandbox.tools.units as u


    def ft(feet, inches=0):  # Converts feet (and inches) to meters
        return feet * u.foot + inches * u.inch


    naca2412 = asb.Airfoil("naca2412")
    naca0012 = asb.Airfoil("naca0012")

    airplane = Airplane(
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
                        airfoil=naca2412,
                        control_surfaces=[
                            asb.ControlSurface(
                                name="aileron",
                                symmetric=False,
                                hinge_point=0.8,
                                deflection=0
                            )
                        ]
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
                        twist=-2,
                        control_surfaces=[
                            asb.ControlSurface(
                                name="elevator",
                                symmetric=True,
                                hinge_point=0.75,
                                deflection=0
                            )
                        ]
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
                        control_surfaces=[
                            asb.ControlSurface(
                                name="rudder",
                                hinge_point=0.75,
                                deflection=0
                            )
                        ]
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
    # airplane.export_XFLR5_xml("test.xml", mass_props=asb.MassProperties(mass=1, Ixx=1, Iyy=1, Izz=1))
