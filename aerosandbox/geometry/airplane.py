from aerosandbox import AeroSandboxObject
from aerosandbox.geometry.common import *
from typing import List, Dict, Any, Union
import aerosandbox.geometry.mesh_utilities as mesh_utils


class Airplane(AeroSandboxObject):
    """
    Definition for an airplane.

    Anatomy of an Airplane:

        An Airplane consists chiefly of a collection of wings and fuselages. These can be accessed with
        `Airplane.wings` and `Airplane.fuselages`, which gives a list of those respective components. Each wing is a
        Wing object, and each fuselage is a Fuselage object.
    """

    def __init__(self,
                 name: str = "Untitled",
                 xyz_ref: Union[np.ndarray, List] = np.array([0, 0, 0]),
                 wings: List['Wing'] = None,
                 fuselages: List['Fuselage'] = None,
                 s_ref: float = None,
                 c_ref: float = None,
                 b_ref: float = None,
                 analysis_specific_options: Dict[type, Dict[str, Any]] = None
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
        if wings is None:
            wings: List['Wing'] = []
        if fuselages is None:
            fuselages: List['Fuselage'] = []
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
             show_kwargs=None,
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

        if backend == "plotly":

            points, faces = self.mesh_body(method="quad", thin_wings=thin_wings)

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

            points, faces = self.mesh_body(method="quad")

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

            points, faces = self.mesh_body(method="tri")

            import trimesh as tri
            fig = tri.Trimesh(points, faces)
            if show:
                fig.show(**show_kwargs)
            return fig
        else:
            raise ValueError("Bad value of `backend`!")

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
