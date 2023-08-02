import aerosandbox.numpy as np
from aerosandbox import ExplicitAnalysis, AeroSandboxObject
from aerosandbox.geometry import *
from aerosandbox.performance import OperatingPoint
from aerosandbox.aerodynamics.aero_3D.singularities.uniform_strength_horseshoe_singularities import \
    calculate_induced_velocity_horseshoe
from typing import Dict, Any, List, Callable, Optional, Union, Tuple
import copy
from functools import cached_property, lru_cache, partial
from collections import namedtuple
from dataclasses import dataclass
from abc import abstractmethod, ABC


### Define some helper functions that take a vector and make it a Nx1 or 1xN, respectively.
# Useful for broadcasting with matrices later.
def tall(array):
    return np.reshape(array, (-1, 1))


def wide(array):
    return np.reshape(array, (1, -1))


immutable_dataclass = partial(dataclass, frozen=True, repr=False)


class LinearPotentialFlow(ExplicitAnalysis):

    def __init__(self,
                 airplane: Airplane,
                 op_point: OperatingPoint,
                 xyz_ref: List[float] = None,
                 run_symmetric_if_possible: bool = False,
                 verbose: bool = False,
                 wing_model: Union[str, Dict[Wing, str]] = "vortex_lattice_all_horseshoe",
                 fuselage_model: Union[str, Dict[Fuselage, str]] = "none",
                 wing_options: Union[Dict[str, Any], Dict[Wing, Dict[str, Any]]] = None,
                 fuselage_options: Union[Dict[str, Any], Dict[Fuselage, Dict[str, Any]]] = None,
                 ):
        import warnings
        warnings.warn("LinearPotentialFlow is under active development and is not yet ready for use.", UserWarning)

        super().__init__()

        ##### Set defaults
        if xyz_ref is None:
            xyz_ref = airplane.xyz_ref
        if wing_options is None:
            wing_options = {}
        if fuselage_options is None:
            fuselage_options = {}

        ##### Initialize
        self.airplane = airplane
        self.op_point = op_point
        self.xyz_ref = xyz_ref
        self.verbose = verbose

        ##### Set up the modeling methods
        if isinstance(wing_model, str):
            wing_model = {wing: wing_model for wing in self.airplane.wings}
        if isinstance(fuselage_model, str):
            fuselage_model = {fuselage: fuselage_model for fuselage in self.airplane.fuselages}

        self.wing_model: Dict[Wing, str] = wing_model
        self.fuselage_model: Dict[Fuselage, str] = fuselage_model

        ##### Set up the modeling options
        ### Check the format of the wing options
        if not (
                all([isinstance(k, str) for k in wing_options.keys()]) or
                all([issubclass(k, Wing) for k in wing_options.keys()])
        ):
            raise ValueError("`wing_options` must be either:\n"
                             "    - A dictionary of the form `{str: value}`, which is applied to all Wings\n"
                             "    - A nested dictionary of the form `{Wing: {str: value}}`, which is applied to the corresponding Wings\n"
                             )
        elif all([isinstance(k, str) for k in wing_options.keys()]):
            wing_options = {wing: wing_options for wing in self.airplane.wings}

        ### Check the format of the fuselage options
        if not (
                all([isinstance(k, str) for k in fuselage_options.keys()]) or
                all([issubclass(k, Fuselage) for k in fuselage_options.keys()])
        ):
            raise ValueError("`fuselage_options` must be either:\n"
                             "    - A dictionary of the form `{str: value}`, which is applied to all Fuselages\n"
                             "    - A nested dictionary of the form `{Fuselage: {str: value}}`, which is applied to the corresponding Fuselages\n"
                             )
        elif all([isinstance(k, str) for k in fuselage_options.keys()]):
            fuselage_options = {fuselage: fuselage_options for fuselage in self.airplane.fuselages}

        ### Set user-specified values
        self.wing_options: Dict[Wing, Dict[str, Any]] = wing_options
        self.fuselage_options: Dict[Fuselage, Dict[str, Any]] = fuselage_options

        ### Set default values
        wing_model_default_options = {
            "none"                        : {},
            "vortex_lattice_all_horseshoe": {
                "spanwise_resolution"              : 10,
                "spanwise_spacing_function"        : np.cosspace,
                "chordwise_resolution"             : 10,
                "chordwise_spacing_function"       : np.cosspace,
                "vortex_core_radius"               : 1e-8,
                "align_trailing_vortices_with_wind": False,
            },
            "vortex_lattice_ring"         : {
                "spanwise_resolution"              : 10,
                "spanwise_spacing_function"        : np.cosspace,
                "chordwise_resolution"             : 10,
                "chordwise_spacing_function"       : np.cosspace,
                "vortex_core_radius"               : 1e-8,
                "align_trailing_vortices_with_wind": False,
            },
            "lifting_line"                : {
                "sectional_data_source": "neuralfoil",
            },
        }

        for wing in self.airplane.wings:
            if self.wing_model[wing] in wing_model_default_options.keys():
                self.wing_options[wing] = {
                    **wing_model_default_options[self.wing_model[wing]],
                    **self.wing_options[wing],
                }
            else:
                raise ValueError(f"Invalid wing model specified: \"{self.wing_model[wing]}\"\n"
                                 f"Must be one of: {list(wing_model_default_options.keys())}")

        fuselage_model_default_options = {
            "none"                  : {},
            "prescribed_source_line": {
                "lengthwise_resolution"      : 1,
                "lengthwise_spacing_function": np.cosspace,
            },
        }

        for fuselage in self.airplane.fuselages:
            if self.fuselage_model[fuselage] in fuselage_model_default_options.keys():
                self.fuselage_options[fuselage] = {
                    **fuselage_model_default_options[self.fuselage_model[fuselage]],
                    **self.fuselage_options[fuselage],
                }
            else:
                raise ValueError(f"Invalid fuselage model specified: \"{self.fuselage_model[fuselage]}\"\n"
                                 f"Must be one of: {list(fuselage_model_default_options.keys())}")

        ### Determine whether you should run the problem as symmetric
        self.run_symmetric = False
        if run_symmetric_if_possible:
            raise NotImplementedError("LinearPotentialFlow with symmetry detection not yet implemented!")
            # try:
            #     self.run_symmetric = (  # Satisfies assumptions
            #             self.op_point.beta == 0 and
            #             self.op_point.p == 0 and
            #             self.op_point.r == 0 and
            #             self.airplane.is_entirely_symmetric()
            #     )
            # except RuntimeError:  # Required because beta, p, r, etc. may be non-numeric (e.g. opti variables)
            #     pass

    def __repr__(self):
        return self.__class__.__name__ + "(\n" + "\n".join([
            f"\tairplane={self.airplane}",
            f"\top_point={self.op_point}",
            f"\txyz_ref={self.xyz_ref}",
        ]) + "\n)"

    @immutable_dataclass
    class Elements(ABC):
        parent_component: AeroSandboxObject
        start_index: int
        end_index: int

        def __repr__(self):
            return self.__class__.__name__ + "(\n" + "\n".join([
                f"\tparent_component={self.parent_component}",
                f"\tstart_index={self.start_index}",
                f"\tend_index={self.end_index}",
                f"\tlength={len(self)}",
            ]) + "\n)"

        @abstractmethod
        def __len__(self):
            pass

        # @abstractmethod
        # def get_induced_velocity_at_points(self,
        #                                    points: np.ndarray
        #                                    ) -> np.ndarray:
        #     pass

    @immutable_dataclass
    class PanelElements(Elements, ABC):
        front_left_vertices: np.ndarray  # Nx3 array of panel corners
        back_left_vertices: np.ndarray  # Nx3 array of panel corners
        back_right_vertices: np.ndarray  # Nx3 array of panel corners
        front_right_vertices: np.ndarray  # Nx3 array of panel corners

        def __len__(self):
            return np.length(self.front_left_vertices)

        @cached_property
        def crosses(self):
            diag1 = self.front_right_vertices - self.back_left_vertices
            diag2 = self.front_left_vertices - self.back_right_vertices
            return np.cross(diag1, diag2)

        @cached_property
        def cross_norms(self):
            return np.linalg.norm(self.crosses, axis=1)

        @cached_property
        def areas(self) -> np.ndarray:
            return self.cross_norms / 2

        @cached_property
        def normal_directions(self):
            return self.crosses / tall(self.cross_norms)

    @immutable_dataclass
    class WingHorseshoeVortexElements(PanelElements):
        trailing_vortex_direction: np.ndarray  # Nx3 array of trailing vortex directions
        vortex_core_radius: float  # [meters]

        @cached_property
        def left_vortex_vertices(self):
            return 0.75 * self.front_left_vertices + 0.25 * self.back_left_vertices

        @cached_property
        def right_vortex_vertices(self):
            return 0.75 * self.front_right_vertices + 0.25 * self.back_right_vertices

        @cached_property
        def vortex_centers(self):
            return (self.left_vortex_vertices + self.right_vortex_vertices) / 2

        @cached_property
        def vortex_bound_legs(self):
            return self.right_vortex_vertices - self.left_vortex_vertices

        @cached_property
        def collocation_points(self):
            return (
                    0.5 * (0.25 * self.front_left_vertices + 0.75 * self.back_left_vertices) +
                    0.5 * (0.25 * self.front_right_vertices + 0.75 * self.back_right_vertices)
            )

        def get_induced_velocity_at_points(self,
                                           points: np.ndarray,
                                           vortex_strengths: np.ndarray,
                                           sum_across_elements: bool = True
                                           ) -> Tuple[np.ndarray]:
            u_induced, v_induced, w_induced = calculate_induced_velocity_horseshoe(
                x_field=tall(points[:, 0]),
                y_field=tall(points[:, 1]),
                z_field=tall(points[:, 2]),
                x_left=wide(self.left_vortex_vertices[:, 0]),
                y_left=wide(self.left_vortex_vertices[:, 1]),
                z_left=wide(self.left_vortex_vertices[:, 2]),
                x_right=wide(self.right_vortex_vertices[:, 0]),
                y_right=wide(self.right_vortex_vertices[:, 1]),
                z_right=wide(self.right_vortex_vertices[:, 2]),
                trailing_vortex_direction=self.trailing_vortex_direction,
                gamma=wide(vortex_strengths),
                vortex_core_radius=self.vortex_core_radius
            )

            if sum_across_elements:
                u_induced = np.sum(u_induced, axis=1)
                v_induced = np.sum(v_induced, axis=1)
                w_induced = np.sum(w_induced, axis=1)

            return u_induced, v_induced, w_induced

    @immutable_dataclass
    class WingLiftingLineElements(WingHorseshoeVortexElements):
        CL0: np.ndarray  # length N array of zero-angle-of-attack lift coefficients
        CLa: np.ndarray  # length N array of lift slopes (1/rad)

        @property
        def collocation_points(self):
            raise NotImplementedError

    @immutable_dataclass
    class FuselagePrescribedSourceLineElements(Elements):
        front_vertices: np.ndarray
        back_vertices: np.ndarray
        strength: np.ndarray

        def __len__(self):
            return np.length(self.front_vertices)

    @cached_property
    def discretization(self):
        """

        Returns: A list of dictionaries, where each item in the list represents a single element.

            Each item in the list is a namedtuple (effectively, a dictionary), and one of the following types:

                    * `wing_vlm_element`
                    * `wing_lifting_line_element`
                    * `fuselage_prescribed_source_line`

        """
        ### Initialize
        discretization = []
        index = 0

        ### Wings
        for wing in self.airplane.wings:
            element_type: str = self.wing_model[wing]
            options = self.wing_options[wing]

            if element_type == "none":
                continue

            elif element_type == "vortex_lattice_all_horseshoe":
                if options["spanwise_resolution"] > 1:
                    subdivided_wing = wing.subdivide_sections(
                        ratio=options["spanwise_resolution"],
                        spacing_function=options["spanwise_spacing_function"],
                    )
                else:
                    subdivided_wing = wing

                points, faces = subdivided_wing.mesh_thin_surface(
                    method="quad",
                    chordwise_resolution=options["chordwise_resolution"],
                    chordwise_spacing_function=options["chordwise_spacing_function"],
                    add_camber=True
                )

                if options["align_trailing_vortices_with_wind"]:
                    raise NotImplementedError("align_trailing_vortices_with_wind not yet implemented!")
                else:
                    trailing_vortex_direction = np.array([1, 0, 0])

                discretization.append(
                    self.WingHorseshoeVortexElements(
                        parent_component=wing,
                        start_index=index,
                        end_index=(index := index + len(faces)),
                        front_left_vertices=points[faces[:, 0], :],
                        back_left_vertices=points[faces[:, 1], :],
                        back_right_vertices=points[faces[:, 2], :],
                        front_right_vertices=points[faces[:, 3], :],
                        trailing_vortex_direction=trailing_vortex_direction,
                        vortex_core_radius=options["vortex_core_radius"],
                    )
                )
            elif element_type == "vortex_lattice_ring":
                raise NotImplementedError("vortex_lattice_ring not yet implemented!")

            elif element_type == "lifting_line":
                raise NotImplementedError("lifting_line not yet implemented!")

            else:
                raise ValueError(f"Invalid wing model specified: \"{element_type}\"")

        ### Fuselages
        for fuselage in self.airplane.fuselages:
            element_type: str = self.fuselage_model[fuselage]
            options = self.fuselage_options[fuselage]

            if element_type == "none":
                continue

            elif element_type == "prescribed_source_line":
                raise NotImplementedError("prescribed_source_line not yet implemented!")

            else:
                raise ValueError(f"Invalid fuselage model specified: \"{element_type}\"")

        return discretization

    @cached_property
    def N_elements(self):
        return sum([len(element_collection) for element_collection in self.discretization])

    @cached_property
    def AIC(self):

        A = np.empty((self.N_elements, self.N_elements)) * np.nan

        for element_collection in self.discretization:
            if isinstance(element_collection, self.WingHorseshoeVortexElements):
                raise NotImplementedError("AIC not yet implemented for horseshoe vortices.")
            elif isinstance(element_collection, self.WingLiftingLineElements):
                raise NotImplementedError("AIC not yet implemented for lifting lines.")
            elif isinstance(element_collection, self.FuselagePrescribedSourceLineElements):
                raise NotImplementedError("AIC not yet implemented for fuselages.")
            else:
                raise ValueError(f"Invalid element type: {type(element_collection)}")

        return A

    def run(self) -> Dict[str, Any]:
        """
        Computes the aerodynamic forces.

        Returns a dictionary with keys:

            - 'F_g' : an [x, y, z] list of forces in geometry axes [N]
            - 'F_b' : an [x, y, z] list of forces in body axes [N]
            - 'F_w' : an [x, y, z] list of forces in wind axes [N]
            - 'M_g' : an [x, y, z] list of moments about geometry axes [Nm]
            - 'M_b' : an [x, y, z] list of moments about body axes [Nm]
            - 'M_w' : an [x, y, z] list of moments about wind axes [Nm]
            - 'L' : the lift force [N]. Definitionally, this is in wind axes.
            - 'Y' : the side force [N]. This is in wind axes.
            - 'D' : the drag force [N]. Definitionally, this is in wind axes.
            - 'l_b', the rolling moment, in body axes [Nm]. Positive is roll-right.
            - 'm_b', the pitching moment, in body axes [Nm]. Positive is pitch-up.
            - 'n_b', the yawing moment, in body axes [Nm]. Positive is nose-right.
            - 'CL', the lift coefficient [-]. Definitionally, this is in wind axes.
            - 'CY', the sideforce coefficient [-]. This is in wind axes.
            - 'CD', the drag coefficient [-]. Definitionally, this is in wind axes.
            - 'Cl', the rolling coefficient [-], in body axes
            - 'Cm', the pitching coefficient [-], in body axes
            - 'Cn', the yawing coefficient [-], in body axes

        Nondimensional values are nondimensionalized using reference values in the LinearPotentialFlow.airplane object.
        """

        raise NotImplementedError

    def get_induced_velocity_at_points(self,
                                       points: np.ndarray
                                       ) -> np.ndarray:
        raise NotImplementedError

    def get_velocity_at_points(self,
                               points: np.ndarray
                               ) -> np.ndarray:
        raise NotImplementedError

    def get_streamlines(self,
                        seed_points: np.ndarray = None,
                        n_steps: int = 300,
                        length: float = None,
                        ):
        raise NotImplementedError

    def draw(self,
             c: np.ndarray = None,
             cmap: str = None,
             colorbar_label: str = None,
             show: bool = True,
             show_kwargs: Dict = None,
             draw_streamlines=True,
             recalculate_streamlines=False,
             backend: str = "pyvista"
             ):
        raise NotImplementedError

    def draw_three_view(self):
        raise NotImplementedError


if __name__ == '__main__':
    ### Import Vanilla Airplane
    import aerosandbox as asb

    from pathlib import Path

    geometry_folder = Path(__file__).parent / "test_aero_3D" / "geometries"

    import sys

    sys.path.insert(0, str(geometry_folder))

    from vanilla import airplane as vanilla

    ### Do the AVL run
    lpf = LinearPotentialFlow(
        airplane=vanilla,
        op_point=asb.OperatingPoint(
            atmosphere=asb.Atmosphere(altitude=0),
            velocity=10,
            alpha=0,
            beta=0,
            p=0,
            q=0,
            r=0,
        ),
    )

    dis = lpf.discretization

    res = lpf.run()

    for k, v in res.items():
        print(f"{str(k).rjust(10)} : {v}")
