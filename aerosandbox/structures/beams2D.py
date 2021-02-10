import casadi as cas
from typing import Union

import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.geometry import *
        

class Beam6DOF(AeroSandboxObject):
    """
    A Euler-Bernoulli 6 DOF FE beam.
    
    (E * I * u(x)'')'' = q(x)

    where:
        * E is the elastic modulus
        * I is the bending moment of inertia
        * u(x) is the local displacement at x.
        * q(x) is the force-per-unit-length at x. (In other words, a dirac delta is a point load.)
        * ()' is a derivative w.r.t. x.

    Governing equation for torsion:
    phi(x)'' = -T / (G * J)

    where:
        * phi is the local twist angle
        * T is the local torque per unit length
        * G is the local shear modulus
        * J is the polar moment of inertia
        * ()' is a derivative w.r.t. x.
        
    Limitations:
        * Does not take buckling into account (fixed cross-section)
    """
    
    def __init__(self,
                 opti:                  asb.Opti,  # type: asb.Opti
                 length:                float,
                 init_geometry:         dict = {},
                 points_per_point_load: int=100,
                 E:                     float=228e9,  # Pa
                 isotropic:             bool=True,
                 poisson_ratio:         float=0.5,
                 max_allowable_stress:  float=570e6 / 1.75,
                 density:               float=1600,
                 G:                     float=None,
                 bending:               bool=True,  # Should we consider beam bending?
                 torsion:               bool=True,  # Should we consider beam torsion?
                 ):
        """
        :param opti: An optimization environment. # type: cas.Opti
        :param length: Length of the beam [m]
        :param points_per_point_load: Number of discretization points to use per point load
        :param E: Elastic modulus [Pa]
        :param isotropic: Is the material isotropic? If so, attempts to find shear modulus from poisson's ratio, or vice versa. [boolean]
        :param poisson_ratio: Poisson's ratio (if isotropic, can't set both poisson_ratio and shear modulus - one must be None)
        :param diameter_guess: Initial guess for the tube diameter [m]. Make this larger for more computational stability, lower for a bit faster speed.
        :param thickness: Tube wall thickness. This will often be set by shell buckling considerations. [m]
        :param max_allowable_stress: Maximum allowable stress in the material. [Pa]
        :param density: Density of the material [kg/m^3]
        :param G: Shear modulus (if isotropic, can't set both poisson_ratio and shear modulus - one must be None)
        :param bending: Should we consider bending? [boolean]
        :param torsion: Should we consider torsion? [boolean]
        """
        
        self.opti = opti
        self.length = length
        self.init_geometry = init_geometry
        self.points_per_point_load = points_per_point_load
        self.E = E
        self.isotropic = isotropic
        self.poisson_ratio = poisson_ratio
        self.max_allowable_stress = max_allowable_stress
        self.density = density
        self.G = G
        self.bending = bending
        self.torsion = torsion
        
    
    #### Properties

    @property
    def I(self):
        """
        :return: np.ndarray
        """
        return self.cross_section.I()
    
    @property
    def J(self):
        raise self.cross_section.J()

    @property
    def volume(self):
        raise NotImplementedError('Volume not specified for object of type ' + str(type(self)))

    @property
    def init_geometry(self):
        return self._init_geometry
    
    req_geometry_vars = []
    @init_geometry.setter
    def init_geometry(self, value):
        for var in self.req_geometry_vars:
            assert var in value.keys()
        
        self._init_geometry = value
        
        for key in value.keys():
            setattr(self, key, value[key])
            
    @property
    def cross_section(self):
        return NotImplementedError('Geometry not specified for object of type ' + str(type(self)))
        
    #### Methods
        
    def add_point_load(self,
                       location,
                       force: np.ndarray = np.zeros((3,)),
                       bending_moment: np.ndarray = np.zeros((2,)),
                       torsional_moment: float = 0,
                       ):
        """
        Adds a point force and/or moment.
        :param location: Location of the point force along the beam [m]
        :param force: Force to add [N]
        :param bending_moment: Bending moment to add [N-m] # TODO make this work
        :return: None (in-place)
        """
        self.point_loads.append(
            {
                "location"        : location,
                "force"           : force,
                "bending_moment"  : bending_moment,
                'torsional_moment': torsional_moment,
            }
        )

    def add_distributed_load(self,
                         force: np.ndarray = np.zeros((3,)),
                         bending_moment: np.ndarray = np.zeros((2,)),
                         torsional_moment: float = 0,
                         type: str = 'distributed'
                         ):
        """
        Adds a uniformly distributed force and/or moment across the entire length of the beam.
        :param force: Total force applied to beam [N]
        :param bending_moment: Bending moment to add [N-m] # TODO make this work
        :param type: distributed/elliptical
        :return: None (in-place)
        """
        self.distributed_loads.append(
            {
                "type"            : type,
                "force"           : force,
                "bending_moment"  : bending_moment,
                'torsional_moment': torsional_moment,
            }
        )
        
    def _extra_constraints(self):
        """
        Geometry-specific constraints.
        :return: None (in-place)
        """
        pass
    
    def _init_opt_vars(self):
        """
        Geometry-specific variables.
        :return: None (in-place)
        """
        raise NotImplementedError()
        
    def _add_loads(self):
        
        self.force_per_unit_length = cas.GenMX_zeros(self.n)
        self.moment_per_unit_length = cas.GenMX_zeros(self.n)
        
        for load in self.distributed_loads:
            if load["type"] == "uniform":
                self.force_per_unit_length += load["force"] / self.length
            elif load["type"] == "elliptical":
                load_to_add = load["force"] / self.length * (
                        4 / cas.pi * cas.sqrt(1 - (self.x/self.length) ** 2)
                )
                self.force_per_unit_length += load_to_add
            else:
                raise ValueError("Bad value of \"type\" for a load within beam.distributed_loads!")
    
    def setup(self,
              bending_BC_type="cantilevered",
              ):
        """
        Sets up the problem. Run this last.
        :return: None (in-place)
        """
        
        # Discretize
        point_load_locations = [load["location"] for load in self.point_loads]
        point_load_locations.insert(0, 0)
        point_load_locations.append(self.length)
        self.x = cas.vertcat(*[
            cas.linspace(
                point_load_locations[i],
                point_load_locations[i + 1],
                self.points_per_point_load)
            for i in range(len(point_load_locations) - 1)
        ])

        # Post-process the discretization
        self.n = self.x.shape[0]
        self.dx = cas.diff(self.x)
        
        # Initialize optimization variables
        self._init_opt_vars(self)
        
        # Add geometry-specific constraints
        self._extra_constraints()
    
        # Add distributed loads
        self._add_loads()

        # Mass
        self.mass = self.volume * self.density
        
        if self.bending:
            # Set up derivatives
            self.u = 1 * self.opti.variable(0, n_vars = self.n)
            self.du = 0.1 * self.opti.variable(0, n_vars = self.n)
            self.ddu = 0.01 * self.opti.variable(0, n_vars = self.n)
            self.dEIddu = 1 * self.opti.variable(0, n_vars = self.n)

            # Define derivatives
            self.opti.subject_to([
                cas.diff(self.u) == np.trapz(self.du) * self.dx,
                cas.diff(self.du) == np.trapz(self.ddu) * self.dx,
                cas.diff(self.E * self.I * self.ddu) == np.trapz(self.dEIddu) * self.dx,
                cas.diff(self.dEIddu) == np.trapz(self.force_per_unit_length) * self.dx + self.point_forces,
            ])

            # Add BCs
            if bending_BC_type == "cantilevered":
                self.opti.subject_to([
                    self.u[0] == 0,
                    self.du[0] == 0,
                    self.ddu[-1] == 0,  # No tip moment
                    self.dEIddu[-1] == 0,  # No tip higher order stuff
                ])
            else:
                raise ValueError("Bad value of bending_BC_type!")

            # Stress
            self.stress_axial()
            
        
        # TODO: Add torsion
        if self.torsion:

            # Set up derivatives
            phi = 0.1 * self.opti.variable(0, n_vars = self.n)
            dphi = 0.01 * self.opti.variable(0, n_vars = self.n)

            # Add forcing term
            ddphi = -self.moment_per_unit_length / (self.G * self.J)

        self.stress = self.stress_axial
        self.opti.subject_to([
            self.stress / self.max_allowable_stress < 1,
            self.stress / self.max_allowable_stress > -1,
        ])
    
    def plot3D(self, displacement=False):
        pass
    
    def draw_bending(self,
                     show=True,
                     for_print=False,
                     equal_scale=True,
                     ):
        """
        Draws a figure that illustrates some bending properties. Must be called on a solved object (i.e. using the substitute_sol method).
        :param show: Whether or not to show the figure [boolean]
        :param for_print: Whether or not the figure should be shaped for printing in a paper [boolean]
        :param equal_scale: Whether or not to make the displacement plot have equal scale (i.e. true deformation only)
        :return:
        """
        import matplotlib.pyplot as plt
        import matplotlib.style as style
        import seaborn as sns
        sns.set(font_scale=1)

        fig, ax = plt.subplots(
            2 if not for_print else 3,
            3 if not for_print else 2,
            figsize=(
                10 if not for_print else 6,
                6 if not for_print else 6
            ),
            dpi=200
        )

        plt.subplot(231) if not for_print else plt.subplot(321)
        plt.plot(self.x, self.u, '.-')
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$u$ [m]")
        plt.title("Displacement (Bending)")
        if equal_scale:
            plt.axis("equal")

        plt.subplot(232) if not for_print else plt.subplot(322)
        plt.plot(self.x, np.arctan(self.du) * 180 / np.pi, '.-')
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"Local Slope [deg]")
        plt.title("Slope")

        plt.subplot(233) if not for_print else plt.subplot(323)
        plt.plot(self.x, self.force_per_unit_length, '.-')
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$q$ [N/m]")
        plt.title("Local Load per Unit Span")

        plt.subplot(234) if not for_print else plt.subplot(324)
        plt.plot(self.x, self.stress_axial / 1e6, '.-')
        plt.xlabel(r"$x$ [m]")
        plt.ylabel("Axial Stress [MPa]")
        plt.title("Axial Stress")

        plt.subplot(235) if not for_print else plt.subplot(325)
        plt.plot(self.x, self.dEIddu, '.-')
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$F$ [N]")
        plt.title("Shear Force")

        plt.subplot(236) if not for_print else plt.subplot(326)
        plt.plot(self.x, self.nominal_diameter, '.-')
        plt.xlabel(r"$x$ [m]")
        plt.ylabel("Diameter [m]")
        plt.title("Optimal Spar Diameter")
        plt.tight_layout()

        plt.show() if show else None
        

class RoundTube(Beam6DOF):
    
    def __init__(self, 
                 *args, 
                 init_geometry: dict = {
                     'diameter': 100,
                     'thickness': 1E-3,
                     'num_eval': 100
                     },
                 **kwargs):
        
        super().__init__(
            *args, 
            init_geometry = init_geometry,
            **kwargs)
    
    # Geometry vars
    req_geometry_vars = ['thickness', 'diameter']    
    
    @property
    def cross_section(self):
        diameter = self.diameter
        thickness = self.thickness
        
        angle = np.linspace( 0 , 2 * np.pi, 
                            self.geometry.get('num_eval', 100))   # Default to 100
         
        xy = np.array([np.cos(angle), np.sin(angle)])
        
        points = np.hstack(
            [
                xy * (diameter - thickness),
                np.flip(xy * diameter, axis=0),
                ]
            ).T
        
        #TODO: Probably quite inefficient for circles (high poly approx)
        poly = asb.Polygon(points)
        
        return poly
    
    
class RectTube(Beam6DOF):
    
    def __init__(self, 
                 *args, 
                 init_geometry: dict = {
                     'height': 100,
                     'width': 100,
                     'thickness': 1E-3,
                     },
                 **kwargs):
        
        super().__init__(
            *args, 
            init_geometry = init_geometry,
            **kwargs)
    
    # Geometry vars
    req_geometry_vars = ['thickness', 'height', 'width']    
    
    @property
    def cross_section(self):
        height = self.height
        width = self.width
        thickness = self.thickness
        
        xy = np.array([
            [0, 0], [1, 0],
            [1, 1], [0, 1],
            [0, 0]
            ])
        width_height = np.array([width, height])
        
        points = np.vstack(
            [
                xy * (width_height - 2 * thickness) +  thickness,
                np.flip(xy * width_height, axis=0),
                ]
            )
        
        poly = asb.Polygon(points)
        
        return poly
    
class RectBar(Beam6DOF):
    
    def __init__(self, 
                 *args, 
                 init_geometry: dict = {
                     'height': 100,
                     'width': 100,
                     },
                 **kwargs):
        
        super().__init__(
            *args, 
            init_geometry = init_geometry,
            **kwargs)
    
    # Geometry vars
    req_geometry_vars = ['height', 'width']    
    
    @property
    def cross_section(self):
        height = self.height
        width = self.width
        
        xy = np.array([
            [0, 0], [1, 0],
            [1, 1], [0, 1],
            [0, 0]
            ])
        
        x = xy[:, 0].reshape((1, -1)) * width.T
        y = xy[:, 1].reshape((1, -1)) * height.T
        
        points = cas.vertcat(x, y).T
        
        poly = asb.Polygon(points)
        
        return poly
    
        
if __name__ == '__main__':
    
    opti = asb.Opti()
    
    # Use default geometry guess
    beam = RectBar(
        opti=opti,
        length=60 / 2,
        points_per_point_load=50,
        bending=True,
        torsion=True
    )