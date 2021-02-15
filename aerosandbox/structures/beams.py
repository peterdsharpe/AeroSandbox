import casadi as cas
from typing import Union

import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.geometry import *
        
# TODO: Check and document
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
        
        # Calculate G
        if isotropic:
            if G is None:
                self.G = E / 2 / (1 + poisson_ratio)
            elif poisson_ratio is None:
                pass  # TODO find poisson?
            else:
                raise ValueError(
                    "You can't uniquely specify shear modulus and Poisson's ratio on an isotropic material!")

        # Create data structures to track loads
        self.point_loads = []
        self.distributed_loads = []
        
    
    #### Properties

    @property
    def I(self):
        """
        :return: np.ndarray
        """
        # TODO: Fix this to matrix form if possible
        return self.cross_section.Ixx().T
    
    @property
    def J(self):
        raise self.cross_section.J()

    @property
    def volume(self):
        #raise NotImplementedError('Volume not specified for object of type ' + str(type(self)))
        # TODO: Check it's correct, it's a bit too late for math
        volume = np.sum(
            np.trapz(
                self.cross_section.area().T
            ) * self.dx
        )
        return volume

    @property
    def init_geometry(self):
        return self._init_geometry
    
    req_geometry_vars = []
    @init_geometry.setter
    def init_geometry(self, value):
        for var in self.req_geometry_vars:
            assert var in value.keys()
        
        self._init_geometry = value
        
        self.n = 1  # Init with 1 cross-section
        for key in value.keys():
            setattr(self, key, np.array(value[key]))
            
    @property
    def cross_section(self):
        return NotImplementedError('Geometry not specified for object of type ' + str(type(self)))

        
    #### Methods
        
    def add_point_load(self,
                       location: float,
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
        force = np.array(force)
        bending_moment = np.array(bending_moment)
        
        assert type(location) in [int, float, cas.MX]
        assert force.shape[0] == 3
        assert bending_moment.shape[0] == 2
        assert type(torsional_moment) in [int, float, cas.MX]
        
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
                         load_type: str = 'distributed'
                         ):
        """
        Adds a uniformly distributed force and/or moment across the entire length of the beam.
        :param force: Total force applied to beam [N]
        :param bending_moment: Bending moment to add [N-m] # TODO make this work
        :param type: distributed/elliptical
        :return: None (in-place)
        """
        
        force = np.array(force)
        bending_moment = np.array(bending_moment)
        
        assert force.shape[0] == 3
        assert bending_moment.shape[0] == 2
        assert type(torsional_moment) in [int, float, cas.MX]
        
        self.distributed_loads.append(
            {
                "type"            : load_type,
                "force"           : force,
                "bending_moment"  : bending_moment,
                'torsional_moment': torsional_moment,
            }
        )
        
    def setup(self,
              bending_BC_type="cantilevered",
              ):
        """
        Sets up the problem. Run this last.
        :return: None (in-place)
        """
        self.bending_BC_type = "cantilevered"
        
        #### Discretize
        self._discretize()

        #### Post-process the discretization
        self.n = self.x.shape[0]
        self.dx = np.diff(self.x)
        
        #### Initialize optimization variables
        self._init_opt_vars()
        
        #### Initialize Stresses
        self.shear_stress_x = np.zeros(self.n)
        self.shear_stress_y = np.zeros(self.n)
        
        self.axial_stress = np.zeros(self.n)
        
        self.bending_stress_x = np.zeros(self.n)
        self.bending_stress_y = np.zeros(self.n)
        
        self.torsional_stress = np.zeros(self.n)
        
        #### Add geometry-specific constraints
        self._extra_constraints()
    
        #### Add distributed loads
        self._add_loads()

        #### Mass
        self.mass = self.volume * self.density
        
        #### Forces
        self.calc_shear_stress()
        self.calc_axial_stress()
        
        #### Bending
        if self.bending:
            self.calc_bending_stress()
            
        #### Torsion
        if self.torsion:
            self.calc_torsional_stress()
            
        #### Calc Max Stress
        self.stress = self.calc_max_stress()
        
        self.opti.subject_to([
            self.stress / self.max_allowable_stress < 1,
            self.stress / self.max_allowable_stress > -1,
        ])
        
    def _discretize(self):
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
        for var in self.req_geometry_vars:
            setattr(self, var, 
                    self.opti.variable(
                        init_guess = self.init_geometry[var],
                        n_vars = self.n
                    )
                )
        
    def _add_loads(self):
        
        # Add point loads
        self.point_forces_x = cas.GenMX_zeros(self.n - 1)
        self.point_forces_y = cas.GenMX_zeros(self.n - 1)
        self.point_forces_z = cas.GenMX_zeros(self.n - 1)
        
        self.point_moments_x = cas.GenMX_zeros(self.n - 1)
        self.point_moments_y = cas.GenMX_zeros(self.n - 1)
        
        self.point_torsional_moments = cas.GenMX_zeros(self.n - 1)
        
        for i in range(len(self.point_loads)):
            load = self.point_loads[i]
            idx = self.points_per_point_load * (i + 1) - 1  # discretized point index
            
            self.point_forces_x[idx] = load["force"][0]
            self.point_forces_y[idx] = load["force"][1]
            self.point_forces_z[idx] = load["force"][2]
            
            self.point_moments_x[idx] = load["bending_moment"][0]
            self.point_moments_y[idx] = load["bending_moment"][1]
            
            self.point_torsional_moments[idx] = load["torsional_moment"]
        
        # Add distributed loads
        self.forces_per_unit_length_x = np.zeros(self.n)
        self.forces_per_unit_length_y = np.zeros(self.n)
        self.forces_per_unit_length_z = np.zeros(self.n)
        
        self.moments_per_unit_length_x = np.zeros(self.n)
        self.moments_per_unit_length_y = np.zeros(self.n)
        
        self.torsional_moments_per_unit_length = np.zeros(self.n)
        
        for load in self.distributed_loads:
            if load["type"] == "uniform":
                scaling = np.array([1])
                
            elif load["type"] == "elliptical":
                scaling = 4 / cas.pi * cas.sqrt(1 - (self.x/self.length) ** 2)
                
            else:
                raise ValueError("Bad value of \"type\" for a load within beam.distributed_loads!")
                
            # TODO: Check if this is correct
            self.forces_per_unit_length_x += load['force'][0] / self.length * scaling
            self.forces_per_unit_length_y += load['force'][1] / self.length * scaling
            self.forces_per_unit_length_z += load['force'][2] / self.length * scaling
            
            self.moments_per_unit_length_x += load['bending_moment'][0] / self.length * scaling
            self.moments_per_unit_length_y += load['bending_moment'][1] / self.length * scaling
            
            self.torsional_moments_per_unit_length += load['torsional_moment'] / self.length * scaling
        
    
    def calc_axial_stress(self):
        """Calculates stresses from shear loads"""
        self.axial_stress += 0  # self.forces_z / self.cross_section.area()
        
    def calc_shear_stress(self):
        """Calculates stresses from axial loads"""
        self.shear_stress_x += 0  # self.forces_x / self.cross_section.area()
        self.shear_stress_y += 0  # self.forces_y / self.cross_section.area()
        
    def calc_bending_stress(self):
        """Calculates stresses from bending loads"""
        
        # Set up derivatives
        # TODO: What are these factors before variables?
        self.u = 1 * self.opti.variable(0, n_vars = self.n)
        self.du = 0.1 * self.opti.variable(0, n_vars = self.n)
        self.ddu = 0.01 * self.opti.variable(0, n_vars = self.n)
        self.dEIddu = 1 * self.opti.variable(0, n_vars = self.n)

        self.v = 1 * self.opti.variable(0, n_vars = self.n)
        self.dv = 0.1 * self.opti.variable(0, n_vars = self.n)
        self.ddv = 0.01 * self.opti.variable(0, n_vars = self.n)
        self.dEIddv = 1 * self.opti.variable(0, n_vars = self.n)

        # Define derivatives
        self.opti.subject_to([
            cas.diff(self.u) == np.trapz(self.du) * self.dx,
            cas.diff(self.du) == np.trapz(self.ddu) * self.dx,
            cas.diff(self.E * self.I * self.ddu) == np.trapz(self.dEIddu) * self.dx,
            cas.diff(self.dEIddu) == np.trapz(self.forces_per_unit_length_x) * self.dx + self.point_forces_x,
            
            cas.diff(self.v) == np.trapz(self.dv) * self.dx,
            cas.diff(self.dv) == np.trapz(self.ddv) * self.dx,
            cas.diff(self.E * self.I * self.ddv) == np.trapz(self.dEIddv) * self.dx,
            cas.diff(self.dEIddv) == np.trapz(self.forces_per_unit_length_y) * self.dx + self.point_forces_y,
        ])

        # Add BCs
        if self.bending_BC_type == "cantilevered":
            self.opti.subject_to([
                self.u[0] == 0,
                self.du[0] == 0,
                self.ddu[-1] == 0,  # No tip moment
                self.dEIddu[-1] == 0,  # No tip higher order stuff
                
                self.v[0] == 0,
                self.dv[0] == 0,
                self.ddv[-1] == 0,  # No tip moment
                self.dEIddv[-1] == 0,  # No tip higher order stuff
            ])
        else:
            raise ValueError("Bad value of bending_BC_type!")

        # Stress functions for x and y based on radius of curvature ddx
        stress_f_x = (lambda x: x * self.E * self.ddu)
        stress_f_y = (lambda y: y * self.E * self.ddv)
        
        # Find the stress at vertices
        self._bending_stress_vertices = np.sqrt(
                stress_f_x(self.cross_section.x())**2 + 
                stress_f_y(self.cross_section.y())**2
            )

        self._min_max_bending_stress = (
            np.min(stress_vertices),  # Compressive
            np.max(stress_vertices),  # Tensile
            )  # TODO: Check
        
    def calc_torsional_stress(self):
        """Calculates stresses from torsional loads"""    
        
        # Set up derivatives
        phi = 0.1 * self.opti.variable(0, n_vars = self.n)
        dphi = 0.01 * self.opti.variable(0, n_vars = self.n)

        # Add forcing term
        ddphi = -self.torsional_moments_per_unit_length / (self.G * self.J)
        
        
        # Define derivatives
        self.opti.subject_to([
            cas.diff(self.u) == np.trapz(self.du) * self.dx,
            cas.diff(self.du) == np.trapz(self.ddu) * self.dx,
            cas.diff(self.dEIddu) == np.trapz(self.force_per_unit_length_x) * self.dx + self.point_forces[0],
        ])
        
        stress_f = (lambda dist: dist)
        
        self._torsional_stress_vertices = stress_f(
                np.sqrt(
                    (self.cross_section.x())**2 + 
                    (self.cross_section.y())**2
                )
            )
        
        self._max_torsional_stress = np.max(self._torsional_stress_vertices)  # TODO: check
    
    def calc_max_stress(self):
        
        # Find max Von Mises Stress
        # TODO: Check this
        self.max_stress = np.max(
            np.sqrt(
                (
                    self.axial_stress
                    +
                    self._bending_stress_vertices
                )**2 
                +
                3 * (
                    self.shear_stress_x**2 
                    + 
                    self.shear_stress_y**2  
                    +
                    3 * self._torsional_stress_vertices**2
                )
                )
            )
        
        return self.max_stress
    
    def plot3D(self, displacement=False):
        sections = self.cross_section.coordinates
    
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
                     'thickness': 5,
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
        
        # TODO: Convert to Casadi
        angle = np.linspace( 0 , 2 * np.pi, 
                            self.init_geometry.get('num_eval', 100))   # Default to 100
         
        xy = np.array([np.cos(angle), np.sin(angle)]).T
        
        x1 = np.vstack([xy[:, 0]]*self.n) * (diameter - thickness)
        y1 = np.vstack([xy[:, 1]]*self.n) * (diameter - thickness)
        x2 = np.flip(np.vstack([xy[:, 0]]*self.n), axis=1) * diameter
        y2 = np.flip(np.vstack([xy[:, 1]]*self.n), axis=1) * diameter
        
        x = cas.horzcat(x1, x2)
        y = cas.horzcat(y1, y2)
        
        #TODO: Probably quite inefficient for circles (high poly approx)
        poly = asb.Polygon(x.T, y.T)
        
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
        
        xy = np.flip(np.array([
            [0, 0], [1, 0],
            [1, 1], [0, 1],
            [0, 0]
            ]), axis=0)
        
        x1 = np.vstack([xy[:, 0]]*self.n) * (width.T - 2 * thickness) +  thickness
        y1 = np.vstack([xy[:, 1]]*self.n) * (height.T - 2 * thickness) +  thickness
        x2 = np.vstack([xy[:, 0]]*self.n) * width.T
        y2 = np.vstack([xy[:, 1]]*self.n) * height.T
        
        x = cas.horzcat(x1, x2)
        y = cas.horzcat(y1, y2)
        
        poly = asb.Polygon(x.T, y.T)
        
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
        
        xy = np.flip(np.array([
            [0, 0], [1, 0],
            [1, 1], [0, 1],
            [0, 0]
            ]), axis=0)
        
        x = np.vstack([xy[:, 0]]*self.n) * width
        y = np.vstack([xy[:, 1]]*self.n) * height
        
        poly = asb.Polygon(x.T, y.T)
        
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
    beam.cross_section.plot()
    
    tube = RoundTube(
        opti=opti,
        length=60 / 2,
        points_per_point_load=50,
        bending=True,
        torsion=True
    )
    tube.cross_section.plot()
    
    # Solve beam
    lift_force = 9.81 * 103.873
    load_location = opti.variable(15)
    opti.subject_to([
        load_location > 2,
        load_location < 60 / 2 - 2,
        load_location == 18,
    ])
    beam.add_point_load(load_location, np.array([-lift_force / 3, 0, 1]))
    beam.add_distributed_load(force= np.array([lift_force / 2, 0, 0]), load_type='uniform')
    beam.setup()

    # Tip deflection constraint
    opti.subject_to([
        # beam.u[-1] < 2,  # Source: http://web.mit.edu/drela/Public/web/hpa/hpa_structure.pdf
        # beam.u[-1] > -2  # Source: http://web.mit.edu/drela/Public/web/hpa/hpa_structure.pdf
        beam.du * 180 / cas.pi < 10,
        beam.du * 180 / cas.pi > -10
    ])
    opti.subject_to([
        cas.diff(cas.diff(beam.nominal_diameter)) < 0.001,
        cas.diff(cas.diff(beam.nominal_diameter)) > -0.001,
    ])

    opti.minimize(beam.mass)

    p_opts = {}
    s_opts = {}
    s_opts["max_iter"] = 1e6  # If you need to interrupt, just use ctrl+c
    opti.solver('ipopt', p_opts, s_opts)

    try:
        sol = opti.solve()
    except:
        print("Failed!")
        sol = opti.debug
        
    sol = beam.substitute_solution(sol)