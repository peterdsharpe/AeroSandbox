import casadi as cas
from typing import Union

import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.geometry import *

import plotly.express as px
import plotly.graph_objects as go

import warnings
        
# TODO: Check and document
# TODO: Add large displacements check

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
                 material:              asb.structures.Material=None,
                 E:                     float=None, #228e9,  # Pa
                 isotropic:             bool=None,  #True,
                 poisson_ratio:         float=None, #0.5,
                 max_allowable_stress:  float=None, #570e6 / 1.75,
                 density:               float=None, #1600,
                 G:                     float=None,
                 bending:               bool=True,  # Should we consider beam bending?
                 torsion:               bool=True,  # Should we consider beam torsion?
                 locked_geometry_vars:  list=[],
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
        
        # TODO: Check this behavior is the one we want
        if material:
            if E == None:
                E = material.E
            if isotropic == None:
                isotropic = material.isotropic
            if poisson_ratio == None:
                poisson_ratio = material.poisson_ratio
            if max_allowable_stress == None:
                max_allowable_stress = material.yield_stress
            if density == None:
                density = material.density
            if G == None:
                G = material.G
            
        self.E = E
        self.isotropic = isotropic
        self.poisson_ratio = poisson_ratio
        self.max_allowable_stress = max_allowable_stress
        self.density = density
        self.G = G
            
        
        self.bending = bending
        self.torsion = torsion
        self.locked_geometry_vars = locked_geometry_vars
        
        # Calculate G  # TODO: Add to materials class
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
    def Ixx(self):
        return self.cross_section.Ixx().T
    
    @property
    def Iyy(self):
        return self.cross_section.Iyy().T
    
    @property
    def Ixy(self):
        return self.cross_section.Ixy().T
    
    @property
    def J(self):
        return self.cross_section.J().T

    @property
    def volume(self):
        #raise NotImplementedError('Volume not specified for object of type ' + str(type(self)))
        # TODO: Check it's correct, it's a bit too late for math
        volume = np.sum(
            np.trapz(
                self.cross_section.area()
            ) * self.dx,
            axis=0
        )
        return volume

    @property
    def init_geometry(self):
        return self._init_geometry
    
    req_geometry_vars = []
    @init_geometry.setter
    def init_geometry(self, value):
        # Make sure all required variables are here
        for var in self.req_geometry_vars:
            assert var in value.keys()
        
        self._init_geometry = value
        
        self.n = 1  # Init with 1 cross-section
        
        # make them properties so we can mess with setting
        for key in self._init_geometry.keys():
            
            # Use function factories to not mess up the keys
            def make_getter(key):
                def getter(self):
                    return getattr(self, "_"+key)
                return getter
            
            def make_setter(key):
                def setter(self, value):
                    if len(value.shape) == 1:
                        value = np.reshape(value, (1, -1))
                    setattr(self, "_"+key, value)
                return setter
            
            prop = property(fget=make_getter(key),
                            fset=make_setter(key))
            
            # Add it to the list of properties
            setattr(self.__class__, key, prop)
            
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
        self.bending_BC_type = bending_BC_type
        
        #### Discretize
        self._discretize()
        
        #### Initialize optimization variables
        self._init_opt_vars()
        
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
        self._bending_stress_vertices = 0
        if self.bending:
            self.calc_bending_stress()
            
        #### Torsion
        self._torsional_stress_vertices_x = 0
        self._torsional_stress_vertices_y = 0
        if self.torsion:
            self.calc_torsional_stress()
            
        #### Calc Stress
        self.calc_stress()
        
        #### Constrain to allowable stress
        self.opti.subject_to([
            cas.reshape(self.stress, (-1, 1)) / self.max_allowable_stress <= 1,
            cas.reshape(self.stress, (-1, 1)) / self.max_allowable_stress >= -1,
        ])
        
    def _discretize(self):
        
        # TODO: Don't add points if load location == length
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

        #### Post-process the discretization
        self.n = self.x.shape[0]
        self.dx = np.diff(self.x)
        
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
            # If locked just stack it
            if var in self.locked_geometry_vars:
                value = np.array(np.tile(self.init_geometry[var], self.n))
                
                setattr(self, var, value)
            
            # Otherwise make it variable
            else:
                value = self.opti.variable(
                            init_guess = self.init_geometry[var],
                            n_vars = self.n
                        ).T
                
                setattr(self, var, value)
            
                # Assuming all geometry parameters are positive
                self.opti.subject_to([
                    getattr(self, var) >= 0,
                    ])
        
    def _add_loads(self):
        
        # Add point loads
        self.point_forces_x = np.zeros(self.n - 1)
        self.point_forces_y = np.zeros(self.n - 1)
        self.point_forces_z = np.zeros(self.n - 1)
        
        self.point_moments_x = np.zeros(self.n - 1)
        self.point_moments_y = np.zeros(self.n - 1)
        
        self.point_torsional_moments = np.zeros(self.n - 1)
        
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
                
            self.forces_per_unit_length_x += load['force'][0] / self.length * scaling
            self.forces_per_unit_length_y += load['force'][1] / self.length * scaling
            self.forces_per_unit_length_z += load['force'][2] / self.length * scaling
            
            self.moments_per_unit_length_x += load['bending_moment'][0] / self.length * scaling
            self.moments_per_unit_length_y += load['bending_moment'][1] / self.length * scaling
            
            self.torsional_moments_per_unit_length += load['torsional_moment'] / self.length * scaling
    
    def calc_axial_stress(self):
        """Calculates stresses from shear loads"""
        self.axial_stress = np.zeros(self.n)
        
        if self.bending_BC_type == 'cantilevered':
            # TODO: Fix hstacks to support casadi types
            per_point_forces_z = self.forces_per_unit_length_z + np.hstack([0, self.point_forces_z])
            forces_z = np.cumsum(per_point_forces_z[::-1])[::-1]
            
            self.axial_stress += forces_z / self.cross_section.area()
        
    def calc_shear_stress(self):
        """Calculates stresses from axial loads"""
        self.shear_stress_x = np.zeros(self.n)
        self.shear_stress_y = np.zeros(self.n)
        
        if self.bending_BC_type == 'cantilevered':
            per_point_forces_x = self.forces_per_unit_length_x + np.hstack([0, self.point_forces_x])
            forces_x = np.cumsum(per_point_forces_x[::-1])[::-1]
            
            per_point_forces_y = self.forces_per_unit_length_y + np.hstack([0, self.point_forces_y])
            forces_y = np.cumsum(per_point_forces_y[::-1])[::-1]
            
            self.shear_stress_x += forces_x / self.cross_section.area()  # self.forces_x / self.cross_section.area()
            self.shear_stress_y += forces_y / self.cross_section.area()  # self.forces_y / self.cross_section.area()
        
    def calc_bending_stress(self):
        """Calculates stresses from bending loads"""
        
        warnings.warn('Bending has not been checked yet.')  # TODO: Check and remove
        
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

        # TODO: Add point moments

        # Define derivatives
        self.opti.subject_to([
            cas.diff(self.u) == np.trapz(self.du) * self.dx,
            cas.diff(self.du) == np.trapz(self.ddu) * self.dx,
            cas.diff(self.E * self.Ixx.T * self.ddu) == np.trapz(self.dEIddu) * self.dx, # + self.point_moments_y,
            cas.diff(self.dEIddu) == np.trapz(self.forces_per_unit_length_x) * self.dx + self.point_forces_x,
            
            cas.diff(self.v) == np.trapz(self.dv) * self.dx,
            cas.diff(self.dv) == np.trapz(self.ddv) * self.dx,
            cas.diff(self.E * self.Iyy.T * self.ddv) == np.trapz(self.dEIddv) * self.dx, # + self.point_moments_y,
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

        # TODO: Check if there's a better way, somewhat inefficient
        # Find the stress at vertices based on radius of curvature ddx
        self._bending_stress_vertices_x = self.cross_section.x() * self.E * self.ddu
        self._bending_stress_vertices_y = self.cross_section.y() * self.E * self.ddv

        
    torsion_formula = None
    def calc_torsional_stress(self):
        """
        Calculates stresses from torsional loads
        
        These might need to be specified  per geometry
        
        http://web.mit.edu/16.20/homepage/6_Torsion/Torsion_files/module_6_with_solutions.pdf
        """  
        
        warnings.warn('Torsion has not been checked yet.')  # TODO: Check and remove
        
        # Set up derivatives
        phi = 0.1 * self.opti.variable(0, n_vars = self.n)
        dphi = 0.01 * self.opti.variable(0, n_vars = self.n)

        # Add forcing term
        ddphi = -self.torsional_moments_per_unit_length / (self.G * self.J)
        
        # TODO: Add point moments
        # Define derivatives
        self.opti.subject_to([
            cas.diff(self.phi) == np.trapz(self.dphi) * self.dx,
            cas.diff(self.dphi) == np.trapz(self.ddphi) * self.dx,
            cas.diff(self.G * self.J.T * self.ddv/self.dphi) == np.trapz(self.dGJddphi) * self.dx, # TODO: This formula is based on no real source
            cas.diff(self.dGJddphi) == np.trapz(self.torsional_moments_per_unit_length) * self.dx, # + self.point_torsional_moments,
        ])
        
        # Add BCs
        if self.bending_BC_type == "cantilevered":
            self.opti.subject_to([
                self.phi[0] == 0,
                self.dphi[0] == 0,
                self.dphi[-1] == 0,  # No tip moment
                self.dEIddphi[-1] == 0,  # No tip higher order stuff
            ])
        else:
            raise ValueError("Bad value of bending_BC_type!")

        
        # TODO: Add torsion formula
        distance = np.sqrt(
            self.cross_section.x()**2 +
            self.cross_section.y()**2
        )
            
        stress = self.G * distance * self.dphi/self.dx
        
        # TODO: cas.sqrt might not work with AD, test cas.norm2?
        self._torsional_stress_vertices_x = 0
        self._torsional_stress_vertices_y = 0
        
        #self._max_torsional_stress = np.max(self._torsional_stress_vertices)  # TODO: check
    
    def calc_stress(self):
        
        # Find max Von Mises Stress
        # TODO: Check this, shear component is wrong
        
        axial_components = (
                    # TODO: This below is not very clean, but works?
                    np.ones(self.cross_section.x().shape) * self.axial_stress.reshape((-1, 1))
                    +
                    (self._bending_stress_vertices_x if self.bending else 0)
                    +
                    (self._bending_stress_vertices_y if self.bending else 0)
                )
        
        shear_components = (
                    np.ones(self.cross_section.x().shape) * self.shear_stress_x.reshape((-1, 1))
                    + 
                    np.ones(self.cross_section.x().shape) * self.shear_stress_y.reshape((-1, 1))
                    +
                    (self._torsional_stress_vertices if self.torsion else 0)
                )
        
        self.stress = \
                np.sqrt(
                    axial_components**2
                    +
                    3 * shear_components**2
                )
        
        return self.stress
    
    def plot3D(self, 
               displacement: bool=True,
               axes_equal: bool=True, 
               show: bool=True, 
               fig: go.Figure=None
               ) -> go.Figure:
        
        if not fig:
            fig = go.Figure()
        
        #### Add scatterplot of vertices
        x = self.cross_section.x()
        y = self.cross_section.y()
        
        assert x.shape == y.shape and self.x.shape[0] == x.shape[0]
        
        x = x - np.ones(x.shape) * self.cross_section.centroid()[0].reshape(-1, 1)
        y = y - np.ones(x.shape) * self.cross_section.centroid()[1].reshape(-1, 1)
        z = np.ones(x.shape) * self.x.reshape(-1, 1)
        
        X = x.flatten()
        Y = y.flatten()
        Z = z.flatten()
        c = self.stress.flatten()
        
        scatterplot_nodisplaceement = \
            go.Scatter3d(
                x = X,
                y = Y,
                z = Z,
                name = 'Beam_Ghost',
                mode = 'lines',
                marker = dict(color='rgba(100,100,100,0.1)'),
                line = dict(color='rgba(100,100,100,0.1)', width=2),
                showlegend=False,
                )
        
        if displacement:
            if self.bending:
                x = x + self.u.reshape(-1, 1)
                y = y + self.v.reshape(-1, 1)
            else:
                warnings.warn('Displacement is 0 if bending is disabled')
        
        X = x.flatten()
        Y = y.flatten()
        Z = z.flatten()
        c = self.stress.flatten()
        
        scatterplot = \
            go.Scatter3d(
                x = X,
                y = Y,
                z = Z,
                name = 'Beam',
                mode = 'lines',
                marker = dict(color=c, opacity=0.5),
                line = dict(color=c, width=2),
                showlegend=True,
                )
        
        #### Add distributed load vectors
        qx = self.forces_per_unit_length_x
        qy = self.forces_per_unit_length_y
        qz = self.forces_per_unit_length_z
        
        distributed_forces = \
            go.Cone(
                x=np.zeros(self.x.shape),
                y=np.zeros(self.x.shape),
                z=self.x,
                u=qx,
                v=qy,
                w=qz,
                colorscale='Reds',
                sizemode="absolute",
                sizeref=100,
                anchor="tip",
                showlegend=False,
                )
        
        #### Add point load vectors
        
        x = []
        y = []
        z = []
        loc = []
        
        for i in range(len(self.point_loads)):
            x.append(self.point_loads[i]['force'][0])
            y.append(self.point_loads[i]['force'][1])
            z.append(self.point_loads[i]['force'][2])
            
            loc.append(self.point_loads[i]['location'])
        
        point_forces = \
            go.Cone(
                x=np.zeros(len(loc)),
                y=np.zeros(len(loc)),
                z=loc,
                u=x,
                v=y,
                w=z,
                colorscale='Blues',
                sizemode="absolute",
                sizeref=3,
                anchor="tip",
                showlegend=False,
                )
        
        #### Show
        if axes_equal:
            # Create cubic bounding box to simulate equal aspect ratio
            max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
            Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
            Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
            Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())

            x = []
            y = []
            z = []
            for xb, yb, zb in zip(Xb, Yb, Zb):
                x.append(xb)
                y.append(yb)
                z.append(zb)
            fig.add_trace(
                go.Scatter3d(
                    x = x,
                    y = y,
                    z = z,
                    name = '',
                    mode = 'markers',
                    marker = dict(size=0, color='rgba(0, 0, 0, 0)'),
                    showlegend=False,
                )
            )
               
        
        # TODO, fix colorbar, add quads/tri meshes
        fig.add_trace(point_forces)
        fig.add_trace(distributed_forces)
        fig.add_trace(scatterplot_nodisplaceement)
        fig.add_trace(scatterplot)
        
        fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1),
                                     camera_eye=dict(x=1.2, y=1.2, z=0.6)))
        
        if show:
            fig.show()
            
        return fig
        
    def draw_geometry_vars(self,
                           show: bool=True,
                           ):
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(font_scale=1)
        
        # TODO: Tile these properly or add to plotly interface
        for var in self.req_geometry_vars:
            fig, ax = plt.subplots()
            
            plt.subplot(111)
            plt.plot(self.x, getattr(self, var).flatten(), '.-')
            plt.xlabel(r"${}$ [m]".format('x'))
            plt.ylabel(r"${}$ [m]".format(var))
            plt.title("{} vs x".format(var.capitalize()))

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
        x2 = np.flip(np.vstack([xy[:, 0]]*self.n), axis=0) * diameter
        y2 = np.flip(np.vstack([xy[:, 1]]*self.n), axis=1) * diameter
        
        x = cas.horzcat(x1, x2)
        y = cas.horzcat(y1, y2)
        
        #TODO: Probably quite inefficient for circles (high poly approx)
        poly = asb.Polygon(x, y)
        
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
        
        poly = asb.Polygon(x, y)
        
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
        
        x = np.vstack([xy[:, 0]]*self.n) * width.T
        y = np.vstack([xy[:, 1]]*self.n) * height.T
        
        poly = asb.Polygon(x, y)
        
        return poly
    
  
#### Main      
if __name__ == '__main__':
    
    opti = asb.Opti()
    material = asb.structures.materials.AISI_1006_Steel_Cold_Drawn()
    
    # Use default geometry guess
    beam = RectBar(
        opti=opti,
        init_geometry = {
            'height': 1,
            'width': 1,
            },
        material = material,
        length=60 / 2,
        points_per_point_load=50,
        bending=True,
        torsion=False
    )
    #beam.cross_section.plot()
    
    # Solve beam
    lift_force = 9.81 * 103.873 * 100
    load_location = 15
    
    beam.add_point_load(load_location, np.array([-lift_force / 3, 0, 0]))
    beam.add_distributed_load(force= np.array([0, lift_force / 2, 0]), load_type='uniform')
    beam.setup()

    # Tip deflection constraint
    opti.subject_to([
        # beam.u[-1] < 2,  # Source: http://web.mit.edu/drela/Public/web/hpa/hpa_structure.pdf
        # beam.u[-1] > -2  # Source: http://web.mit.edu/drela/Public/web/hpa/hpa_structure.pdf
        (beam.du * 180 / np.pi) < 10,
        (beam.du * 180 / np.pi) > -10,
        (beam.dv * 180 / np.pi) < 10,
        (beam.dv * 180 / np.pi) > -10,
    ])
    
    # Some sensible boundaries to avoid crazy beams
    opti.subject_to([
        beam.height < 1000,
        beam.width < 1000,
        beam.height > 0.01,
        beam.width > 0.01,
        ])
    
    # Add some profile change constraints
    opti.subject_to([
        cas.diff(cas.diff(beam.height)) < 0.001,
        cas.diff(cas.diff(beam.height)) > -0.001,
        
        cas.diff(cas.diff(beam.width)) < 0.001,
        cas.diff(cas.diff(beam.width)) > -0.001,
        
        # cas.diff(beam.height) < 0.1,
        # cas.diff(beam.height) > -0.1,
        
        # cas.diff(beam.width) < 0.1,
        # cas.diff(beam.width) > -0.1,
    ])

    opti.minimize(beam.mass)

    p_opts = {}
    s_opts = {}
    s_opts["max_iter"] = 1e6  # If you need to interrupt, just use ctrl+c
    opti.solver('ipopt', p_opts, s_opts)

    try:
        sol = opti.solve()
        beam.substitute_solution(sol)
        beam.plot3D(axes_equal=True)
        beam.draw_geometry_vars()
    except:
        print("Failed!")
        sol = opti.debug
        print(sol)
        raise