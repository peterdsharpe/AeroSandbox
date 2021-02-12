import aerosandbox.numpy as np


class MassProps:
    def __init__(self,
                 mass_components=[]  # list of MassComponent objects
                 ):
        self.mass_components = mass_components

    def get_mass(self):
        """ Mass of the aircraft according the mass of the components
        
        Args:
            self.mass_components (iterable): Iterable of components each with a mass property in kg
        Returns:
            float: Total mass of the aircraft in kg
        """
        total_mass = 0.
        for component in self.mass_components:
            total_mass = total_mass + component.mass
        return total_mass

    def get_cg(self):
        total_mass = 0.
        total_mass_times_position = 0.
        for component in self.mass_components:
            total_mass = total_mass + component.mass
            total_mass_times_position = total_mass_times_position + component.mass * component.xyz_cg
        return total_mass_times_position / total_mass

    def get_inertia_tensor(self):
        total_inertia = np.zeros((3, 3))
        cg = self.get_cg()
        for component in self.mass_components:
            total_inertia = total_inertia + component.get_inertia_tensor_about_point(cg)
        return total_inertia


class MassComponent:
    def __init__(self,
                 name=None,  # Totally optional, not used for anything.
                 mass=0,  # Mass of the component.
                 xyz_cg=np.array([0, 0, 0]),  # Location of the component's CG. Axes are in geometry axes.
                 Ixx=0,  # About the component's center of mass. 0 if this is a point mass.
                 Iyy=0,  # About the component's center of mass. 0 if this is a point mass.
                 Izz=0,  # About the component's center of mass. 0 if this is a point mass.
                 Iyz=0,  # About the component's center of mass. 0 if this is symmetric about x.
                 Ixz=0,  # About the component's center of mass. 0 if this is symmetric about y.
                 Ixy=0,  # About the component's center of mass. 0 if this is symmetric about z.
                 ):
        self.name = name
        self.mass = mass
        self.xyz_cg = xyz_cg
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.Ixy = Ixy
        self.Ixz = Ixz
        self.Iyz = Iyz

    def get_inertia_tensor(self):
        # Returns the inertia tensor about the component's centroid.
        return np.array(
            [[self.Ixx, self.Ixy, self.Ixz],
             [self.Ixy, self.Iyy, self.Iyz],
             [self.Ixz, self.Iyz, self.Izz]]
        )

    def get_inertia_tensor_about_point(self, point):
        # Returns the inertia tensor about an arbitrary point.
        # Using https://en.wikipedia.org/wiki/Parallel_axis_theorem#Tensor_generalization

        R = point - self.xyz_cg
        I = self.get_inertia_tensor()
        m = self.mass
        J = I + m * (np.dot(R, R) * np.eye(3) - np.outer(R, R))

        return J
