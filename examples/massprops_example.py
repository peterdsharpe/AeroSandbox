from aerosandbox import *

a = MassComponent(
    mass = 1,
    xyz_cg=[1,0,0],
)
b = MassComponent(
    mass = 1,
    xyz_cg=[0,1,0],
)
c = MassComponent(
    mass = 1,
    xyz_cg = [-1,0,0]
)

m = MassProps(mass_components=[a,b,c])