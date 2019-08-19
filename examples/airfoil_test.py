from aerosandbox import *
from autograd import grad

# Test on some of my favorite airfoils
naca0012 = Airfoil("NACA0012")
naca4410 = Airfoil("naca4410")
s1223 = Airfoil("s1223")
clarky = Airfoil("clarky")
goe803h = Airfoil("goe803h")
e423 = Airfoil("e423")
ag12 = Airfoil("ag12")
mh80 = Airfoil("mh80")
e625 = Airfoil("e625")

# And some that are not my favorite, but I suppose we need to stress-test this...
diamond = Airfoil(name = "Diamond", coordinates = np.array([
    [1, 0],
    [0.5, 0.15],
    [0, 0],
    [0.5, -0.15],
    [1, 0]
]))

# Some operations
s1223.draw()

s1223_flapped = s1223.add_control_surface(deflection = 20)
s1223_flapped.draw()

# Reverse-mode AD test
def flapped_Iyy(deflection):
    e625_flapped = e625.add_control_surface(deflection=deflection)
    return e625_flapped.Iyy()

grad_flapped_Iyy = grad(flapped_Iyy)
dIyyddeflection = grad_flapped_Iyy(0.)

print(naca0012.get_sharp_TE_airfoil().coordinates)