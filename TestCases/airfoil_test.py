from aerosandbox import *

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
x= np.linspace(0,1,1000)
c = s1223.get_camber_at_chord_fraction(x)
plt.plot(x,c)
n = s1223.get_normal_direction_at_chord_fraction(x)