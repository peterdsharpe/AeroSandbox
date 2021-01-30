from aerosandbox import *

# A few different ways to make airfoils

a = Airfoil(name="naca2412")
b = Airfoil(name="e216")
c = Airfoil(name="Apogee 10", coordinates="./ag10.dat")
d = Airfoil(name="Ellipsish",
            coordinates=np.hstack([
                [0.5 + 0.5 * np.cos(theta), 0.05 * np.sin(theta) + 0.01 * np.cos(theta / 2)]
                for theta in np.linspace(0, 2*np.pi, 100)
            ]).reshape((-1, 2))
            )

# And some commands

a.draw()
b.draw()

