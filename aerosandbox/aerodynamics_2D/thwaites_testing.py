from aerosandbox.geometry.airfoil import *

# Inputs
x = np.linspace(0, 1, 50)
ue = np.linspace(1, 1, 50)
nu = 1.5e-5

# Thwaites
delta_FS = np.sqrt(nu * x / ue)
theta = delta_FS
duedx = np.hstack((
    0,
    np.diff(ue) / np.diff(x)
))
lambd = theta ** 2 / nu * duedx

