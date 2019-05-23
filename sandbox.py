import numpy as np
from Classes import *
import examples

a = examples.conventional()
#a.draw()

ringpanel = Panel(vertices=np.array([
    [1, 1, 1],
    [1, 1.05, 1],
    [1.05, 1.05, 1],
    [1.05, 1, 1]
]))

#ringpanel.draw()

tripanel = Panel(vertices=np.array([
    [1, 1, 1],
    [1, 1.05, 1],
    [1.05, 1.05, 1]
]))
tripanel.colocation_point=np.array([0,1,2])
tripanel.set_colocation_point_at_centroid()
tripanel.draw()

v=HorseshoeVortex(vertices=np.array([[1,1,1],[1,1.5,1]]))
print(vars(v))

print(v.calculate_unit_influence(np.array([0,0,0])))

test=AeroProblem(aircraft=a)
print(vars(test))