from aerosandbox import *
import autograd.numpy as np
from autograd import grad, jacobian

def f(x):
    return np.sin(x)

dfdx = grad(f)
print(dfdx(0.0))

class myclass:
    def __init__(self):
        pass
    def sqrprop(self):
        self.property *= self.property

def g(x):
    mc = myclass()
    mc.property = 3*x
    mc.sqrprop()
    return mc.property

dgdx = grad(g)
print(dgdx(3.0))

def h(x):
    a=x[0]
    b=x[1]
    c=x[2]
    d=x[3]



    vec1 = np.array([a,2*b])
    vec2 = np.array([3*c,4*d])
    vec3 = np.concatenate((vec1, vec2))
    #return np.linalg.norm(vec3)
    return np.sum(vec3)

dhdx = grad(h)
print(dhdx(np.array([0.0,0.0,0.0,0.0])))
