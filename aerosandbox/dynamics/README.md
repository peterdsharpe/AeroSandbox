# AeroSandbox Dynamics

This module provides dynamics engines (e.g., tools for computing equations of motions) for a variety of free-flying dynamical systems.

Each dynamics engine is given within the context of a Python class. These dynamics engines can be broadly categorized into three groups, which increasing fidelity:

* Point mass
* 3 DoF (i.e., 2D)
* 6 DoF (i.e., 3D). These models can use either Euler angle parameterization or quaternion parameterization for the underlying state variables.

