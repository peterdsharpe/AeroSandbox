"""AeroSandbox NumPy-like interface with CasADi backend support.

This module provides a NumPy-compatible API that automatically dispatches to
either NumPy or CasADi backends at runtime based on input types. This allows
the same code to work for both numerical evaluation and symbolic optimization.

Key Features
------------
- All standard NumPy functions work with both NumPy arrays and CasADi symbols
- Type aliases for annotating hybrid (NumPy/CasADi) and concrete (NumPy-only) code
- Smooth optimization utilities (softmax, softmin, etc.)
- Numerical differentiation and integration

Example
-------
>>> import aerosandbox.numpy as np
>>> x = np.array([1, 2, 3])  # NumPy array
>>> np.sum(x)  # Returns NumPy scalar
6
>>> # With CasADi optimization:
>>> opti = asb.Opti()
>>> x = opti.variable(init_guess=1)
>>> np.sin(x)  # Returns CasADi symbolic expression
"""
### Import everything from NumPy

from numpy import *

### Export type aliases for hybrid (NumPy/CasADi) and concrete (NumPy-only) types
from aerosandbox.numpy.typing import (
    Scalar,
    Vector,
    Array,
    Vectorizable,
    ConcreteScalar,
    ConcreteVector,
    ConcreteArray,
    ConcreteVectorizable,
    VectorLike,
    ArrayLike,
    PointLike,
    ConcreteArrayLike,
)

### Export type detection utilities
from aerosandbox.numpy.determine_type import is_casadi_type, is_iterable

### Overwrite some functions
from aerosandbox.numpy.array import *
from aerosandbox.numpy.arithmetic_monadic import *
from aerosandbox.numpy.arithmetic_dyadic import *
from aerosandbox.numpy.calculus import *
from aerosandbox.numpy.conditionals import *
from aerosandbox.numpy.finite_difference_operators import *
from aerosandbox.numpy.integrate import *
from aerosandbox.numpy.interpolate import *
from aerosandbox.numpy.linalg_top_level import *
import aerosandbox.numpy.linalg as linalg
from aerosandbox.numpy.logicals import *
from aerosandbox.numpy.rotations import *
from aerosandbox.numpy.spacing import *
from aerosandbox.numpy.surrogate_model_tools import *
from aerosandbox.numpy.trig import *

### Force-overwrite built-in Python functions.

from numpy import round  # TODO check that min, max are properly imported
