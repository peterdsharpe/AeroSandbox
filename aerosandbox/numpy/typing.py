"""Type aliases for AeroSandbox's numpy-like interface.

Design Philosophy
-----------------
AeroSandbox operates in two computational modes:

1. **Hybrid Mode**: Functions that work with either NumPy or CasADi,
   dispatched at runtime based on input types. Use "hybrid types" here.

2. **Concrete Mode**: Functions that require actual numeric values
   (external tools like XFoil/AVL, file I/O, plotting). Use "concrete types" here.

Type Naming Conventions
-----------------------
- Types WITHOUT prefix: Hybrid (NumPy OR CasADi)
  - `Scalar`, `Vector`, `Array`, `Vectorizable`

- Types WITH `Concrete` prefix: NumPy-only (for I/O, external tools)
  - `ConcreteScalar`, `ConcreteVector`, `ConcreteArray`

- Types WITH `Like` suffix: Permissive input types (accept sequences)
  - `ArrayLike`, `VectorLike`, `PointLike`

Type Hierarchy
--------------
Types are built progressively to ensure consistency::

    ConcreteScalar → Scalar (adds CasADi)
    ConcreteArray → Array (adds CasADi)
    ConcreteVectorizable → Vectorizable (adds CasADi)
    ConcreteArrayLike → ArrayLike (adds CasADi)

Internal Types
--------------
- `_CasADiType`: Internal only - should not be used outside aerosandbox.numpy.
  This exists to keep the library backend-agnostic at the API level.
"""

from typing import Sequence
import numpy as _onp
import casadi as _cas

# =============================================================================
# BUILDING BLOCKS (internal use)
# =============================================================================

_PythonScalar = int | float
"""Python's native scalar types."""

_NumPyScalar = _onp.integer | _onp.floating
"""NumPy's scalar types (e.g., np.int64, np.float64)."""

_SequenceOfScalars = Sequence[int | float]
"""A sequence (list, tuple) of numeric scalars."""

# =============================================================================
# CasADi types (internal use only)
# =============================================================================

_CasADiType = _cas.MX | _cas.DM
"""CasADi array types compatible with AeroSandbox's Opti class. Internal use only.

Note: cas.SX is intentionally excluded. While aerosandbox.numpy operations support SX
at runtime, the Opti class is MX-based and incompatible with SX. Type hints using
_CasADiType reflect what works in the optimization context.
"""

# =============================================================================
# CONCRETE TYPES - For external tools, I/O, plotting (NumPy only)
# These are the base types; hybrid types extend these by adding CasADi.
# =============================================================================

ConcreteScalar = _PythonScalar | _NumPyScalar
"""A concrete scalar value: Python int/float or numpy scalar. No CasADi.

Use for:
- External tool inputs (XFoil, AVL, MSES)
- File I/O
- Matplotlib plotting
- Anywhere concrete numeric values are required
"""

ConcreteVector = _onp.ndarray
"""A concrete 1D NumPy array. No CasADi.

Semantically indicates 1D data in concrete-only contexts.
"""

ConcreteArray = _onp.ndarray
"""A concrete NumPy array. No CasADi.

Use for:
- External tool inputs/outputs
- Interpolation lookup tables (points, values in interpn)
- File I/O
- Matplotlib plotting
"""

ConcreteVectorizable = ConcreteScalar | ConcreteArray
"""A concrete value that broadcasts element-wise: scalar OR array, NumPy only.

The NumPy-only equivalent of Vectorizable. Use for external tool parameters
that accept either scalars or arrays but cannot accept CasADi types.

Examples:
- MSES: alpha, Re, mach parameters
- Any external tool that broadcasts over multiple operating points
"""

ConcreteArrayLike = ConcreteScalar | Sequence["ConcreteScalar | ConcreteArrayLike"] | ConcreteArray
"""Permissive input for concrete array parameters. No CasADi.

Accepts scalars, sequences (including nested sequences), or ndarrays.
Use for function INPUTS to external tools that accept array-like values
but not CasADi.
"""

ConcreteVectorLike = ConcreteArrayLike
"""Permissive input for concrete vector parameters. No CasADi.

Semantically indicates 1D data; structurally identical to ConcreteArrayLike.
"""

ConcretePointLike = _SequenceOfScalars | ConcreteArray
"""A concrete 3D point [x, y, z] as sequence or ndarray. No CasADi.

Use for spatial point inputs in external tools or file I/O.
"""

# =============================================================================
# HYBRID TYPES - For bridge-layer functions (NumPy OR CasADi)
# These extend their Concrete counterparts by adding CasADi support.
# =============================================================================

Scalar = ConcreteScalar | _CasADiType
"""A scalar numeric value: Python int/float, numpy scalar, or CasADi scalar.

Use when a single numeric value is expected, but may be symbolic in optimization.
"""

Vector = ConcreteVector | _CasADiType
"""A 1D numeric array (vector): NumPy 1D ndarray or CasADi column vector.

Semantically indicates 1D data. Note that both NumPy and CasADi arrays
are the underlying type; this alias communicates dimensional intent.
"""

Array = ConcreteArray | _CasADiType
"""An N-dimensional numeric array: NumPy ndarray or CasADi array.

Use when an array of any dimensionality is expected.
Note: CasADi is limited to 2D arrays; NumPy can be N-dimensional.
"""

Vectorizable = ConcreteVectorizable | _CasADiType
"""A value that broadcasts element-wise: scalar OR array, NumPy OR CasADi.

Use for parameters that accept EITHER scalar OR array, where the function
will broadcast/vectorize element-wise. Examples:

- OperatingPoint: velocity, alpha, beta, p, q, r
- Atmosphere: altitude
- MassProperties: mass, x_cg, y_cg, z_cg, Ixx, etc.
- Dynamics state variables: x_e, z_e, speed, gamma, etc.

These parameters accept scalars for single-point calculations,
or arrays of the same shape for vectorized/broadcast calculations.
"""

ArrayLike = ConcreteArrayLike | _CasADiType
"""Permissive input for array parameters: anything castable to an array.

Accepts scalars, sequences, ndarrays, or CasADi arrays. Use for function
INPUTS where the value will be converted to an array internally via
``asarray()`` or similar.
"""

VectorLike = ArrayLike
"""Permissive input for vector parameters: anything castable to a 1D array.

Semantically indicates 1D data; structurally identical to ArrayLike.
"""

PointLike = ConcretePointLike | _CasADiType
"""A 3D point [x, y, z] as sequence, ndarray, or CasADi array.

Use for spatial point inputs in geometry functions.
"""
