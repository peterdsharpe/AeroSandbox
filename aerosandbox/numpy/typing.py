"""
Type aliases for AeroSandbox's numpy-like interface.

## Design Philosophy

AeroSandbox operates in two computational modes:

1. **Hybrid Mode**: Functions that work with either NumPy or CasADi,
   dispatched at runtime based on input types. Use "hybrid types" here.

2. **Concrete Mode**: Functions that require actual numeric values
   (external tools like XFoil/AVL, file I/O, plotting). Use "concrete types" here.

## Type Naming Conventions

- Types WITHOUT prefix: Hybrid (NumPy OR CasADi)
  - `Scalar`, `Vector`, `Array`, `Vectorizable`

- Types WITH `Concrete` prefix: NumPy-only (for I/O, external tools)
  - `ConcreteScalar`, `ConcreteVector`, `ConcreteArray`

- Types WITH `Like` suffix: Permissive input types (accept sequences)
  - `ArrayLike`, `VectorLike`, `PointLike`

## Internal Types

- `CasADiType`: Internal only - should not be used outside aerosandbox.numpy.
  This exists to keep the library backend-agnostic at the API level.
"""

from typing import Sequence
import numpy as _onp
import casadi as _cas

# =============================================================================
# Core CasADi types (internal use only)
# =============================================================================
CasADiType = _cas.MX | _cas.DM | _cas.SX
"""Any CasADi array type (MX, DM, or SX). Internal use only."""

# =============================================================================
# HYBRID TYPES - For bridge-layer functions (NumPy OR CasADi)
# =============================================================================

Scalar = int | float | _onp.integer | _onp.floating | CasADiType
"""A scalar numeric value: Python int/float, numpy scalar, or CasADi scalar.

Use when a single numeric value is expected, but may be symbolic in optimization.
"""

Vector = _onp.ndarray | CasADiType
"""A 1D numeric array (vector): NumPy 1D ndarray or CasADi column vector.

Semantically indicates 1D data. Note that both NumPy and CasADi arrays
are the underlying type; this alias communicates dimensional intent.
"""

Array = _onp.ndarray | CasADiType
"""An N-dimensional numeric array: NumPy ndarray or CasADi array.

Use when an array of any dimensionality is expected.
Note: CasADi is limited to 2D arrays; NumPy can be N-dimensional.
"""

Vectorizable = int | float | _onp.integer | _onp.floating | _onp.ndarray | CasADiType
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

# =============================================================================
# CONCRETE TYPES - For external tools, I/O, plotting (NumPy only)
# =============================================================================

ConcreteScalar = int | float | _onp.integer | _onp.floating
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

ConcreteVectorizable = int | float | _onp.integer | _onp.floating | _onp.ndarray
"""A concrete value that broadcasts element-wise: scalar OR array, NumPy only.

The NumPy-only equivalent of Vectorizable. Use for external tool parameters
that accept either scalars or arrays but cannot accept CasADi types.

Examples:
- MSES: alpha, Re, mach parameters
- Any external tool that broadcasts over multiple operating points
"""

# =============================================================================
# INPUT TYPES - Permissive types for function parameters (hybrid)
# =============================================================================

VectorLike = int | float | _onp.integer | _onp.floating | Sequence[int | float] | _onp.ndarray | CasADiType
"""Permissive input for vector parameters: scalar, sequence, ndarray, or CasADi.

Use for function INPUTS that will be converted to Vector internally.
Includes scalars so that `Scalar | Array` return values can be passed directly.
"""

ArrayLike = int | float | _onp.integer | _onp.floating | Sequence[int | float] | _onp.ndarray | CasADiType
"""Permissive input for array parameters: scalar, sequence, ndarray, or CasADi.

Use for function INPUTS that will be converted to Array internally.
Includes scalars so that `Scalar | Array` return values can be passed directly.
"""

PointLike = Sequence[float] | _onp.ndarray | CasADiType
"""A 3D point [x, y, z] as sequence, ndarray, or CasADi array.

Use for spatial point inputs in geometry functions.
"""

# =============================================================================
# INPUT TYPES - Permissive types for function parameters (concrete only)
# =============================================================================

ConcreteArrayLike = Sequence[float] | Sequence[int] | _onp.ndarray
"""Permissive input for concrete array parameters. No CasADi.

Use for function INPUTS to external tools that accept sequences but not CasADi.
"""
