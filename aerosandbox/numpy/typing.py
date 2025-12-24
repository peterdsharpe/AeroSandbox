"""
Type aliases for AeroSandbox's numpy-like interface.

These types account for the fact that AeroSandbox functions can accept both
standard numeric types (int, float, np.ndarray) and CasADi symbolic types
(MX, DM, SX) for use in optimization.

This module is the ONLY place outside of aerosandbox.numpy internals where
CasADi should be directly imported for typing purposes.

## Type Semantics:

- **Scalar**: A single numeric value (int, float, or 0-d CasADi type)
- **Vector**: A 1D array (ndarray or CasADi type with vector shape)
- **Array**: Any-dimensional array (ndarray or CasADi type)
- **Vectorizable**: For parameters that accept EITHER scalar OR array,
  where the function will broadcast/vectorize element-wise.
  Example: OperatingPoint.velocity can be a single value or an array
  of values that broadcast with other parameters.
"""

from typing import Sequence
import numpy as _onp
import casadi as _cas

# =============================================================================
# Core CasADi types
# =============================================================================
CasADiType = _cas.MX | _cas.DM | _cas.SX
"""Any CasADi array type (MX, DM, or SX)."""

# =============================================================================
# Scalar types - for truly scalar values
# =============================================================================
Scalar = int | float | _onp.floating | CasADiType
"""A scalar numeric value: int, float, numpy scalar, or CasADi scalar."""

# =============================================================================
# Array types - for array values
# =============================================================================
Vector = _onp.ndarray | CasADiType
"""A 1D array: NumPy ndarray or CasADi array. Semantically a vector."""

Array = _onp.ndarray | CasADiType
"""An N-dimensional array: NumPy ndarray or CasADi array."""

# =============================================================================
# Vectorizable types - for element-wise broadcasting parameters
# =============================================================================
# Use these for parameters that can be EITHER scalar OR array,
# where the function broadcasts/vectorizes element-wise.
Vectorizable = int | float | _onp.floating | _onp.ndarray | CasADiType
"""
A value that can be scalar or array for element-wise vectorization.

Use for parameters like:
- OperatingPoint: velocity, alpha, beta, p, q, r
- Atmosphere: altitude  
- MassProperties: mass, x_cg, y_cg, z_cg, Ixx, etc.
- Dynamics state variables: x_e, z_e, speed, gamma, etc.

These parameters accept scalars for single-point calculations,
or arrays of the same shape for vectorized/broadcast calculations.
"""

# =============================================================================
# Input types for list/sequence parameters
# =============================================================================
PointLike = Sequence[float] | _onp.ndarray | CasADiType
"""A 3D point: [x, y, z] as sequence, ndarray, or CasADi array."""

VectorLike = Sequence[float] | Sequence[int] | _onp.ndarray | CasADiType
"""Input for vector parameters: sequence, ndarray, or CasADi array."""

ArrayLike = Sequence[float] | Sequence[int] | _onp.ndarray | CasADiType
"""Input for array parameters: sequence, ndarray, or CasADi array."""
