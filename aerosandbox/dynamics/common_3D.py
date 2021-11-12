# import aerosandbox.numpy as np
# from aerosandbox.dynamics.common import _DynamicsBaseClass
# from abc import ABC, abstractmethod, abstractproperty
# from typing import Union, Tuple, Dict, List
#
# class _DynamicsBaseClass3D(_DynamicsBaseClass):
#     @abstractmethod
#     def convert_axes(self,
#                      x_from: float,
#                      y_from: float,
#                      z_from: float,
#                      from_axes: str,
#                      to_axes: str,
#                      ) -> Tuple[float, float, float]:
#         """
#         Converts a vector [x_from, y_from, z_from], as given in the `from_axes` frame, to an equivalent vector [x_to,
#         y_to, z_to], as given in the `to_axes` frame.
#
#         Identical to OperatingPoint.convert_axes(), but adds in "earth" as a valid axis frame. For more documentation,
#         see the docstring of OperatingPoint.convert_axes().
#
#         Both `from_axes` and `to_axes` should be a string, one of:
#                 * "geometry"
#                 * "body"
#                 * "wind"
#                 * "stability"
#                 * "earth"
#
#         Args:
#                 x_from: x-component of the vector, in `from_axes` frame.
#                 y_from: y-component of the vector, in `from_axes` frame.
#                 z_from: z-component of the vector, in `from_axes` frame.
#                 from_axes: The axes to convert from.
#                 to_axes: The axes to convert to.
#
#         Returns: The x-, y-, and z-components of the vector, in `to_axes` frame. Given as a tuple.
#
#         """
#         pass
#
#     @abstractmethod
#     def add_force(self,
#                   Fx: Union[np.ndarray, float] = 0,
#                   Fy: Union[np.ndarray, float] = 0,
#                   Fz: Union[np.ndarray, float] = 0,
#                   axes="body",
#                   ) -> None:
#         pass