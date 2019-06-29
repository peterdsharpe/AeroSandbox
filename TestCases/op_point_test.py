from aerosandbox import *

o = OperatingPoint(
    density = 1.225,
    velocity=10,
    alpha=5,
    beta=5
)

print(o.compute_rotation_matrix_wind_to_geometry())
print()
print(o.compute_freestream_velocity_geometry_axes())
