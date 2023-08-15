import aerosandbox as asb
import aerosandbox.numpy as np
import svgpathtools as svg

paths, _ = svg.svg2paths("drela_opt6_90_dof.svg")
path: svg.path.Path = paths[0]
points = [path[0].start] + [s.end for s in path]
points = np.array(points)

coordinates = np.stack((
    np.real(points),
    np.imag(points),
), axis=1)

af = asb.Airfoil(
    name="Drela Opt6 90DoF",
    coordinates=coordinates,
).normalize()

af.write_dat("drela_opt6_90_dof.dat")