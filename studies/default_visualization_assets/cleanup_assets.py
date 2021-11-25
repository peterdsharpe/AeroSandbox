import pyvista as pv

def plot(mesh):
    plotter = pv.Plotter()
    plotter.add_mesh(mesh)
    plotter.camera.up = (0, 0, -1)
    plotter.camera.Azimuth(90)
    plotter.camera.Elevation(60)
    plotter.show_axes()
    plotter.show_grid()
    plotter.show()

talon = pv.read("talon-original.stl")
talon.rotate_z(180)
talon.rotate_x(180)
talon.translate([0, 0, 27])
talon.points /= -talon.bounds[0]
talon.save("talon.stl")
plot(talon)

yf = pv.read("yf23-original.stl")
yf.rotate_z(-90)
yf.rotate_x(180)
yf.translate([yf.bounds[0], 0, 0])
yf.points /= -yf.bounds[0]
yf.save("yf23.stl")
plot(yf)
