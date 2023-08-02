import aerosandbox as asb
import aerosandbox.numpy as np

theta = np.linspace(0, 2 * np.pi, 361)

circle_airfoil = asb.Airfoil(
    name="Circle",
    coordinates=np.stack([
        0.5 + 0.5 * np.cos(theta),
        0.5 * np.sin(theta)
    ], axis=1)
).to_kulfan_airfoil()

if __name__ == '__main__':
    circle_airfoil.draw()
