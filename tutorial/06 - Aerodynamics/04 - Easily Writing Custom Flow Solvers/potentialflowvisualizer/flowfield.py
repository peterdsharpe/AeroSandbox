import aerosandbox.numpy as np
from potentialflowvisualizer.objects import Singularity
from typing import List


class Flowfield:
    def __init__(self,
                 objects: List[Singularity] = None
                 ):
        if objects is None:
            objects = []

        self.objects = objects

    def get_potential_at(self, points: np.ndarray):
        return sum([object.get_potential_at(points) for object in self.objects])

    def get_streamfunction_at(self, points: np.ndarray):
        return sum([object.get_streamfunction_at(points) for object in self.objects])

    def get_x_velocity_at(self, points: np.ndarray):
        return sum([object.get_x_velocity_at(points) for object in self.objects])

    def get_y_velocity_at(self, points: np.ndarray):
        return sum([object.get_y_velocity_at(points) for object in self.objects])

    def draw(self,
             scalar_to_plot: str = "potential",  # "potential", "streamfunction", "xvel", "yvel", "velmag", "Cp"
             x_points: np.ndarray = np.linspace(-10, 10, 400),
             y_points: np.ndarray = np.linspace(-10, 10, 300),
             percentiles_to_include: float = 99.7,
             set_equal: bool = True,
             show: bool = True,
             ):
        import aerosandbox.tools.pretty_plots as p

        X, Y = np.meshgrid(x_points, y_points)
        X_r = np.reshape(X, -1)
        Y_r = np.reshape(Y, -1)
        points = np.vstack((X_r, Y_r)).T

        if scalar_to_plot == "potential":
            scalar_to_plot_value = sum([object.get_potential_at(points) for object in self.objects])
        elif scalar_to_plot == "streamfunction":
            scalar_to_plot_value = sum([object.get_streamfunction_at(points) for object in self.objects])
        elif scalar_to_plot == "xvel":
            scalar_to_plot_value = sum([object.get_x_velocity_at(points) for object in self.objects])
        elif scalar_to_plot == "yvel":
            scalar_to_plot_value = sum([object.get_y_velocity_at(points) for object in self.objects])
        elif scalar_to_plot == "velmag":
            x_vels = sum([object.get_x_velocity_at(points) for object in self.objects])
            y_vels = sum([object.get_y_velocity_at(points) for object in self.objects])
            scalar_to_plot_value = np.sqrt(x_vels ** 2 + y_vels ** 2)
        elif scalar_to_plot == "Cp":
            x_vels = sum([object.get_x_velocity_at(points) for object in self.objects])
            y_vels = sum([object.get_y_velocity_at(points) for object in self.objects])
            V = np.sqrt(x_vels ** 2 + y_vels ** 2)
            scalar_to_plot_value = 1 - V ** 2
        else:
            raise ValueError("Bad value of `scalar_to_plot`!")

        min = np.nanpercentile(scalar_to_plot_value, 50 - percentiles_to_include / 2)
        max = np.nanpercentile(scalar_to_plot_value, 50 + percentiles_to_include / 2)

        p.contour(
            x_points, y_points, scalar_to_plot_value.reshape(X.shape),
            levels=np.linspace(min, max, 80),
            linelabels=False,
            cmap=p.mpl.colormaps.get_cmap("rainbow"),
            contour_kwargs={
                "linestyles": 'solid',
                "alpha"     : 0.4
            }
        )

        if set_equal:
            p.equal()

        p.show_plot(
            f"Potential Flow: {scalar_to_plot}",
            "$x$",
            "$y$",
            show=show
        )

    def draw_streamlines(self,
                         x_points: np.ndarray = np.linspace(-10, 10, 400),
                         y_points: np.ndarray = np.linspace(-10, 10, 300),
                         cmap=None,
                         norm=None,
                         set_equal: bool = True,
                         show: bool = True,
                         ):

        X, Y = np.meshgrid(x_points, y_points)
        U = np.reshape(
            self.get_x_velocity_at(np.stack([
                X.flatten(),
                Y.flatten()
            ], axis=1)),
            X.shape
        )
        V = np.reshape(
            self.get_y_velocity_at(np.stack([
                X.flatten(),
                Y.flatten()
            ], axis=1)),
            X.shape
        )

        velmag = np.sqrt(U ** 2 + V ** 2)

        import matplotlib.pyplot as plt
        import aerosandbox.tools.pretty_plots as p

        if cmap is None:
            cmap = "coolwarm_r"

        if norm is None:
            from matplotlib.colors import LogNorm
            norm = LogNorm(
                np.quantile(velmag, 0.05),
                np.quantile(velmag, 0.95)
            )

        plt.streamplot(
            X, Y, U, V,
            color=velmag,
            linewidth=1,
            minlength=0.02,
            cmap=cmap,
            norm=norm,
            broken_streamlines=False
        )

        plt.colorbar(label="Velocity Magnitude [m/s]")

        if set_equal:
            p.equal()

        p.show_plot(
            "Potential Flow: Streamlines",
            "$x$",
            "$y$",
            show=show
        )
