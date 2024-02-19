import aerosandbox as asb
import aerosandbox.numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

airfoil_database_path = asb._asb_root / "geometry" / "airfoil" / "airfoil_database"
current_airfoil_database_names = airfoil_database_path.glob("*.dat")


class QualityError(Exception):
    pass


def compute_airfoil_quality(af: asb.Airfoil):
    if af.name in current_airfoil_database_names:
        raise QualityError("Airfoil already exists in the database.")

    # Check that the airfoil has coordinates
    if af.coordinates is None:
        raise QualityError("Airfoil has no coordinates (None).")
    if len(af.coordinates) == 0:
        raise QualityError("Airfoil has no coordinates (0).")

    # Check that the airfoil has enough coordinates
    if len(af.coordinates) <= 25:
        raise QualityError(f"Airfoil has too few coordinates ({len(af.coordinates)}).")

    # Check that airfoil is roughly normalized
    if np.any(af.x() <= -0.05):
        raise QualityError("Airfoil has abnormally low x-coordinates.")
    if np.any(af.x() >= 1.05):
        raise QualityError("Airfoil has abnormally high x-coordinates.")
    if np.any(af.y() <= -0.5):
        raise QualityError("Airfoil has abnormally low y-coordinates.")
    if np.any(af.y() >= 0.5):
        raise QualityError("Airfoil has abnormally high y-coordinates.")

    # Check if the airfoil is self-intersecting
    if not af.as_shapely_polygon().is_valid:
        raise QualityError("Airfoil is self-intersecting.")

    # Check that no edge lengths are longer than 0.2
    dx = np.diff(af.x())
    dy = np.diff(af.y())
    ds = (dx ** 2 + dy ** 2) ** 0.5

    if np.any(ds == 0):
        i = np.argwhere(ds == 0)[0][0]
        raise QualityError(f"Doubled point at ({af.x()[i]}, {af.y()[i]}).")

    if np.any(ds > 0.18):
        raise QualityError(f"Airfoil has abnormally long edge lengths ({np.max(ds)}).")

    d_angles = np.concatenate([ # Angle changes between adjacent edges at each point
        [0],
        np.abs(np.diff(np.arctan2d(dy, dx), period=360)),
        [0],
    ], axis=0
    )

    allowable_d_angle = np.where(
        af.x() < 0.05,
        160,  # At the leading edge
        np.where(
            af.x() < 0.98,
            20,  # In the middle
            45  # At the trailing edge
        )
    )

    if np.any(d_angles > allowable_d_angle):
        i = np.argmax(d_angles - allowable_d_angle)
        raise QualityError(f"Airfoil has abnormally large changes in angle at ("
                           f"{af.x()[i]:.6g}, "
                           f"{af.y()[i]:.6g}"
                           f"), {d_angles[i]:.3g} deg.")

    # Normalize the airfoil
    af = af.normalize()

    # Check that the airfoil is representable by a Kulfan airfoil
    ka: asb.KulfanAirfoil = af.to_kulfan_airfoil()

    indices_upper = np.logical_and(
        af.upper_coordinates()[:, 0] >= 0,
        af.upper_coordinates()[:, 0] <= 1
    )
    y_deviance_upper = af.upper_coordinates()[indices_upper, 1] - ka.upper_coordinates(
        af.upper_coordinates()[indices_upper, 0])[:, 1]
    indices_lower = np.logical_and(
        af.lower_coordinates()[:, 0] >= 0,
        af.lower_coordinates()[:, 0] <= 1
    )
    y_deviance_lower = af.lower_coordinates()[indices_lower, 1] - ka.lower_coordinates(
        af.lower_coordinates()[indices_lower, 0])[:, 1]

    if np.max(np.abs(y_deviance_upper)) > 0.01:
        i = np.argmax(np.abs(y_deviance_upper))
        raise QualityError(
            f"Airfoil is not representable by a Kulfan airfoil (upper deviance of {y_deviance_upper[i]:.3g} at x={af.upper_coordinates()[i, 0]:.3g}).")

    if np.max(np.abs(y_deviance_lower)) > 0.01:
        i = np.argmax(np.abs(y_deviance_lower))
        raise QualityError(
            f"Airfoil is not representable by a Kulfan airfoil (lower deviance of {y_deviance_lower[i]:.3g} at x={af.lower_coordinates()[i, 0]:.3g}).")

    # if not ka.as_shapely_polygon().is_valid:
    #     raise QualityError("Kulfan airfoil is self-intersecting.")
    #
    # if af.jaccard_similarity(ka.to_airfoil()) < 0.97:
    #     raise QualityError("Airfoil is not representable by a Kulfan airfoil.")


if __name__ == '__main__':
    airfoil_database_path = asb._asb_root / "geometry" / "airfoil" / "airfoil_database"
    airfoil_database = [
        asb.Airfoil(
            name=filename.stem,
            coordinates=filename,
        )
        for filename in airfoil_database_path.glob("*.dat")
    ]

    for af in airfoil_database:
        try:
            compute_airfoil_quality(af)
        except QualityError as e:
            print(f"Airfoil {af.name.ljust(20)} failed quality checks: {e}")
            af.draw()

    airfoil_database_kulfan = [
        af.to_kulfan_airfoil()
        for af in airfoil_database
    ]

    all_kulfan_parameters = {
        k: np.stack([
            af.kulfan_parameters[k]
            for af in airfoil_database_kulfan
        ], axis=0)
        for k in airfoil_database_kulfan[0].kulfan_parameters.keys()
    }
    # for i in np.argsort(np.abs(all_kulfan_parameters["TE_thickness"]))[-10:]:
    #     fig, ax = plt.subplots()
    #     plt.plot(
    #         airfoil_database[i].normalize().x(),
    #         airfoil_database[i].normalize().y(),
    #         ".k", markersize=5, zorder=4,
    #     )
    #     plt.plot(
    #         airfoil_database_kulfan[i].x(),
    #         airfoil_database_kulfan[i].y(),
    #         "-r", linewidth=2,
    #     )
    #     plt.title("Airfoil: " + airfoil_database[i].name)
    #     plt.show()
