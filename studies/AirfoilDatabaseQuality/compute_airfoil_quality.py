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
    # if len(af.coordinates) < 40:
    #     raise QualityError("Airfoil has too few coordinates.")

    # Check that airfoil is roughly normalized
    if np.any(af.x() <= -0.2):
        raise QualityError("Airfoil has abnormally low x-coordinates.")
    if np.any(af.x() >= 1.2):
        raise QualityError("Airfoil has abnormally high x-coordinates.")
    if np.any(af.y() <= -0.7):
        raise QualityError("Airfoil has abnormally low y-coordinates.")
    if np.any(af.y() >= 0.7):
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

    # if np.any(ds > 0.12):
    #     raise QualityError("Airfoil has abnormally long edge lengths.")

    angles = np.arctan2d(dy, dx)
    if np.any(np.abs(np.diff(angles, period=360)) > 150):
        raise QualityError("Airfoil has abnormally large changes in angle.")

    is_mid_section = np.logical_and(
        af.x() > 0.05,
        af.x() < 0.95
    )

    if np.any(np.abs(np.diff(angles, period=360)[is_mid_section[1:-1]]) > 30):
        # print(np.abs(np.diff(angles, period=360)]))
        raise QualityError("Airfoil has abnormally large changes in angle in mid-section.")

    # Normalize the airfoil
    af = af.normalize()

    # Check that the airfoil is representable by a Kulfan airfoil
    ka = af.to_kulfan_airfoil()

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
            print(f"Airfoil {af.name} failed quality checks: {e}")
            af.draw()

    # all_kulfan_parameters = {
    #     k: np.stack([
    #         af.kulfan_parameters[k]
    #         for af in UIUC_airfoils
    #     ], axis=0)
    #     for k in UIUC_airfoils[0].kulfan_parameters.keys()
    # }
    # for i in np.argsort(np.abs(kulfan_parameters["leading_edge_weight"]))[-10:]:
    #     UIUC_airfoils[i].draw()
    #     asb.Airfoil(UIUC_airfoils[i].name).draw()
