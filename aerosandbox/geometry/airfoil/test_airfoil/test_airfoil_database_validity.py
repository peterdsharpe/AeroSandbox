import aerosandbox as asb
import aerosandbox.numpy as np
import pytest
from typing import List


def get_airfoil_database() -> List[asb.Airfoil]:
    airfoil_database_root = asb._asb_root / "geometry" / "airfoil" / "airfoil_database"

    afs = [
        asb.Airfoil(
            name=airfoil_file.stem,
        )
        for airfoil_file in airfoil_database_root.glob("*.dat")
    ]

    return afs


def check_validity(af: asb.Airfoil):
    if af.n_points() < 4:
        raise ValueError(f"Airfoil {af.name} has too few points (n_points = {af.n_points()})!")

    if af.area() < 0:
        raise ValueError(f"Airfoil {af.name} has negative area (area = {af.area()})!")

    if af.area() == 0:
        raise ValueError(f"Airfoil {af.name} has zero area!")

    if af.area() > 0.6:
        raise UserWarning(f"Airfoil {af.name} has unusually large area (area = {af.area()})!")

    if af.x().max() > 1.1:
        raise UserWarning(f"Airfoil {af.name} has unusually high x_max (x_max = {af.x().max()})!")

    if af.x().max() < 0.9:
        raise UserWarning(f"Airfoil {af.name} has unusually low x_max (x_max = {af.x().max()})!")

    if af.x().min() < -0.1:
        raise UserWarning(f"Airfoil {af.name} has unusually low x_min (x_min = {af.x().min()})!")

    if af.x().min() > 0.1:
        raise UserWarning(f"Airfoil {af.name} has unusually high x_min (x_min = {af.x().min()})!")

    if af.y().max() > 0.5:
        raise UserWarning(f"Airfoil {af.name} has unusually high y_max (y_max = {af.y().max()})!")

    if af.y().min() < -0.5:
        raise UserWarning(f"Airfoil {af.name} has unusually low y_min (y_min = {af.y().min()})!")

    ds = np.linalg.norm(np.diff(af.coordinates, axis=0), axis=1)

    if ds.max() > 0.8:
        raise UserWarning(f"Airfoil {af.name} has unusually large ds_max (ds_max = {ds.max()})!")

    x_thicknesses = np.linspace(af.x().min(), af.x().max(), 501)
    thicknesses = af.local_thickness(x_over_c=x_thicknesses)
    if np.any(thicknesses < 0):
        raise ValueError(f"Airfoil {af.name} has negative thickness @ x = {x_thicknesses[np.argmin(thicknesses)]}!")


def test_airfoil_database_validity():
    afs = get_airfoil_database()

    failed_airfoils_and_errors = {}

    for af in afs:
        try:
            check_validity(af)
        except UserWarning as e:  # If a UserWarning is raised, print it and continue.
            print(e)
        except ValueError as e:
            failed_airfoils_and_errors[af.name] = e

    if len(failed_airfoils_and_errors) > 0:
        raise ValueError(
            f"The following airfoils failed the validity test:\n"
            "\n".join(f"{af_name}: {error}" for af_name, error in failed_airfoils_and_errors.items())
        )


if __name__ == '__main__':

    afs = get_airfoil_database()
    for af in afs:
        try:
            check_validity(af)
        except (ValueError, UserWarning) as e:
            print(e)
            af.draw()

    pytest.main()
