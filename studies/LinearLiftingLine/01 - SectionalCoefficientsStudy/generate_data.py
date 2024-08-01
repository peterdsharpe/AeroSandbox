import aerosandbox as asb
from pathlib import Path

airfoil_database_path = Path(
    asb._asb_root / "geometry" / "airfoil" / "airfoil_database"
)

afs = [
    asb.Airfoil(name=filename.stem)
    for filename in airfoil_database_path.iterdir() if filename.suffix == ".dat"
]

# if __name__ == '__main__':

    # af = asb.Airfoil("goe187")
