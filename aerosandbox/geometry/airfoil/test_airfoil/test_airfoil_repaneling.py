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

def test_repaneling_validity():
    try:
        import shapely
    except ModuleNotFoundError:
        pytest.skip("Shapely not installed; skipping this test.")

    afs = get_airfoil_database()

    for af in afs:
        try:
            similarity = af.jaccard_similarity(
                af.repanel()
            )
        except shapely.errors.GEOSException:
            similarity = np.nan
        assert similarity > 1 - 3 / af.n_points(), f"Airfoil {af.name} failed repaneling validity check with similarity {similarity}!"

def debug_draw(af: asb.Airfoil):
    if isinstance(af, str):
        af = asb.Airfoil(af)

    for af in [af, af.repanel()]:
        af.draw(
            draw_mcl=False, backend='plotly', show=False
        ).update_layout(
            yaxis=dict(scaleanchor=None)
        ).show()




if __name__ == '__main__':
    test_repaneling_validity()