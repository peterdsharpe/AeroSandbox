import aerosandbox as asb
import aerosandbox.numpy as np

af = asb.Airfoil("naca0012")
cache = asb._asb_root / "geometry/airfoil/test_airfoil/naca0012.json"


def make_cache():
    af.generate_polars(cache_filename=cache)


def test_load_cache():
    af.generate_polars(cache_filename=cache)


if __name__ == '__main__':
    make_cache()
    test_load_cache()
