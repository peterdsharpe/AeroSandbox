import aerosandbox as asb
import aerosandbox.numpy as np

af = asb.Airfoil("naca0012")

def make_cache():
    af.generate_polars(cache_filename="naca0012.json")

def test_load_cache():
    af.generate_polars(cache_filename="naca0012.json")

if __name__ == '__main__':
    make_cache()
    test_load_cache()
