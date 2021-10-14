import aerosandbox as asb
import aerosandbox.numpy as np

if __name__ == '__main__':
    af1 = asb.Airfoil("naca0012")
    af1.generate_polars(cache_filename="airfoil_data_cache.json")
    af2 = asb.Airfoil("naca0012")
    af2.generate_polars(cache_filename="airfoil_data_cache.json")
