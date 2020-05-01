from aerosandbox import *

if __name__ == '__main__':
    a = Airfoil("sd7032")
    a.get_xfoil_data(
        n_Res=6,
        a_step = 1,
        parallel=True
    )
    a.plot_xfoil_data_all_polars()