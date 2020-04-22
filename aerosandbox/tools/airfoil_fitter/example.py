from aerosandbox.tools.airfoil_fitter.airfoil_fitter import *

if __name__ == '__main__':

    a = Airfoil(name="HALE_03 (Thiqboi)", coordinates="C:/Projects/Github/Airfoils/HALE_03.dat")
    # a = Airfoil("naca0012")

    try:
        with open("%s.pkl" % a.name, "rb") as f:
            af = pickle.load(f)
    except:
        a.get_xfoil_data()
        af = AirfoilFitter(a)

        with open("%s.pkl" % a.name, "wb+") as f:
            pickle.dump(af, f)


    af.airfoil.plot_xfoil_data_contours()
    af.airfoil.plot_xfoil_data_polars()
    # af.plot_xfoil_alpha_Re('Cl')
    # af.plot_xfoil_alpha_Re('Cd', log_z=True)
    # func = af.fit_xfoil_data_Cl(plot_fit=True)
    # func = af.fit_xfoil_data_Cd(plot_fit=True)

    # with open("func.pkl", "wb+") as f:
    #     pickle.dump(func, f)
    # print(
    #     func(0, 1e6)
    # )
