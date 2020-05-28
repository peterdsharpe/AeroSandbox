import matplotlib.pyplot as plt
import matplotlib.style as style
from aerosandbox import *

style.use("seaborn")

if __name__ == '__main__':

    afs = ["e216", "dae11", "e63", "naca4410", "naca662415"]

    for af_name in afs:
        af = Airfoil(af_name, repanel=False)
        xfoil_data = af.xfoil_aseq(
            0,
            15,
            0.25,
            repanel=True,
            Re=200e3,
            max_iter=50,
        )

        #     plt.plot(alphas, cls/cds, ".-", label=af_name)
        # plt.xlabel("Angle of Attack [deg]")
        # plt.ylabel("CL/CD")
        # plt.title("CL/CD Polars of Candidate Airfoils @ Re = 100k")
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        plt.plot(
            remove_nans(xfoil_data["Cd"]),
            remove_nans(xfoil_data["Cl"]),
            "-", label=af_name
        )
    plt.xlabel("CD")
    plt.xlim(0, 0.05)
    plt.ylim(0, 2)
    plt.ylabel("CL")
    plt.title("CL vs. CD of Candidate Airfoils @ Re = 200k")
    plt.legend()
    plt.tight_layout()
    plt.savefig("C:/Users/User/Downloads/airfoil_choice.png", dpi=600)
    plt.show()
