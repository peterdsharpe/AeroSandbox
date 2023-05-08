import aerosandbox as asb
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import pytest
import aerosandbox.numpy as np
from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.UniqueWing import airplane
def test_lifting_line():
    op_point = asb.OperatingPoint(
            atmosphere=asb.Atmosphere(altitude=0),
            velocity=10,  # m/s
            alpha=np.linspace(-8,8,8)
        )
    LL_aeros = [asb.NlLiftingLine(
        airplane=airplane,
        op_point=op,
        verbose=True,
        spanwise_resolution=8,
    ).run()
            for op in op_point
    ]

    LL_aero = {}

    for k in LL_aeros[0].keys():
        LL_aero[k] = np.array([
            aero[k]
            for aero in LL_aeros
        ])

    fig, ax = plt.subplots(1, 3)

    plt.sca(ax[0])
    plt.plot(op_point.alpha, LL_aero["CL"])

    plt.sca(ax[1])
    plt.plot(op_point.alpha, LL_aero["CD"])

    plt.sca(ax[2])
    plt.plot(op_point.alpha, LL_aero["Cm"])

    vlm_aeros = [asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=op,
        verbose=True,
        spanwise_resolution=10,
        chordwise_resolution=10,
    ).run()
       for op in op_point
    ]

    vlm_aero = {}

    for k in vlm_aeros[0].keys():
        vlm_aero[k] = np.array([
            aero[k]
            for aero in vlm_aeros
        ])

    plt.sca(ax[0])
    plt.plot(op_point.alpha, vlm_aero["CL"])

    plt.sca(ax[1])
    plt.plot(op_point.alpha, vlm_aero["CD"])

    plt.sca(ax[2])
    plt.plot(op_point.alpha, vlm_aero["Cm"])

    AVL_aeros = [asb.AVL(
        airplane=airplane,
        op_point=op,
        verbose=True,
    ).run()
                 for op in op_point
                 ]

    AVL_aero = {}

    for k in AVL_aeros[0].keys():
        AVL_aero[k] = np.array([
            aero[k]
            for aero in AVL_aeros
        ])

    plt.sca(ax[0])
    plt.plot(op_point.alpha, AVL_aero["CL"])
    plt.xlabel(r"$\alpha$ [deg]")
    plt.ylabel(r"$C_L$")
    p.set_ticks(2, 0.5, 0.1, 0.01)
    plt.legend(labels=["NL LL", "VLM", "AVL"])

    plt.sca(ax[1])
    plt.plot(op_point.alpha, AVL_aero["CD"])
    plt.xlabel(r"$\alpha$ [deg]")
    plt.ylabel(r"$C_D$")
    p.set_ticks(2, 0.5, 0.005, 0.001)


    plt.sca(ax[2])
    plt.plot(op_point.alpha, AVL_aero["Cm"])
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$C_m$")
    # p.set_ticks(2, 0.5, 0.005, 0.001)
    plt.legend(labels=["NL LL", "VLM", "AVL"])
    p.show_plot(
        title="`NL LL, VLM, AVL` Aircraft",
        # savefig="NL_LL&VLM&AVL_0015_Notwist.png"
    )

if __name__ == '__main__':
    test_lifting_line()
    airplane.draw()
     # pytest.main()
