from aerosandbox.tools.airfoil_fitter.airfoil_fitter import *

if __name__ == '__main__':

    # a = Airfoil(name="HALE_03 (Thiqboi)", coordinates="C:/Projects/Github/Airfoils/HALE_03.dat")
    a = Airfoil("e216")
    # a = Airfoil("naca0008")

    try:
        with open("%s.pkl" % a.name, "rb") as f:
            af = pickle.load(f)
    except:
        a.get_xfoil_data()
        af = AirfoilFitter(a)

        with open("%s.pkl" % a.name, "wb+") as f:
            pickle.dump(af, f)

    # af.airfoil.plot_xfoil_data_contours()
    # af.airfoil.plot_xfoil_data_polars()
    # af.plot_xfoil_alpha_Re('Cl')
    # af.plot_xfoil_alpha_Re('Cd', log_z=True)
    Cl = af.fit_xfoil_data_Cl(plot_fit=True)
    Cd = af.fit_xfoil_data_Cd(plot_fit=True)

    # Get data
    y_data = af.airfoil.xfoil_data_1D['Cl'] / af.airfoil.xfoil_data_1D['Cd']

    # Get model
    n = 60
    linspace = lambda x: np.linspace(np.min(x), np.max(x), n)
    logspace = lambda x: np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), n)
    x1 = linspace(af.airfoil.xfoil_data_1D['alpha'])
    x2 = logspace(af.airfoil.xfoil_data_1D['Re'])
    X1, X2 = np.meshgrid(x1, x2)
    y_model = np.array(Cl(X1, X2) / Cd(X1, X2))

    fig = go.Figure()
    # fig.add_trace(
    #     go.Scatter3d(
    #         x=af.airfoil.xfoil_data_1D['alpha'],
    #         y=af.airfoil.xfoil_data_1D['Re'],
    #         z=af.airfoil.xfoil_data_1D['Cl']/af.airfoil.xfoil_data_1D['Cd'],
    #         mode="markers",
    #         marker=dict(
    #             size=2,
    #             color="black"
    #         )
    #     )
    # )
    # fig.add_trace(
    #     go.Surface(
    #         contours={
    #             # "x": {"show": True, "start": -20, "end": 20, "size": 1},
    #             # "y": {"show": True, "start": 1e4, "end": 1e6, "size": 1e5},
    #             # "z": {"show": True, "start": -5, "end": 5, "size": 0.1}
    #         },
    #         x=x1,
    #         y=x2,
    #         z=y_model,
    #         surfacecolor=y_model,
    #         colorscale="plasma",
    #         # flatshading=True
    #     )
    # )
    # fig.update_layout(
    #     scene=dict(
    #         xaxis=dict(
    #             title="Alpha"
    #         ),
    #         yaxis=dict(
    #             type='log',
    #             title="Re"
    #         ),
    #         zaxis=dict(
    #             type='linear',
    #             title="L/D"
    #         ),
    #     ),
    #     title="L/D"
    # )
    fig.add_trace(
        go.Mesh3d(
            x=af.airfoil.xfoil_data_1D["alpha"],
            y=af.airfoil.xfoil_data_1D["Re"],
            z=np.array(
                (Cl(af.airfoil.xfoil_data_1D["alpha"], af.airfoil.xfoil_data_1D["Re"]) / Cd(af.airfoil.xfoil_data_1D["alpha"], af.airfoil.xfoil_data_1D["Re"])) /
                (af.airfoil.xfoil_data_1D["Cl"] / af.airfoil.xfoil_data_1D["Cd"])
            ).reshape(-1),
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="Alpha"
            ),
            yaxis=dict(
                type='log',
                title="Re"
            ),
            zaxis=dict(
                type='linear',
                title="L/D",
                range=(0,2)
            ),
        ),
        title="L/D"
    )
    fig.show()

    # with open("func.pkl", "wb+") as f:
    #     pickle.dump(func, f)
    # print(
    #     func(0, 1e6)
    # )
