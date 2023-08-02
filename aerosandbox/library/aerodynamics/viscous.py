import aerosandbox.numpy as np


def Cd_cylinder(
        Re_D: float,
        mach: float = 0.,
        include_mach_effects=True,
        subcritical_only=False
) -> float:
    """
    Returns the drag coefficient of a cylinder in crossflow as a function of its Reynolds number and Mach.

    Args:
        Re_D: Reynolds number, referenced to diameter
        mach: Mach number
        include_mach_effects: If this is set False, it assumes Mach = 0, which simplifies the computation.
        subcritical_only: Determines whether the model models purely subcritical (Re < 300k) cylinder flows. Useful, since
    this model is now convex and can be more well-behaved.

    Returns:

    # TODO rework this function to use tanh blending, which will mitigate overflows

    """

    ##### Do the viscous part of the computation
    csigc = 5.5766722118597247
    csigh = 23.7460859935990563
    csub0 = -0.6989492360435040
    csub1 = 1.0465189382830078
    csub2 = 0.7044228755898569
    csub3 = 0.0846501115443938
    csup0 = -0.0823564417206403
    csupc = 6.8020230357616764
    csuph = 9.9999999999999787
    csupscl = -0.4570690347113859

    x = np.log10(np.abs(Re_D) + 1e-16)

    if subcritical_only:
        Cd_mach_0 = 10 ** (csub0 * x + csub1) + csub2 + csub3 * x
    else:
        log10_Cd = (
                (np.log10(10 ** (csub0 * x + csub1) + csub2 + csub3 * x))
                * (1 - 1 / (1 + np.exp(-csigh * (x - csigc))))
                + (csup0 + csupscl / csuph * np.log(np.exp(csuph * (csupc - x)) + 1))
                * (1 / (1 + np.exp(-csigh * (x - csigc))))
        )
        Cd_mach_0 = 10 ** log10_Cd

    ##### Do the compressible part of the computation
    if include_mach_effects:
        m = mach
        p = {'a_sub'    : 0.03458900259594298,
             'a_sup'    : -0.7129528087049688,
             'cd_sub'   : 1.163206940186374,
             'cd_sup'   : 1.2899213533122527,
             's_sub'    : 3.436601777569716,
             's_sup'    : -1.37123096976983,
             'trans'    : 1.022819211244295,
             'trans_str': 19.017600596069848}

        Cd_over_Cd_mach_0 = np.blend(
            p["trans_str"] * (m - p["trans"]),
            p["cd_sup"] + np.exp(p["a_sup"] + p["s_sup"] * (m - p["trans"])),
            p["cd_sub"] + np.exp(p["a_sub"] + p["s_sub"] * (m - p["trans"]))
        ) / 1.1940010047391572

        Cd = Cd_mach_0 * Cd_over_Cd_mach_0

    else:
        Cd = Cd_mach_0

    return Cd


def Cf_flat_plate(
        Re_L: float,
        method="hybrid-sharpe-convex"
) -> float:
    """
    Returns the mean skin friction coefficient over a flat plate.

    Don't forget to double it (two sides) if you want a drag coefficient.

    Args:

        Re_L: Reynolds number, normalized to the length of the flat plate.

        method: The method of computing the skin friction coefficient. One of:

            * "blasius": Uses the Blasius solution. Citing Cengel and Cimbala, "Fluid Mechanics: Fundamentals and
            Applications", Table 10-4.

                Valid approximately for Re_L <= 5e5.

            * "turbulent": Uses turbulent correlations for smooth plates. Citing Cengel and Cimbala,
            "Fluid Mechanics: Fundamentals and Applications", Table 10-4.

                Valid approximately for 5e5 <= Re_L <= 1e7.

            * "hybrid-cengel": Uses turbulent correlations for smooth plates, but accounts for a
            non-negligible laminar run at the beginning of the plate. Citing Cengel and Cimbala, "Fluid Mechanics:
            Fundamentals and Applications", Table 10-4. Returns: Mean skin friction coefficient over a flat plate.

                Valid approximately for 5e5 <= Re_L <= 1e7.

            * "hybrid-schlichting": Schlichting's model, that roughly accounts for a non-negligtible laminar run.
            Citing "Boundary Layer Theory" 7th Ed., pg. 644

            * "hybrid-sharpe-convex": A hybrid model that blends the Blasius and Schlichting models. Convex in
            log-log space; however, it may overlook some truly nonconvex behavior near transitional Reynolds numbers.

            * "hybrid-sharpe-nonconvex": A hybrid model that blends the Blasius and Cengel models. Nonconvex in
            log-log-space; however, it may capture some truly nonconvex behavior near transitional Reynolds numbers.

    Returns:

        C_f: The skin friction coefficient, normalized to the length of the plate.

    You can view all of these functions graphically using
    `aerosandbox.library.aerodynamics.test_aerodynamics.test_Cf_flat_plate.py`

    """
    Re_L = np.abs(Re_L)

    if method == "blasius":
        return 1.328 / Re_L ** 0.5
    elif method == "turbulent":
        return 0.074 / Re_L ** (1 / 5)
    elif method == "hybrid-cengel":
        return 0.074 / Re_L ** (1 / 5) - 1742 / Re_L
    elif method == "hybrid-schlichting":
        return 0.02666 * Re_L ** -0.139
    elif method == "hybrid-sharpe-convex":
        return np.softmax(
            Cf_flat_plate(Re_L, method="blasius"),
            Cf_flat_plate(Re_L, method="hybrid-schlichting"),
            hardness=1e3
        )
    elif method == "hybrid-sharpe-nonconvex":
        return np.softmax(
            Cf_flat_plate(Re_L, method="blasius"),
            Cf_flat_plate(Re_L, method="hybrid-cengel"),
            hardness=1e3
        )


def Cl_flat_plate(alpha, Re_c=None):
    """
    Returns the approximate lift coefficient of a flat plate, following thin airfoil theory.
    :param alpha: Angle of attack [deg]
    :param Re_c: Reynolds number, normalized to the length of the flat plate.
    :return: Approximate lift coefficient.
    """
    if Re_c is not None:
        from warnings import warn
        warn("`Re_c` input will be deprecated in a future version.")

    alpha_rad = alpha * np.pi / 180
    return 2 * np.pi * alpha_rad


def Cd_flat_plate_normal():
    """
    Returns the drag coefficient of a flat plat oriented normal to the flow (i.e., alpha = 90 deg).

    Uses results from Tian, Xinliang, Muk Chen Ong, Jianmin Yang, and Dag Myrhaug. “Large-Eddy Simulation of the Flow
    Normal to a Flat Plate Including Corner Effects at a High Reynolds Number.” Journal of Fluids and Structures 49 (
    August 2014): 149–69. https://doi.org/10.1016/j.jfluidstructs.2014.04.008.

    Note: Cd for this case is effectively invariant of Re.

    Returns: Drag coefficient

    """
    return 2.202


import warnings


def Cl_2412(alpha, Re_c):
    # A curve fit I did to a NACA 2412 airfoil, 2D XFoil data
    # Within -2 < alpha < 12 and 10^5 < Re_c < 10^7, has R^2 = 0.9892

    warnings.warn(
        "This function is deprecated. Use `asb.Airfoil.get_aero_from_neuralfoil()` instead.",
        DeprecationWarning,
    )
    return 0.2568 + 0.1206 * alpha - 0.002018 * alpha ** 2


def Cd_profile_2412(alpha, Re_c):
    # A curve fit I did to a NACA 2412 airfoil in incompressible flow.
    # Within -2 < alpha < 12 and 10^5 < Re_c < 10^7, has R^2 = 0.9713

    warnings.warn(
        "This function is deprecated. Use `asb.Airfoil.get_aero_from_neuralfoil()` instead.",
        DeprecationWarning,
    )
    Re_c = np.maximum(Re_c, 1)
    log_Re = np.log(Re_c)

    CD0 = -5.249
    Re0 = 15.61
    Re1 = 15.31
    alpha0 = 1.049
    alpha1 = -4.715
    cx = 0.009528
    cxy = -0.00588
    cy = 0.04838

    log_CD = CD0 + cx * (alpha - alpha0) ** 2 + cy * (log_Re - Re0) ** 2 + cxy * (alpha - alpha1) * (
            log_Re - Re1)  # basically, a rotated paraboloid in logspace
    CD = np.exp(log_CD)

    return CD


def Cl_e216(alpha, Re_c):
    # A curve fit I did to a Eppler 216 (e216) airfoil, 2D XFoil data. Incompressible flow.
    # Within -2 < alpha < 12 and 10^4 < Re_c < 10^6, has R^2 = 0.9994
    # Likely valid from -6 < alpha < 12 and 10^4 < Re_c < Inf.
    # See: C:\Projects\GitHub\firefly_aerodynamics\Gists and Ideas\XFoil Drag Fitting\e216

    warnings.warn(
        "This function is deprecated. Use `asb.Airfoil.get_aero_from_neuralfoil()` instead.",
        DeprecationWarning,
    )
    Re_c = np.fmax(Re_c, 1)
    log10_Re = np.log10(Re_c)

    # Coeffs
    a1l = 3.0904412662858878e-02
    a1t = 9.6452654383488254e-02
    a4t = -2.5633334023068302e-05
    asl = 6.4175433185427011e-01
    atr = 3.6775107602844948e-01
    c0l = -2.5909363461176749e-01
    c0t = 8.3824440586718862e-01
    ctr = 1.1431810545735890e+02
    ksl = 5.3416670116733611e-01
    rtr = 3.9713338634462829e+01
    rtr2 = -3.3634858542657771e+00
    xsl = -1.2220899840236835e-01

    a = alpha
    r = log10_Re

    Cl = (c0t + a1t * a + a4t * a ** 4) * 1 / (1 + np.exp(ctr - rtr * r - atr * a - rtr2 * r ** 2)) + (
            c0l + a1l * a + asl / (1 + np.exp(-ksl * (a - xsl)))) * (
                 1 - 1 / (1 + np.exp(ctr - rtr * r - atr * a - rtr2 * r ** 2)))

    return Cl


def Cd_profile_e216(alpha, Re_c):
    # A curve fit I did to a Eppler 216 (e216) airfoil, 2D XFoil data. Incompressible flow.
    # Within -2 < alpha < 12 and 10^4 < Re_c < 10^6, has R^2 = 0.9995
    # Likely valid from -6 < alpha < 12 and 10^4 < Re_c < 10^6.
    # see: C:\Projects\GitHub\firefly_aerodynamics\Gists and Ideas\XFoil Drag Fitting\e216

    warnings.warn(
        "This function is deprecated. Use `asb.Airfoil.get_aero_from_neuralfoil()` instead.",
        DeprecationWarning,
    )
    Re_c = np.fmax(Re_c, 1)
    log10_Re = np.log10(Re_c)

    # Coeffs
    a1l = 4.7167470806940448e-02
    a1t = 7.5663005080888857e-02
    a2l = 8.7552076545610764e-04
    a4t = 1.1220763679805319e-05
    atr = 4.2456038382581129e-01
    c0l = -1.4099657419753771e+00
    c0t = -2.3855286371940609e+00
    ctr = 9.1474872611212135e+01
    rtr = 3.0218483612170434e+01
    rtr2 = -2.4515094313899279e+00

    a = alpha
    r = log10_Re

    log10_Cd = (c0t + a1t * a + a4t * a ** 4) * 1 / (1 + np.exp(ctr - rtr * r - atr * a - rtr2 * r ** 2)) + (
            c0l + a1l * a + a2l * a ** 2) * (1 - 1 / (1 + np.exp(ctr - rtr * r - atr * a - rtr2 * r ** 2)))

    Cd = 10 ** log10_Cd

    return Cd


def Cd_wave_e216(Cl, mach, sweep=0.):
    r"""
    A curve fit I did to Eppler 216 airfoil data.
    Within -0.4 < CL < 0.75 and 0 < mach < ~0.9, has R^2 = 0.9982.
    See: C:\Projects\GitHub\firefly_aerodynamics\MSES Interface\analysis\e216
    :param Cl: Lift coefficient
    :param mach: Mach number
    :param sweep: Sweep angle, in deg
    :return: Wave drag coefficient.
    """

    warnings.warn(
        "This function is deprecated. Use `asb.Airfoil.get_aero_from_neuralfoil()` instead.",
        DeprecationWarning,
    )
    mach = np.fmax(mach, 0)
    mach_perpendicular = mach * np.cosd(sweep)  # Relation from FVA Eq. 8.176
    Cl_perpendicular = Cl / np.cosd(sweep) ** 2  # Relation from FVA Eq. 8.177

    # Coeffs
    c0 = 7.2685945744797997e-01
    c1 = -1.5483144040727698e-01
    c3 = 2.1305118052118968e-01
    c4 = 7.8812272501525316e-01
    c5 = 3.3888938102072169e-03
    l0 = 1.5298928303149546e+00
    l1 = 5.2389999717540392e-01

    m = mach_perpendicular
    l = Cl_perpendicular

    Cd_wave = (np.fmax(m - (c0 + c1 * np.sqrt(c3 + (l - c4) ** 2) + c5 * l), 0) * (l0 + l1 * l)) ** 2

    return Cd_wave


def Cl_rae2822(alpha, Re_c):
    # A curve fit I did to a RAE2822 airfoil, 2D XFoil data. Incompressible flow.
    # Within -2 < alpha < 12 and 10^4 < Re_c < 10^6, has R^2 = 0.9857
    # Likely valid from -6 < alpha < 12 and 10^4 < Re_c < 10^6.
    # See: C:\Projects\GitHub\firefly_aerodynamics\Gists and Ideas\XFoil Drag Fitting\rae2822

    warnings.warn(
        "This function is deprecated. Use `asb.Airfoil.get_aero_from_neuralfoil()` instead.",
        DeprecationWarning,
    )
    Re_c = np.fmax(Re_c, 1)
    log10_Re = np.log10(Re_c)

    # Coeffs
    a1l = 5.5686866813855172e-02
    a1t = 9.7472055628494134e-02
    a4l = -7.2145733312046152e-09
    a4t = -3.6886704372829236e-06
    atr = 8.3723547264375520e-01
    atr2 = -8.3128119739031697e-02
    c0l = -4.9103908291438701e-02
    c0t = 2.3903424824298553e-01
    ctr = 1.3082854754897108e+01
    rtr = 2.6963082864300731e+00

    a = alpha
    r = log10_Re

    Cl = (c0t + a1t * a + a4t * a ** 4) * 1 / (1 + np.exp(ctr - rtr * r - atr * a - atr2 * a ** 2)) + (
            c0l + a1l * a + a4l * a ** 4) * (1 - 1 / (1 + np.exp(ctr - rtr * r - atr * a - atr2 * a ** 2)))

    return Cl


def Cd_profile_rae2822(alpha, Re_c):
    # A curve fit I did to a RAE2822 airfoil, 2D XFoil data. Incompressible flow.
    # Within -2 < alpha < 12 and 10^4 < Re_c < 10^6, has R^2 = 0.9995
    # Likely valid from -6 < alpha < 12 and 10^4 < Re_c < Inf.
    # see: C:\Projects\GitHub\firefly_aerodynamics\Gists and Ideas\XFoil Drag Fitting\e216

    warnings.warn(
        "This function is deprecated. Use `asb.Airfoil.get_aero_from_neuralfoil()` instead.",
        DeprecationWarning,
    )
    Re_c = np.fmax(Re_c, 1)
    log10_Re = np.log10(Re_c)

    # Coeffs
    at = 8.1034027621509015e+00
    c0l = -8.4296746456429639e-01
    c0t = -1.3700609138855402e+00
    kart = -4.1609994062600880e-01
    kat = 5.9510959342452441e-01
    krt = -7.1938030052506197e-01
    r1l = 1.1548628822014631e-01
    r1t = -4.9133662875044504e-01
    rt = 5.0070459892411696e+00

    a = alpha
    r = log10_Re

    log10_Cd = (c0t + r1t * (r - 4)) * (
            1 / (1 + np.exp(kat * (a - at) + krt * (r - rt) + kart * (a - at) * (r - rt)))) + (
                       c0l + r1l * (r - 4)) * (
                       1 - 1 / (1 + np.exp(kat * (a - at) + krt * (r - rt) + kart * (a - at) * (r - rt))))

    Cd = 10 ** log10_Cd

    return Cd


def Cd_wave_rae2822(Cl, mach, sweep=0.):
    r"""
    A curve fit I did to RAE2822 airfoil data.
    Within -0.4 < CL < 0.75 and 0 < mach < ~0.9, has R^2 = 0.9982.
    See: C:\Projects\GitHub\firefly_aerodynamics\MSES Interface\analysis\rae2822
    :param Cl: Lift coefficient
    :param mach: Mach number
    :param sweep: Sweep angle, in deg
    :return: Wave drag coefficient.
    """

    warnings.warn(
        "This function is deprecated. Use `asb.Airfoil.get_aero_from_neuralfoil()` instead.",
        DeprecationWarning,
    )
    mach = np.fmax(mach, 0)
    mach_perpendicular = mach * np.cosd(sweep)  # Relation from FVA Eq. 8.176
    Cl_perpendicular = Cl / np.cosd(sweep) ** 2  # Relation from FVA Eq. 8.177

    # Coeffs
    c2 = 4.5776476424519119e+00
    mc0 = 9.5623337929607111e-01
    mc1 = 2.0552787101770234e-01
    mc2 = 1.1259268018737063e+00
    mc3 = 1.9538856688443659e-01

    m = mach_perpendicular
    l = Cl_perpendicular

    Cd_wave = np.fmax(m - (mc0 - mc1 * np.sqrt(mc2 + (l - mc3) ** 2)), 0) ** 2 * c2

    return Cd_wave


def fuselage_upsweep_drag_area(
        upsweep_angle_rad: float,
        fuselage_xsec_area_max: float,
) -> float:
    """
    Calculates the drag area (in m^2) of the aft end of a fuselage with a given upsweep angle.

    Upsweep is the characteristic shape seen on the aft end of many fuselages in transport aircraft, where the
    centerline of the fuselage is angled upwards near the aft end. This is done to reduce the required landing gear
    height for adequate takeoff rotation, which in turn reduces mass. This nonzero centerline angle can cause some
    separation drag, which is predicted here.

    Equation is from Raymer, Aircraft Design: A Conceptual Approach, 5th Ed., Eq. 12.36, pg. 440.

    Args:
        upsweep_angle_rad: The upsweep angle of the aft end of the fuselage relative to the centerline, in radians.

        fuselage_xsec_area_max: The maximum cross-sectional area of the fuselage, in m^2.

    Returns: The drag area of the aft end of the fuselage [m^2]. This is equivalent to D/q, where D is the drag force
    and q is the dynamic pressure.
    """
    return 3.83 * np.abs(upsweep_angle_rad) ** 2.5 * fuselage_xsec_area_max


if __name__ == "__main__":
    pass
    # # Run some checks
    # import matplotlib.pyplot as plt
    # import matplotlib.style as style
    # import plotly.express as px
    # from plotly import io
    #
    # io.renderers.default = "browser"
    #
    # # # E216 checks
    # alpha_inputs = np.linspace(-6, 12, 200)
    # Re_inputs = np.logspace(4, 6, 200)
    # alphas = []
    # Res = []
    # CLs = []
    # CDs = []
    # for alpha in alpha_inputs:
    #     for Re in Re_inputs:
    #         alphas.append(alpha)
    #         Res.append(Re)
    #         CLs.append(Cl_e216(alpha, Re))
    #         CDs.append(Cd_profile_e216(alpha, Re))
    # px.scatter_3d(
    #     x=alphas,
    #     y=Res,
    #     z=CLs,
    #     size=np.ones_like(alphas),
    #     color=CLs,
    #     log_y=True,
    #     labels={"x": "alphas", "y": "Re", "z": "CL"}
    # ).show()
    # px.scatter_3d(
    #     x=alphas,
    #     y=Res,
    #     z=CDs,
    #     size=np.ones_like(alphas),
    #     color=CDs,
    #     log_y=True,
    #     labels={"x": "alphas", "y": "Re", "z": "CD"}
    # ).show()
    # px.scatter_3d(
    #     x=alphas,
    #     y=Res,
    #     z=np.array(CLs) / np.array(CDs),
    #     size=np.ones_like(alphas),
    #     color=np.array(CLs) / np.array(CDs),
    #     log_y=True,
    #     labels={"x": "alphas", "y": "Re", "z": "CL/CD"}
    # ).show()
    #
    # # # rae2822 checks
    # alpha_inputs = np.linspace(-6, 12)
    # Re_inputs = np.logspace(4, 6)
    # alphas = []
    # Res = []
    # CLs = []
    # CDs = []
    # for alpha in alpha_inputs:
    #     for Re in Re_inputs:
    #         alphas.append(alpha)
    #         Res.append(Re)
    #         CLs.append(Cl_rae2822(alpha, Re))
    #         CDs.append(Cd_profile_rae2822(alpha, Re))
    # px.scatter_3d(
    #     x=alphas,
    #     y=Res,
    #     z=CLs,
    #     size=np.ones_like(alphas),
    #     color=CLs,
    #     log_y=True,
    #     labels={"x": "alphas", "y": "Re", "z": "CL"}
    # ).show()
    # px.scatter_3d(
    #     x=alphas,
    #     y=Res,
    #     z=CDs,
    #     size=np.ones_like(alphas),
    #     color=CDs,
    #     log_y=True,
    #     labels={"x": "alphas", "y": "Re", "z": "CD"}
    # ).show()
    # px.scatter_3d(
    #     x=alphas,
    #     y=Res,
    #     z=np.array(CLs) / np.array(CDs),
    #     size=np.ones_like(alphas),
    #     color=np.array(CLs) / np.array(CDs),
    #     log_y=True,
    #     labels={"x": "alphas", "y": "Re", "z": "CL/CD"}
    # ).show()
    #
    # # Cd_wave_e216 check
    # CL_inputs = np.linspace(-0.4, 1)
    # mach_inputs = np.linspace(0.3, 1)
    # CLs = []
    # machs = []
    # CD_waves = []
    # for CL in CL_inputs:
    #     for mach in mach_inputs:
    #         CLs.append(CL)
    #         machs.append(mach)
    #         CD_waves.append(Cd_wave_e216(CL, mach))
    # px.scatter_3d(
    #     x=CLs,
    #     y=machs,
    #     z=CD_waves,
    #     size=np.ones_like(CD_waves),
    #     color=CD_waves,
    #     title="E216 Fit",
    #     labels={"x": "CL", "y": "Mach", "z": "CD_wave"},
    #     range_z=(0, 200e-4)
    # ).show()
    #
    # # Cd_wave_rae2822 check
    # CL_inputs = np.linspace(-0.4, 1)
    # mach_inputs = np.linspace(0.3, 1)
    # CLs = []
    # machs = []
    # CD_waves = []
    # for CL in CL_inputs:
    #     for mach in mach_inputs:
    #         CLs.append(CL)
    #         machs.append(mach)
    #         CD_waves.append(Cd_wave_rae2822(CL, mach))
    # px.scatter_3d(
    #     x=CLs,
    #     y=machs,
    #     z=CD_waves,
    #     size=np.ones_like(CD_waves),
    #     color=CD_waves,
    #     title="RAE2822 Fit",
    #     labels={"x": "CL", "y": "Mach", "z": "CD_wave"},
    #     # range_z=(0, 200e-4)
    # ).show()
    #
    # # Cd_wave_Korn check
    # CL_inputs = np.linspace(-0.4, 1)
    # mach_inputs = np.linspace(0.3, 1)
    # CLs = []
    # machs = []
    # CD_waves = []
    # for CL in CL_inputs:
    #     for mach in mach_inputs:
    #         CLs.append(CL)
    #         machs.append(mach)
    #         CD_waves.append(float(Cd_wave_Korn(CL, t_over_c=0.121, mach=mach, kappa_A=0.95)))
    # px.scatter_3d(
    #     x=CLs,
    #     y=machs,
    #     z=CD_waves,
    #     size=list(np.ones_like(CD_waves)),
    #     color=CD_waves,
    #     title="Korn Equation",
    #     labels={"x": "CL", "y": "Mach", "z": "CD_wave"},
    #     range_z=(0, 200e-4)
    # ).show()
    #
    # # # Cylinder Drag Check
    # Res = np.logspace(-1, 8, 300)
    # CDs = Cd_cylinder(Res)
    # CDs_s = Cd_cylinder(Res, True)
    #
    # plt.loglog(Res, CDs, label="Full Model")
    # plt.loglog(Res, CDs_s, label="Subcrit. Only Model")
    # plt.xlabel("Re")
    # plt.ylabel("CD")
    # plt.title("Cylinder Drag Checking")
    # plt.legend()
    # plt.show()
