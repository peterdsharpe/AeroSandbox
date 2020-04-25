from aerosandbox import *
import dill as pickle

if __name__ == '__main__': # If you're a Windows user, you must run within a main block if you want to use any parallel functions.

    ########## First, define the airfoils that you want to use. ##########
    # For a lifting line analysis like CasLL1, you need differentiable 2D sectional data for the airfoils.
    # There are two ways you can do this:
    #   1. You can automatically fit functions to automated XFoil runs.
    #   2. You can provide explicit (possibly nonlinear) functions for CL, CDp, and Cm for each airfoil.
    # In this example, we're using an Eppler 216 airfoil for the wing and NACA0008 airfoils everywhere else.
    # I'll show you what both methods to get data look like!

    ### Method 1: XFoil Fitting

    # I've wrapped these in a simple caching script, since the XFoil runs are slow!
    try:
        with open("e216.pkl", "rb") as f: e216 = pickle.load(f)
    except:
        e216 = Airfoil("e216") # You can use the string of any airfoil from the UIUC database here!
        e216.populate_sectional_functions_from_xfoil_fits()
        with open("e216.pkl", "wb+") as f: pickle.dump(e216, f)

    try:
        with open("naca0008.pkl", "rb") as f: naca0008 = pickle.load(f)
    except:
        naca0008 = Airfoil("naca0008") # You can also give NACA airfoils!
        # You can also load from a .dat file (see Airfoil constructor docstring for syntax)!
        naca0008.populate_sectional_functions_from_xfoil_fits()
        with open("naca0008.pkl", "wb+") as f: pickle.dump(naca0008, f)

    ### Method 2: Explicit fits (look here in the library to see what these look like)
    from aerosandbox.library.airfoils import e216, naca0008

    ########## Now, we're ready to start putting together our 3D CasLL1 run! ##########

    opti = cas.Opti()  # Initialize an analysis/optimization environment

    ### Define the 3D geometry you want to analyze/optimize.
    # Here, all distances are in meters and all angles are in degrees.
    airplane = Airplane(
        name="Peter's Glider",
        x_ref=0,  # CG location
        y_ref=0,  # CG location
        z_ref=0,  # CG location
        wings=[
            Wing(
                name="Main Wing",
                x_le=0,  # Coordinates of the wing's leading edge
                y_le=0,  # Coordinates of the wing's leading edge
                z_le=0,  # Coordinates of the wing's leading edge
                symmetric=True, # Should this wing be mirrored across the XZ plane?
                xsecs=[  # The wing's cross ("X") sections
                    WingXSec(  # Root
                        x_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                        y_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                        z_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                        chord=0.18,
                        twist=2,  # degrees
                        airfoil=e216,  # Airfoils are blended between a given XSec and the next one.
                        control_surface_type='symmetric', # Flap (ctrl. surfs. applied between this XSec and the next one.)
                        control_surface_deflection=0,  # degrees
                    ),
                    WingXSec(  # Mid
                        x_le=0.01,
                        y_le=0.5,
                        z_le=0,
                        chord=0.16,
                        twist=0,
                        airfoil=e216,
                        control_surface_type='asymmetric',  # Aileron
                        control_surface_deflection=0,
                    ),
                    WingXSec(  # Tip
                        x_le=0.08,
                        y_le=1,
                        z_le=0.1,
                        chord=0.08,
                        twist=-2,
                        airfoil=e216,
                    ),
                ]
            ),
            Wing(
                name="Horizontal Stabilizer",
                x_le=0.6,
                y_le=0,
                z_le=0.1,
                symmetric=True,
                xsecs=[
                    WingXSec(  # root
                        x_le=0,
                        y_le=0,
                        z_le=0,
                        chord=0.1,
                        twist=-10,
                        airfoil=naca0008,
                        control_surface_type='symmetric',  # Elevator
                        control_surface_deflection=0,
                    ),
                    WingXSec(  # tip
                        x_le=0.02,
                        y_le=0.17,
                        z_le=0,
                        chord=0.08,
                        twist=-10,
                        airfoil=naca0008
                    )
                ]
            ),
            Wing(
                name="Vertical Stabilizer",
                x_le=0.6,
                y_le=0,
                z_le=0.15,
                symmetric=False,
                xsecs=[
                    WingXSec(
                        x_le=0,
                        y_le=0,
                        z_le=0,
                        chord=0.1,
                        twist=0,
                        airfoil=naca0008,
                        control_surface_type='symmetric',  # Rudder
                        control_surface_deflection=0,
                    ),
                    WingXSec(
                        x_le=0.04,
                        y_le=0,
                        z_le=0.15,
                        chord=0.06,
                        twist=0,
                        airfoil=naca0008
                    )
                ]
            )
        ]
    )
    # airplane.draw() # You can use this to quickly preview your geometry!
    airplane.set_spanwise_paneling_everywhere(20)  # Set the resolution of your analysis
    ap = Casll1(  # Set up the AeroProblem
        airplane=airplane,
        op_point=OperatingPoint(
            density=1.225,  # kg/m^3
            viscosity=1.81e-5,  # kg/m-s
            velocity=10,  # m/s
            mach=0,  # Freestream mach number
            alpha=5,  # In degrees
            beta=0,  # In degrees
            p=0,  # About the body x-axis, in rad/sec
            q=0,  # About the body y-axis, in rad/sec
            r=0,  # About the body z-axis, in rad/sec
        ),
        opti=opti  # Pass it an optimization environment to work in
    )

    # Solver options
    p_opts = {}
    s_opts = {}
    # s_opts["mu_strategy"] = "adaptive"
    opti.solver('ipopt', p_opts, s_opts)

    ### Solve
    try:
        sol = opti.solve()
    except RuntimeError:
        sol = opti.debug
        raise Exception("An error occurred!")

    ### Postprocess

    # Create solved object
    ap_sol = copy.deepcopy(ap)
    ap_sol.substitute_solution(sol)

    ap_sol.draw()  # Generates

    print("CL:", ap_sol.CL)
    print("CD:", ap_sol.CD)
    print("CY:", ap_sol.CY)
    print("Cl:", ap_sol.Cl)
    print("Cm:", ap_sol.Cm)
    print("Cn:", ap_sol.Cn)

    # Answer from XFLR5 Viscous VLM2
    # CL = 1.112
    # CD = 0.057
    # CL/CD = 19.499
    #   Note that XFLR5 will overpredict lift for this case compared to reality, since a VLM method
    #   (which is fundamentally linear) doesn't take into account any kind of viscous decambering
    #   at high CL, and XFLR5 makes no adjustments for this. This will also mean that XFLR5 will
    #   overpredict drag (in particular induced drag), since the circulation is overpredicted.