import aerosandbox.numpy as np
from aerosandbox.geometry import Airfoil
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates
from scipy import optimize
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

if __name__ == '__main__':

    ### Design Conditions
    Re_des = 3e5  # Re to design to
    Cl_start = 1.0  # Lower bound of CLs that you care about
    Cl_end = 1.5  # Upper bound of CLs that you care about (Effectively, CL_max)
    Cm_min = -0.08  # Worst-allowable pitching moment that you'll allow
    TE_thickness = 0.0015  # Sets trailing edge thickness
    enforce_continuous_LE_radius = True  # Should we force the leading edge to have continous curvature?

    ### Guesses for airfoil CST parameters; you usually don't need to change these
    lower_guess = -0.05 * np.ones(30)
    upper_guess = 0.25 * np.ones(30)
    upper_guess[0] = 0.15
    upper_guess[1] = 0.20

    # lower_guess = [-0.21178419, -0.05500152, -0.04540216, -0.03436429, -0.03305599,
    #                -0.03121454, -0.04513736, -0.05491045, -0.02861083, -0.05673649,
    #                -0.06402239, -0.05963394, -0.0417384, -0.0310728, -0.04983729,
    #                -0.04211283, -0.04999657, -0.0632682, -0.07226548, -0.03604782,
    #                -0.06151112, -0.04030985, -0.02748867, -0.02705322, -0.04279788,
    #                -0.04734922, -0.033705, -0.02380217, -0.04480772, -0.03756881]
    # upper_guess = [0.17240303, 0.26668075, 0.21499604, 0.26299318, 0.22545807,
    #                0.24759903, 0.31644402, 0.2964658, 0.15360716, 0.31317824,
    #                0.27760982, 0.23009955, 0.24045039, 0.37542525, 0.21361931,
    #                0.18678503, 0.23466624, 0.20630533, 0.16191541, 0.20453953,
    #                0.14370825, 0.13428077, 0.15387739, 0.13767285, 0.15173257,
    #                0.14042002, 0.11336701, 0.35640688, 0.10953915, 0.08167446]

    ### Packing/Unpacking functions
    n_lower = len(lower_guess)
    n_upper = len(upper_guess)
    pack = lambda lower, upper: np.concatenate((lower, upper))
    unpack = lambda pack: (pack[:n_lower], pack[n_lower:])


    def make_airfoil(x):
        """
        A function that constructs an airfoil from a packed design vector.
        :param x:
        :return:
        """
        lower, upper = unpack(x)
        return Airfoil(
            name="Optimization Airfoil",
            coordinates=get_kulfan_coordinates(
                lower_weights=lower,
                upper_weights=upper,
                enforce_continuous_LE_radius=enforce_continuous_LE_radius,
                TE_thickness=TE_thickness,
                n_points_per_side=80
            )
        )


    ### Initial guess construction
    x0 = pack(lower_guess, upper_guess)
    initial_airfoil = make_airfoil(x0)

    ### Initialize plotting
    fig = plt.figure(figsize=(15, 2.5))
    ax = fig.add_subplot(111)
    trace_initial, = ax.plot(
        initial_airfoil.coordinates[:, 0],
        initial_airfoil.coordinates[:, 1],
        ':r',
        label="Initial Airfoil"
    )
    trace_current, = ax.plot(
        initial_airfoil.coordinates[:, 0],
        initial_airfoil.coordinates[:, 1],
        "-b",
        label="Current Airfoil"
    )
    plt.axis("equal")
    plt.xlabel(r"$x/c$")
    plt.ylabel(r"$y/c$")
    plt.title("Airfoil Optimization")
    plt.legend()


    def draw(
            airfoil  # type: Airfoil
    ):
        """
        Updates the "current airfoil" line on the plot with the given airfoil.
        :param airfoil:
        :return:
        """
        trace_current.set_xdata(airfoil.coordinates[:, 0])
        trace_current.set_ydata(airfoil.coordinates[:, 1])
        plt.draw()
        plt.pause(0.001)


    ### Utilities for tracking the design vector and objective throughout the optimization run
    iteration = 0
    xs = []
    fs = []


    def augmented_objective(x):
        """
        Objective function with constraints added via a multiplicative external penalty method
        :param x: Packed design vector
        :return: Value of the augmented objective
        """
        airfoil = make_airfoil(x)
        xfoil = airfoil.xfoil_cseq(
            cl_start=Cl_start,
            cl_step=0.02,
            cl_end=Cl_end,
            Re=Re_des,
            verbose=False,
            max_iter=40,
            repanel=False
        )
        if np.isnan(xfoil["Cd"]).any():
            return np.Inf

        objective = np.sqrt(np.mean(xfoil["Cd"] ** 2))  # RMS

        penalty = 0
        penalty += np.sum(np.minimum(0, (xfoil["Cm"] - Cm_min) / 0.01) ** 2)  # Cm constraint
        penalty += np.minimum(0, (airfoil.TE_angle() - 5) / 1) ** 2  # TE angle constraint
        penalty += np.minimum(0, (airfoil.local_thickness(0.90) - 0.015) / 0.005) ** 2  # Spar thickness constraint
        penalty += np.minimum(0, (airfoil.local_thickness(0.30) - 0.12) / 0.005) ** 2  # Spar thickness constraint

        xs.append(x)
        fs.append(objective)

        return objective * (1 + penalty)


    def callback(x):
        global iteration
        iteration += 1
        print(
            f"Iteration {iteration}: Cd = {fs[-1]:.6f}"
        )
        if iteration % 1 == 0:
            airfoil = make_airfoil(x)
            draw(airfoil)
            ax.set_title(f"Airfoil Optimization: Iteration {iteration}")
            airfoil.write_dat("optimized_airfoil.dat")


    draw(initial_airfoil)

    initial_simplex = (
            (0.5 + 1 * np.random.random((len(x0) + 1, len(x0))))
            * x0
    )
    initial_simplex[0, :] = x0  # Include x0 in the simplex
    print("Initializing simplex (give this a few minutes)...")
    res = optimize.minimize(
        fun=augmented_objective,
        x0=pack(lower_guess, upper_guess),
        method="Nelder-Mead",
        callback=callback,
        options={
            'maxiter'        : 10 ** 6,
            'initial_simplex': initial_simplex,
            'xatol'          : 1e-8,
            'fatol'          : 1e-6,
            'adaptive'       : False,
        }
    )

    final_airfoil = make_airfoil(res.x)
