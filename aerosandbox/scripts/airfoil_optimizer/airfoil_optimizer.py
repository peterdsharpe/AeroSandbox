from aerosandbox.geometry import *
from scipy import optimize
import dill as pickle

HALE_03 = Airfoil(name="HALE_03", coordinates=r"C:\Projects\GitHub\Airfoils\HALE_03.dat")

Re_des = 2e5
Cl_start = 1.15
Cl_end = 1.45
Cm_min = -0.04

lower_guess = -0.05 * np.ones(30)
upper_guess = 0.25 * np.ones(30)
upper_guess[0] = 0.15
upper_guess[1] = 0.20

lower_guess = [-0.21178419, -0.05500152, -0.04540216, -0.03436429, -0.03305599,
       -0.03121454, -0.04513736, -0.05491045, -0.02861083, -0.05673649,
       -0.06402239, -0.05963394, -0.0417384 , -0.0310728 , -0.04983729,
       -0.04211283, -0.04999657, -0.0632682 , -0.07226548, -0.03604782,
       -0.06151112, -0.04030985, -0.02748867, -0.02705322, -0.04279788,
       -0.04734922, -0.033705  , -0.02380217, -0.04480772, -0.03756881]
upper_guess = [0.17240303, 0.26668075, 0.21499604, 0.26299318, 0.22545807,
       0.24759903, 0.31644402, 0.2964658 , 0.15360716, 0.31317824,
       0.27760982, 0.23009955, 0.24045039, 0.37542525, 0.21361931,
       0.18678503, 0.23466624, 0.20630533, 0.16191541, 0.20453953,
       0.14370825, 0.13428077, 0.15387739, 0.13767285, 0.15173257,
       0.14042002, 0.11336701, 0.35640688, 0.10953915, 0.08167446]

n_lower = len(lower_guess)
n_upper = len(upper_guess)
pack = lambda lower, upper: np.concatenate((lower, upper))
unpack = lambda pack: (pack[:n_lower], pack[n_lower:])


def make_airfoil(x):
    lower, upper = unpack(x)
    return Airfoil(
        name="Optimization Airfoil",
        coordinates=kulfan_coordinates(
            lower_weights=lower,
            upper_weights=upper,
            enforce_continuous_LE_radius=False,
            TE_thickness=0.0015,
            n_points_per_side=80
        )
    )


x0 = pack(lower_guess, upper_guess)
initial_airfoil = make_airfoil(x0)

fig = plt.figure(figsize=(15, 2.5))
ax = fig.add_subplot(111)
trace_initial, = ax.plot(
    initial_airfoil.coordinates[:, 0],
    initial_airfoil.coordinates[:, 1],
    ':',
    label="Initial Airfoil"
)
trace_current, = ax.plot(
    initial_airfoil.coordinates[:, 0],
    initial_airfoil.coordinates[:, 1],
    "-",
    label="Current Airfoil"
)
plt.axis("equal")


def draw(airfoil):
    trace_current.set_xdata(
        airfoil.coordinates[:, 0]
    )
    trace_current.set_ydata(
        airfoil.coordinates[:, 1]
    )
    plt.draw()
    plt.pause(0.001)

xs = []
fs = []


def augmented_objective(x):
    lower, upper = unpack(x)
    airfoil = make_airfoil(x)
    xfoil = airfoil.xfoil_cseq(
        cl_start=Cl_start,
        cl_step=0.02,
        cl_end=Cl_end,
        Re=Re_des,
        verbose=False,
        max_iter=40,
        repanel=True
    )
    if np.isnan(xfoil["Cd"]).any():
        return np.Inf

    # objective = np.mean(xfoil["Cd"])
    objective = np.sqrt(np.mean(xfoil["Cd"]**2)) # RMS

    penalty = 0
    penalty += np.sum(np.minimum(0, xfoil["Cm"] - Cm_min) ** 2 / 0.01)  # Cm constraint
    penalty += np.minimum(0, airfoil.TE_angle() - 5) ** 2 / 1  # TE angle constraint
    penalty += np.minimum(0, airfoil.local_thickness(0.90) - 0.015) ** 2 / 0.005  # Spar thickness constraint
    penalty += np.minimum(0, airfoil.local_thickness(0.30) - 0.12) ** 2 / 0.005  # Spar thickness constraint
    # penalty += np.sum(np.maximum(0, lower + 0.01) ** 2 / 0.01)
    # penalty += np.sum(np.minimum(0, upper - 0.01) ** 2 / 0.01)

    xs.append(x)
    fs.append(objective)

    return objective * (1 + penalty)


iteration = 0


def callback(x):
    global iteration
    iteration += 1
    print(
        "Iteration %i: Cd = %.6f" % (iteration, fs[-1])
    )
    if iteration % 1 == 0:
        airfoil = make_airfoil(x)
        draw(airfoil)
        airfoil.write_dat("optimized_airfoil.dat")


if __name__ == '__main__':
    draw(initial_airfoil)

    initial_simplex = (
            (0.5 + 1 * np.random.random((len(x0) + 1, len(x0))))
            * x0
    )
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
    # res = optimize.minimize(
    #     fun=augmented_objective,
    #     x0=pack(lower_guess, upper_guess),
    #     method="trust-constr",
    #     callback=callback,
    #     options={
    #         'maxiter'          : 10 ** 6,
    #         'verbose'          : 2,
    #         'initial_tr_radius': 0.01,
    #         'xtol'            : 1e-8,
    #     }
    # )

    final_airfoil = make_airfoil(res.x)
