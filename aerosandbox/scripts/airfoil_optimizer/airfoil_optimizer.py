from aerosandbox.geometry import *
from scipy import optimize

Re = 2e5
Cl = 0.75
Cm = -0.133

lower_guess = -0.1 * np.ones(10)
upper_guess = 0.1 * np.ones(10)

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
            TE_thickness=0,
            n_points_per_side=200
        )
    )


x0 = pack(lower_guess, upper_guess)
initial_airfoil = make_airfoil(x0)


def draw(airfoil):
    fig = plt.figure(figsize=(15, 2.5))
    plt.plot(
        initial_airfoil.coordinates[:, 0],
        initial_airfoil.coordinates[:, 1],
        ':',
        label="Initial Airfoil"
    )
    plt.plot(
        airfoil.coordinates[:, 0],
        airfoil.coordinates[:, 1],
        "-",
        label="Current Airfoil"
    )
    plt.axis("equal")
    plt.show()

fs = []

def augmented_objective(x):
    lower, upper = unpack(x)
    airfoil = make_airfoil(x)
    # xfoil = airfoil.xfoil_cseq(
    #     cl_start=0.2,
    #     cl_step=0.05,
    #     cl_end=1,
    #     Re=Re,
    #     verbose=False,
    #     max_iter=30,
    #     reset_bls=True,
    #     repanel=True
    # )
    # xfoil = airfoil.xfoil_a(
    #     alpha=6,
    #     Re=Re,
    #     verbose=False,
    #     max_iter=30,
    #     reset_bls=True,
    #     repanel=True
    # )
    xfoil = airfoil.xfoil_cl(
        cl=Cl,
        Re=Re,
        verbose=False,
        max_iter=30,
        reset_bls=True,
        repanel=True
    )
    # draw(airfoil)
    Cd = xfoil["Cd"]
    if np.isnan(Cd):
        Cd = 0.1

    objective = Cd

    penalty = 0
    penalty += np.sum(np.maximum(0, lower + 0.01) ** 2 / 0.01)
    penalty += np.sum(np.minimum(0, upper - 0.01) ** 2 / 0.01)

    fs.append(objective+penalty)

    return objective + penalty

iteration = 0

def callback(x):
    global iteration
    iteration += 1
    airfoil = make_airfoil(x)
    draw(airfoil)
    print(
        """Iteration %i: Cd = %.6f
        \tx = %s""" % (iteration, fs[-1], np.array2string(x, separator=","))
    )


if __name__ == '__main__':
    draw(initial_airfoil)

    method = "Nelder-Mead"

    if method == "Nelder-Mead":
        initial_simplex = (
                (0.5 + 1 * np.random.random((len(x0) + 1, len(x0))))
                * x0
        )
        res = optimize.minimize(
            fun=augmented_objective,
            x0=pack(lower_guess, upper_guess),
            method=method,
            callback=callback,
            options={
                'maxiter'        : 10 ** 6,
                'initial_simplex': initial_simplex,
                'xatol'          : 1e-3,
                'fatol'          : 1e-2,
                'adaptive'       : True,
            }
        )
        final_airfoil = make_airfoil(res.x)
    elif method == "TNC":
        res = optimize.minimize(
            fun=augmented_objective,
            x0=pack(lower_guess, upper_guess),
            method=method,
            callback=callback,
            options={
                'maxiter': 10 ** 6,
                'eps'    : 1e-3,
                'stepmx' : 0.05
            }
        )
        final_airfoil = make_airfoil(res.x)

