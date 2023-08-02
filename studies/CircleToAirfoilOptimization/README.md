## Airfoil Optimization from a Circle to an Airfoil

Recent literature ([He, et al., "Robust Aerodynamic Shape Optimization - From a Circle to an Airfoil"](https://www.sciencedirect.com/science/article/abs/pii/S1270963818319072)) has proposed an airfoil optimization test case that stress-tests any airfoil optimization procedure for robustness to bad initial guesses.

In short, this test case is to optimize a transonic airfoil starting from an initial guess of a circle-shaped airfoil.

More precisely, we reproduce the optimization problem from [He, et al.](https://www.sciencedirect.com/science/article/abs/pii/S1270963818319072):

* Minimize $C_D$ at $\mathrm{Re}_c = 6.5\mathrm{M}$ and $\mathrm{M}_\infty = 0.734$
* By changing airfoil shape and angle of attack $\alpha$
* Subject to:
    * $C_L = 0.824$
    * $C_M \geq -0.092$
    * $1\degree \leq \alpha \leq 5\degree$
    * Airfoil area $A \leq A_\mathrm{initial}$
    * Airfoil thickness $t/c \geq 0$ everywhere

