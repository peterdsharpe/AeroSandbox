# Linear Lifting Line Studies

Peter Sharpe

The goal here is to develop a linear lifting line method, which has a key advantage over nonlinear methods in the context of an automatic-differentiable optimization framework: a linear method can be represented as a single (easily differentiatable) linear solve, whereas a nonlinear method requires an iterative solver, which is not as easily to differentiate. A linear solve is also much faster than an iterative solve. Finally, a linear solve tends to be more stable in the context of a simultaneous-analysis-and-design (SAND) optimization framework, allowing for worse initial guesses. 

At the same time, traditional linear lifting line methods have a weakness in that they (usually) assume a $C_L(\alpha)$ relationship in the form: $C_L(\alpha) = 2\pi\alpha$, which is inaccurate for cambered airfoils, very thick airfoils, or low-Reynolds number airfoils.  The goal here is to develop a linear lifting line method that leverages affine $C_L(\alpha)$ relationships, which preserves the nice mathematical properties of the linear solve, but injects a bit more realism at the sectional level.

## Task List

- [ ] Using XFoil, develop a function that does the following mapping:
  - Inputs: `asb.Airfoil` object (basically, section shape), local angle of attack, local chord, freestream `asb.OperatingPoint` (i.e., freestream velocity, air properties, etc.)
  - Intermediates to be computed: Local Reynolds number
  - Outputs: `CL0` and `CLa`, which are the zero-alpha lift coefficient and the lift-curve slope, respectively
  - This sub-study is done in `./01 - SectionalCoefficientsStudy/`