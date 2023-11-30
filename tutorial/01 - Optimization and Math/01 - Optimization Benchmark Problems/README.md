# Optimization Benchmark Problems

This folder compares optimization performance with AeroSandbox to various other common optimization paradigms.

## AeroSandbox vs. Black-Box Optimization Methods

This chart shows optimization performance on the [N-dimensional Rosenbrock problem](https://en.wikipedia.org/wiki/Rosenbrock_function#Multidimensional_generalizations). Here, $N$ is the number of design variables, which is a convenient knob to dial up or down the difficulty of the problem. The problem is defined as:

* minimize $\sum\limits_{i=1}^{N-1} [ 100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]$

For all $N$, the global optimum is at $\vec{x} = \vec{1}$, where the objective function evaluates to $0$. 

This problem is chosen here because it shares many difficult aspects with engineering design optimization problems: it is nonlinear, nonconvex, and poorly-scaled. Furthermore, **we deliberately choose awful initial guesses**, with each element of the vector of initial guesses drawn from a random uniform distribution in the interval $[-10, 10]$.

The performance of AeroSandbox (with CasADi backend) is compared against existing methods using black-box optimization techniques. AeroSandbox offers faster practical and asymptotic optimization performance than existing black-box optimization methods, demonstrating the magnitude of acceleration that is possible.

![benchmark_nd_rosenbrock](./nd_rosenbrock/benchmark_nd_rosenbrock.png)

## AeroSandbox vs. Disciplined Optimization Methods

In this chart, runtime is used instead of function evaluations, because the GPkit API doesn't easily expose this information from the underlying solver.

![benchmark_gp_beam](./gp_beam/benchmark_gp_beam.png)