# Optimization Benchmark Problems

This folder compares optimization performance with AeroSandbox to various other common optimization paradigms.

## AeroSandbox vs. Black-Box Optimization Methods

This chart shows optimization performance on the [N-dimensional Rosenbrock problem](https://en.wikipedia.org/wiki/Rosenbrock_function#Multidimensional_generalizations). Here, $N$ is the number of design variables, which is a convenient knob to dial up or down the difficulty of the problem. The problem is defined as:

* minimize $\sum_{i=1}^{N-1} [ 100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]$


![benchmark_nd_rosenbrock](./nd_rosenbrock/benchmark_nd_rosenbrock.png)


