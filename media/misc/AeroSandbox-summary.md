> AeroSandbox is a computational tool for solving design optimization problems for large, multidisciplinary engineered systems. Its key contribution is an object-oriented optimization framework paired with a numerics library. Both the optimization framework and numerics library are accelerated with modern automatic differentiation, which dramatically improves optimization capabilities on large problems: design problems with tens of thousands of decision variables solve in seconds on a laptop.
>
> AeroSandbox focuses on abstracting away the underlying optimization numerics, which yields design code that looks nearly-identical to that of a typical engineering point analysis. Additionally, the AeroSandbox numerics library exactly mirrors syntax with the popular NumPy scientific computing library, which further reduces the tool's learning curve. This enables engineers without optimization experience to quickly pose problems in natural syntax while still taking advantage of powerful optimization tools.
>
> In addition, AeroSandbox has many features that make it much easier to pose aircraft design optimization problems:
>
> 1. Hundreds of mutually-compatible differentiable aerospace physics models across dozens of disciplines
> 2. Surrogate modeling tools to create new differentiable models from user data (e.g. CFD, flight test)
> 3. An object-oriented aircraft geometry framework and performance stack
>
>When these features are combined, AeroSandbox enables users to simultaneously optimize an aircraft's aerodynamics, structures, propulsion, mission trajectory, stability, and more in mere seconds. As a result, interactive design optimization with a human in the loop becomes viable, where engineers iteratively pose various design questions to better understand the mission space.