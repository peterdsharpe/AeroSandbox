# Interfaces to External Aerodynamics Tools

AeroSandbox provides interfaces to various external aerodynamics tools. Note that these tools are non-differentiable, so they cannot be used in conjunction with the `asb.Opti` optimization module. (Unless you wrap outputs with a finite-difference surrogate model; see the Surrogate Modeling tutorial section.)

However, they are still very useful for a variety of purposes:
* Generating data for use with surrogate modeling
* General scripting, as tools are accessible through a unified Python interface with shared data structures and all the scripting tools of the Python language.
* Validation of new aerodynamics analyses in AeroSandbox, by cross-checking point designs.