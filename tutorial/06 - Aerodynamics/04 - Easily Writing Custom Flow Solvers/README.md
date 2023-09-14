# Easily Writing Custom Aerodynamics Solvers

The tutorials in this folder make extensive use of potential flow theory, so it may be useful to have a working knowledge of potential flow theory.

## Introduction to Aerodynamic Potential Flow Theory

> “When a flow is both frictionless and irrotational, pleasant things happen.”
> 
> –White, Fluid Mechanics 4th ed.

Potential flow theory is a specialized area of aerodynamics that focuses on flows meeting specific assumptions: they are inviscid, incompressible, irrotational, and steady. The theory has its origins in the full Navier-Stokes equations. First, disregarding compressibility and unsteadiness reduces these to the incompressible steady Navier-Stokes equations. Eliminating viscosity brings us to the Euler equations. Lastly, an irrotationality constraint leads to the simplified potential flow equation, $\nabla^2 \phi = 0$. This equation is linear, which is a powerful property as it allows for the superposition of elementary flows like uniform flow, sources, sinks, vortices, and doublets. 

Because of its linearity, complex flow patterns can be synthesized by algebraically summing these elementary solutions. This enables us to model intricate aerodynamic phenomena, such as the flow around an airfoil by combining a uniform flow, vortex, and doublet. This makes potential flow theory a powerful tool for studying lift, induced drag, and pressure distributions in a computationally efficient manner - orders of magnitude faster than Navier-Stokes-based methods.

The linearity of the governing equation also allows us to fulfill boundary conditions with relative ease. Typical boundary conditions include the no-penetration condition, which specifies that fluid does not flow through a solid boundary. The solutions to potential flow equations can therefore often be found as boundary value problems. However, it's essential to recognize that potential flow omits viscous effects, which are crucial for predicting phenomena like flow separation and drag. To address this shortcoming, integral boundary layer methods can be utilized to "bring viscosity back in." For example, XFoil uses integral boundary layer methods along with potential flow solutions to offer a more comprehensive aerodynamic analysis around airfoils. Such approaches bridge the gap between idealized and real-world aerodynamic scenarios, providing a more complete picture of fluid behavior around bodies.

