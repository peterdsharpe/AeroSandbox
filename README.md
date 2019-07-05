# [AeroSandbox](https://peterdsharpe.github.io/AeroSandbox/) 
by [Peter Sharpe](https://peterdsharpe.github.io) (<pds [at] mit [dot] edu>)


## About
AeroSandbox is a Python package for aircraft design optimization, fully-coupled viscous/inviscid 3D aerodynamics, and reverse-mode automatic differentiation for computing gradients of design variables. 

Work in progress!


![AeroSandbox Image](media/images/vlm2_with_control_surfaces.png)
*VLM2 simulation of a glider, aileron deflections of +-30°*

## Getting Started

There are several easy ways to get started with AeroSandbox! (Assuming you already have Python >=3.7 installed, preferably via the [Anaconda distribution](https://www.anaconda.com/distribution/#download-section).)

1. (Recommended) Download the latest release here: [https://github.com/peterdsharpe/AeroSandbox/releases](https://github.com/peterdsharpe/AeroSandbox/releases). Then, run "pip install AeroSandbox" in your command prompt to ensure that all dependencies are satisfied.

2. If you just want the raw package (and no test cases or examples), install by simply typing "pip install AeroSandbox" into your terminal. 

3. Both of the above options will download released versions of AeroSandbox. If you'd rather get a nightly/dev version (which has more features but may be buggy), clone or download directly from [the AeroSandbox GitHub page](https://github.com/peterdsharpe/AeroSandbox).

There are many example cases you can try out in the /examples/ directory! Specifically, try running "/examples/vlm2_conventional.py".

### Usage
AeroSandbox is designed to have extremely intuitive, high-level, and human-readable code. For example, here is all the code that is needed to design a glider and analyze its aerodynamics in flight (found in "/examples/vlm2_conventional.py"):

```python
from aerosandbox import *

glider = Airplane(
    name="Peter's Glider",
    xyz_ref=[0, 0, 0], # CG location
    wings=[
        Wing(
            name="Main Wing",
            xyz_le=[0, 0, 0], # Coordinates of the wing's leading edge
            symmetric=True,
            xsecs=[ # The wing's cross ("X") sections
                WingXSec(  # Root
                    xyz_le=[0, 0, 0], # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    chord=0.18,
                    twist=2, # degrees
                    airfoil=Airfoil(name="naca4412"),
                    control_surface_type='symmetric',  # Flap # Control surfaces are applied between a given XSec and the next one.
                    control_surface_deflection=0, # degrees
                    control_surface_hinge_point=0.75 # as chord fraction
                ),
                WingXSec(  # Mid
                    xyz_le=[0.01, 0.5, 0],
                    chord=0.16,
                    twist=0,
                    airfoil=Airfoil(name="naca4412"),
                    control_surface_type='asymmetric',  # Aileron
                    control_surface_deflection=0,
                    control_surface_hinge_point=0.75
                ),
                WingXSec(  # Tip
                    xyz_le=[0.08, 1, 0.1],
                    chord=0.08,
                    twist=-2,
                    airfoil=Airfoil(name="naca4412"),
                )
            ]
        ),
        Wing(
            name="Horizontal Stabilizer",
            xyz_le=[0.6, 0, 0.1],
            symmetric=True,
            xsecs=[
                WingXSec(  # root
                    xyz_le=[0, 0, 0],
                    chord=0.1,
                    twist=-10,
                    airfoil=Airfoil(name="naca0012"),
                    control_surface_type='symmetric',  # Elevator
                    control_surface_deflection=0,
                    control_surface_hinge_point=0.75
                ),
                WingXSec(  # tip
                    xyz_le=[0.02, 0.17, 0],
                    chord=0.08,
                    twist=-10,
                    airfoil=Airfoil(name="naca0012")
                )
            ]
        ),
        Wing(
            name="Vertical Stabilizer",
            xyz_le=[0.6, 0, 0.15],
            symmetric=False,
            xsecs=[
                WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=0.1,
                    twist=0,
                    airfoil=Airfoil(name="naca0012"),
                    control_surface_type='symmetric',  # Rudder
                    control_surface_deflection=0,
                    control_surface_hinge_point=0.75
                ),
                WingXSec(
                    xyz_le=[0.04, 0, 0.15],
                    chord=0.06,
                    twist=0,
                    airfoil=Airfoil(name="naca0012")
                )
            ]
        )
    ]
)

aero_problem = vlm2( # Analysis type: Vortex Lattice Method, version 2
    airplane=glider,
    op_point=OperatingPoint(
        velocity=10,
        alpha=5,
        beta=0,
        p=0,
        q=0,
        r=0,
    ),
)

aero_problem.run() # Runs and prints results to console
aero_problem.draw() # Creates an interactive display of the surface pressures and streamlines
```

The best part is that by adding just a few more lines of code, you can not only get the performance at a specified design point, but also the derivatives of any performance variable with respect to any design variable. And, thanks to reverse-mode automatic differentiation, you can calculate 10,000 design derivatives in roughly the same amount of time that it takes to do a single point analysis. For an example of this, see "/examples/gradient_test_vlm2.py".

One final point to note: as we're all civilized human beings here, all inputs and outputs to AeroSandbox are expressed in base metric units (meters, Newtons, kilograms, etc.).

### Dependencies

The fastest way to ensure that all dependencies are satisfied is by simply running "pip install AeroSandbox" in your command prompt. However, you can also install dependencies on your own if you'd like. You'll need the following libraries:
* numpy
* scipy
* matplotlib
* numba
* autograd
* pyvista

If you installed Python via the [Anaconda distribution](https://www.anaconda.com/distribution/#download-section), you likely have all of these except autograd and pyvista.

OpenGL is also required for visualization, though this should already be installed on nearly every computer. (No promises if you try to run AeroSandbox on a Raspberry Pi!)

### Current Features
* User-friendly, concise, high-level, object-oriented structure for airplane geometry definition and analysis.
* Very fast vortex-lattice method flow solver ("VLM1") fully compatible with arbitrary combinations of lifting surfaces.
* Fully reverse-mode AD compatible vortex-lattice method flow solver ("VLM2")! With this, you can get the gradient of a design space with arbitrary dimensionality almost instantly.

### Purpose
The primary purpose for this repository is to explore existing methods for aerodynamic analysis and develop new methods within a unified code base.

The "holy grail" of aerodynamics that this package seeks to develop is:
An aerodynamics tool that models flow around any general triangulated 3D shape (with non-separated flow) using strongly-coupled viscous/inviscid methods. If successful, this could be orders of magnitude faster than volume-mesh-based CFD while retaining high accuracy (XFoil is a 2D example of this). This is very difficult and likely years away, and it's likely that AeroSandbox won't be the toolbox to develop this - but there's no harm in taking a stab at it, if only to understand the problem space better!

This code is made open-source in hopes that the aerodynamics community can benefit from this work. I've benefitted so much from open-source aerodynamics tools that came before me (XFoil, AVL, QProp, GPKit, XFLR5, OpenVSP, SU2, and SUAVE, just to name a few), so I hope to pay it forward, at least in small part!

## Future Goals
In descending order of priority/feasibility:
* (DONE) Finish implementing a traditional VLM for simulating multiple thin lifting surfaces.
* (DONE) Implement proper stability derivative calculation (i.e. not using finite-differencing).
* (SKIPPING) Perhaps implement a viscous drag buildup on wings from interpolated 2D XFOIL data (a la XFLR5's method for approximation of viscous drag).
* (SKIPPING) Perhaps implement a hybrid ring/horseshoe vortex VLM (a la XFLR5's VLM2) for simulating multiple thin lifting surfaces (hopefully with improved speed and robustness over the VLM1 approach).
* (SKIPPING) Implement a viscous drag buildup on nearly-axisymmetric bodies (using the method detailed in Drela's TASOPT v2.00 documentation, Appendix E)
* (SKIPPING) Perhaps consider implementing a free-wake compatible VLM model?
* (IN PROGRESS) Implement an inviscid 3D panel method for simulating multiple objects of arbitrary thickness.
* Make the aforementioned 3D panel method able to use triangular panels for use with generalized geometries (e.g. blended wing bodies), given prescribed trailing edge stagnation points.
* Implement a 2.5D coupled viscous/inviscid method directly using the viscous methods described in Drela's paper "Viscous-Inviscid Analysis of Transonic and Low Reynolds Number Airfoils". Inviscid flow would be fully 3D, while viscous flow would make the assumption of negligible spanwise flow.
* Implement a fully 3D coupled viscous/inviscid method, compatible with triangular panels (a la Drela's IBL3 approach detailed in his paper "Three-Dimensional Integral Boundary Layer Formulation for General Configurations"). Ideally, the trailing edge stagnation points will be automatically identified, and nothing more than a surface triangulation along with freestream conditions will be required to compute forces and moments.


## Usefulness
AeroSandbox attempts to improve over existing conceptual-level aerodynamics tools. The following strengths and weaknesses are identified with existing tools, based purely off the author's experience:

Strengths:
* XFLR5: Reliability, speed, accuracy, visualization
* AVL: Reliability, speed, accuracy, scriptability
* Tornado: Implementation in a high-level language
* VSPAero: Rapid CAD/geometry integration, geometric flexibility

Weaknesses:
* XFLR5: Lack of scriptability, limited geometric flexibility
* AVL: Single-precision calculation (low gradient accuracy), bottlenecking due to file I/O
* Tornado: Speed, user-friendliness
* VSPAero: Robustness, speed, accuracy, and reliability

With any luck, the list of strengths and weaknesses here will help to drive AeroSandbox development to retain positive qualities and eliminate negative ones. 

Specifically, the following desirable qualities (and associated quantitative metrics) have been identified:
* Fast (for point analysis, VLM1 should yield a solution (CL, CDi) within 5% of the "Richardson-extrapolated" solution in less than 1 second for the ExampleAirplanes.conventional() airplane on a typical desktop computer)
* Accurate (in the limit of high panel density, the solution (CL, CDi) given by VLM1 must match AVL or XFLR5 to within 1%)
* Reliable/Robust (gradients of the outputs w.r.t. inputs are always finite and sensible - specifically, this implies that all vortex kernels must be artificially made to have no singularities)
* User-friendly (eventually, a GUI will be created, and AeroSandbox will optionally ship as a packaged executable)
* Scriptable (the code will be object-oriented; the GUI will contain a CLI)
* Readable (every class and function will be documented; code will be PEP-8-compatible where reasonable)
* Optimizer-friendly (design gradients and stability derivatives will be efficiently computed through automatic differentiation, not finite differencing - perhaps with the autograd library?)
* Visualization (visualization will be provided through an OpenGL-compatible library - perhaps PyVista?)

## Donating
If you like this software, please consider donating to support development via PayPal at [paypal.me/peterdsharpe](https://paypal.me/peterdsharpe)!

## Bugs
Please, please report all bugs by creating a new issue at [https://github.com/peterdsharpe/AeroSandbox/issues](https://github.com/peterdsharpe/AeroSandbox/issues)!

Please note that, while the entirety of the codebase should be cross-platform compatible, AeroSandbox has only been tested on Windows 10.

## Contributing

Thanks for your interest in helping with the development of AeroSandbox - contributions are always so welcome! 

If you have a change you'd like to make, the easiest way to do that is by submitting a pull request. However, please let me know before you do this (pds at mit dot edu), because:

Right now, branching is basically nonexistent. This is because there's currently only one contributor - me. As soon as this changes, we'll need to implement [proper branching](https://nvie.com/posts/a-successful-git-branching-model/). 

If you've made several additions and would like to be involved in a more long-term capacity, please message me at (pds at mit dot edu) and we can add you as a collaborator here on Github!

## License

MIT License

Copyright (c) 2019 Peter Sharpe

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

