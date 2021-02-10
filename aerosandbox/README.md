# AeroSandbox User Guide

by Peter Sharpe

----------

Hello there, dear user!

Welcome to the inner workings of AeroSandbox. Come in and stay awhile - we're so glad you're here! :smile:

## Map

There's a big world in here filled with functions, classes, analyses, and more - let me show you around; we wouldn't want you to get lost! Here's your map of the place, in the order of how you should explore these folders to learn how you can harness the power of AeroSandbox. First, let's look at the core pieces:

### The Core Tools

These are the two key pieces of AeroSandbox to understand - once you get these, you're 90% of the way there.

* `/optimization/`: This folder contains only one thing, and it's the single most important class in AeroSandbox: the `Opti` stack. The `Opti` class is an object-oriented way to formulate and solve an optimization problem, with syntax specifically aimed at engineering design. 

	One of the core principles of AeroSandbox is that *everything* is an optimization problem. Even for problems that look like pure analysis ("I already have a design, how well does it perform?"), there's a beautiful duality between optimization and analysis through something called "Simultaneous Analysis and Design" - more on this later. Because of this, the `Opti` stack is truly ubiquitous throughout AeroSandbox.

	Extensive documentation with examples is provided in `aerosandbox.optimization.opti` - please read this!

* `/numpy/`: One of the coolest things about the `Opti` stack is that it's fast - really, **really** fast. You can solve nonlinear, nonconvex optimization problems with thousands of variables in mere seconds on a laptop, thanks to automatic differentiation (AD) provided by CasADi and modern optimization methods via IPOPT. 

	In order for AD to work, we need to be able to make a list (more precisely, a directed graph) of each mathematical operation (think +, -, *, /, **, log(), fabs(), etc.) that's applied throughout our optimization formulation (some call this list a "trace" in the literature). This means we can't just use NumPy out of the box like we'd like to, because some of its functions break our trace.

	Instead, we need to use a custom math library, which sounds scary at first. However, the AeroSandbox development team has tried to make this as seamless to you as possible - by writing our own NumPy with identical syntax! Here's how this works:

	* `aerosandbox.numpy` imports the entirety of NumPy.
	* For NumPy functions that break our AD trace (e.g. `np.sum()`), we've rewritten our own versions of them. This means:
		* If you pass normal NumPy arrays to these functions, they'll work 100% exactly the same as they would in original NumPy - same result and same speed.
		* If you pass optimization variables to these functions, they'll intelligently switch over to a version of the function that allows us to preserve the AD trace.
	* **So what does this mean for you, dear user?** It means that when working with AeroSandbox, all you need to do is replace `import numpy as np` with `import aerosandbox.numpy as np`, and you're good to go!
	* Caveat: Not all NumPy functions that should be overwritten have been overwritten - we've done our best, but there are *sooo* many obscure NumPy functions! If you get an error on a function you want to use, raise an issue ticket [here](https://github.com/peterdsharpe/AeroSandbox/issues)!

Before continuing, I'd recommend practicing a bit using the `Opti()` stack and `aerosandbox.numpy` to solve a few canonical optimization problems. A good starter problem is finding the minimum of the 2D [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function) - for extra credit, add a constraint that the point has to lie inside the unit circle!

### Modeling Tools

* `/geometry/`: The main goal of AeroSandbox is to make engineering design optimization more harmonious. Engineering design can look like a lot of things depending on what you're trying to design - airplanes, cars, bridges, et cetera. 

	However, all of these objects have one thing in common - geometry! They're all *physical* objects that we're trying to optimize the shape of - remember that **engineering design is the process of finding an optimal mapping from an object's function to that object's form** (in the words of my excellent advisor, Prof. John Hansman).

	The `geometry` folder therefore is a self-contained object-oriented framework for representing the geometry of engineered systems in Python. Right now, it's primarily used for aircraft - you can build a nested data structure all the way from an entire aircraft down to each point that defines the shape of an individual airfoil. Once you have that data structure, you can do all sorts of useful things with - output it to various filetypes, draw it in an interactive 3D window so that you can see it, and pass it to all kinds of analysis tools.

	In the future, we'll hopefully generalize this `geometry` stack with more general representations (`Polygon`, `Polyhedron`, etc.) to represent the geometry of arbitrary types of engineered systems (not just aircraft).

* `/modeling/` is all about one thing - making surrogate models (which is also called "curve fitting" if we're being honest or "machine learning" if we're trying to convince someone to give us money). Jokes aside, the point of `modeling` is to make simple, continuous, automatic-differentiable models based on either:

	* synthetic "data" created by running either:
		* high-fidelity analysis that we'd prefer not to run "online" with the rest of our optimization models (e.g. RANS CFD)
		* black-box analysis that we can't run "online" with the rest of our optimization models (e.g. legacy Fortran code)
	* real data from real-life experiments (the highest-fidelity analysis, in a sense!)

### Discipline-Specific Tools

We also have a collections of aircraft analysis tools categorized by discipline:

* `aerodynamics`: Contains analysis tools related to aerodynamics both in 2D and 3D.
* `propulsion`: Contains analysis tools related to propulsion. Work in progress.
* `structures`: Contains analysis tools related to structures, like beam models.
* `atmosphere`: Contains a few models of standard atmospheres, so that you can get atmospheric properties at different altitudes.
* `library`: Contains tons of simple surrogate models for a wide variety of relevant phenomena. Some models are physics-based, others are empirical fits - all are documented in their docstring.

### Miscellaneous Tools

You can ignore all of these folders.

* `tools`: Miscellaneous tools.
* `visualization`: Tools for making prettier plots.

* `in_progress`: Here be dragons, beware! But seriously, this is work in progress, ignore it.