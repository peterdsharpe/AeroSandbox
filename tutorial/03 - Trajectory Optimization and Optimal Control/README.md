# Trajectory Optimization and Optimal Control

Because AeroSandbox can efficiently optimize systems with thousands of design variables and constraints, AeroSandbox can easily be used as a tool for optimizing dynamic engineering systems.

## Optimizing Parameters of Dynamical Systems

At its most basic level, AeroSandbox can be used to optimize parameters of dynamical systems. An example of this would be optimizing the wing area of an airplane throughout a prescribed mission - a single value must be chosen for adequate operation at all flight conditions.

This capability to optimize systems that must operate at many different operating points is so important, especially for aircraft design! With rare exceptions, nearly all aircraft operate at very different operating points throughout their missions. A few examples that have come up from users' AeroSandbox studies:

* Commercial transport aircraft lose a significant fraction (~30%) of their take-off mass over the course of their flight due to fuel burn. This fuel burn leads to a large variation in the cruise $C_L$ over the course of the flight.

* Solar aircraft performance is intensely driven by the day-night energy cycle. Due to the cyclic injection of power into the system, trajectories may involve differing cruise altitudes or airspeeds during day and night. Naturally, any optimized aircraft design must then be robust to this wider variety of flight conditions.

* Rocket-propelled UAVs lose a significant fraction of their mass over the course of their flight. In addition to the changes in $C_L$ as mentioned previously, changes in center of gravity location force designers to allow for a more robust static margin to ensure adequate stability.

All of these kind of get back at the same idea, which is that engineering systems typically must operate well at a variety of conditions. In other words, optimality at a point condition is insufficient - designs must be *robust* across some range of expected operating conditions.

Optimizing across all operating conditions that a system might experience is quite difficult (and indeed, only possible under certain assumptions) - this is because there are an infinite number of unique operating conditions that a system will experience over a fixed time window. (This is kind of harkening back to the idea that function spaces are infinite-dimensional.) There are several ways that this is traditionally handled in engineering design optimization:

### Method 1: Reduce Dynamics to a Point

In the simplest form, we can reduce the dynamics down to a single operating point that is expected to be representative. This is what we did in the [Design Optimization chapter of this tutorial](../02%20-%20Design%20Optimization) - for example, we picked a single value of $\rho$ and $\mu$, assuming that the airplane exclusively operates there.

In the [SimPleAC example of that chapter](../02%20-%20Design%20Optimization/03%20-%20Aircraft%20Design%20-%20SimPleAC.ipynb), you will notice that the $L=W$ constraint is written as:

```python
W_0 + W_w + 0.5 * W_f <= 0.5 * rho * S * C_L * V ** 2,
```

In other words, we assume that the weight of the airplane throughout cruise is equal to the weight of the airplane with half of its fuel burnt. This is a point reduction that is a crude approximation to the [Breguet Range Equation](https://web.mit.edu/16.unified/www/FALL/thermodynamics/notes/node98.html) (and is actually due to the fact that the geometric programming framework that SimPleAC comes from is unable to model logarithms).

### Method 2: Multiple Segments with Point Reduction (Multipoint Optimization based on Segments)

For a step up in fidelity, we could split the mission up into segments; for an aircraft, a prototypical mission might consist of the following segments:

* Takeoff and Climb
* Cruise
* Loiter
* Cruise
* Descent and Landing

For each of these segments, a representative point design could be chosen.

Performance is analyzed at each of the points chosen. In order to obtain an objective function, some reduction is then done by combining performance metrics across design points. (For example, one might choose an objective that is a linear combination of fuel burn rates, weighted by the expected duration of each segment.)

This is a basic example of an idea called *multipoint optimization*, where we optimize a design based on its performance at a finite number of operating conditions.

### Method 3: Full Dynamics

For even more fidelity, we can directly simulate the system throughout its mission, without any kind of a priori segmentation. For an aircraft, this might involve prescribing a trajectory for the airspeed $V(t)$ and altitude $h(t)$ and discretizing time $t$ by 100 discrete points or so.

If we had our aircraft mission above (climb, cruise, loiter, cruise, descend), we could discretize this in two ways:

* Multiphase optimization, where we discretize each mission segment into a series of points. This is good if the segments involve fundamentally different physics or discrete events (for example, an aircraft that drops a payload during the loiter phase)

* Single-phase optimization, where we do away with all labels of climb/cruise/etc., and we just tell the vehicle to go from point A to point B. Phases like climb and cruise will naturally arise out of these dynamics, and often will give us more insight into how to optimal trajectories rather than prescribed dynamics.

In either case, because the full trajectory is simulated, this is suitable not only for simulation but also trajectory optimization and optimal control.

This method of "full dynamics" (along with applications to trajectory optimization and optimal control) is what we'll cover in this chapter.

## Guides

Before learning how to do trajectory optimization and optimal control in AeroSandbox, you should learn the math behind these concepts.

Matthew Peter Kelly does an incredible job introducing these concepts. Copied from [his website](http://www.matthewpeterkelly.com/tutorials/trajectoryOptimization/index.html):

* [Trajectory Optimization Tutorial: Video (YouTube)](https://youtu.be/wlkRYMVUZTs)
* [An Introduction to Trajectory Optimization (.pdf)](https://epubs.siam.org/doi/10.1137/16M1062569)

Russ Tedrake also has an excellent presentation of trajectory optimization topics on his course website for MIT's Underactuated Robotics: [see here](http://underactuated.mit.edu/trajopt.html) (Really, this whole course webpage is gold.)