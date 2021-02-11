"""

Let's review before continuing. We've gone through how the `Opti` stack works and how you can use `aerosandbox.numpy`
to create complex nonlinear expressions.

Now, let's put it to use on something a bit more practical than just a math problem.

Let's use some classical aircraft design formulas to minimize the drag on a wing.

Let's say we have a payload of known mass that we want to lift, so we want to design a wing that can provide the
required lift force with as little drag as possible. For pedagogical purposes, assume the wing has no mass and assume
the wing is rectangular.

"""
import aerosandbox as asb
import aerosandbox.numpy as np

# Let's define some constants.
density = 1.225  # Note: AeroSandbox uses base SI units (or derived units thereof) everywhere, with no exceptions.
viscosity = 1.81e-5  # So the units here are kg/(m*s).

# Time to start constructing our problem!
opti = asb.Opti()

aspect_ratio = opti.variable(init_guess=10, log_transform=True)
wing_area = opti.variable(init_guess=1, log_transform=True)

"""
A couple interesting things to note, right off the bat!

-----

First of all, we chose to parameterize the wing sizing with the parameters of aspect ratio and wing area. The 
combination of these two variables uniquely determine the size and shape of our wing planform (recall that the wing 
is assumed to be rectangular). 

However, wing span and wing chord could also uniquely determine wing sizing. Why didn't we use those two as our 
variables? 

Honestly, the truth is that we could, and we would have been totally fine. The aspect ratio - wing area 
parameterization is perhaps a hair nicer because: 

    a) one of our parameters, aspect ratio, is a nondimensional parameter - this can help eliminate scaling issues. 
    
    b) we have some engineering intuition that quantities of interest, such as lift force and induced drag, 
    are perhaps more directly connected in some sense to aspect ratio and wing area than chord and span. 
    
However, like I said, it really doesn't matter too much - just interesting observations. 

-----

The second interesting thing is that we chose to log-transform these variables. This basically means that internally, 
the optimizer is really optimizing log(wing area) rather than the wing area itself. 

One implication of this is that wing area can never go negative (whereas if we had specified it as a normal 
optimization variable, nothing would prevent it from going negative unless we manually constrained it.) Of course, 
wing area should always be positive anyway, so this isn't a bad thing - in fact, it saves us one constraint. Just an 
important observation. 

Log-transforming has some pros and cons, although its usefulness can vary widely problem-to-problem. Some general notes:

Pros: 

* A lot of engineering problems become convex or more-convex when log-transformed, which can lead to faster, 
more stable solutions and global optimality guarantees (although this requires one to first prove convexity, 
which isn't easily without something like Disciplined Convex Programming, which can be overly restrictive.) For more 
on this idea of log-transforming engineering problems, see work by former MIT Prof. Hoburg here: 
https://www2.eecs.berkeley.edu/Pubs/TechRpts/2015/EECS-2015-22.html . 

* Many scaling issues disappear under log-transformation, as many orders of magnitude can be spanned with relatively 
little change in the underlying log-transformed variable. 

* We more faithfully represent our design intent. If we think about how "significant" a design change is, we usually 
think multiplicitively, whether we realize it or not. A 10% drag decrease is roughly equally significant if our 
previous drag was 100 N as if it were 1e6 N. But if someone instead says "We decreased drag by 1000 N", your first 
question is probably "Well what was it before?". 

* You get rid of one constraint for quantities that need to be positive, as you've transformed your constrained problem
into an unconstrained one.

Cons:

* Log-transformation is yet another nonlinearity, so it can make the problem more difficult to solve. This is a big 
one, especially on large (thousands of variables w/ nonlinear shenanigans) problems. 

* If the optimal value of a variable goes to zero or negative, the log-transformed problem will a) not be correct and 
b) probably report that the problem is unbounded.

All in all, I find that it's usually best to start without using log-transforms, and to add it in later if need be. 
We're using it in this example mostly just for illustration and to introduce the concept. 

Let's continue.

"""

# We don't know the best airspeed to fly at, so let's make that an unknown, too.
airspeed = opti.variable(init_guess=30, lower_bound=0)

"""

Airspeed is a quantity that should always be positive - we could have log-transformed this! Here, we invoke a new 
parameter (`lower_bound`) to illustrate that you can apply a lower bound of zero without log-transforming your problem.

A functionally-identical way to represent this would be to declare airspeed as a variable (without the `lower_bound` 
flag) and then to add in `opti.subject_to(airspeed > 0)`. 

"""
