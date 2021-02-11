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

    a) one of our parameters, aspect ratio, is a nondimensional parameter - this can eliminate scaling issues.

"""