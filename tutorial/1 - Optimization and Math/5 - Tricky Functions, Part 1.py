"""

Great, so you've now learned how to solve optimization problems with nonlinear objectives and nonlinear constraints
by using overloaded expressions. To review:

* You can use +, -, *, /, **, etc. for typical arithmetic operations.

* For more complicated expressions like sin(), log(), etc., you can:

    1. `import aerosandbox.numpy as np`

    2. Proceed to use overloaded NumPy-like expressions: `np.sin(x)`, `np.log(x)`, etc.


*But wait!* We can't set you loose on the whole wide world of mathematical optimization just yet! We're actually in a
bit of a dangerous place here - we've given you just enough information to potentially get yourself into a real
pickle if you're not careful.

This is because nonlinear functions can present some real challenges if you're not aware of what you're doing.

Let's give a very very simple example:

Suppose we want to find the value of x that minimizes abs(x). Seems like a simple enough problem.

NumPy (and aerosandbox.numpy, by extension) provides us with the `np.fabs()` ("floating absolute value") function
that lets us do this:

"""
import aerosandbox as asb
import aerosandbox.numpy as np

opti = asb.Opti()

x = opti.variable(init_guess=1)

opti.minimize(np.fabs(x))

"""

Great! This seems easy enough to set up. But let's try to solve it...

(Spoiler, this will throw an error, hence we wrap it in a try/except block in order to catch it)

"""

try:
    sol = opti.solve() # This is going to fail to solve and raise a RuntimeError
except RuntimeError as e:
    print(e)

"""

Whoa! Hang on, why did this fail to solve? The problem looked so easy - so easy, in fact, that we can identify the solution
just by inspection: `x = 0`!

So what happened here?

Well, let's think about this from the perspective of the optimizer:



"""

# TODO write about abs and sqrt and log nastiness
