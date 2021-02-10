"""
In the previous example, we solved the constrained Rosenbrock problem. This was a 2-dimensional problem,
so we created two variables: x and y.

However, imagine we had a problem with 100 variables. It'd be pretty tedious to create these variables individually
and do the math on each variable one-by-one. (Not to mention the fact that it'd be slow - rule #1 of scientific Python
is to vectorize everything!)

So, what we can do instead is, you guessed it, create variables that are vectors. Think of a vectorized variable as a box that
contains n entries, each of which is a scalar variable.

Let's demonstrate this by finding the minimum of the n-dimensional sphere problem.

"""
import aerosandbox as asb
import aerosandbox.numpy as np # Whoa! What is this? Why are we writing this instead of `import numpy as np`? Don't worry, we'll talk about this in the next tutorial :)

N = 100 # Let's optimize in 100-dimensional space.

opti = asb.Opti()

# Define optimization variables
x = opti.variable(
    init_guess=np.ones(shape=N) # Creates a variable with an initial guess that is [1, 1, 1, 1,...] with N entries.
) # Note that the fact that we're declaring a vectorized variable was *inferred* automatically the shape of our initial guess.

# Define objective
f = np.sum(x ** 2)
opti.minimize(f)

# Optimize
sol = opti.solve()

# Extract values at the optimum
x_opt = sol.value(x)

# Print values
print(f"x = {x_opt}")
