"""
Let's do another example with the Rosenbrock problem that we just solved.

Let's try adding a constraint to the problem that we previously solved. Recall that the unconstrained optimum that we
found occured at (x, y) == (1, 1).

What if we want to know the minimum function value that is still within the unit circle. Clearly, (1,
1) is not inside the unit circle, so we expect to find a different answer.

Let's see what we get:

"""
import aerosandbox as asb

opti = asb.Opti()

# Define optimization variables
x = opti.variable(init_guess=1)  # Let's change our initial guess to the value we found before, (1, 1).
y = opti.variable(init_guess=1)  # As above, change 0 -> 1

# Define objective
f = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
opti.minimize(f)

# Define constraint
r = (x ** 2 + y ** 2) ** 0.5  # r is the distance from the origin
opti.subject_to(
    r <= 1  # Constrain the distance from the origin to be less than or equal to 1.
)

"""
Note that in continuous optimization, there is no difference between "less than" and "less than or equal to". (At 
least, not when we solve them numerically on a computer - contrived counterexamples can be made.) So, use < or <=, 
whatever feels best to you.
"""

# Optimize
sol = opti.solve()

# Extract values at the optimum
x_opt = sol.value(x)
y_opt = sol.value(y)

# Print values
print(f"x = {x_opt}")
print(f"y = {y_opt}")

"""
Now, we've found a new optimum that lies on the unit circle.

The point that we've found is (0.786, 0.618). Let's check that it lies on the unit circle.
"""

r_opt = sol.value(r)  # Note that sol.value() can be used to evaluate any object, not just design variables.
print(f"r = {r_opt}")

"""
This prints out 1, so we're satisfied that the point indeed lies within (in this case, on, the unit circle).

If we look close, the value isn't exactly 1 - mine says `r = 1.0000000099588466`; yours may be different. This is due
to convergence error and isn't a big deal.

"""
