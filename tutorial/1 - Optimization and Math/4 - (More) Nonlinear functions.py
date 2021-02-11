"""
In the previous examples, we solved a constrained Rosenbrock problem and the sphere problem.

While both of these problems are nonlinear, they're really just a mix of simple polynomials. This means that we can
express it with overloaded Python operators: [+, -, *, /, **, ...] and so on.

But what if we want to use a function that's not part of simple arithmetic - something like cos(x), for example?
That's what we'll explore here!

-----

One of the coolest things about the `Opti` stack is that it's fast - really, **really** fast. You can solve
nonlinear, nonconvex optimization problems with thousands of variables in mere seconds on a laptop, thanks to
automatic differentiation (AD) provided by CasADi and modern optimization methods via IPOPT.

In order for AD to work, we need to be able to make a list (more precisely, a directed graph) of each mathematical
operation (think +, -, *, /, **, log(), fabs(), etc.) that's applied throughout our optimization formulation (some
call this list a "trace" in the literature). This means we can't just use NumPy out of the box like we'd like to,
because some of its functions break our trace.

Instead, we need to use a custom math library, which sounds scary at first. However, the AeroSandbox development team
has tried to make this as seamless to you as possible - by writing our own NumPy with identical syntax! Here's how
this works:

	* `aerosandbox.numpy` imports the entirety of NumPy.

	* For NumPy functions that break our AD trace (e.g. `np.sum()`), we've rewritten our own versions of them. This
	means:

		* If you pass normal NumPy arrays to these functions, they'll work 100% exactly the same as they would in
		original NumPy - same result and same speed.

		* If you pass optimization variables to these functions, they'll intelligently switch over to a version of the
		function that allows us to preserve the AD trace.

	* **So what does this mean for you, dear user?** It means that when working with AeroSandbox, all you need to do
	is replace `import numpy as np` with `import aerosandbox.numpy as np`, and you're good to go!

	* Caveat: Not all NumPy functions that should be overwritten have been overwritten - we've done our best,
	but there are *sooo* many obscure NumPy functions! If you get an error on a function you want to use,
	raise an issue ticket!

You'll notice that in our last example, we imported `aerosandbox.numpy` in order to use the `sum()` function.

Here, let's do an example with some other functions.
"""
import aerosandbox as asb
import aerosandbox.numpy as np

opti = asb.Opti()

x = opti.variable(init_guess=3)

f = np.exp(  # You can use normal operations from NumPy like this!
    np.cos(  # These functions are intelligently overloading in the background...
        x
    )
)

opti.minimize(f)

opti.subject_to([
    x >= 0,
    x <= np.pi / 2
])
"""
Note another feature we just introduced here: you can give `opti.subject_to()` a list of constraints, not just a 
single constraint like we did before! Often, this makes for cleaner, more readable code. 

Also, note that you can declare variables, constraints, and objectives in any order. As long as they're all set in 
place by the time you call `sol = opti.solve()`, you're good. Speaking of, let's solve! 
"""
sol = opti.solve()

x_opt = sol.value(x)

print(f"x = {x_opt}")
"""

Nice, it solved! The value of x at the optimum turns out to be equal to pi / 2.

Note that there are tons and tons of nonlinear functions you can use - everything from logarithms to vector norms to 
linear solves to eigenvalue decompositions. The list is quite extensive and can be viewed at:
`aerosandbox/numpy/test_numpy/test_all_operations_run.py`, where many of the valid operations are listed.

This would not be possible without tons of hard work by the CasADi team!

"""
