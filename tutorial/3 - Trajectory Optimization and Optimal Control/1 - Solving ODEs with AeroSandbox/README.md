# Solving ODEs with AeroSandbox

Any kind of dynamical simulation or optimal control problem consists of an ordinary differential equation of some sort. Let's talk at the most basic level about how to solve systems of ordinary differential equations using AeroSandbox.

Before we get into examples, let's make some observations that will be helpful to think about:

## All Higher-Order ODEs can be Decomposed into First-Order ODEs

Let's realize that any ODE of higher order (such as our third-order ODE here) can be broken up into a system of first-order ODEs. For example, consider the Falkner-Skan ODE, which is detailed [here](https://en.wikipedia.org/wiki/Falkner%E2%80%93Skan_boundary_layer):

With $f(\eta)$ as a function of $\eta$, and $f' = \frac{df}{d\eta}$, and so on:

$$ f''' + f f'' + \beta (1- (f' )^2) = 0 $$

Then, define three new variables $F, U, S$:

$$ F = f  $$

$$ U = f' $$

$$ S = f''$$

Then, define:

$$ \vec{y} = [F, U, S]'$$

From this, we can represent the derivative as:

$$ \frac{d\vec{y}}{d\eta} = [U, S, -F S - \beta (1 - U^2)]'$$

which is a first-order ODE of three variables $F, U, S$.

## Function Spaces are Infinite-Dimensional

To represent an arbitrary function $f(x)$ takes an infinite number of degrees of freedom. In other words, an infinite amount of information is required to describe any possible function - it is infinite-dimensional.

Furthermore, even functions with a finite domain are infinite-dimensional. Consider a function $f(x)$ defined only in the range $0 < x < 1$. Even here, an infinite amount of information is required if one desires a format that can represent any possible function on this finite domain.

(Contrast this to the finite-dimensional design space of a rectangular prism - any member of the set of all rectangular prisms can be described with just three numbers for length, width, and height.)

Computers do not like infinities, so we need to find some finite-dimensional representation of $f(x)$ - we call this process "discretization", and it necessarily introduces error. (If you don't want to discretize, put down the computer and brush off your calculus of variations!)

There are tons of ways to discretize functions; a few are detailed here:

### Finite Differences

The simplest way, and the one we'll generally use here, involves tracking a finite number of sampled points from the function. Sometimes this approach is referred to as "finite differences" because of how derivatives and integrals are typically computed with this approach.

With this approach, we don't often think about what happens "between" points, but if we had to, we'd probably draw a line between the two neighboring points - a process called "linear reconstruction". This approach works well on functions without discontinuities.

### Finite Element

Suppose we had a finite-difference representation of a function, but we wanted to make it more accurate. One way would be to simply sample more points - that's called "h-adaption". (Remember back to your calculus days when $h$ represented the step size when we were computing a derivative, and we took the limit as $h \rightarrow 0$?)

Another way to increase fidelity would be to increase the *order* of the reconstruction. Instead of using the linear reconstruction above, we could use a quadratic, cubic, or even higher-order polynomial reconstruction. This is called "p-adaption", and this idea of representing a function as a piecewise patchwork of polynomials is called "finite-element" modeling.

### Spectral Methods

If we take this idea of p-adaption to the extremes, we could use only one extremely-high-order polynomial to represent the entire function, completely getting rid of the "piecewise" part. This approach is referred to as a "spectral" or "pseudospectral" method for reasons that will become apparent.

(A note about this approach: if you do this approach, be sure to use a set of [orthogonal polynomials](https://en.wikipedia.org/wiki/Orthogonal_polynomials) such as the Legendre, Chebyshev, or Bernstein polynomials as your basis rather than directly tracking extremely-high-order coefficients - this will eliminate many nasty issues from floating-point arithmetic that will rear their head.)

However, extremely-high-order polynomials are not the only way to represent a function. Just as a Taylor series represents a high-order polynomial, recall that there are other series - one that comes to mind is a Fourier series. So, one could represent a function as a linear combination of sinusoids with increasingly high frequency. (Spectral methods got their name because this sinusoid approach effectively means that you're storing a discrete approximation of the spectrum of the function.) Other series exist as well; the chosen family of "modes of functions" that are to be linearly combined are referred to as the *basis* of the spectral method.

One nice thing about spectral methods is that it can be quite easy to compute exact derivatives and integrals of $f(x)$. Indeed, usually both the derivative and antiderivative (which are both functions of $x$ as well) can also be represented in the exact same basis: this is true for both orthogonal polynomial families and sinusoids. This can significantly reduce discretization error. (Note that error still exists due to the fact that we are representing an infinite-dimensional space in finite dimensions.)

However, one tricky thing about spectral methods is path constraints. It can be quite difficult, if not impossible, to make any guarantees that a function discretely represented by orthogonal polynomials or sinusoids does not exceed some threshold in a given range of inputs. This is a root-finding problem, and indeed for the case of orthogonal polynomials with degree greater than 4, no solution exists (see Abel-Ruffini theorem, 1823) and only approximate iterative methods suffice - this is unacceptable for the purposes of implementing path constraints.

So, we generally do not use spectral or pseudospectral methods to discretize functions in AeroSandbox (although that's not to say that it's not possible).

