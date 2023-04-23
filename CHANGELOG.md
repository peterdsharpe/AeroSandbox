AeroSandbox Changelog
=====================

This log keeps track of any major or breaking changes to AeroSandbox. It is not intended to be a comprehensive log of all changes (for that, see [the commit history](https://github.com/peterdsharpe/AeroSandbox/commits/master)), but attempts to catalog all highlights major changes that practical industry users might care about. For this reason, patch versions might have no changelog entries at all.

#### Guidance for writing code that doesn't break

Note that AeroSandbox development makes a good effort to adhere to [semantic versioning](https://semver.org/), so significant breaking changes will be accompanied by a major version bump (e.g., 3.x.x -> 4.x.x). If a breaking change is extremely minor, it might only be accompanied by a minor version bump (e.g., 4.0.x -> 4.1.x). Every effort will be made to keep breaking changes out of patch version (e.g., 4.0.0 -> 4.0.1), but this is impossible to guarantee, especially since Python is dynamically typed and lets you do whatever you want with it. So, it's recommended that serious industrial users write a test or two and run them after updating AeroSandbox. (If you find a breaking change, please [open an issue on GitHub](https://github.com/peterdsharpe/AeroSandbox/issues)!)

Note that AeroSandbox does not consider reordering keyword arguments to be a breaking change - so use keyword arguments whenever possible! (This is a good practice anyway, since it makes your code more readable.) In other words, replace this:

```python
def my_function(a, b):
    pass


my_function(1, 2)
```

with this:

```python
def my_function(a, b):
    pass


my_function(
    a=1,
    b=2
)
```

Also, for at least one version before a breaking change, AeroSandbox development will make a good effort to provide a deprecation warning for any breaking changes.

# In-progress (develop) version

#### 4.0.8

-----

# Latest (master / release), and previous versions

#### 4.0.7

- BREAKING: Critical bugfix in `asb.OperatingPoint.compute_rotation_matrix_wind_to_geometry()`, which in turn affects `asb.OperatingPoint.compute_freestream_direction_geometry_axes()`: A sign was reversed in the rotation matrix calculation for the sideslip angle beta. This resulted in the sign convention for beta (sideslip) being flipped, ultimately causing a sign error. In practice, this bug affected `asb.VortexLatticeMethod` analyses with nonzero beta, causing flipped signs for `CY, Cn`, etc. This bug was introduced in AeroSandbox v4.0.0, and was present in all versions of AeroSandbox v4.0.0-v4.0.6. This bug did not affect `asb.AeroBuildup` or any other aerodynamics analysis methods. (Thanks to @carlitador for catching this.)
- Added `asb.AVL.open_interactive()`, to interactively launch an AVL session.
- Improved `__repr__` methods throughout for better readability.
- Updated `asb.AeroBuildup` to add induced drag on a whole-airplane level, not per-lifting-object. In general, this will result in slightly higher induced drag, and also improves optimization pressure - for example, tandem-wing configurations are no longer unrealistically attractive, since the induced drag scales superlinearly with respect to total lift.
- Improved `asb.Airfoil.generate_polars()` to not error out when a cache filename is provided where the containing directory does not yet exist. (Instead, it now creates the directory.) Fixed a bug in `generate_polars()` that now allows any Reynolds number input list to be specified when calling XFoil.
- Added `asb.Airfoil.plot_polars()` to make polar functions more interpretable.
- Added an optional `color` attribute to `Wing`, `Fuselage`, and `Propulsor` objects, which can be set during instantiation or manually set afterwards. This will control the color that the component is drawn with during visualization; currently only works with the `Airplane.draw_wireframe()` and `Airplane.draw_three_view()` methods, but will be extended to others later.
- Updated pinned CasADi version to 3.6.1. Note that CasADi MUST be v3.6.0 or higher, otherwise automatic differentiation errors may occur due to undefined primitives. This shouldn't be an issue, as this is now pinned in ASB's setup.py.

#### 4.0.6

- Better documentation. Added a hotfix to support CasADi 3.6.0, which made a breaking change by removing `casadi.mod`.

#### 4.0.5

- Better documentation only, improved README, etc.

#### 4.0.4

- Baseline version for changelog (started tracking here)

#### 4.0.2

- [Online-hosted documentation](https://aerosandbox.readthedocs.io/en/master/) set up