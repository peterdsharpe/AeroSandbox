AeroSandbox Changelog
=====================

This log keeps track of any major or breaking changes to AeroSandbox. It is not intended to be a comprehensive log of all changes (for that, see [the commit history](https://github.com/peterdsharpe/AeroSandbox/commits/master)), but attempts to catalog all highlights major changes that practical industry users might care about. For this reason, patch versions might have no changelog entries at all.

#### Guidance for writing code that doesn't break

Note that AeroSandbox development makes a good effort to adhere to [semantic versioning](https://semver.org/), so significant breaking changes will be accompanied by a major version bump (e.g., 3.x.x -> 4.x.x). If a breaking change is extremely minor, it might only be accompanied by a minor version bump (e.g., 4.0.x -> 4.1.x). Every effort will be made to keep breaking changes out of patch version (e.g., 4.0.0 -> 4.0.1), but this is impossible to guarantee, especially since Python is dynamically typed and lets you do whatever you want with it. So, it's recommended that serious industrial users write a test or two and run them after updating AeroSandbox. (If you find a breaking change, please [open an issue on GitHub](https://github.com/peterdsharpe/AeroSandbox/issues)!)

Note that AeroSandbox does not consider reordering keyword arguments or adding new ones to be a breaking change - so use keyword arguments whenever possible! (This is a good practice anyway, since it makes your code more readable.) In other words, replace this:

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

-----

# Latest (master / release), and previous versions

#### 4.2.8

- Fixes minor bug that could occur with various functions in `aerosandbox.tools.inspect_tools`, where having `stacklevel` too high could cause the frame to pop off the top of the stack.

#### 4.2.7

- Fixes an AVL bug found by @JohannesKimphove.
- Adds version compatibility with NeuralFoil 0.3.0, which offers large speed improvements using caching of neural network parameters.
- Drops support for Python 3.8, to allow better typing and f-strings.

#### 4.2.6

- Verified compatibility with NumPy 2.0.0+; version pin removed.
- Converted formatting to Black, for improved readability.
- Fixed a bunch of minor things using Ruff; removed dead code branches.

#### 4.2.5

- Improvements to AeroSandbox plotting tools (`aerosandbox.tools.pretty_plots`)
- New features in AeroSandbox profiling tools (`aerosandbox.tools.code_benchmarking`)
- In tutorials, added new validation studies for aerodynamics solvers including wind tunnel data

#### 4.2.4

- Fixed a bug in `aerosandbox.numpy.gradient` which resulted in inconsistent answers with `numpy.gradient` (dropped an argument). Added fix for the `period` argument.
- Adds a new tutorial on visualizing flight test data.
- Adds a new tutorial on aircraft noise impact minimization.
- Improved accuracy of transonic modeling with `asb.Airfoil.get_aero_from_neuralfoil()` based on correlation to full-potential and RANS solutions.
- Adds `softmax_scalefree` and `softmin_scalefree` as new surrogate modeling tools. Similar to softmax and softmin, but with a scale dependent on arguments.
- Adds in ability to do `.leading_edge_radius()` calculations in `asb.Airfoil` and `asb.KulfanAirfoil`, which can be used during optimization to prevent sharp leading edges from forming.
- Adds `softplus` and `swish` functions to support neural surrogate modeling.
- Adds preparations for NumPy 2.0 by fixing soon-to-be-deprecated calls.

#### 4.2.3

- Synchronized update: ASB 4.2.3 and NeuralFoil 0.2.0.

#### 4.2.2

- Added Python 3.12 support, after completing investigation of small floating-point math differences between Python 3.11 and 3.12 (caused by backend changes in NumPy 1.24.3 and 1.25.0).
- Updated version pins in `requirements.txt` to match those used in local pre-commit testing.
- Breaking change from 4.2.1: in `asb.XFoil`, to access boundary layer data (new feature from 4.2.1), give `include_bl_data=True` in the `asb.XFoil` constructor rather than in `asb.XFoil.alpha()`.
- In `asb.XFoil`, added more aggressive suppression of pop-up window generation from X11, allowing it to run in the background more reliably.
- Increased the default `asb.XFoil` paneling resolution to 320 points when `xfoil_repanel=True` is given.
- Fixed gimbal lock singularity issues in the solar energy library (`aerosandbox.library.power_solar`), which could cause optimization to fail when the sun is directly overhead.

#### 4.2.1

- Added ability to extract boundary layer data from XFoil via the `asb.XFoil` interface. See `asb.XFoil.alpha(alpha, include_bl_data=True)` for details. This is useful for boundary layer analysis, especially for high-lift devices.
- Added geometry export capabilities to OpenVSP using the *.vspscript interface. Accessible via `asb.Airplane.export_OpenVSP_vspscript(filename="/path/to/my/file.vspscript")`. To import into OpenVSP, use this function to generate a VSPscript file, and then import it via File -> Run Script... in OpenVSP.
- Overhaul of the airfoil database at `aerosandbox.geometry.airfoil.airfoil_database`. This includes the addition of around 500 new high-quality airfoils, the removal of around 15 exceptionally-low-quality airfoils from the UIUC database, the correction of some airfoil coordinates for transcription errors, and fixes for file encodings that caused errors on Linux.

#### 4.2.0

- Fixed I/O error with XFoil. XFoil outputs may have mislabeled columns if both CP and hinge moments are requested. This is a bug in XFoil, but AeroSandbox now detects and corrects for it.
- Added `Atmosphere.density_altitude()`, which returns the density altitude.
- Added a `Timer` context manager in `aerosandbox.tools.code_benchmarking`, for code profiling.
- Improved visualization of Dynamics-class instances with PyVista; now, an altitude drape, ground plane, and wingtip ribbon can be drawn.
- Fixed a bug in `asb.AeroBuildup` where wave drag could continue to be added via airfoil aerodynamics even if the user specified `include_wave_drag=False`.
- Added a `period` argument to `asb.numpy.diff()` which allows for `diff`ing periodic variables (e.g., heading angle) without wraparound error.
- Added `asb.numpy.integrate_discrete` module, which has higher-order integrators for pre-sampled functions (e.g., useful when doing direct collocation trajectory optimization).
- Added `asb.numpy.integrate` module, which copies `scipy.integrate` and provides `quad` (quadrature) and `solve_ivp` (ODE solve) functions.
- Added `asb.numpy.gradient()` as a new function.
- Fixed a bug in the `asb.numpy.mod()` function, which previous had slight discrepancies against NumPy's implementation for negative inputs.
- Fixed singularities in some dynamics instances when converting to speed-gamma-track representations due to zero-speed points.
- Added new tools for uncertainty quantification of time-series datasets, available in `aerosandbox.tools.statistics`.
- Fixed an issue where trying to put `asb.OperatingPoint` objects into a NumPy object array would result in an infinite recursion (due to iterability of singleton `OperatingPoint` objects). Same fix applied to `asb.MassProperties` and ASB Dynamics-class instances.
- Added support for `order` argument in `asb.numpy.reshape()`.

-----

#### 4.1.7

- Minimal changes; mostly more documentation and backend upgrades (e.g., new PyPI upload process).
- Minor refactoring of `np.reshape` methods from array methods to functions, for clarity.
- Exploring reasons why NumPy 1.25.0 seems to have different optimization behavior than versions preceding. Currently ASB is pinned to NumPy<1.25.0a0; hoping to release pin in upcoming version.

#### 4.1.6

- Fixed I/O error with AVL, where too many digits of precision would cause AVL to read .mass files incorrectly
- Added `asb.Opti.maximize()`, a convenience function that is a simple pass-through to `.minimize(-1 * x)`. Improves readability for users who are less familiar with optimization.
- Added support for the `keepdims` argument in aerosandbox.numpy.linalg.norm().
- Added optimization benchmarks in the tutorials, in `/tutorials/01 - Optimization and Math/01 - Optimization Benchmark Problems/`

#### 4.1.5

- PENDING DEPRECATION added: `asb.Airplane.export_XFLR()` has been renamed to `asb.Airplane.export_XFLR5_xml()`. This clarifies that the output is an XFLR5 XML file (which needs to be imported through the Plane menu in XFLR5), not an XFLR5 .xfl file - a point of user confusion. For now, both will work, but the old name will trigger a warning, and eventually will be removed.
- Added improvements to `asb.LiftingLine` to ensure mixed-backend compatibility.

#### 4.1.4

- Public release of `asb.NonlinearLiftingLine`, which is a new 3D aerodynamics analysis method implemented by Yashil Choony (@yashil99) at Politecnico di Milano. It is implicitly solved (i.e., by iteration), subclassing `asb.ImplicitAnalysis`. This is a nonlinear lifting line theory method (with sweep and dihedral accounting), where the CL(alpha) function used to drive the nonlinear closure loop is taken from NeuralFoil. Fuselage influences can be optionally handled using a source-line (i.e. nonlifting) method. Preliminary testing on full-aircraft configurations indicates good agreement with other solvers, but production use should likely wait until more testing is performed in future versions.
- Added `asb.LiftingLine`, which is an experimental new 3D aerodynamics analysis method. It is explicitly solved by linearizing about the "naive" incidence angle on each surface, then doing a linear solve. Basic method is a lifting line theory method (with sweep and dihedral accounting), where the linearized sectional data is taken from NeuralFoil results for improved accuracy. Fuselage modeling is handled through a semi-empirical method (essentially, falling back on `asb.AeroBuildup`). Preliminary testing on full-aircraft configurations indicates good agreement with other solvers, but production use should likely wait until more testing is performed in future versions. Stability derivatives with respect to alpha and beta appear accurate, but rate derivatives (p, q, r) are not yet tested.
- Added save/load capabilities for AeroSandboxObjects (parent class of Airplane, Wing, Airfoil, etc.) via `AeroSandboxObject.save()` and `asb.load()`.

#### 4.1.3

- Various plotting-related syntax changes to address deprecation warnings in Matplotlib 3.8.0.

#### 4.1.2

- Added cost modeling capabilities for electric aircraft, in `aerosandbox.library.costs`.
- Rework of fuselage aerodynamics calculation to be much more accurate in moment prediction (and stability derivative prediction). This involves a much more precise moment integration based on slender body theory (potential flow around a line-source-doublet), and removing various unnecessary coordinate system conversions.
- Added deprecation warning on `/aerosandbox/geometry/airfoil/default_airfoil_aerodynamics.py`, which is superseded by `asb.Airfoil.get_aero_from_neuralfoil()`.
- BREAKING: In aircraft cost model (the DAPCA IV model in `/aerosandbox/library/costs.py`, adjusted key names on the dictionary output to be consistent with each other. This is a breaking change, but because this is a relatively new feature and buried very deep into the cost library, it's expected that this will affect very few users.

#### 4.1.1

- Added `asb.KulfanAirfoil`, which is a new subclass of `asb.Airfoil` with the underlying parameterization as Kulfan (CST) parameters rather than a coordinate array. Useful for design optimization, as CST parameters can be optimized through, but raw coordinate arrays cannot.
- Added transonic airfoils from TASOPT to the AeroSandbox airfoil database. See them, for example, with `asb.Airfoil("tasopt-c090")`.
- Extend capabilities for `aerosandbox.numpy` functions `np.max`, `np.min`, `np.diag`.
- Added new tutorials on airfoil analysis and airfoil shape optimization. (Roughly in the path `/tutorials/Aerodynamics/...2D Aero Tools/NeuralFoil`)
- Begin work on a unified potential flow model for 3D aerodynamics, which will allow easy switching between different modeling assumptions.
- Added an improvement to Airplane geometry export to STEP thanks to @Zcaic, which now makes a leading-edge line part of each Wing. Improves STEP accuracy.

#### 4.1.0

- In general, uses NeuralFoil throughout instead of XFoil. `asb.Airfoil` no longer needs to have its `generate_polars()` method called before assembling an airplane object. In general, the old "polar functions" of `Airfoil` (e.g., `Airfoil.CL_function()`) are no longer used.
- Added `asb.Airfoil.get_aero_from_neuralfoil()`, which is the standard rapid airfoil aerodynamics analysis method in AeroSandbox now.
- Significant backend change in `asb.AeroBuildup` to use NeuralFoil instead of XFoil for airfoil analysis. This is a significant mathematical change, but it should be transparent to the user. Nevertheless, bumping minor version because this change has the potential to cause differences in power-users' workflows (i.e., if they're hacking Airfoil.xfoil_data or Airfoil.generate_polars() to make custom polars).
- Made a change in `asb.AeroBuildup` within the `.fuselage_aerodynamics()` method that causes fuselage potential flow lift (via slender body theory) to act in the local-wind-normal direction, not halfway between the body-axes-$z$ and wind-axes-$z$ vector as before. The new implementation agrees with Drela's Flight Vehicle Aerodynamics (and intuition), but disagrees with Jorgensen 1977, a NASA TR on fuselage aero. In general, new aero predictions are verified as more realistic (no longer generate a suction force from potential flow in edge cases, satisfying d'Alembert). In general, this results in slightly higher (~5%) fuselage drag prediction, and negligible changes to fuselage lift prediction. For most airplanes this will make negligible difference, but for lifting-body-type aircraft this may be perceptible. As a bonus, optimization should be better-behaved through `asb.AeroBuildup` analyses with fuselages now.
- Added lots of new tools for computing drag of various miscellaneous components in `aerosandbox.library.aerodynamics.components`.

-----

#### 4.0.11

- Fixed a bug in `asb.Airfoil.add_control_surface()` that resulted in duplicated leading-edge nodes.
- Added new ability to draw shaded 3D renderings of airplanes with Matplotlib backend, using Matplotlib 3.7.0 features. Available through `asb.Airplane.draw_three_view()` or `asb.Airplane.draw(backend="matplotlib", ...)`.
- Recast the Kulfan airfoil parameterization problem as a least-squares problem, resulting in a 20x speedup in converting an airfoil coordinate set to CST parameters. Available in `aerosandbox.geometry.airfoil.airfoil_families` in `get_kulfan_parameters()`. New behavior is added as the default, but the old behavior is still available as `get_kulfan_parameters(method="opti")` (as opposed to `"least_squares"`).

#### 4.0.10

- Added lots of new tools and refinements to support airfoil conversion between raw coordinates and Kulfan (CST) coefficients, in `aerosandbox/geometry/airfoil/airfoil_families.py`. Conversions are now bidirectional to/from coordinates and CST coefficients.
- Lots of modifications to the airfoil database stored in `aerosandbox/geometry/airfoil/airfoil_database`. A few (<5) particularly unusable airfoils were removed. With many others (~50), airfoil coordinates were tweaked to prevent self-intersection (i.e., parts of the airfoil where the upper and lower surfaces cross each other). Generally, modifications were near the trailing edge, and on the lower surface when possible (less sensitive boundary layer), and no more than 0.1% of the chord. Several airfoils with duplicated coordinates had these removed. Several airfoils that were egregiously not normalized (e.g., scaled for x/c from 0 to 100, not 0 to 1) were rescaled appropriately. Added new unit tests for airfoil database validity.
- Reworked the `asb.Airfoil.repanel()` method to use `scipy.interpolate.CubicSpline` rather than `PChipInterpolator`. This allows for more precise control over LE/TE boundary conditions, so that it is less likely (but not impossible) to produce self-intersecting geometries when repaneling very-low-resolution airfoils. Upsampled airfoils will be ever-so-slightly different, which may have miniscule but nonzero changes on existing code. Also rewrote code for better readability.
- Improved the plate buckling structural model in `aerosandbox/structures/buckling.py` to take into account Poisson ratio.

#### 4.0.9

- Added turboshaft modeling capabilities in `aerosandbox.library.power_turboshaft`. Added turbine engine database in `aerosandbox/library/datasets/turbine_engines`.
- Added tools for uncertainty quantification of time-series datasets.
- BREAKING: Renamed `aerosandbox.library.propulsion_jet` to `propulsion_turbofan` for clarification. Very minor change, as the old module only contained one function (`mass_turbofan`) that was not used in any other modules or known production code.
- Added turbofan/turbojet modeling capabilities in `aerosandbox.library.power_turbofan`.
- Better docs throughout.

#### 4.0.8

- Added first experimental ability to support black-box functions in optimization, available in `aerosandbox.modeling.black_box`. This is a very early prototype, and the API is subject to change.
- Fixed a bug in `asb.Airplane` auto-generation of s_ref, c_ref, and b_ref that would cause hard-coded s_ref values in the constructor to not be applied to the Airplane instance. Also, adds the capability for auto-generation of reference values from fuselages, if no wing is present.
- Fixed a bug within `asb.AeroBuildup` effective span calculation for symmetric wings that were used to represent doubled vertical stabilizers. Now, if you create an aircraft with twin vertical stabilizers, the Trefftz-plane wake will correctly not carry over between the two vstabs. (If you have a symmetric horizontal stabilizer, it will still carry over - currently this is a function of effective dihedral angle.)
- Fixed Torenbeek wing weight model to use a simpler model for speedbrake / spoiler weight. This resolves a (possible?) error in the wing weight model due to an ambiguity in the source text (Torenbeek's "Synthesis of Subsonic Airplane Design", Appendix C) about what area they were referring to.
- Fixed a bug with uncertainty bootstrapping in `aerosandbox.tools.pretty_plots`, when normalizing input data.

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