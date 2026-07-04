# AeroSandbox Deep Review — Master List

*Generated 2026-07-04 against `develop` @ `ecd4d676` by a 198-agent parallel review (283 raw
findings, 3 refuted, 263 after de-duplication), each bug-class finding adversarially verified.
UPDATED 2026-07-04 after the implementation pass: **216 of 263 findings are now fixed on the
`cleanup` branch** (194 commits; full test suite green: 761 passed / 6 skipped locally, and the
package test job green across Python 3.10–3.13 in CI). Implemented items have moved to the
[Implemented section](#implemented-on-the-cleanup-branch-2026-07-04) at the bottom; the
47 items below remain open — 36 deliberately left (each carries a
**Status** line explaining why: needs a maintainer decision, changes numerics beyond the verified
bug, or requires a breaking change), and the rest simply unclaimed by any implementation batch.*

**Legend** — every entry lists: location · category · verification status · backwards-compat risk.
"verified" = independent agents reproduced or confirmed the claim against the code (✓ in compact
lists). Remaining counts: 0 critical · 1 high · 10 medium · 36 low.

---

## 1. Critical — remaining

None — all items in this tier were implemented.


---

## 2. High severity — remaining


#### API design

### 1. Dependency floors pinned to latest-at-time patch releases, over-constraining installs

`pyproject.toml:42` · api-design · not independently verified · backwards-compat risk: none

Floors like numpy>=2.2.6, matplotlib>=3.10.7, pandas>=2.3.3, scipy>=1.15.3, casadi>=3.7.2 mirror whatever was newest when `uv add` ran, not the true minimum the code needs (4.2.6 changelog says NumPy 2.0.0+ works). For a widely-used library this forces upgrades and creates resolver conflicts with users' pinned scientific stacks; nothing validates the floors.

**Fix:** Lower floors to actual tested minimums (e.g. numpy>=2.0) and add a CI job using `uv run --resolution lowest-direct` to validate them.

**Status (2026-07-04):** left unimplemented — Per instructions: maintainer decision; lowering floors needs a lowest-resolution CI job to validate.


---

## 3. Medium severity — remaining


#### Bugs (wrong results or crashes)

### NEW. thrust_turbofan() fit coefficient appears unit-inconsistent (~150x high)

`aerosandbox/library/propulsion_turbofan.py` · bug · found during docs build (2026-07-04) · backwards-compat risk: low

While writing the propulsion chapter of the docs book, the author found `thrust_turbofan()`'s
regression coefficient inconsistent with a refit of the bundled dataset: the leading coefficient
should be ~71.9 rather than 12050.7 (~150x discrepancy, consistent with a lbf/N or kg/ton unit slip
in the original fit). `mass_turbofan` and the TSFC models were verified consistent and are used in
the book instead.

**Fix:** Refit the thrust regression against `studies/` source data and pin with a test; until then
treat `thrust_turbofan()` outputs as suspect.

### 2. FuselageXSec.compute_frame() returns non-unit axes for tilted normals, shrinking cross-sections

`aerosandbox/geometry/fuselage.py:854` · bug · verified · backwards-compat risk: none

zg_local is Gram-Schmidt-projected but never renormalized; yg_local = cross(zg, xg) inherits the deficit. Verified: FuselageXSec(xyz_normal=[1,0,1], radius=1) yields perimeter points at radius 0.707 (scaled by cos(tilt)). Affects get_3D_coordinates, meshing, drawing, and CAD export whenever xyz_normal is not x-aligned. Propulsor.compute_frame (propulsor.py:88) has the identical defect.

**Fix:** After projection: `zg_local = zg_local / np.linalg.norm(zg_local)` in both classes.

**Status (2026-07-04):** left unimplemented — Explicitly DO NOT TOUCH per batch instructions; maintainer says the non-unit-axes pattern is plausibly intentional.

### 3. n_gear misread as gear count; Raymer's N_gear is the gear load factor (~3)

`aerosandbox/library/weights/raymer_cargo_transport_weights.py:288` · bug · verified · backwards-compat risk: HIGH (defer to v5)

Raymer defines N_l = N_gear * 1.5 where N_gear is the gear load factor from Table 11.5 (typically ~3), an aircraft-level constant. Code (lines 288, 334; also raymer_general_aviation_weights.py lines 246, 285) uses 'number of landing gear' (defaults 2 and 1), giving N_l = 3 and 1.5 instead of ~4.5, and inconsistent main/nose values — underestimating nose gear mass up to ~46%.

**Fix:** Add a gear_load_factor parameter (default 3.0) and compute N_l = gear_load_factor * 1.5; deprecate the n_gear-based load factor. Semantics change, so defer to v5.

**Status (2026-07-04):** left unimplemented — Explicitly deferred to v5 (semantics/signature change); Raymer's N_gear is a gear load factor (~3), not gear count; equations also use landing weight.

### 4. GA mass_hydraulics uses fuselage width where Raymer's equation uses design gross weight

`aerosandbox/library/weights/raymer_general_aviation_weights.py:458` · bug · verified · backwards-compat risk: HIGH (defer to v5)

Raymer's GA hydraulics equation (5th Ed., Sec. 15.3.3) is W_hyd = K_h * W_dg^0.8 * M^0.5 with W_dg in lb (the K_h values quoted in the code's own comment come from that equation). The code substitutes fuselage_width^0.8, yielding ~0.01 lb for a typical light aircraft instead of a few lb — off by orders of magnitude.

**Fix:** Use (design_mass_TOGW/u.lbm)**0.8; requires adding a design_mass_TOGW parameter and deprecating fuselage_width.

**Status (2026-07-04):** left unimplemented — Explicitly deferred to v5 (requires new parameter); Raymer Eq. 15.55 uses design gross weight, code substitutes fuselage width.


#### Latent bugs & fragile code

### 5. indicated_airspeed() uses incompressible inversion of compressible impact pressure

`aerosandbox/performance/operating_point.py:308` · latent-bug · verified · backwards-compat risk: low

Computes sqrt(2*(P_t - P_s)/rho_0): total pressure is computed compressibly but inverted with incompressible Bernoulli. Standard calibrated airspeed is CAS = a0*sqrt((2/(gamma-1))*((qc/P0 + 1)**((gamma-1)/gamma) - 1)) (standard pitot equation). At M=0.8 sea level the current formula gives ~294 m/s vs. correct 272 m/s (~8% high); error grows with Mach.

**Fix:** Implement the compressible CAS relation above (a0, P0 from sea-level ISA); document that IAS is approximated as CAS.

**Status (2026-07-04):** left unimplemented — As instructed: correct CAS needs compressible (Saint-Venant) inversion, which changes numeric outputs and requires an authoritative source and maintainer decision.

### 6. Import-time side effect: module globally overrides plotly's default renderer to 'browser'

`aerosandbox/visualization/plotly.py:7` · latent-bug · verified · backwards-compat risk: low

`pio.renderers.default = "browser"` runs at module import. aerosandbox/geometry/airfoil/airfoil.py:902 imports this module inside Airfoil.draw(backend='plotly'), so calling that method silently changes the user's global plotly renderer for the whole session — e.g., breaking inline figure rendering in Jupyter/VSCode notebooks for all subsequent plotly usage.

**Fix:** Remove the module-level assignment, or set it only inside spy()/plot_point_cloud, or only when pio.renderers.default is unset.

**Status (2026-07-04):** left unimplemented — Instructed leave: intentional design.


#### Testing gaps

### 7. library/, structures/, and visualization/ subsystems are ~0% tested

`aerosandbox/library/winds.py:1` · testing · not independently verified · backwards-compat risk: none

Coverage run: overall 51%; aerosandbox/library has ~20 modules at 0% (winds, power_solar, costs, mass_structural, field_lengths, all propulsion_*, all weights/raymer_* and torenbeek_weights), structures/buckling.py and tube_spar_bending.py 0%, visualization 0%, weights/mass_properties.py (top-level MassProperties class) only 33%. These are formula-heavy modules where regressions are invisible.

**Fix:** Add spot-check tests pinning known values from the cited references (Raymer, Torenbeek, ISA winds) and unit tests for MassProperties arithmetic (parallel-axis, __add__).


#### API design

### 8. Airplane.export_AVL's include_fuselages parameter has no effect

`aerosandbox/geometry/airplane.py:1016` · api-design · not independently verified · backwards-compat risk: none

export_AVL accepts `include_fuselages: bool = True` but never uses it; the AVL writer unconditionally writes BFILEs for all fuselages (avl.py:596). Passing include_fuselages=False silently exports fuselages anyway. Method also has no docstring, only TODO comments.

**Fix:** Wire the flag through to AVL (or strip fuselages from a copied airplane), and add a docstring.

**Status (2026-07-04):** left unimplemented — Behavior deliberately unchanged per instructions; parameter documented as having no effect; wiring it through the AVL writer is a maintainer design decision.

### 9. No `__all__` in aerosandbox.numpy submodules: typing helpers leak into the public np namespace

`aerosandbox/numpy/__init__.py:49` · api-design · not independently verified · backwards-compat risk: low

The chained star-imports (lines 49-63) leak non-underscore module-level names. Verified present on `aerosandbox.numpy`: `Any`, `Callable`, `Literal`, `Sequence`, `cast` (typing.cast — especially confusing since NumPy 1.x had a different `np.cast`), `overload`, `OrderACF`, `OrderKACF`, and a circular self-reference `np.np`. These autocomplete as public API and invite accidental dependence.

**Fix:** Add explicit `__all__` to each aerosandbox/numpy submodule (or underscore-alias typing imports, e.g. `from typing import cast as _cast`).


#### Modernization

### 10. Publish workflow fires on every master push with no version guard, and uses a long-lived API token

`.github/workflows/publish-on-master-push.yml:7` · modernization · not independently verified · backwards-compat risk: none

Any master push without a version bump fails at `uv publish` (verified: run 15663835892, a README-only push, failed). Workflow also authenticates with secrets.PYPI_API_TOKEN instead of PyPI Trusted Publishing (OIDC), has no `permissions:` block, and pins setup-uv@v4 (current major is much newer).

**Fix:** Trigger on release/tag (or add `uv publish --check-url` guard), switch to Trusted Publishing with `id-token: write`, add least-privilege permissions, bump action versions.

**Status (2026-07-04):** left unimplemented — Per instructions: maintainer decision; recommend version guard (uv publish --check-url or tag trigger) plus PyPI Trusted Publishing instead of long-lived token.


#### Dead code

### 11. Airfoil optimizer script uses removed v3 API Airfoil.xfoil_cseq and stale result keys

`aerosandbox/aerodynamics/aero_2D/airfoil_optimizer/airfoil_optimizer.py:122` · dead-code · not independently verified · backwards-compat risk: none

`airfoil.xfoil_cseq(...)` does not exist on the v4 Airfoil class (no xfoil_* methods remain; grep confirms), and result keys 'Cd'/'Cm' are the v3 casing (v4 XFoil returns 'CD'/'CM'). The script crashes with AttributeError when executed. It also passes deprecated `enforce_continuous_LE_radius` to get_kulfan_coordinates.

**Fix:** Port the script to asb.XFoil(...).cl(...) with current keys, or delete/move it to a tutorials directory.

**Status (2026-07-04):** left unimplemented — Script is dead (v3 xfoil_cseq gone); porting to current XFoil API requires design decisions about an unmaintained study script.


---

## 4. Low severity — remaining


#### Bugs (wrong results or crashes)

- `aerosandbox/aerodynamics/aero_3D/singularities/point_source.py:68` — **Dimensional error in viscous_radius regularization: smoothing radius is sqrt(viscous_radius), not viscous_radius** ✓ — smoothed_x_15_inv is called with x = r_squared (units L^2), but the denominator adds x**2.5 (L^5) to viscous_radius**2.5 (L^2.5) — dimensionally inconsistent. Smoothing kicks in at physical distance r = sqrt(viscous_radius), not r = viscous_radius. LiftingLine (lifting_line.py:1017) and NonlinearLiftingLine (:693) pass viscous_radius=0.0001, so velocities are silently attenuated out to ~1 cm (50% error at 1 cm) instead of 0.1 mm. *Fix:* Change denominator to `x**2.5 + viscous_radius**5` so smoothing activates at r ~ viscous_radius, matching the documented meaning. **Status (2026-07-04): left —** Explicitly instructed to leave; changing the regularization denominator would change numeric outputs of LiftingLine/NonlinearLiftingLine.
- `aerosandbox/library/aerodynamics/unsteady.py:227` — **Wagner-function derivative coefficient typo: 0.00750075 should be 0.0075075** — dW_ds is the derivative of Jones' Wagner approximation used in wagners_function (line 71): d/ds[-0.165 e^{-0.0455 s}] = 0.165*0.0455 e^{-0.0455 s} = 0.0075075 e^{-0.0455 s}. Code has 0.00750075 (digit transposition), a ~0.1% error in that term of the Duhamel integral in calculate_lift_due_to_pitching_profile. *Fix:* Change 0.00750075 to 0.0075075 (or write 0.165 * 0.0455 explicitly). **Status (2026-07-04): left —** Assigned LEAVE: 0.0075075 digit transposition suspected, but confirming intent needs the original Wagner-approximation source; ~0.1% effect.
- `aerosandbox/library/weights/raymer_cargo_transport_weights.py:39` — **Wing/vstab mass uses minimum t/c across sections; Raymer's equations specify root t/c** — mass_wing (line 39) and mass_vstab (line 156), plus the GA-file counterparts, take np.min of all section airfoil thicknesses. Raymer's (t/c) term is the root value; since transport wings are thickest at the root, min() selects the tip t/c and the (t/c)^-0.4 / (t/c)^-0.5 terms overestimate mass ~10-15% versus the cited reference. *Fix:* Use wing.xsecs[0].airfoil.max_thickness() (root section), as torenbeek_weights.py already does; document the choice. **Status (2026-07-04): left —** Explicitly deferred per instructions; Raymer specifies root t/c but code takes min over sections; changing alters mass outputs.
- `aerosandbox/visualization/plotly.py:26` — **spy() destructively mutates the caller's matrix and plots dead all-ones data** ✓ — Line 26 writes log10 magnitudes into the input matrix in place, corrupting the caller's array (verified: input is modified after the call). Then line 33's `val = matrix[sparsity_pattern]` is immediately overwritten by `val = np.ones_like(i_index)` (line 34), so the heatmap z is constant 1 and the log10 computation is entirely dead code. *Fix:* Operate on a copy (matrix = np.array(matrix, dtype=float)); delete one of the two `val` assignments depending on intended intensity (log-magnitude vs. binary sparsity).

#### Latent bugs & fragile code

- `aerosandbox/aerodynamics/aero_3D/aero_buildup_submodels/softmax_scalefree.py:9` — **softmax_scalefree crashes on array-valued inputs under NumPy >= 1.24 (inhomogeneous array)** — `np.array([1e-6] + x)` prepends a scalar to the user's list. If entries of x are ndarrays (e.g., array-valued geometry, which Vectorizable typing permits), NumPy 2.x (project requires numpy>=2.2.6) raises ValueError for the inhomogeneous list. Also, an empty list x reaches np.softmax() with zero args and raises a confusing ValueError (e.g., AeroBuildup on an airplane with no wings/fuselages). *Fix:* Compute softness as `np.max(np.concatenate([np.ravel(np.array(xi)) for xi in ([1e-6] + x)]))*0.01` or per-element maximum; guard len(x)==0 with a clear error. **Status (2026-07-04): left —** Not in batch's assigned items; robustness rewrite requires design judgment about which Vectorizable inputs to support.
- `aerosandbox/aerodynamics/aero_3D/singularities/uniform_strength_horseshoe_singularities.py:126` — **vortex_core_radius applied dimensionally inconsistently across Biot-Savart terms** — smoothed_inv(x) = x/(x^2 + r_c^2) is applied to norm_a (length), to norm_a*norm_b + a_dot_b (length^2, term1), and to norm_a - a_dot_u (~h^2/2|a|, terms 2-3). For term1 the effective bound-leg core radius is ~sqrt(r_c) (e.g. 1e-4 m for VLM's default r_c=1e-8 m), and for the trailing legs it grows as sqrt(2|a|*r_c) downstream — contradicting the docstring's claim that the parameter "governs the radius" of a Kaufmann vortex. *Fix:* Regularize distances instead: replace norms with sqrt(|a|^2 + r_c^2) etc., or scale the added constant per-term to have matching dimensions (r_c^4 for term1). **Status (2026-07-04): left —** Explicitly instructed to leave viscous_radius/vortex_core_radius dimensional questions; changes would alter solver numeric outputs.
- `aerosandbox/common.py:45` — **AeroSandboxObject defines __eq__ without __hash__, making all ASB objects unhashable** ✓ — Defining __eq__ sets __hash__ to None, so every Airplane, Wing, Airfoil, Atmosphere, etc. is unhashable. Verified: hash(asb.Airfoil('naca0012')) raises TypeError; sets, dict keys, and functools.lru_cache over these objects all crash. Introduced in commit f07007fc (2024-03-29), shipped in v4.2.x. *Fix:* Add `__hash__ = object.__hash__` to AeroSandboxObject (restores pre-2024 identity hashing; document that equal-by-value objects may hash differently, acceptable for mutable objects). **Status (2026-07-04): left —** Instructed leave: design decision.
- `aerosandbox/dynamics/point_mass/common_point_mass.py:627` — **Loop variable 'i' shadowed inside draw(): inner axes-color loop reuses outer trajectory index** — The outer loop 'for i in np.linspace(0, len(self)-1, n_vehicles_to_draw)' (line 555) has its index rebound by the inner 'for i, c in enumerate(["r","g","b"])' (line 627). Currently harmless because the outer body no longer uses i afterwards, but any future code appended after the axes block would silently read i=2. *Fix:* Rename the inner loop variable, e.g. 'for ax_index, c in enumerate(["r", "g", "b"])' and use rot[:, ax_index]. **Status (2026-07-04): left —** Explicitly instructed to leave unless a provable bug; nothing reads the outer index after the inner loop, so latent style issue only.
- `aerosandbox/dynamics/rigid_body/rigid_3D/body_euler.py:415` — **Rigid-body speed lacks the gradient-singularity guard used by point-mass classes** — DynamicsRigidBody3DBodyEuler.speed computes (u^2+v^2+w^2)**0.5 without the +1e-200 epsilon that DynamicsPointMass3DCartesian.speed (cartesian.py:106) deliberately adds. Under the CasADi backend, sqrt(0) yields NaN derivatives at the common all-zero initial guess, stalling the optimizer; alpha/beta arctan2d(0,0) at the origin compound this. *Fix:* Add the same +1e-200 term inside the square root, matching the point-mass implementation. **Status (2026-07-04): left —** Explicitly instructed to leave; adding the guard changes numeric output of a public property for non-bug inputs.
- `aerosandbox/library/aerodynamics/unsteady.py:70` — **wagners_function/kussners_function return NaN (with overflow warning) for large negative reduced time** — Both compute expr * np.where(s >= 0, 1, 0); since both operands are evaluated, exp(-0.3*s) overflows to inf for very negative s and inf*0 = NaN. Verified: wagners_function(-5000.0) returns nan plus RuntimeWarnings, instead of the intended 0. *Fix:* Clamp first (s = np.fmax(reduced_time, 0)) or use np.where(s >= 0, expr_with_clamped_s, 0). **Status (2026-07-04): left —** Assigned LEAVE: fix requires a judgment call on overflow clamping strategy across both backends.
- `aerosandbox/library/weights/raymer_cargo_transport_weights.py:94` — **mass_hstab classifies a stabilizer with zero control surfaces as all-moving (+14.3% mass)** — all_moving initializes to True and is only set False when a control surface is found with a partial hinge. A conventional fixed hstab modeled without ControlSurface objects (a very common simplification) therefore gets Raymer's K_uht=1.143 all-moving-tail factor, overestimating hstab mass by 14.3%. Also, the `break` only exits the inner loop. *Fix:* Treat hstabs with no control surfaces as not all-moving (or require explicit user flag); use a flag or for/else to break both loops. **Status (2026-07-04): left —** Explicitly deferred per instructions; Raymer's default is non-all-moving (K_uht=1.0) but code defaults all_moving=True.
- `aerosandbox/optimization/opti.py:1360` — **OptiSol.value on sets: array-valued members crash, frozenset silently becomes set** — The set branch builds {self.value(i) for i in x}; vector variables evaluate to np.ndarray, which is unhashable, raising TypeError. frozenset input also returns a mutable set, contradicting the 'preserve the type as best as possible' comment. *Fix:* Fall back to returning a list (or tuple) when converted members are unhashable; reconstruct frozenset via frozenset(...) when input was frozenset. **Status (2026-07-04): left —** Fixing array-valued set members changes the public return type (set to list/tuple), a design decision; the frozenset-becomes-set half was fixed.
- `aerosandbox/structures/tube_spar_bending.py:261` — **Variable-scaling heuristic is dimensionally inconsistent by one factor of length** ✓ — np.sum(np.trapz(distributed_force)*dy) is the total force F [N], so scale=F*length**4/EI_guess has units m^2, not m; a cantilever tip deflection scales as F*L^3/(3EI). All four scales (u, du, ddu, dEIddu at lines 260, 266, 274, 282) carry the same extra factor of length, inflating scales ~17x for a typical 17 m half-span and degrading IPOPT conditioning. Solution values are unaffected. *Fix:* Divide each scale by length (u: F*L^3/EI, du: F*L^2/EI, ddu: F*L/EI, dEIddu: F); compute F once. **Status (2026-07-04): left —** Explicitly instructed to leave: rescaling changes IPOPT solver numerics/conditioning even though solutions are unaffected.

#### API design

- `aerosandbox/__init__.py:1` — **Top-level namespace leaks `asb.Path` and `asb.version`; no `__all__`** — `from pathlib import Path` (line 1) and `from importlib.metadata import version` (line 57) leak into the package namespace — verified `asb.Path` is pathlib.Path and `asb.version` is importlib.metadata.version (a function needing a distribution name, easily confused with `asb.__version__`). No `__all__` is defined, so `from aerosandbox import *` exports these too. *Fix:* Alias as `from pathlib import Path as _Path`, `from importlib.metadata import version as _version`, and add an explicit `__all__` of intended exports. **Status (2026-07-04): left —** Instructed leave: backwards-compatibility risk.
- `aerosandbox/atmosphere/atmosphere.py:56` — **_valid_altitude_range stored but never enforced; silent extrapolation beyond 0-80 km** — __init__ sets self._valid_altitude_range = (0, 80000), but no library code checks it (only the test file reads it). Users querying e.g. 150 km silently get extrapolated fit values with no warning, despite the model being documented for 0-100 km MAE only. *Fix:* Either document the extrapolation behavior in the class docstring or emit a warnings.warn for NumPy-backend altitudes outside the range (skip for CasADi symbolics). **Status (2026-07-04): left —** As instructed: choosing warning vs documentation for out-of-range altitudes is a design-intent call; warnings could break optimization loops.
- `aerosandbox/geometry/nosecone_shapes/haack.py:4` — **nosecone_shapes package exports nothing and haack functions lack docstrings/typing** — nosecone_shapes/__init__.py is empty, so `haack_series`, `karman`, `LV_haack`, and `tangent` are reachable only via the full submodule path. None of the four public functions have docstrings, the `C` parameter is unannotated, and there are no return annotations. (Formula itself checks out against the standard Haack-series equation.) *Fix:* Re-export the functions in __init__.py; add docstrings (units, C-parameter meaning, normalization R=1 at x/L=1) and type hints.
- `aerosandbox/tools/pretty_plots/__init__.py:33` — **pretty_plots mutates global seaborn theme and mpl rcParams as an import side effect** — Importing the package runs `sns.set_theme(...)` (line 33) and overwrites `mpl.rcParams` figure.dpi/useoffset/negative_linestyle (lines 37-39). Any transitive import restyles a user's unrelated figures with no opt-out or way to re-apply after `mpl.rcdefaults()`. The behavior is the package's purpose, so keep the default, but it should also be exposed as a callable. *Fix:* Wrap the theming in a public `set_theme()`/`apply_style()` function called once at import, and document the side effect in the module docstring. **Status (2026-07-04): left —** Explicitly instructed to LEAVE in the batch assignment.
- `aerosandbox/tools/statistics/time_series_uncertainty_quantification.py:150` — **Unconditional print() in library code paths without verbose gating** — Of 59 print() calls outside __main__ blocks, most are properly gated by `if self.verbose`. Ungated exceptions: bootstrap_fits prints estimated noise stdev (lines 150, 153; the function has no verbose parameter), mses.py:250-251 dumps subprocess stdout/stderr on error regardless of verbosity, xfoil.py:583 prints an interactive-mode notice. *Fix:* Add a `verbose: bool = True` parameter to bootstrap_fits; attach the mses stdout/stderr to the raised exception message instead of printing. **Status (2026-07-04): left —** Adding a verbose parameter is a public signature change and design-intent call, prohibited by batch rules; prints arguably intentional feedback during noise estimation.

#### Performance

- `aerosandbox/library/winds.py:61` — **winds.py loads .npy/.csv datasets and builds interpolation models at import time** — Lines 61-134 and 156-187 run np.load x3, np.genfromtxt, array reshaping, and construct two InterpolatedModel objects at module import. Any `import aerosandbox.library.winds` pays this I/O cost even if only wind_speed_conus_summer_99 is needed, and a missing/corrupt dataset breaks import entirely. *Fix:* Lazily build winds_95_world_model / tropopause_altitude_model on first call (cached factory or module __getattr__), keeping existing names as accessors. **Status (2026-07-04): left —** Assigned LEAVE: lazy loading changes public module-level attribute semantics; design-intent decision.

#### Modernization

- `aerosandbox/aerodynamics/aero_3D/aero_buildup_submodels/softmax_scalefree.py:5` — **Duplicate, undocumented softmax_scalefree shadows the public asb.numpy.softmax_scalefree** — aerosandbox.numpy already exposes softmax_scalefree (aerosandbox/numpy/surrogate_model_tools.py:135) with a documented *args/relative_softness API (norm-based scaling). This submodule version has no docstring, a different signature (list argument), and subtly different semantics (max-based scaling with a 1e-6 floor). Two same-named public-ish functions with different behavior is a maintenance and confusion hazard. *Fix:* Add a docstring now; in v5 (or after validating numerics), delegate to np.softmax_scalefree or rename to clarify the max-based, floored variant. **Status (2026-07-04): left —** Dedupe not done: variants numerically non-identical (max-based vs L2-norm softness); merging would change AeroBuildup outputs; documented the intentional difference instead.
- `aerosandbox/numpy/linalg.py:268` — **Exception chaining lost in 37 raise-without-from sites; worst prints exception to stdout** — Ruff B904 flags 37 `raise` statements inside `except` blocks without `from e`, losing tracebacks. Worst example: linalg.py:268 does `print(e)` to stdout then raises a ValueError with no chain, so the real cause is not attached to the exception. Also 29 B028 warnings.warn calls without stacklevel, which misattribute warning locations to library internals. *Fix:* Use `raise ValueError(...) from e` (remove print(e)); add stacklevel=2 to warns; consider enabling ruff B rules in pyproject.toml.
- `pyproject.toml:11` — **License metadata uses deprecated table form instead of PEP 639 SPDX string** — `license = { text = "MIT" }` plus the 'License :: OSI Approved :: MIT License' classifier is the pre-PEP-639 style; current hatchling deprecates both in favor of an SPDX expression and license-files, and future build backends will warn or error. *Fix:* Use `license = "MIT"` and `license-files = ["LICENSE.txt"]`; drop the license classifier. **Status (2026-07-04): left —** Per instructions: PEP 639 migration is a build-chain risk; left as license table form.
- `pyproject.toml:102` — **No [tool.pytest.ini_options] config and pytest-cov installed but coverage never measured in CI** — There is no pytest configuration: no testpaths, minversion, or filterwarnings policy (93 warnings pass silently, including the pytest-9-breaking return-not-None pattern). pytest-cov is a declared test dependency but CI never runs coverage, so the 51% overall figure is invisible to maintainers. *Fix:* Add [tool.pytest.ini_options] with testpaths='aerosandbox', selected filterwarnings=error entries, and add --cov=aerosandbox with a report/upload step in CI. **Status (2026-07-04): left —** Per instructions: suggestion only; recommend testpaths and filterwarnings config, left for maintainer.

#### Type hints

- `aerosandbox/geometry/airfoil/airfoil.py:240` — **Shared mutable ndarray defaults and missing annotations on public signatures** — generate_polars (alphas, Res), plot_polars, local_camber/local_thickness, max_camber/max_thickness, and KulfanAirfoil.upper/lower_coordinates all use module-level np arrays as defaults (shared mutable state; one in-place edit corrupts all future calls). generate_polars' alphas/Res are also unannotated and crash on plain lists (`Res.min()`, `attached_alphas_to_use.min()`). *Fix:* Use None defaults constructed inside functions (or tuples), add type hints, and `np.asarray` list inputs in generate_polars.

#### Documentation & docstrings

- `CONTRIBUTING.md:117` — **Docstring style is inconsistent across public API: Google, numpydoc, reST, and freeform mixed** — CONTRIBUTING prescribes Google-style, and most modules comply (opti.py, wing.py, aero_buildup.py), but common.py and numpy/calculus.py use numpydoc, while library/aerodynamics/viscous.py, structures/legacy/beams.py, and OperatingPoint.reynolds use Sphinx reST `:param:` fields. Sampled ~25 public functions/20 modules: many public members are freeform or missing (Airfoil.get_aero_from_neuralfoil, MassProperties.inertia_tensor, most DynamicsPointMass3DCartesian methods). *Fix:* Incrementally convert reST/numpydoc docstrings to Google style; napoleon already renders both so this is zero-risk.
- `aerosandbox/atmosphere/thermodynamics/choked_flow.py:4` — **mass_flow_rate() has no docstring, units, reference, or type hints** — Public function computing compressible-flow mass flow (formula verified correct: mdot = A*P_t*sqrt(gamma/(R*T_t))*M*(1+(gamma-1)/2*M^2)^(-(gamma+1)/(2(gamma-1)))) documents nothing: no units (SI assumed), no arg descriptions, no return description, no annotations. *Fix:* Add docstring with SI units per arg, return units [kg/s], a reference (e.g., Anderson, Modern Compressible Flow), and Vectorizable/float annotations.
- `aerosandbox/performance/operating_point.py:10` — **OperatingPoint, a top-level asb export, has no class docstring** — `asb.OperatingPoint` is one of the primary user-facing classes (paired with Airplane in every aero analysis) but the class has no docstring, so `help(asb.OperatingPoint)` and the generated API docs show nothing about alpha/beta being in degrees, velocity units, or axes conventions — exactly the unit pitfalls the README promises docstrings will clarify. *Fix:* Add a class docstring covering constructor parameters, units (alpha/beta in degrees), and axes conventions.
- `aerosandbox/structures/legacy/simple_beam_opt.py:43` — **TODO triage: 76 TODO/HACK comments; author-flagged wrong physics in legacy beam models** — 76 TODO/FIXME/HACK comments in library code. Highest-value: simple_beam_opt.py:43 and simple_beam_opt_daedalus_calibration.py:42 use isotropic G = E/2/(1+nu) with nu=0.5 for CFRP with the author's own note 'TODO fix this!!! CFRP is not isotropic!' (composite shear modulus is not derivable from the isotropic relation); both live in structures/legacy/. *Fix:* Add a docstring warning about the isotropic-G approximation in the legacy beam modules, or deprecate them formally; burn down remaining TODOs opportunistically.

#### Dead code

- `.gitignore:1` — **Stale local build artifacts: dist/ holds old releases; AeroSandbox.egg-info/ is a setuptools-era leftover** — dist/ (aerosandbox-4.2.8 wheel, 4.2.9 sdist) and AeroSandbox.egg-info/ exist in the repo root. Both are correctly gitignored and untracked, so no repo pollution, but egg-info will never be regenerated under hatchling and a stale one can feed wrong metadata to pip during editable installs. CI publishes from a clean checkout, so releases are unaffected. *Fix:* Delete dist/ and AeroSandbox.egg-info/ locally; keep the .gitignore entries. **Status (2026-07-04): left —** Artifacts are untracked and already gitignored; they exist only in the maintainer's working directory, which must not be touched.
- `aerosandbox/aerodynamics/aero_2D/IBL2.py:21` — **IBL2 is an empty stub: __init__ body is `pass`, silently doing nothing** — IBL2's __init__ accepts streamwise_coordinate, edge_velocity, viscosity, theta_0, H_0 and does nothing with them; instantiating it silently succeeds and produces a useless object. It is not exported in aero_2D/__init__.py but is importable and looks like a working analysis. *Fix:* Raise NotImplementedError in __init__ (with a docstring note that it is a placeholder), or remove the stub. **Status (2026-07-04): left —** Fix (raise NotImplementedError vs delete) is a design-intent judgment call that would change instantiation behavior; left as-is per instructions.
- `aerosandbox/aerodynamics/aero_2D/mses.py:81` — **MSES constructor param mset_alpha is stored but never used** — self.mset_alpha is assigned in __init__ but run() always meshes at alphas[0] (line 248) and re-meshes at next_alpha; the user-supplied mset_alpha has no effect, silently ignoring an apparently functional public parameter. *Fix:* Use self.mset_alpha for the initial mset() call (falling back to alphas[0] if None), or deprecate the parameter with a warning. **Status (2026-07-04): left —** mset_alpha stored but unused; wiring it in would change MSES meshing behavior; documented as 'currently unused' in the rewritten docstring instead.
- `aerosandbox/atmosphere/atmosphere.py:18` — **Unused module constant effective_collision_diameter; unused PerfectGas attribute** — effective_collision_diameter in atmosphere.py is referenced nowhere in the repo (mean_free_path uses the viscosity-based formula instead). Similarly PerfectGas.effective_collision_diameter (gas.py:51) is stored and copied through process() but never used in any computation. *Fix:* Remove the atmosphere.py constant (or mark deprecated since it is importable); either implement PerfectGas.mean_free_path using the attribute or document it as reserved. **Status (2026-07-04): left —** As instructed: both symbols are unused but public importable API; removal is a backwards-compat risk with no correctness gain.
- `aerosandbox/library/weights/torenbeek_weights.py:407` — **torenbeek mass_fuselage is an unfinished stub that always raises NotImplementedError** — Public function mass_fuselage computes two fineness_ratio() results, discards them, then unconditionally raises NotImplementedError; it has no docstring and appears complete in autocomplete/docs alongside working siblings. power_nuclear_rtg.po210_specific_power has the same discarded-expression stub pattern. *Fix:* Either finish the Torenbeek Appendix D implementation or remove the dead statements and add a docstring stating it is not yet implemented, pointing to mass_fuselage_simple. **Status (2026-07-04): left —** Assigned LEAVE: still an unfinished stub raising NotImplementedError; completing it requires implementing Torenbeek Appendix D, a design-intent judgment call.
- `aerosandbox/numpy/calculus.py:13` — **Ruff triage: 3 F401 unused imports are the only default-rule violations** — `uv run ruff check aerosandbox --statistics` reports only 3 F401s: unused `typing.cast` (calculus.py:13), unused `_CasADiType` (linalg.py:12), plus one in a test file (out of scope). All autofixable with --fix. *Fix:* Run `ruff check aerosandbox --fix` to remove the two library-file unused imports.
- `aerosandbox/structures/legacy/simple_beam_opt.py:151` — **stress_von_mises_squared is misnamed, not von Mises, and dead code** — In both simple_beam_opt.py (line 151) and simple_beam_opt_daedalus_calibration.py (line 139), stress_von_mises_squared = np.sqrt(stress_axial**2 + 0*stress_shear**2) is (a) a square root, not a square; (b) not the von Mises criterion (which is sqrt(sigma^2 + 3*tau^2)); (c) has shear zeroed out; and (d) is never used (stress = stress_axial). *Fix:* Delete the variable in both scripts, or implement sqrt(sigma^2 + 3*tau^2) and actually use it. **Status (2026-07-04): left —** Instructed to leave and report; misnamed variable is unused dead code in a legacy demo script; renaming/removing is a judgment call.
- `qodana.yaml:29` — **qodana.yaml is orphaned — no workflow runs Qodana** — .github/workflows/ contains only run-pytest.yml and publish-on-master-push.yml; nothing references Qodana, so this starter-profile config (with an unpinned jetbrains/qodana-python:latest linter) is dead configuration, and ruff already covers linting per pyproject. *Fix:* Delete qodana.yaml, or add a qodana workflow if JetBrains analysis is still wanted. **Status (2026-07-04): left —** Per instructions: report only; orphaned config with unpinned linter image; ruff already covers linting.

---

## 5. Deferred to v5 (requires breaking changes)

3 findings need breaking changes; repeated here for planning. The Raymer items were
source-verified against the actual text during the implementation pass (notes below).

- `aerosandbox/library/weights/raymer_cargo_transport_weights.py:288` — **n_gear misread as gear count; Raymer's N_gear is the gear load factor (~3)** ✓ — Raymer defines N_l = N_gear * 1.5 where N_gear is the gear load factor from Table 11.5 (typically ~3), an aircraft-level constant. Code (lines 288, 334; also raymer_general_aviation_weights.py lines 246, 285) uses 'number of landing gear' (defaults 2 and 1), giving N_l = 3 and 1.5 instead of ~4.5, and inconsistent main/nose values — underestimating nose gear mass up to ~46%. *Fix:* Add a gear_load_factor parameter (default 3.0) and compute N_l = gear_load_factor * 1.5; deprecate the n_gear-based load factor. Semantics change, so defer to v5.
  
  **Source verification (Raymer 6th ed. PDF, 2026-07-04):** terminology p. 578: 'N_l: ultimate landing load factor; = N_gear x 1.5'; Eq. 11.11 (p. 356): 'N_gear = L/W_landing', 'typically equals three'; Table 11.5 gear load factors: large bomber 2.0-3, commercial 2.7-3, GA 3, USAF fighter 3.0-4, Navy fighter 5.0-6. Confirms the code misreads N_gear as gear count.
- `aerosandbox/library/weights/raymer_general_aviation_weights.py:458` — **GA mass_hydraulics uses fuselage width where Raymer's equation uses design gross weight** ✓ — Raymer's GA hydraulics equation (5th Ed., Sec. 15.3.3) is W_hyd = K_h * W_dg^0.8 * M^0.5 with W_dg in lb (the K_h values quoted in the code's own comment come from that equation). The code substitutes fuselage_width^0.8, yielding ~0.01 lb for a typical light aircraft instead of a few lb — off by orders of magnitude. *Fix:* Use (design_mass_TOGW/u.lbm)**0.8; requires adding a design_mass_TOGW parameter and deprecating fuselage_width.
  
  **Source verification (Raymer 6th ed. PDF, 2026-07-04):** Eq. 15.55 (p. 576): 'W_hydraulics = K_h W_dg^0.8 M^0.5' with W_dg = design gross weight in lb; the code's fuselage_width substitution is unsupported by the source.
- `aerosandbox/dynamics/point_mass/point_3D/cartesian.py:225` — **add_force default 'axes' differs per subclass ('earth'/'wind'/'body'), contradicting base-class signature** — The abstract base declares axes: Literal[...] = "wind", but DynamicsPointMass3DCartesian defaults to "earth", speed-gamma classes to "wind", rigid-body to "body". Code written against the base-class docs silently applies forces in the wrong frame when switching dynamics classes. Subclass overrides also drop the Literal annotation entirely. *Fix:* Document each subclass's native default in its docstring, restore Literal annotations now; unify defaults (or require explicit axes) in v5 since changing them silently alters physics.

---

## Implemented on the `cleanup` branch (2026-07-04)

216 findings fixed across 194 commits (each commit: one logical fix + regression
test; authored by Peter Sharpe). Grouped by severity; commit hashes refer to `cleanup`.


### Critical (5)

- `aerosandbox/aerodynamics/aero_3D/lifting_line.py:589` — LiftingLine.run() crashes with NameError on any airplane with a symmetric wing (`01ee4b00`)
- `aerosandbox/aerodynamics/aero_3D/nonlinear_lifting_line.py:211` — NonlinearLiftingLine.run() crashes with NameError on any symmetric wing (same TYPE_CHECKING regression) (`01ee4b00`)
- `aerosandbox/aerodynamics/aero_3D/test_aero_3D/test_vlm/test_airplane_optimization.py:38` — CI on develop is red: VLM regression (IndexError) introduced by latest numpy-typing commit (`c83ecdf7`)
- `aerosandbox/geometry/airfoil/airfoil.py:1` — Base install cannot `import aerosandbox`: module-level plotly import of an optional dependency (`ef7b55bf`)
- `.github/workflows/run-pytest.yml:30` — CI has been red on develop since 2025-12-25; two genuine test failures ship unnoticed (`c83ecdf7`)

### High (12)

- `aerosandbox/aerodynamics/aero_3D/avl.py:405` — AVL wrapper silently ignores all ControlSurface.deflection values; first control always deflected 1.0 deg (`901ccc45`)
- `aerosandbox/dynamics/point_mass/common_point_mass.py:502` — draw() crashes on NumPy 2.x: float() on 1-element array when auto-scaling vehicle model (`d7a23dab`)
- `aerosandbox/geometry/openvsp_io/asb_to_openvsp/fuselage_vspscript_generator.py:73` — OpenVSP fuselage export double-counts the first xsec's position offset (`26bf918e`)
- `aerosandbox/library/gust_pitch_control.py:52` — TransverseGustPitchControl crashes on construction: np.sum() called on a generator expression (`4e5a37d3`)
- `aerosandbox/library/weights/raymer_cargo_transport_weights.py:413` — mass_nacelles returns pounds-mass, not kg: missing final `* u.lbm` conversion (`cf27ad64`)
- `aerosandbox/library/weights/raymer_cargo_transport_weights.py:582` — mass_instruments and mass_hydraulics multiply L_f*B_w where Raymer specifies (L_f + B_w) (`b8794d8e`)
- `aerosandbox/library/weights/raymer_general_aviation_weights.py:251` — GA landing-gear length term divided by 12 twice, underestimating gear mass 2.8-8x (`a4abde13`)
- `aerosandbox/geometry/wing.py:1566` — WingXSec deprecated control_surface_* kwargs silently ignored: `in locals()` never true (`0e168d4f`)
- `.github/workflows/publish-on-master-push.yml:24` — PyPI publish workflow has no test gate; any master push publishes even with failing tests (`3b09eb43`)
- `aerosandbox/aerodynamics/aero_3D/test_aero_3D/test_lifting_line.py:1` — LiftingLine and NonlinearLiftingLine (public top-level solvers) have zero executable tests (`b2267a75`)
- `INSTALLATION.md:12` — Install docs state wrong minimum Python (3.8) and link to nonexistent setup.py (`f11086d1`)
- `requirements.txt:1` — requirements.txt is stale and mutually unsatisfiable with pyproject.toml (`fcf6a57d`)

### Medium (87)

- `.readthedocs.yaml:6` — Read the Docs build is guaranteed to fail: Python 3.8 pinned but package requires >=3.10 (`409131d6`)
- `aerosandbox/aerodynamics/aero_2D/mses.py:249` — MSET failure silently swallowed unless stderr contains the X11 'BadName' string (`7c478354`)
- `aerosandbox/aerodynamics/aero_2D/mses.py:323` — MSES 'terminate' behavior silently ignored when verbosity=0 due to misindented break (`b69f44ca`)
- `aerosandbox/aerodynamics/aero_2D/xfoil.py:225` — XFoil(hinge_point_x=None) crashes despite docstring saying None disables hinge moment (`b219f281`)
- `aerosandbox/aerodynamics/aero_2D/xfoil.py:626` — alpha()/cl() unconditionally sort inputs, contradicting documented start_at=None behavior (`2cc9c05f`)
- `aerosandbox/aerodynamics/aero_3D/avl.py:595` — AVL.write_avl(filepath=None) crashes despite docstring promising string return; None-check is dead code (`820256c5`)
- `aerosandbox/atmosphere/_isa_atmo_functions.py:11` — ISA model uses g=9.81 instead of standard g0=9.80665; docstring claims 'exactly reproduced' (`6c49dd6b`)
- `aerosandbox/atmosphere/thermodynamics/gas.py:240` — PerfectGas.process() with enthalpy_addition_* yields pressure=None or silently ignores input (`f75384f3`)
- `aerosandbox/common.py:78` — AeroSandboxObject.__eq__ raises ValueError when array attributes have different shapes (`32c6e609`)
- `aerosandbox/dynamics/point_mass/common_point_mass.py:470` — draw() with a filepath string: pv.read() result discarded, then AttributeError (`98918288`)
- `aerosandbox/dynamics/point_mass/common_point_mass.py:539` — draw() crashes for trajectories of length 1-3: spline degree k not reduced below point count (`c81036de`)
- `aerosandbox/dynamics/point_mass/common_point_mass.py:778` — kinetic_energy and rotational_kinetic_energy raise AttributeError on all point-mass classes (`175b0bd4`)
- `aerosandbox/dynamics/rigid_body/common_rigid_body.py:64` — rotational_kinetic_energy ignores products of inertia (Ixy, Iyz, Ixz) (`0814ba71`)
- `aerosandbox/geometry/airfoil/airfoil.py:106` — Airfoil constructor transposes the shape tuple, not the coordinates, for 2xN input (`f9e403ee`)
- `aerosandbox/geometry/airfoil/airfoil.py:1003` — TE_angle() dot-product typo returns wrong trailing-edge angle (`95452779`)
- `aerosandbox/geometry/airfoil/airfoil.py:1296` — add_control_surface() raises AttributeError for any normally-constructed Airfoil (`f6a31825`)
- `aerosandbox/geometry/airfoil/airfoil_families.py:416` — get_kulfan_parameters(method='opti') crashes with NameError: 'asb' not defined (`1ec8925f`)
- `aerosandbox/geometry/fuselage.py:257` — fineness_ratio(assumed_shape="sears-haack") is 2x too large (uses radius, not diameter) (`dc9badbb`)
- `aerosandbox/geometry/polygon.py:263` — Polygon.J() computes polar moment about origin, not centroid as documented (`094183fc`)
- `aerosandbox/geometry/test_geometry/test_cadquery_export.py:22` — test_cadquery_export references non-existent asb.np, failing every CI run since Nov 2025 (`2bc5eb8e`)
- `aerosandbox/geometry/wing.py:1187` — Wing.mesh_line with iterable x_nondim and add_camber=True crashes or gives wrong points (`5b59c6e5`)
- `aerosandbox/library/aerodynamics/unsteady.py:132` — Gust-entry offset mixes dimensional chord with nondimensional reduced time (`ad29b8c1`)
- `aerosandbox/library/aerodynamics/unsteady.py:308` — pitching_through_transverse_gust crashes with float angle_of_attack despite advertised support (`d5756f75`)
- `aerosandbox/library/propulsion_small_solid_rocket.py:163` — thrust_coefficient crashes with default arguments; docstring promises None handling (`eef246bd`)
- `aerosandbox/library/propulsion_turbofan.py:139` — Nacelle exit-area weight term uses `2.5 * 0.0363` instead of `2.5 + 0.0363` per TASOPT (`04b2e784`)
- `aerosandbox/modeling/interpolation.py:128` — InterpolatedModel.x_data and y_data use mismatched flattening orders, scrambling point pairing (`494d63e5`)
- `aerosandbox/modeling/interpolation_unstructured.py:92` — Unsorted 1D point-cloud input constructs successfully but crashes at call time (`5ced1786`)
- `aerosandbox/modeling/interpolation_unstructured.py:93` — 1D/structured shortcut path silently discards fill_value and interpolated_model_kwargs (`5ced1786`)
- `aerosandbox/modeling/surrogate_model.py:121` — SurrogateModel.plot() crashes for 1D models with dict x_data: dict_keys is not subscriptable (`b41c8e08`)
- `aerosandbox/numpy/__init__.py:55` — trapz deprecation message directs users to `asb.numpy.integrate_discrete_intervals`, which is not importable (`c93c7277`)
- `aerosandbox/numpy/arithmetic_dyadic.py:137` — mod() CasADi branch returns divisor instead of 0 when x1 is a negative exact multiple (`71f063ab`)
- `aerosandbox/numpy/arithmetic_monadic.py:92` — mean(axis=None) on a 2D CasADi matrix returns a vector, not a scalar (`ee47cbe0`)
- `aerosandbox/numpy/calculus.py:65` — diff() with CasADi input ignores n: loop body uses `a` instead of `result` (`423fbcc8`)
- `aerosandbox/numpy/integrate.py:4` — `aerosandbox.numpy.integrate` attribute is scipy.integrate, shadowing the ASB submodule (`76ca1067`)
- `aerosandbox/numpy/integrate.py:286` — solve_ivp CasADi backend silently ignores t_eval (`677a8ede`)
- `aerosandbox/numpy/interpolate.py:41` — interp() with period and CasADi input mishandles x outside [0, period) (negative x) (`5fff9dc5`)
- `aerosandbox/numpy/linalg.py:254` — norm() returns wrong matrix norms for ord=1 and ord=2 under CasADi backend (`0d144c2f`)
- `aerosandbox/numpy/linalg.py:262` — norm() CasADi branch silently ignores axis for ord=inf (and 'fro') (`0d144c2f`)
- `aerosandbox/numpy/linalg.py:276` — norm(..., keepdims=True) crashes with TypeError for CasADi matrix input (`0d144c2f`)
- `aerosandbox/numpy/spacing.py:203` — geomspace with CasADi inputs returns linearly-spaced, not geometrically-spaced values (`1fee0f8c`)
- `aerosandbox/numpy/surrogate_model_tools.py:289` — sigmoid(sigmoid_type="logistic") raises ValueError due to `== ("tanh" or "logistic")` (`6883ef61`)
- `aerosandbox/optimization/opti.py:421` — Constraint declaration tracking off by one frame on Python 3.12+ for constraint lists (`37f8d88b`)
- `aerosandbox/structures/buckling.py:192` — plate_buckling_critical_load double-counts the pi^2/(12(1-nu^2)) factor; results ~9.5% low (`23b607f3`)
- `aerosandbox/structures/tube_spar_bending.py:260` — TubeSparBendingStructure crashes on default (zero) or net-downward distributed load (`ed9a8c7a`)
- `aerosandbox/tools/pretty_plots/formatting.py:433` — show_plot(legend_inline=True) crashes with IndexError if any axes has no lines (e.g., colorbar) (`051651ff`)
- `aerosandbox/tools/pretty_plots/plots/pie.py:41` — pie() crashes with ValueError when sort_by is a NumPy array (documented-valid input) (`b15840fe`)
- `pyproject.toml:95` — Dev dependency "ptest>=2.0.3" is a typo for pytest; installs unrelated abandoned package (`d62b9fe4`)
- `aerosandbox/aerodynamics/aero_3D/lifting_line.py:874` — bound_leg_YZ aliases vortex_bound_leg; in-place zeroing corrupts self.vortex_bound_leg after run() (`505fa63d`)
- `aerosandbox/geometry/airfoil/airfoil.py:337` — generate_polars crashes when cache_filename has no directory component (`d98ddb58`)
- `aerosandbox/geometry/airfoil/airfoil_families.py:226` — Deprecated kwarg mutates get_kulfan_coordinates' shared mutable default array (`91965854`)
- `aerosandbox/geometry/airfoil/kulfan_airfoil.py:39` — KulfanAirfoil.__init__ wipes all global warning filters via warnings.resetwarnings() (`67e18151`)
- `aerosandbox/geometry/fuselage.py:97` — Deprecated `symmetric` kwarg check uses locals(); never fires, arg silently swallowed (`ca9b71d2`)
- `aerosandbox/geometry/polygon.py:142` — Polygon.perimeter() omits the closing edge for non-closed coordinate lists (`f7e2b5d2`)
- `aerosandbox/library/aerodynamics/unsteady.py:54` — calculate_reduced_time silently truncates to integers when `time` is an int array (`5e3ba3cc`)
- `aerosandbox/library/field_lengths.py:180` — field_length_analysis_torenbeek divides by unguarded climb angle; negative/zero T/W margin gives nonsense (`5168ea5d`)
- `aerosandbox/modeling/fitting.py:250` — In-place-mutation detection broken for dict x_data: zip iterates keys, not arrays (`bf81d6c9`)
- `aerosandbox/numpy/interpolate.py:193` — interpn all-zero bspline workaround returns zeros with shape of xi, not of the result (`013bf65e`)
- `aerosandbox/numpy/interpolate.py:223` — interpn with fill_value=None mutates the caller's xi array in place (`5af3c208`)
- `aerosandbox/optimization/opti.py:359` — log_transform variable with nonpositive lower_bound silently creates NaN constraint (`0f24b565`)
- `aerosandbox/optimization/opti.py:645` — variable_categories_to_freeze as str breaks cache loading in solve() (iterates characters) (`f0ed017a`)
- `aerosandbox/performance/operating_point.py:13` — OperatingPoint.__init__ shares one mutable default Atmosphere across all instances (`0e2df4ac`)
- `aerosandbox/tools/pretty_plots/labellines/core.py:272` — Uses Axis.converter attribute, deprecated in Matplotlib 3.10 and removed in 3.12 (`50485d8b`)
- `aerosandbox/tools/statistics/time_series_uncertainty_quantification.py:205` — bootstrap_fits can loop forever if spline fits keep producing NaN (`a280d175`)
- `aerosandbox/weights/mass_properties.py:199` — MassProperties.__array__ lacks NumPy 2.0 'copy' keyword; deprecated, will become TypeError (`d36f2ca9`)
- `.github/workflows/run-pytest.yml:20` — Python support matrix incomplete: 3.11/3.13/3.14 untested, classifiers omit versions (`18d8bc0c`)
- `.github/workflows/run-pytest.yml:30` — "Without extras" CI job silently installs the dev dependency-group, so minimal install is never tested (`18d8bc0c`)
- `aerosandbox/aerodynamics/aero_2D/xfoil.py:599` — XFoil interface 10% covered; March 2026 list-input bugfix shipped with no regression test (`2cc9c05f`)
- `aerosandbox/aerodynamics/aero_3D/test_aero_3D/test_avl.py:17` — AVL tests silently 'pass' when avl binary is absent and return non-None (pytest 9 break) (`b283a28a`)
- `aerosandbox/aerodynamics/aero_3D/test_aero_3D/test_vortex_lattice_method.py:14` — Core VLM solver validated only by 'assert aero is not None'; external-data validation not wired to pytest (`97a69cfd`)
- `aerosandbox/geometry/test_geometry/test_airplane.py:7` — Five test_*.py files collect zero tests, giving phantom coverage (`379fb4aa`)
- `aerosandbox/geometry/airfoil/airfoil.py:103` — List coordinates input silently discarded; shape validation via assert disappears under -O (`f9e403ee`)
- `aerosandbox/geometry/openvsp_io/asb_to_openvsp/propulsor_vspscript_generator.py:35` — Propulsor export runs an IPOPT optimization to compute two closed-form Euler angles (`10160a5b`)
- `aerosandbox/structures/tube_spar_bending.py:260` — Internal library code calls deprecated asb.numpy.trapz (9 call sites) (`dfa41fac`)
- `aerosandbox/library/aerodynamics/components.py:219` — CDA_perpendicular_sheet_metal_joint Literal type omits 6 of 12 supported `kind` values (`bc7f8333`)
- `CONTRIBUTING.md:20` — Contributor setup instructions reference removed setup.py (`1f5ca232`)
- `aerosandbox/aerodynamics/aero_2D/mses.py:90` — MSES docstrings are copy-pasted from XFoil: wrong class name, nonexistent params, broken usage example (`cafbc9dd`)
- `aerosandbox/aerodynamics/aero_3D/aero_buildup.py:582` — wing_aerodynamics/fuselage_aerodynamics docstrings state wrong moment reference point and a nonexistent arg (`56cc02a4`)
- `aerosandbox/aerodynamics/aero_3D/aero_buildup_submodels/fuselage_aerodynamics_utilities.py:15` — critical_mach docstring off by factor of 2: parameter is L_n/d, not 2*L_n/d (`623737ef`)
- `aerosandbox/aerodynamics/aero_3D/singularities/point_source.py:33` — point_source docstring documents nonexistent parameters (copy-pasted from horseshoe function) (`dda8280c`)
- `aerosandbox/dynamics/flight_dynamics/airplane.py:7` — Public get_modes() has no docstring, untyped 'aero' dict, and undocumented accuracy caveats (`6b19f5e8`)
- `aerosandbox/library/field_lengths.py:272` — Public function field_length_analysis has no docstring (`919ebb3e`)
- `aerosandbox/modeling/fitting.py:78` — Docstrings swap 'dependent' and 'independent': x_data called dependent, y_data independent (`d9cf6d14`)
- `aerosandbox/optimization/opti.py:492` — Opti.minimize and Opti.maximize — core public API — have no docstrings (`0c9d9934`)
- `aerosandbox/optimization/opti.py:1048` — derivative_of/constrain_derivative docstrings document integrator methods that raise exceptions (`ce747417`)
- `aerosandbox/structures/legacy/beams.py:21` — torsion=True (the default) is silently ignored; docstring claims torsion is simulated (`d4fcb0d1`)
- `aerosandbox/structures/tube_spar_bending.py:120` — Docstring says None is valid for load/modulus functions, but None crashes with TypeError (`416e01d6`)
- `docs/source/conf.py:47` — autoapi_ignore misses *_derivations scratch directories, polluting published API docs (`6cb005c9`)

### Low (112)

- `aerosandbox/aerodynamics/aero_3D/linear_potential_flow.py:65` — LinearPotentialFlow crashes on construction: dicts keyed by unhashable Wing/Fuselage instances (`e2ede446`)
- `aerosandbox/atmosphere/atmosphere.py:118` — Atmosphere.__len__ resets detected vector length to 1 on trailing length-1 arrays (`ec464f01`)
- `aerosandbox/library/aerodynamics/components.py:285` — Dict key typo: 'lap joint_...' makes a documented joint kind raise ValueError (`bc7f8333`)
- `aerosandbox/library/propulsion_propeller.py:102` — style.use('seaborn') crashes on matplotlib >= 3.8 (`d7104f7d`)
- `aerosandbox/numpy/integrate_discrete.py:110` — integrate_discrete_intervals(method="midpoint") raises instead of warning (`d4eeba49`)
- `aerosandbox/optimization/opti.py:38` — freeze_style type hint says Literal["parameter", "frozen"] but code implements "float" (`a28b908e`)
- `aerosandbox/optimization/opti.py:1118` — derivative_of does not forward _stacklevel or category to self.variable() (`e382b98e`)
- `aerosandbox/optimization/opti.py:1455` — OptiSol.show_infeasibilities crashes on problems with a single scalar constraint (`c630578b`)
- `aerosandbox/structures/legacy/simple_beam_opt_daedalus_calibration.py:231` — Hardcoded Windows user path in plt.savefig breaks the demo script on other machines (`5bbaaedd`)
- `aerosandbox/tools/code_benchmarking.py:83` — Timer.__enter__ does not return self, so `with Timer() as t:` binds t=None (`b5db15ee`)
- `aerosandbox/tools/pretty_plots/labellines/__init__.py:3` — labellines `__all__` contains function objects, not strings — star import raises TypeError (`284ba6f7`)
- `aerosandbox/tools/pretty_plots/labellines/core.py:251` — labelLines ignores its `lines` argument and labels all legend handles instead (`e1defc26`)
- `aerosandbox/tools/string_formatting.py:76` — eng_string() drops the space before unit when exponent is outside SI-prefix range (`e2037293`)
- `aerosandbox/visualization/carpet_plot_utils.py:112` — patch_nans bridging loop: `continue` where `break` was intended, so last pair overwrites first (`35ec1c0f`)
- `aerosandbox/aerodynamics/aero_2D/mses.py:368` — MSES.run() crashes with KeyError('Ma') when zero runs converge (`8ca35eb5`)
- `aerosandbox/aerodynamics/aero_2D/xfoil.py:337` — except subprocess.CalledProcessError block is dead code; curated XFoil crash diagnostics never fire (`2b93333a`)
- `aerosandbox/aerodynamics/aero_2D/xfoil.py:399` — UnboundLocalError masks intended XFoilError when polar output has no separator line (`b46ee2ac`)
- `aerosandbox/aerodynamics/aero_3D/aero_buildup.py:750` — Mirrored-section branch mutates shared aerodynamic_centers arrays in place (`c9cb43de`)
- `aerosandbox/aerodynamics/aero_3D/avl.py:774` — parse_unformatted_data_output checks string bounds after indexing; IndexError/negative-index wraparound possible (`d4d99229`)
- `aerosandbox/aerodynamics/aero_3D/linear_potential_flow.py:78` — issubclass() called on Wing/Fuselage instances raises TypeError when validating per-component options (`e2ede446`)
- `aerosandbox/aerodynamics/aero_3D/vortex_lattice_method.py:483` — run_with_stability_derivatives uses locals() lookup and lacks the zero-division guard its siblings have (`64970ddc`)
- `aerosandbox/atmosphere/thermodynamics/gas.py:57` — PerfectGas.__repr__ crashes for array-valued pressure/temperature (`edcc2db8`)
- `aerosandbox/common.py:490` — ExplicitAnalysis.get_options returns geometry's options dict by reference in one branch (`7ed280f4`)
- `aerosandbox/dynamics/point_mass/common_point_mass.py:242` — __array__ missing NumPy 2 'copy' keyword; np.array(dyn) already fails under -W error (`63382a21`)
- `aerosandbox/dynamics/point_mass/common_point_mass.py:301` — constrain_derivatives: bad state name in 'which' leaks raw KeyError; friendly ValueError is dead code (`94d244af`)
- `aerosandbox/geometry/airfoil/airfoil.py:363` — make_symmetric_polars corrupts Cpmin, Xcpmin, Top_Xtr, Bot_Xtr in xfoil_data and cache (`1a8dd289`)
- `aerosandbox/geometry/airfoil/airfoil.py:1085` — Misparenthesized duplicate-point check in repanel(): np.all(np.diff(x)) > 0 (`89007b11`)
- `aerosandbox/library/aerodynamics/inviscid.py:86` — oswalds_efficiency raises UnboundLocalError for unknown `method` instead of ValueError (`b1c6516a`)
- `aerosandbox/library/aerodynamics/unsteady.py:51` — assert used for input validation in public library functions (stripped under python -O) (`2da5dc5d`)
- `aerosandbox/library/power_solar.py:232` — solar_flux silently swallows unknown keyword arguments via **deprecated_kwargs (`ee093751`)
- `aerosandbox/library/propulsion_electric.py:369` — mass_motor_electric silently returns None for an unrecognized `method` (`ffdcb13a`)
- `aerosandbox/modeling/black_box.py:115` — n_positional_args is always 0: len(parameters) minus len(parameters.values()) (`2f5cc107`)
- `aerosandbox/modeling/interpolation_unstructured.py:138` — UnstructuredInterpolatedModel mutates the caller's x_data_resample dict in place (`a57a6ffd`)
- `aerosandbox/modeling/splines/bezier.py:55` — quadratic_bezier_patch_from_tangents divides by zero when end tangents are parallel (`b94c076d`)
- `aerosandbox/modeling/surrogate_model.py:62` — SurrogateModel.__call__ catches NameError but missing x_data raises AttributeError (`b41c8e08`)
- `aerosandbox/numpy/__init__.py:67` — `np.round` crashes on CasADi types, violating the dual-backend contract (`3db59392`)
- `aerosandbox/numpy/array.py:741` — zeros_like/ones_like/full_like return wrong shape for 2D CasADi arrays (`5720355a`)
- `aerosandbox/numpy/trig.py:12` — Conversion constants _deg2rad and _rad2deg are named backwards (`b1051cab`)
- `aerosandbox/optimization/opti.py:658` — freeze_style="float" plus cache loading crashes: floats never get is_manually_frozen attribute (`2a00a1e0`)
- `aerosandbox/optimization/opti.py:675` — Direct parameter_mapping size mismatch produces misleading 'cached solution' error (`f1b1a37c`)
- `aerosandbox/optimization/opti.py:727` — Invalid behavior_on_failure value causes UnboundLocalError instead of clear ValueError (`3604e3e2`)
- `aerosandbox/optimization/opti.py:1357` — OptiSol.value does not propagate warn_on_unknown_types (or recursive) in recursive calls (`e2a2d8bf`)
- `aerosandbox/structures/legacy/beams.py:168` — Point loads added out of location order silently produce a non-monotonic mesh and wrong results (`5d57ad91`)
- `aerosandbox/structures/legacy/beams.py:306` — TubeBeam1.setup() crashes with AttributeError when bending=False (`21a05b74`)
- `aerosandbox/tools/inspect_tools.py:313` — Argument-name parser truncates args containing ':' outside braces (slices, lambdas) (`5f1e3ffe`)
- `aerosandbox/tools/pretty_plots/plots/contour.py:156` — contour(z_log_scale=True) with constant Z crashes with cryptic geomspace error (`f9905312`)
- `aerosandbox/tools/pretty_plots/plots/plot_smooth.py:148` — plot_smooth raises UnboundLocalError instead of ValueError for invalid `function_of` (`427eab68`)
- `aerosandbox/tools/python/importing.py:19` — lazy_import uses importlib.util without importing importlib.util (`8ceb3468`)
- `aerosandbox/tools/python/io.py:22` — convert_ipynb_to_py opens files without encoding='utf-8' (`1367f408`)
- `aerosandbox/visualization/carpet_plot_utils.py:38` — time_limit leaks the SIGALRM handler and rejects float durations (`a9b946ea`)
- `aerosandbox/visualization/carpet_plot_utils.py:62` — patch_nans() mutates its input array in place and prints progress unconditionally (`35ec1c0f`)
- `aerosandbox/visualization/plotly_Figure3D.py:72` — Figure3D mirror=True crashes on tuple/list points, the exact format the docstrings show (`544b55bb`)
- `aerosandbox/weights/mass_properties.py:497` — generate_possible_set_of_point_masses: dimensionally wrong radius estimate (`975217de`)
- `aerosandbox/atmosphere/thermodynamics/test/test_gas.py:6` — Thermodynamics test file is an empty stub: 'def test_isentropic(): pass' (`52b1d940`)
- `aerosandbox/geometry/test_geometry/test_wingxsec.py:6` — test_wingxsec has 'TODO actually test this', no assertions, and exercises only deprecated kwargs (`0e168d4f`)
- `aerosandbox/aerodynamics/aero_3D/singularities/uniform_strength_horseshoe_singularities.py:64` — assert_equal_shape rejects valid broadcasting the docstring explicitly promises (`9718142c`)
- `aerosandbox/atmosphere/atmosphere.py:148` — Invalid `method` errors say "Bad value of 'type'!" and are deferred to first property call (`6197e334`)
- `aerosandbox/dynamics/point_mass/common_point_mass.py:797` — potential_energy is a property with an unusable 'g' parameter and misleading Args docstring (`2c519458`)
- `aerosandbox/dynamics/point_mass/point_3D/cartesian.py:225` — add_force default 'axes' differs per subclass ('earth'/'wind'/'body'), contradicting base-class signature (`f51e6d28`)
- `aerosandbox/geometry/airplane.py:360` — Airplane.draw(): plotly show_kwargs set inside face loop; matplotlib branch returns None (`8f611fbc`)
- `aerosandbox/library/propulsion_electric.py:273` — electric_propeller_propulsion_analysis returns locals(), contradicting dict[str, float] annotation (`68cbf0a5`)
- `aerosandbox/modeling/fitting.py:231` — Model-evaluation failure re-raised as bare Exception, discarding the original error (`4649489a`)
- `aerosandbox/structures/buckling.py:135` — plate_buckling_critical_load `length` parameter is documented but never used (`d2d43d7f`)
- `aerosandbox/weights/__init__.py:2` — Shape mass-properties helpers not re-exported despite sibling being top-level API (`929d160d`)
- `.github/workflows/run-pytest.yml:6` — CI workflow modernization: duplicate runs, fail-fast cancellation, deprecated actions, Python 3.13 untested (`7cec59b0`)
- `INSTALLATION.md:102` — Troubleshooting advice recommends 'sudo pip install', which fails on modern Linux (PEP 668) (`f11086d1`)
- `aerosandbox/aerodynamics/aero_2D/airfoil_inviscid.py:141` — AirfoilInviscid internally calls asb.numpy.trapz, which is slated for removal and warns on every call (`3e0263a9`)
- `aerosandbox/common.py:182` — Deprecation warnings library-wide lack `stacklevel=2` (and one lacks a category) (`95c25f23`)
- `aerosandbox/common.py:557` — ImplicitAnalysis.initialize wrapper loses subclass __init__ signature/docstring (`2bcc086e`)
- `aerosandbox/library/aerodynamics/viscous.py:179` — Cl_flat_plate deprecation of Re_c uses bare UserWarning without category or stacklevel (`5531ba87`)
- `aerosandbox/library/winds.py:10` — Invalid escape sequences in docstrings emit SyntaxWarning on Python 3.12+ (4 files) (`12fd44e4`)
- `aerosandbox/tools/pretty_plots/colors.py:78` — get_last_line_color reads private Line2D._color instead of get_color() (`8615b694`)
- `aerosandbox/aerodynamics/aero_2D/mses.py:80` — behavior_after_unconverged_run accepts any string; typos silently disable both behaviors (`cafbc9dd`)
- `aerosandbox/aerodynamics/aero_3D/lifting_line.py:58` — model_size typed as bare str instead of Literal; NonlinearLiftingLine spanwise_resolution unannotated (`4788369c`)
- `aerosandbox/aerodynamics/aero_3D/singularities/point_source.py:13` — Typing gaps: viscous_radius untyped, fuselage_form_factor missing return annotation (`dda8280c`)
- `aerosandbox/atmosphere/atmosphere.py:139` — Public Atmosphere/OperatingPoint methods lack parameter and return type annotations (`ff710b3f`)
- `aerosandbox/common.py:38` — _asb_metadata annotated as dict[str, str] but defaulted to None (`95c25f23`)
- `aerosandbox/modeling/fitting.py:341` — goodness_of_fit: 'type' parameter shadows builtin and lacks Literal/return annotations (`a0af4096`)
- `aerosandbox/structures/tube_spar_bending.py:22` — Minor typing gaps on TubeSparBendingStructure public surface (`10488c96`)
- `README.md:117` — README typo 'discplines' and inconsistent master/develop branch links (`4d0cff4e`)
- `README.md:207` — README claims base install is 'headless with minimal dependencies' but base deps include matplotlib+seaborn (`4d0cff4e`)
- `aerosandbox/README.md:74` — Developer README 'Map' references removed in_progress folder and omits three real modules (`4d0cff4e`)
- `aerosandbox/__init__.py:64` — `asb.docs()` opens the GitHub source tree instead of the hosted docs site that pyproject declares (`3ad17844`)
- `aerosandbox/aerodynamics/aero_2D/airfoil_polar_functions.py:17` — airfoil_coefficients_post_stall docstring lists wrong args and type hint understates accepted input (`2b178806`)
- `aerosandbox/aerodynamics/aero_3D/avl.py:208` — Public AVL.run() docstring says 'Private function to run AVL' (`3e88c1bd`)
- `aerosandbox/aerodynamics/aero_3D/lifting_line.py:374` — LiftingLine docstrings: 'alp ha' split mid-word, nonexistent 'opti' arg documented, wrong class names (`4788369c`)
- `aerosandbox/atmosphere/atmosphere.py:93` — IndexError message missing space: 'while theparent has length' (`f11793cd`)
- `aerosandbox/atmosphere/atmosphere.py:173` — density_altitude() docstring omits `method` arg and unimplemented 'exact' option (`ff710b3f`)
- `aerosandbox/atmosphere/thermodynamics/gas.py:193` — process() docstring says inplace=True returns nothing, but code returns self (`55e7a72e`)
- `aerosandbox/dynamics/point_mass/common_point_mass.py:394` — add_gravity_force docstring says '-z direction' but the force is applied in +z_e (down, NED) (`2c519458`)
- `aerosandbox/dynamics/point_mass/point_1D/vertical.py:19` — Docstring errors in point-mass classes: wrong axis label, nonexistent method name, u/v typo (`2c519458`)
- `aerosandbox/geometry/airfoil/airfoil.py:1439` — rotate() docstring contradicts itself: summary says clockwise, behavior is counterclockwise (`c33e0ab3`)
- `aerosandbox/geometry/airfoil/airfoil.py:1526` — write_dat() and draw() docstrings misstate return values (`c33e0ab3`)
- `aerosandbox/geometry/airfoil/airfoil_families.py:462` — least_squares unknown-vector packing comment states upper-first; code packs lower-first (`b784d061`)
- `aerosandbox/geometry/fuselage.py:626` — FuselageXSec docstring superellipse equation wrong by factor of 2 (`1c7745f6`)
- `aerosandbox/geometry/polygon.py:92` — Polygon.rotate() docstring self-contradicts: "clockwise" summary vs "counterclockwise" Args (`1c7745f6`)
- `aerosandbox/geometry/propulsor.py:28` — Public Propulsor.__init__ docstring is a placeholder ("TODO add docs") (`1c7745f6`)
- `aerosandbox/geometry/wing.py:610` — Broken reference URL in Wing.aerodynamic_center docstring ("downloattttd") (`1c7745f6`)
- `aerosandbox/library/aerodynamics/viscous.py:25` — Docstring gaps: empty Returns + stray TODO in Cd_cylinder; Korn functions missing param docs (`74eaf4fe`)
- `aerosandbox/modeling/black_box.py:18` — black_box public API has empty docstring args and undocumented fd_step/fd_step_iter (`b3ee5b63`)
- `aerosandbox/modeling/surrogate_model.py:47` — Minor docstring errors: 'R^1 -> R^2' typo and invalid usage example in FittedModel (`d9cf6d14`)
- `aerosandbox/numpy/array.py:49` — array() docstring says dtype is ignored for CasADi contents, but code crashes instead (`d6b3236e`)
- `aerosandbox/optimization/opti.py:39` — Opti.__init__ parameters entirely undocumented (marked '# TODO document') (`a28b908e`)
- `aerosandbox/optimization/opti.py:742` — solve_sweep is public but has no docstring and incomplete type annotations (`0c9d9934`)
- `aerosandbox/structures/buckling.py:139` — poissons_ratio missing from docstring Args; buckling functions lack return-type annotations (`d2d43d7f`)
- `aerosandbox/visualization/plotly_Figure3D.py:114` — Figure3D docstrings reference nonexistent add_face() and wrong examples (`fb9d2764`)
- `aerosandbox/weights/mass_properties.py:150` — Malformed user-facing error messages in MassProperties.__getitem__ and __len__ (`d24b54c4`)
- `docs/source/conf.py:53` — conf.py points at nonexistent _static/_templates dirs and has stale 2023 copyright (`6cb005c9`)
- `pyproject.toml:41` — neuralfoil<0.4.0 upper bound is justified but undocumented (`81e6b6d3`)
- `aerosandbox/dynamics/rigid_body/rigid_3D/body_euler.py:317` — sincos() helper duplicated verbatim; alpha/beta/speed re-declarations duplicate base class (`b68d1e24`)
- `aerosandbox/library/aerodynamics/unsteady.py:232` — Dead no-op statement inside calculate_lift_due_to_pitching_profile integrand (`f7f2a15a`)
- `aerosandbox/tools/statistics/time_series_uncertainty_quantification.py:181` — Dead code in bootstrap_fits: no-op expression and unreachable branch (`8359a209`)

### Integration fixes (found during implementation)

- `aerosandbox/tools/test_tools/test_inspect_tools.py` — codegen tests broke on Python 3.13
  (PEP 667 exec/eval frame-locals change); fixed with explicit namespaces (`7cec59b0`)
- `aerosandbox/numpy/test_numpy/test_all_operations_run.py` — module-level
  `np.seterr(all="raise")` leaked strict error state into every later test, making full-suite
  results order-dependent; scoped to a per-test fixture (`747133a9`)
- `.github/workflows/run-pytest.yml` — `fail-fast: false` so one Python version's failure doesn't
  cancel the rest of the matrix (`7cec59b0`)


---

## Appendix: reviewer coverage notes

- **aero-2D**: Reviewed all six non-test files in aero_2D (xfoil.py, mses.py, airfoil_inviscid.py, linear_strength_line_singularities.py, airfoil_polar_functions.py, IBL2.py, airfoil_optimizer.py), verifying claims against aerosandbox.numpy and Airfoil APIs. Did not re-derive Katz-Plotkin Eq 11.99/11.100 algebra or Truong post-stall constants. No findings dropped by cap.
- **aero-3D-solvers**: All six top-level aero_3D solver files reviewed; three crashes confirmed by execution, AVL deflection bug confirmed via git history. Not covered: singularities/, aero_buildup_submodels/, test dirs (out of scope); physics formulas spot-checked only. Dropped for cap: tuple-vs-list return-doc mismatches, dict[str,float] hint inaccuracies, untyped bool args.
- **aero-3D-internals**: Covered all non-test files in singularities/ and aero_buildup_submodels/ plus aerodynamics/__init__.py (clean), verifying formulas against Mason Eq. 7-8, the fit study, and the standard horseshoe Biot-Savart form, and tracing all production callers. __main__ demo blocks and empty subpackage __init__ files not flagged. No findings dropped.
- **atmo-perf-prop**: Covered all non-test files in atmosphere/ and performance/; propulsion/ contains only /ignore/ files (skipped per instructions). Top bugs verified by execution, including CasADi-backend spot checks and comparison against official 1976 Standard Atmosphere values. Nothing dropped due to the 15-finding cap.
- **dynamics**: Reviewed all 12 non-test files under aerosandbox/dynamics/; crashes verified by running code (NumPy 2.4.6, pyvista, CasADi). Body-Euler EOM and speed-gamma-track derivatives checked against inertia/rotation conventions and found correct. Not verified: get_modes coefficients against the FVA book. Dropped: minor typos (duplicate self.bank, __len__ message grammar).
- **geometry-airfoil**: Reviewed all 5 in-scope source files (airfoil.py, kulfan_airfoil.py, airfoil_families.py, default_airfoil_aerodynamics.py, nosecone_shapes/haack.py, plus utils/convert script and __init__ files); top 6 bugs verified by execution. Checked Haack/NACA/CST/Korn formulas against references (kulfan mach_dd 0.1/320 is self-consistent, not flagged). Nothing dropped by cap.
- **geometry-3D**: Read all 7 scoped geometry modules plus all 5 openvsp_io/asb_to_openvsp files; top bugs verified by execution. geometry/airfoil, nosecone_shapes, tests out of scope. Dropped for cap: control_surface_area's `type: ...|None` crashes on None; untyped bool params; subdivide_sections sharing ControlSurface objects between new xsecs.
- **library-aero**: Covered all scoped files (aerodynamics/*.py incl. __init__, airfoils.py, winds.py, field_lengths.py, regulations/far_part_23.py); verified crash/warning claims by execution and formulas against DATCOM/Korn/Cengel/normal-shock references. Dropped (minor): isinstance misses numpy-int AoA scalars; diamond_airfoil missing docstring; normal_shock_relations lacking type hints.
- **library-power-prop**: Covered all in-scope files; crashes verified at runtime; Raymer/TASOPT/Torenbeek formulas cross-checked. Dropped for cap: untyped `method`/`aircraft_type` params, removed matplotlib "seaborn" style in __main__ demos, peak_sun_hours docstring ghost param, costs.py Returns missing "aircraft_interiors" key, mass_structural `type` shadows builtin. Torenbeek C-10/propeller constants unverified against source.
- **modeling**: Reviewed all 8 non-test files in aerosandbox/modeling/ (fitting, interpolation, interpolation_unstructured, surrogate_model, black_box, splines/bezier, splines/hermite, __init__s). Top 7 findings verified by execution; hermite/bezier math checked against Bernstein-form Hermite formulas (no formula errors). aerosandbox/numpy internals (e.g., interpn xi mutation) out of scope. Nothing dropped to the 15-cap.
- **numpy-wrapper**: Reviewed every file in aerosandbox/numpy/ (excluding test_numpy and derivation dirs); all 15 findings verified by execution. Dropped for the cap: overly-broad `except (TypeError, Exception)` in array.array(), missing quad() docstring, logicals.all/any returning False for symbolic MX, dot() 2D NumPy/CasADi semantic divergence, softmax hardness=0 error path.
- **optimization**: Full scope covered: opti.py (all methods, Opti + OptiSol) and __init__.py; all findings runtime-verified on Python 3.12/CasADi 3.7.2. Dropped as marginal: solve_sweep get_vals np.vectorize edge cases, max_runtime maps to ipopt CPU-time not wall-time, typing.Sequence isinstance modernization.
- **structures**: Reviewed all 5 non-test files in aerosandbox/structures/ (buckling.py, tube_spar_bending.py, legacy/beams.py, both legacy simple_beam_opt scripts). Verified crashes and the plate-buckling factor numerically at runtime. __main__ demo blocks skimmed, not exhaustively audited. No findings dropped due to the 15-item cap.
- **tools**: Read all 30 non-test .py files in aerosandbox/tools/; verified crashes/behavior empirically. units.py constants checked and correct. Dropped for the 15-cap: pretty_plots/__init__ import-time rcParams/theme mutation, show_plot ylabel parameter shadowing, tuple[float] typing nits, codegen ndarray-repr truncation, plot_color_by_value LineCollection performance, time_function runs repeats+1.
- **misc-modules**: Read all files in scope fully; key crashes/behaviors verified by execution (numpy 2.4.6). Dropped for the 15-cap: missing stacklevel in substitute_solution's DeprecationWarning, Figure3D.draw showscale truthiness check, plot_point_cloud lacking show/return options, and docs() pointing at GitHub (existing TODO).
- **packaging-ci**: Covered all scoped files plus adjacent .readthedocs.yaml (broken docs build); verified ptest/neuralfoil on PyPI and live CI logs via gh. _config.yml and CHANGELOG.md reviewed, no findings worth the cap. Did not deep-audit uv.lock contents or docs/ folder. No findings dropped by the 15-cap.
- **test-quality**: Covered: all 22 test dirs mapped, full suite run with coverage (461 pass/2 fail, 51%), CI workflows and run history, zero-assert/skip/phantom-test audit, external-reference validation assessment. Not covered: tutorial notebook execution, per-test flakiness/runtime. Dropped (cap/minor): dataset_temperature bare-import fragility, inconsistent shapely-skip idioms, DynamicsRigidBody2DBody untested.
- **antipattern-sweep**: Swept aerosandbox/ (grep + AST) for all scoped patterns: mutable defaults (1, benign), bare except (0), ==None (0), asserts, prints, eval/exec/os.system (0), NumPy-2.x/scipy/mpl deprecations; ran ruff (default + B/NPY/PLW/RUF triage). Verified norm/hash/seaborn findings by execution. Did not individually triage all 76 TODOs or audit tools/pretty_plots exhaustively; nothing dropped for the cap.
- **docs-and-docstrings**: Covered README.md, INSTALLATION.md, CONTRIBUTING.md, aerosandbox/README.md, tutorial structure/READMEs, docs/source (conf.py, index.rst), .readthedocs.yaml, requirements.txt/pyproject config, all README relative links verified, ~25 public docstrings sampled across 20 modules. Did not open individual tutorial notebooks or docs/Makefile internals. No findings dropped by the 15-cap.
- **public-api-bc**: Covered: top-level and all 48 subpackage __init__.py files, runtime namespace inspection of aerosandbox and aerosandbox.numpy, all deprecation shims, optional-dependency import chain (simulated base install), README/pyproject consistency. Not audited: deep module internals beyond the API surface, tutorial notebooks, docs build config. No findings dropped by the 15-cap.
