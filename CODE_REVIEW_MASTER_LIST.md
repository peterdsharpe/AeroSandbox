# AeroSandbox Deep Review — Master List

*Generated 2026-07-04 against `develop` @ `ecd4d676`. Produced by a 198-agent parallel review
(20 scoped reviewers over ~47k LOC + adversarial verification of every bug-class finding at
medium severity or above; 283 raw findings, 3 refuted, 263 after de-duplication).*

**Legend** — every entry lists: location · category · verification status · backwards-compat risk.
"verified" = independent agents reproduced or confirmed the claim against the code (✓ in compact
lists). "not independently verified" applies to non-bug findings (docs, testing, modernization),
which were not adversarially re-checked. All suggested fixes are backwards-compatible unless
marked otherwise; anything marked **defer to v5** requires a breaking change.

**Counts:** 5 critical · 13 high · 97 medium · 148 low.
By category: 79 bugs, 68 latent bugs, 12 testing,
18 API design, 9 typing, 48 docs,
14 modernization, 13 dead code, 2 performance.

---

## Executive summary

**The develop branch is not currently releasable.** CI has been red since 2025-12-25 with two
genuine failures, and additional unreleased regressions were confirmed by execution:

1. **VLM crashes under the NumPy backend** — `IndexError` at `vortex_lattice_method.py:160`. Root
   cause (bisected by verifiers): commit `f45b11a4`'s `asarray` changes broke
   `aerosandbox/numpy/array.py::length()` for lists of CasADi MX — `len(MX)` raises `TypeError`,
   `length()` returns 1, and `Wing.mesh_body()` produces empty face arrays (`wing.py:1084`).
2. **`import aerosandbox` fails on a base install** — `airfoil.py:1` imports plotly at module
   level, but plotly moved to the `[full]` extra during the packaging migration. Plain
   `pip install aerosandbox` built from develop would get `ModuleNotFoundError`.
3. **LiftingLine and NonlinearLiftingLine crash with `NameError` on any symmetric wing** (i.e.,
   almost every airplane) — a lint-fix commit (`e0f8ce07`) added `ControlSurface` annotations to
   nested functions while the import is `TYPE_CHECKING`-only. Zero test coverage let it land.
4. `test_cadquery_export.py` references `asb.np`, failing every CI run since Nov 2025.

**A cluster of wrong-number bugs in the weights/aero libraries affects released versions.** These
return plausible but wrong engineering results: Raymer instruments/hydraulics use `L_f * B_w`
where Raymer specifies `(L_f + B_w)` (~7.5×/~45× too heavy for a 737-class aircraft); GA
landing-gear lengths are divided by 12 twice (2.8–8× too light); `mass_nacelles` returns lbm
while documented as kg (2.2× heavy); the TASOPT nacelle weight term multiplies where it should
add; the ISA atmosphere uses g=9.81 instead of 9.80665; plate buckling is ~9.5% low from a
double-counted factor. None of these are caught by tests — the weights/library modules are ~0%
covered.

**A cluster of silently-ignored inputs.** The AVL wrapper ignores all `ControlSurface.deflection`
values (always deflects control 1 by exactly 1°); `WingXSec`/`Fuselage` deprecation shims test
`in locals()` against `**kwargs` so deprecated arguments are silently dropped with no warning;
`solve_ivp` ignores `t_eval` under CasADi; `norm()` ignores `axis` for several `ord` values.

**Packaging/modernization is largely already done** (pyproject.toml + hatchling + uv are in
place — `setup.py` is already gone), but the migration left sharp edges: the dev dependency
group installs **`ptest` (an abandoned, unrelated package) instead of `pytest`**;
`requirements.txt` is stale and mutually unsatisfiable with pyproject; **the PyPI publish
workflow has no test gate** (any push to master publishes, even with tests failing); Read the
Docs pins Python 3.8, which can no longer install the package; INSTALLATION.md and
CONTRIBUTING.md still reference `setup.py` and Python 3.8.

**Suggested order of attack for the minor release:**
1. Fix the five critical items (§1) and get CI green.
2. Gate the publish workflow on tests; fix `ptest`→`pytest`; delete or regenerate `requirements.txt`.
3. Fix the high-severity wrong-number library bugs (§2) — users get bad designs from these today.
4. Sweep the medium bugs (§3), prioritizing the `aerosandbox/numpy` dual-backend defects since
   everything downstream depends on them.
5. Batch the docs/typing/low items (§4) opportunistically; many are one-line fixes.
6. Add regression tests as each fix lands — the test-quality findings (LiftingLine untested, five
   test files that collect zero tests, no coverage measurement in CI) explain how most of these
   bugs shipped in the first place.

---

## 1. Critical — release blockers

These must be fixed before tagging the minor release. Two are live regressions on `develop` that are not in any released version; the rest break the released package.

### 1. LiftingLine.run() crashes with NameError on any airplane with a symmetric wing

`aerosandbox/aerodynamics/aero_3D/lifting_line.py:589` · bug · verified · backwards-compat risk: none

Nested function `mirror_control_surface(surf: ControlSurface) -> ControlSurface` is defined inside `if wing.symmetric:`. `ControlSurface` is imported only under TYPE_CHECKING (line 17-18), and Python 3.10-3.13 evaluates function annotations at def-time, so run() raises NameError for symmetric wings (the most common case). Confirmed by execution; regression from lint-fix commit e0f8ce07 (Dec 2025, develop only).

**Fix:** Quote the annotations (`surf: "ControlSurface"`), add `from __future__ import annotations`, or import ControlSurface at runtime.

> Verifier: Confirmed by execution: LiftingLine.run() with a symmetric wing raises NameError at lifting_line.py:589; ControlSurface only imported under TYPE_CHECKING (lifting_line.py:17-18), no future-annotations import, repo requires Python >=3.10 (annotations eval at def-time through 3.13).

### 2. NonlinearLiftingLine.run() crashes with NameError on any symmetric wing (same TYPE_CHECKING regression)

`aerosandbox/aerodynamics/aero_3D/nonlinear_lifting_line.py:211` · bug · verified · backwards-compat risk: none

Identical defect to lifting_line.py: nested `def mirror_control_surface(surf: ControlSurface) -> ControlSurface` inside `if wing.symmetric:` raises NameError at def-time because ControlSurface is TYPE_CHECKING-only (line 14-15). Confirmed by execution: any airplane with a symmetric wing crashes. Introduced by commit e0f8ce07.

**Fix:** Quote the annotations or add `from __future__ import annotations` to the module.

> Verifier: Confirmed by execution on project Python 3.12: run() with a symmetric wing raises NameError: name 'ControlSurface' is not defined. Annotation at nonlinear_lifting_line.py:211 is evaluated at def-time; ControlSurface only imported under TYPE_CHECKING (line 15).

### 3. CI on develop is red: VLM regression (IndexError) introduced by latest numpy-typing commit

`aerosandbox/aerodynamics/aero_3D/test_aero_3D/test_vlm/test_airplane_optimization.py:38` · bug · verified · backwards-compat risk: none

test_airplane_optimization fails with 'IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed' at aerosandbox/aerodynamics/aero_3D/vortex_lattice_method.py:160. Reproduced locally and in CI run 28706043488 on commit ecd4d676 (2026-07-04). A core public solver crashes on valid input; develop CI is failing.

**Fix:** Bisect ecd4d676's aerosandbox/numpy changes (likely gradient/array 1-D handling feeding VLM line 160); fix and keep this test as the regression guard.

> Verifier: Confirmed: IndexError at aerosandbox/aerodynamics/aero_3D/vortex_lattice_method.py:160 reproduces locally at ecd4d676; CI run 28706043488 failed. Bisect shows regression introduced by f45b11a4 (asarray changes), not ecd4d676; f37ef646 passes.

### 4. Base install cannot `import aerosandbox`: module-level plotly import of an optional dependency

`aerosandbox/geometry/airfoil/airfoil.py:1` · bug · verified · backwards-compat risk: none

airfoil.py line 1 does `import plotly.graph_objects` at module level, but plotly is only in the `[full]` extra (pyproject.toml line 52). Import chain aerosandbox/__init__.py -> geometry -> airfoil pulls this in, so `pip install aerosandbox` (base) fails with ModuleNotFoundError on `import aerosandbox`. Verified by simulating a blocked plotly import. The import is only used for a return-type annotation (line 865); draw() already lazy-imports plotly.

**Fix:** Move `import plotly.graph_objects` under `if TYPE_CHECKING:` and quote the annotation at line 865 (or add `from __future__ import annotations`).

> Verifier: Confirmed: airfoil.py:1 eagerly imports plotly, used only for annotation at airfoil.py:865; plotly is only in [full] extra (pyproject.toml:53). Simulated missing plotly: `import aerosandbox` raises ModuleNotFoundError via aerosandbox/__init__.py:15. Critical stands.

### 5. CI has been red on develop since 2025-12-25; two genuine test failures ship unnoticed

`.github/workflows/run-pytest.yml:30` · testing · not independently verified — found independently by 2 reviewers · backwards-compat risk: none

Every push to develop since Dec 25, 2025 fails (verified via gh run logs, latest run 28706043488): test_vlm/test_airplane_optimization.py fails with 'IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed' (likely NumPy-backend regression from recent aerosandbox/numpy changes), and test_cadquery_export.py fails with "module 'aerosandbox' has no attribute 'np'". A red baseline masks all new regressions.

**Fix:** Fix the two failing tests/regressions before any release; consider branch protection requiring green CI on develop.


---

## 2. High severity

Wrong engineering results or broken public functionality. All are backwards-compatible fixes.


#### Bugs (wrong results or crashes)

### 6. AVL wrapper silently ignores all ControlSurface.deflection values; first control always deflected 1.0 deg

`aerosandbox/aerodynamics/aero_3D/avl.py:405` · bug · verified · backwards-compat risk: low

CONTROL lines (line 540) write gain=1 with per-surface names, but the run keystrokes hardcode `d1 d1 1`, so AVL deflects control variable 1 by exactly 1.0 degree and all others by 0, regardless of user-specified deflections. Commit 8704f5df changed gain from `{surf.deflection}` (shared 'all_deflections' variable, where `d1 d1 1` was correct) without updating the keystroke. Affects released versions.

**Fix:** Track unique control names in write_avl order and emit `d{i} d{i} {deflection}` per control; or restore gain=surf.deflection with a shared variable name.

> Verifier: Confirmed: avl.py:540 emits gain=1 per-surface CONTROL names, but avl.py:405 hardcodes "d1 d1 1", so control 1 gets 1.0 deg and others 0; surf.deflection ignored (regression from 8704f5df). High is correct.

### 7. draw() crashes on NumPy 2.x: float() on 1-element array when auto-scaling vehicle model

`aerosandbox/dynamics/point_mass/common_point_mass.py:502` · bug · verified · backwards-compat risk: none

vehicle_length = np.diff(vehicle_bounds[0, :]) is a length-1 ndarray; float() on it raises TypeError ('only 0-dimensional arrays can be converted to Python scalars') on NumPy >= 2.0 (deprecated 1.25). Any dyn.draw() call with default scale_vehicle_model=None on a multi-point trajectory crashes. Verified with repo's NumPy 2.4.6.

**Fix:** Use vehicle_length = np.diff(vehicle_bounds[0, :]).item() (or index [0]) before the division, so float() receives a scalar.

> Verifier: Confirmed: reproduced TypeError at common_point_mass.py:502 on NumPy 2.4.6 via DynamicsPointMass3DCartesian.draw() with a 10-point trajectory; np.diff at line 501 yields shape-(1,) array, float() rejects it. Severity high stands.

### 8. OpenVSP fuselage export double-counts the first xsec's position offset

`aerosandbox/geometry/openvsp_io/asb_to_openvsp/fuselage_vspscript_generator.py:73` · bug · verified · backwards-compat risk: none

X/Y/Z_Rel_Location is set to xsecs[0].xyz_c, but XLocPercent/YLocPercent/ZLocPercent are computed as `xsec.xyz_c[i] / length` without subtracting xsecs[0].xyz_c. Any fuselage whose first xsec is not at the origin (e.g., after .translate()) exports shifted/skewed geometry. Also divides by zero if length == 0.

**Fix:** Use `(xsec.xyz_c[i] - fuselage.xsecs[0].xyz_c[i]) / length`; guard length == 0.

> Verifier: Confirmed: fuselage_vspscript_generator.py:51-53 sets Rel_Location to xsecs[0].xyz_c, yet lines 73-75 use absolute xyz_c/length for LocPercent, double-counting offset; wing_vspscript_generator.py:54-58 shows correct relative-delta approach. length==0 divide also real.

### 9. TransverseGustPitchControl crashes on construction: np.sum() called on a generator expression

`aerosandbox/library/gust_pitch_control.py:52` · bug · verified · backwards-compat risk: none

Lines 52 and 96 pass a generator expression to aerosandbox.numpy.sum(), whose asarray() produces a 0-d object array; is_casadi_type() then raises 'TypeError: iteration over a 0-d array'. Verified at runtime: constructing TransverseGustPitchControl with any valid input crashes, so the entire public class (and calculate_transients) is unusable.

**Fix:** Use Python builtin sum(...) or an explicit accumulation loop (as already done at lines 59-62) instead of np.sum(generator).

> Verifier: Confirmed at runtime: gust_pitch_control.py:52 (and :96) crash with TypeError via asarray/is_casadi_type on generator (determine_type.py:74). Class is fully unusable, but it is a peripheral untested library class, so high, not critical.

### 10. mass_nacelles returns pounds-mass, not kg: missing final `* u.lbm` conversion

`aerosandbox/library/weights/raymer_cargo_transport_weights.py:413` · bug · verified · backwards-compat risk: low

Every other function in this module converts the Raymer fps-unit result to kg via a trailing `* u.lbm`, but mass_nacelles' return expression (lines 403-413) omits it. The function therefore returns nacelle mass in lbm while documented as kg, overestimating by a factor of 2.205 in any consistent-SI weights buildup.

**Fix:** Append `* u.lbm` to the returned expression, matching all sibling functions.

> Verifier: Confirmed: raymer_cargo_transport_weights.py:403-413 return lacks trailing `* u.lbm`; all sibling functions (e.g., lines 344, 434) include it. Docstring line 386 promises kg; result is lbm, ~2.205x overestimate. No caller/test compensates.

### 11. mass_instruments and mass_hydraulics multiply L_f*B_w where Raymer specifies (L_f + B_w)

`aerosandbox/library/weights/raymer_cargo_transport_weights.py:582` · bug · verified · backwards-compat risk: low

Raymer 5th Ed. Eqs. 15.41-15.42 use (L_f + B_w): W_instr = 4.509*Kr*Ktp*Nc^0.541*Nen*(Lf+Bw)^0.5 and W_hyd = 0.2673*Nf*(Lf+Bw)^0.937. Lines 582 and 612 compute the product Lf*Bw instead. For a 737-class airplane (Lf~124ft, Bw~112ft) hydraulics comes out ~45x too heavy (~14,000 lb) and instruments ~7.5x too heavy.

**Fix:** Replace `fuselage.length()/u.foot * main_wing.span()/u.foot` with `fuselage.length()/u.foot + main_wing.span()/u.foot` in both functions.

> Verifier: Confirmed: raymer_cargo_transport_weights.py:582,612 compute Lf*Bw where Raymer Eqs. 15.41-15.42 require (Lf+Bw). For 737-class geometry, hydraulics ~45x and instruments ~7.7x too heavy; no tests catch it. Severity high stands.

### 12. GA landing-gear length term divided by 12 twice, underestimating gear mass 2.8-8x

`aerosandbox/library/weights/raymer_general_aviation_weights.py:251` · bug · verified · backwards-compat risk: low

Raymer GA equations use (L_m/12)^0.409 and (L_n/12)^0.845 with L in inches, i.e. length in feet. Lines 251 and 290 compute `(length / u.foot / 12)`, converting to feet and dividing by 12 again. The main-gear term is 12^0.409 = 2.8x too small; the nose-gear term 12^0.845 = 8.2x too small.

**Fix:** Change to `(main_gear_length / u.foot) ** 0.409` and `(nose_gear_length / u.foot) ** 0.845` (equivalently length/u.inch/12).

> Verifier: Confirmed: units.py:6 defines foot=12*inch, so length/u.foot is already feet; raymer_general_aviation_weights.py:251,290 divide by 12 again. Compare correct `/u.inch` in raymer_cargo_transport_weights.py:295. Main gear 2.76x, nose gear 8.16x underestimated.


#### Latent bugs & fragile code

### 13. WingXSec deprecated control_surface_* kwargs silently ignored: `in locals()` never true

`aerosandbox/geometry/wing.py:1566` · latent-bug · verified — found independently by 2 reviewers · backwards-compat risk: none

`"control_surface_is_symmetric" in locals()` (and lines 1567-1568, 1576-1580) is always False because these arrive in `**deprecated_kwargs`, never as locals. Verified: `WingXSec(chord=2, control_surface_deflection=10)` produces control_surfaces=[] with zero warnings — silently wrong aero results for users of the deprecated API that the docstring (lines 1495-1531) still documents as supported.

**Fix:** Check/read `deprecated_kwargs` instead of `locals()`; emit the DeprecationWarning and build the ControlSurface from kwarg values. Also reject unrecognized leftover kwargs.

> Verifier: Confirmed: __init__ takes **deprecated_kwargs (wing.py:1443), so `in locals()` checks at wing.py:1566-1580 are always False. Reproduced: control_surface_deflection=10 gives control_surfaces=[] and no warning. Regression from commit cc1d756f; twist_angle branch fixed, this wasn't.


#### Testing gaps

### 14. PyPI publish workflow has no test gate; any master push publishes even with failing tests

`.github/workflows/publish-on-master-push.yml:24` · testing · not independently verified · backwards-compat risk: none

publish-on-master-push.yml runs 'uv build && uv publish' on every master push with no dependency on the Tests workflow. Given develop CI is currently red, a merge to master would ship a broken release of a 1200-star PyPI package.

**Fix:** Add a test job in the same workflow and make deploy 'needs' it, or trigger via workflow_run on Tests with conclusion==success; add a protected environment.

### 15. LiftingLine and NonlinearLiftingLine (public top-level solvers) have zero executable tests

`aerosandbox/aerodynamics/aero_3D/test_aero_3D/test_lifting_line.py:1` · testing · not independently verified — found independently by 2 reviewers · backwards-compat risk: none

test_lifting_line.py is entirely commented out, and no other test references these classes. Coverage: lifting_line.py 14%, nonlinear_lifting_line.py 7%, despite both being exported in aerosandbox/__init__.py. Any regression in these solvers would ship silently.

**Fix:** Add real tests: run both solvers on the shared test geometries and assert CL/CDi against VLM/analytic elliptic-wing results within tolerance.


#### API design

### 16. Dependency floors pinned to latest-at-time patch releases, over-constraining installs

`pyproject.toml:42` · api-design · not independently verified · backwards-compat risk: none

Floors like numpy>=2.2.6, matplotlib>=3.10.7, pandas>=2.3.3, scipy>=1.15.3, casadi>=3.7.2 mirror whatever was newest when `uv add` ran, not the true minimum the code needs (4.2.6 changelog says NumPy 2.0.0+ works). For a widely-used library this forces upgrades and creates resolver conflicts with users' pinned scientific stacks; nothing validates the floors.

**Fix:** Lower floors to actual tested minimums (e.g. numpy>=2.0) and add a CI job using `uv run --resolution lowest-direct` to validate them.


#### Documentation & docstrings

### 17. Install docs state wrong minimum Python (3.8) and link to nonexistent setup.py

`INSTALLATION.md:12` · docs · not independently verified — found independently by 2 reviewers · backwards-compat risk: none

Line 12 says the minimum Python version is in `./setup.py` and is 3.8 'at the time of writing (July 2023)'. The repo migrated to pyproject.toml/hatchling — setup.py does not exist (broken relative link) — and `requires-python = ">=3.10"`. Users on 3.8/3.9 following this doc will hit an unexplained pip resolution failure. The list numbering (1. then nested 2.) is also malformed.

**Fix:** Reference pyproject.toml's `requires-python` and state Python >=3.10; fix list numbering.


#### Dead code

### 18. requirements.txt is stale and mutually unsatisfiable with pyproject.toml

`requirements.txt:1` · dead-code · not independently verified — found independently by 2 reviewers · backwards-compat risk: low

Pins directly conflict with pyproject dependencies: neuralfoil~=0.2.4 vs >=0.3.0,<0.4.0; casadi~=3.6.5 vs >=3.7.2; numpy~=2.0.1 vs >=2.2.6; dill~=0.3.8 vs >=0.4.0; trimesh~=3.22.4 vs >=4.9.0. It also lists ipyvtklink (deprecated/archived, absent from pyproject). Users following `pip install -r requirements.txt` get an environment that cannot run the package as declared.

**Fix:** Delete requirements.txt (uv.lock is the lockfile), or regenerate via `uv export --no-hashes -o requirements.txt` in CI.


---

## 3. Medium severity

Real defects with narrower blast radius: edge-case crashes, contract violations between docstring and behavior, silently-ignored arguments.


#### Bugs (wrong results or crashes)

### 19. Read the Docs build is guaranteed to fail: Python 3.8 pinned but package requires >=3.10

`.readthedocs.yaml:6` · bug · verified — found independently by 2 reviewers · backwards-compat risk: none

RTD config pins `python: "3.8"` and `ubuntu-20.04` (EOL image), then runs `pip install .[docs]`. pyproject.toml declares `requires-python = ">=3.10"` and docs extras (`sphinx>=8.1.3`, `furo>=2025.9.25`) require Python >=3.10, so pip refuses to install and every docs build fails. README's docs link and 'Documentation Status' badge (readthedocs.io/en/master) therefore point at stale/failing docs.

**Fix:** Set `os: ubuntu-24.04` and `python: "3.12"` (or any >=3.10) in .readthedocs.yaml; verify a docs build.

### 20. MSET failure silently swallowed unless stderr contains the X11 'BadName' string

`aerosandbox/aerodynamics/aero_2D/mses.py:249` · bug · verified · backwards-compat risk: none

The `except subprocess.CalledProcessError` around mset() only raises when 'BadName (named color or font does not exist)' is in stderr; any other MSET failure (bad geometry, missing executable via shell exit 127, license issue) is printed and swallowed, so run() proceeds without a mesh and fails later with a confusing MSES error. Also typo 'becausee'.

**Fix:** Add `else: raise` (or re-raise e) after the BadName check; fix the typo.

### 21. MSES 'terminate' behavior silently ignored when verbosity=0 due to misindented break

`aerosandbox/aerodynamics/aero_2D/mses.py:323` · bug · verified · backwards-compat risk: none

In run(), the `break` for behavior_after_unconverged_run='terminate' is indented inside `if self.verbosity >= 1:`. With verbosity=0, an unconverged run does not terminate the sweep; all subsequent points are still executed on the stale unconverged solution, contradicting the requested behavior.

**Fix:** Dedent `break` so it executes regardless of verbosity; keep only the print inside the verbosity check.

### 22. XFoil(hinge_point_x=None) crashes despite docstring saying None disables hinge moment

`aerosandbox/aerodynamics/aero_2D/xfoil.py:225` · bug · verified · backwards-compat risk: none

Docstring (line 92-93) says 'If this is None, the hinge moment is not calculated', and alpha()/cl() guard on None (lines 633, 711). But _default_keystrokes always emits hinc/fnew/fmom: `float(self.hinge_point_x)` and `self.airfoil.local_camber(None)` raise TypeError when hinge_point_x=None.

**Fix:** Wrap the hinc/fnew/fmom keystrokes (lines 224-229) in `if self.hinge_point_x is not None:`, and change the type hint to `float | None`.

### 23. alpha()/cl() unconditionally sort inputs, contradicting documented start_at=None behavior

`aerosandbox/aerodynamics/aero_2D/xfoil.py:626` · bug · verified · backwards-compat risk: low

Docstrings promise that with start_at=None 'the alpha inputs are run as a single sequence in the order given', but `alphas = np.sort(alphas)` (line 626) and `cls = np.sort(cls)` (line 704) always sort, destroying user-specified run ordering that matters for XFoil convergence continuation.

**Fix:** Only sort when the start_at split path is taken; otherwise preserve input order (outputs are already re-sorted by alpha before returning).

### 24. AVL.write_avl(filepath=None) crashes despite docstring promising string return; None-check is dead code

`aerosandbox/aerodynamics/aero_3D/avl.py:595` · bug · verified · backwards-compat risk: none

Docstring says 'If None, this function returns the .avl file as a string', but `filepath = Path(filepath)` at line 595 raises TypeError on None, making the `if filepath is not None` at line 617 unreachable; airfoil sidecar files are also written to literal 'None.af0' paths (line 560). Identical defect in write_avl_bfile (line 642). Return annotation `-> None` also contradicts the docstring.

**Fix:** Guard `Path(filepath)` behind a None check, return the string when filepath is None, and fix the return annotations/docstrings.

### 25. ISA model uses g=9.81 instead of standard g0=9.80665; docstring claims 'exactly reproduced'

`aerosandbox/atmosphere/_isa_atmo_functions.py:11` · bug · verified · backwards-compat risk: low

US Standard Atmosphere 1976 / ISO 2533 define g0'=9.80665 m/s^2. Verified deviations vs. official table: -0.051% at 11 km, -0.163% at 32 km, -0.346% at 71 km. This also seeds the 'differentiable' model knot points. Atmosphere docstring (atmosphere.py:47) claims the ISA is 'exactly reproduced', and density_altitude already uses 9.80665 (inconsistent).

**Fix:** Set g = 9.80665. Also replace hardcoded 9.81 in OperatingPoint.energy_altitude (operating_point.py:330) with a shared constant.

### 26. PerfectGas.process() with enthalpy_addition_* yields pressure=None or silently ignores input

`aerosandbox/atmosphere/thermodynamics/gas.py:240` · bug · verified · backwards-compat risk: low

temperature_specified is computed (line 198) before the enthalpy branches set new_temperature (lines 227-238), so T_ratio is never computed and no process branch fires. Verified: process('isochoric', enthalpy_addition_at_constant_volume=10e3) returns a gas with pressure=None (TypeError on .density); same for isentropic/polytropic. process('isothermal', enthalpy_addition_*) silently returns an unchanged gas instead of raising.

**Fix:** After computing new_temperature from enthalpy, set temperature_specified=True and compute T_ratio; isothermal then correctly raises and other processes propagate pressure.

### 27. AeroSandboxObject.__eq__ raises ValueError when array attributes have different shapes

`aerosandbox/common.py:78` · bug · verified · backwards-compat risk: none

__eq__ does np.all(self.__dict__[key] == other.__dict__[key]). With NumPy 2.x, comparing non-broadcastable arrays raises ValueError instead of returning False. Verified: asb.Airfoil('naca0012') == Airfoil('naca0012', coordinates=coords[::2]) raises 'operands could not be broadcast together with shapes (399,2) (200,2)'. Every subclass (Airfoil, Wing, Airplane, MassProperties) crashes on == against a same-type object whose array attributes differ in length.

**Fix:** Wrap the per-key comparison in try/except (ValueError, TypeError): return False, or compare np.shape() of both values before elementwise ==.

### 28. draw() with a filepath string: pv.read() result discarded, then AttributeError

`aerosandbox/dynamics/point_mass/common_point_mass.py:470` · bug · verified · backwards-compat risk: none

In the str branch, pv.read(filename=vehicle_model) is called but not assigned, so vehicle_model stays a string and vehicle_model.bounds at line 488 raises AttributeError: 'str' object has no attribute 'bounds' (verified). The type hint (line 427) and the TypeError message (line 475) also omit str even though it is intended to be supported.

**Fix:** Assign: vehicle_model = pv.read(filename=vehicle_model). Add str to the vehicle_model type hint and the TypeError message.

### 29. draw() crashes for trajectories of length 1-3: spline degree k not reduced below point count

`aerosandbox/dynamics/point_mass/common_point_mass.py:539` · bug · verified · backwards-compat risk: none

InterpolatedUnivariateSpline requires number of points m > degree k, but k=np.clip(len(self),1,3) sets k equal to m for len(self) in {1,2,3}, raising ValueError 'm must be > k' (verified). So drawing any dynamics instance with 1, 2, or 3 time points always fails.

**Fix:** Use k=int(np.clip(len(self)-1, 1, 3)) and special-case len(self)==1 (skip interpolation, use constant values).

### 30. kinetic_energy and rotational_kinetic_energy raise AttributeError on all point-mass classes

`aerosandbox/dynamics/point_mass/common_point_mass.py:778` · bug · verified · backwards-compat risk: none

The point-mass base class's rotational_kinetic_energy references self.p/q/r, which no point-mass subclass defines. So dyn.rotational_kinetic_energy and dyn.kinetic_energy raise AttributeError on every DynamicsPointMass* instance (verified). A point mass has zero rotational KE; the property belongs only on the rigid-body base, which already overrides it.

**Fix:** Return 0 from rotational_kinetic_energy in _DynamicsPointMassBaseClass (rigid-body override already provides the inertia-based version).

### 31. rotational_kinetic_energy ignores products of inertia (Ixy, Iyz, Ixz)

`aerosandbox/dynamics/rigid_body/common_rigid_body.py:64` · bug · verified · backwards-compat risk: none

Correct formula is KE = 0.5*w^T I w. With MassProperties' tensor-element convention (I12=Ixy etc., mass_properties.py:20-22), that is 0.5*(Ixx p^2 + Iyy q^2 + Izz r^2) + Ixy p q + Iyz q r + Ixz p r. The cross terms are omitted, giving wrong energy whenever products of inertia are nonzero (Ixz is typically nonzero for aircraft).

**Fix:** Add + Ixy*p*q + Iyz*q*r + Ixz*p*r terms (outside the 0.5 factor, per the stored tensor-element sign convention).

### 32. Airfoil constructor transposes the shape tuple, not the coordinates, for 2xN input

`aerosandbox/geometry/airfoil/airfoil.py:106` · bug · verified · backwards-compat risk: none

In __init__, `coordinates = np.transpose(shape)` operates on the shape tuple instead of the array. Passing a 2xN array (documented as supported by the shape check on line 104) silently sets `self.coordinates` to a 2-element 1D array of the dimensions. Verified live: `Airfoil('test', coordinates=coords.T)` yields `coordinates == [2, 399]`.

**Fix:** Change to `coordinates = np.transpose(coordinates)`.

### 33. TE_angle() dot-product typo returns wrong trailing-edge angle

`aerosandbox/geometry/airfoil/airfoil.py:1003` · bug · verified · backwards-compat risk: none

The second arctan2d argument uses `upper_TE_vec[1] * upper_TE_vec[1]` where the dot product u·l requires `upper_TE_vec[1] * lower_TE_vec[1]`. Silently wrong result for any airfoil whose TE vectors have differing y-components. Verified: naca2412 returns 15.095 deg vs correct 15.939 deg.

**Fix:** Replace `upper_TE_vec[1] * upper_TE_vec[1]` with `upper_TE_vec[1] * lower_TE_vec[1]`.

### 34. add_control_surface() raises AttributeError for any normally-constructed Airfoil

`aerosandbox/geometry/airfoil/airfoil.py:1296` · bug · verified · backwards-compat risk: low

Modern Airfoils (constructed without deprecated kwargs) have no CL/CD/CM_function attributes. `add_control_surface(modify_polars=False)` crashes immediately at `self.CL_function`; with default modify_polars=True, the returned airfoil's polar functions crash at first call. Verified live both ways. Also, line 1300 passes deprecated kwargs back into Airfoil(), emitting a spurious DeprecationWarning.

**Fix:** Only wrap/copy polar functions when `hasattr(self, 'CL_function')`; otherwise skip. Set functions via attribute assignment instead of deprecated constructor kwargs.

### 35. get_kulfan_parameters(method='opti') crashes with NameError: 'asb' not defined

`aerosandbox/geometry/airfoil/airfoil_families.py:416` · bug · verified · backwards-compat risk: none

The 'opti' branch calls `asb.Opti()` but `aerosandbox` is never imported in module or function scope (only inside the __main__ block). The documented Literal['least_squares','opti'] option 'opti' is therefore unusable. Verified live: NameError.

**Fix:** Add `import aerosandbox as asb` (or `from aerosandbox.optimization import Opti`) at the top of the 'opti' branch.

### 36. fineness_ratio(assumed_shape="sears-haack") is 2x too large (uses radius, not diameter)

`aerosandbox/geometry/fuselage.py:257` · bug · verified · backwards-compat risk: low

Docstring defines FR = length / max_diameter, and the cylinder branch matches, but the Sears-Haack branch returns `length / r_max`. Verified numerically: an exact Sears-Haack body with L/d=5 returns 10.0. Also note caller torenbeek_weights.py:422 passes "sears_haack" (underscore), which raises ValueError.

**Fix:** Return `length / (2 * r_max)`; optionally accept the "sears_haack" spelling.

### 37. FuselageXSec.compute_frame() returns non-unit axes for tilted normals, shrinking cross-sections

`aerosandbox/geometry/fuselage.py:854` · bug · verified · backwards-compat risk: none

zg_local is Gram-Schmidt-projected but never renormalized; yg_local = cross(zg, xg) inherits the deficit. Verified: FuselageXSec(xyz_normal=[1,0,1], radius=1) yields perimeter points at radius 0.707 (scaled by cos(tilt)). Affects get_3D_coordinates, meshing, drawing, and CAD export whenever xyz_normal is not x-aligned. Propulsor.compute_frame (propulsor.py:88) has the identical defect.

**Fix:** After projection: `zg_local = zg_local / np.linalg.norm(zg_local)` in both classes.

### 38. Polygon.J() computes polar moment about origin, not centroid as documented

`aerosandbox/geometry/polygon.py:263` · bug · verified · backwards-compat risk: low

Docstring says "taken about the centroid", but J() sums origin-referenced Ixx+Iyy without the parallel-axis correction that Ixx()/Iyy() apply. Verified: unit square centered at (10.5,10.5) returns 220.67 instead of 1/6. Line 257 is also a dead statement (`0.5 * np.sum(a)` discarded).

**Fix:** Return `self.Ixx() + self.Iyy()` (centroidal), and delete the dead statement.

### 39. test_cadquery_export references non-existent asb.np, failing every CI run since Nov 2025

`aerosandbox/geometry/test_geometry/test_cadquery_export.py:22` · bug · verified · backwards-compat risk: none

The test builds geometry with asb.np.array(...), but aerosandbox/__init__.py has never exported 'np' (removed 2021, commit beea45aa). AttributeError at line 22 makes this test fail unconditionally; it has kept develop CI red for ~8 months (commit 657015af, 2025-11-05), and PRs (e.g. #177) were merged over red CI, masking new failures.

**Fix:** Use 'import aerosandbox.numpy as np' in the test (or plain lists). Then treat red CI as merge-blocking via branch protection.

### 40. Wing.mesh_line with iterable x_nondim and add_camber=True crashes or gives wrong points

`aerosandbox/geometry/wing.py:1187` · bug · verified · backwards-compat risk: none

Inside the per-xsec loop, camber is evaluated as `local_camber(x_over_c=x_nondim)` using the full input array instead of the per-xsec scalar `xsec_x_nondim`. Verified: with 4 xsecs and x_nondim of length 4, raises broadcast ValueError; with exactly 3 xsecs it silently adds camber values into the wrong xyz components.

**Fix:** Change to `local_camber(x_over_c=xsec_x_nondim)`.

### 41. Gust-entry offset mixes dimensional chord with nondimensional reduced time

`aerosandbox/library/aerodynamics/unsteady.py:132` · bug · verified · backwards-compat risk: low

indicial_gust_response (line 132) and calculate_lift_due_to_transverse_gust's integrand (line 180) compute offset = chord/2 * (1 - cos(alpha)) in METERS, then subtract it from reduced time (semichords). Reduced time is already chord-normalized: the offset in semichords is (1 - cos(alpha)), independent of chord. For chord != 2 m the offset is scaled wrongly; results at nonzero alpha depend spuriously on chord units.

**Fix:** Use offset = (1 - np.cos(angle_of_attack_radians)) (dimensionless semichords) in both places.

### 42. pitching_through_transverse_gust crashes with float angle_of_attack despite advertised support

`aerosandbox/library/aerodynamics/unsteady.py:308` · bug · verified · backwards-compat risk: none

Signature/docstring accept `Callable | float` for angle_of_attack, but it is forwarded to added_mass_due_to_pitching, which calls angle_of_attack(s) unconditionally (line 263). Passing a float raises TypeError: 'float' object is not callable (verified).

**Fix:** Normalize once at top: if not callable, wrap as `lambda s: value` before passing to the three sub-models.

### 43. thrust_coefficient crashes with default arguments; docstring promises None handling

`aerosandbox/library/propulsion_small_solid_rocket.py:163` · bug · verified · backwards-compat risk: low

Docstring says p_a and er are optional ('If None, then p_a = exit_pressure'), but the None-handling code is commented out and line 163 unconditionally computes `er * (exit_pressure - p_a) / chamber_pressure`. Verified: thrust_coefficient(2e6, 1e5, 1.2) raises TypeError. Any call omitting p_a or er crashes.

**Fix:** Implement documented behavior: if p_a is None, set p_a = exit_pressure; skip (or zero) the pressure-thrust term when er is None; raise ValueError if only one is given.

### 44. Nacelle exit-area weight term uses `2.5 * 0.0363` instead of `2.5 + 0.0363` per TASOPT

`aerosandbox/library/propulsion_turbofan.py:139` · bug · verified · backwards-compat risk: low

Drela's TASOPT nacelle weight model (TASOPT_doc.pdf, 'Turbofan Weight Model from Historical Data', after Onat & Klees) gives unit weights: inlet 2.5 + 0.0238*d_fan, exit 2.5 + 0.0363*d_fan (lb/ft^2, d_fan in inches). Line 137 correctly uses '+' for the inlet, but line 139 uses '*' for the exit, overweighting the exit cowl ~24% for CFM56-class fans and mis-scaling with diameter.

**Fix:** Change `(2.5 * 0.0363 * d_fan_in)` to `(2.5 + 0.0363 * d_fan_in)`.

### 45. n_gear misread as gear count; Raymer's N_gear is the gear load factor (~3)

`aerosandbox/library/weights/raymer_cargo_transport_weights.py:288` · bug · verified · backwards-compat risk: HIGH (defer to v5)

Raymer defines N_l = N_gear * 1.5 where N_gear is the gear load factor from Table 11.5 (typically ~3), an aircraft-level constant. Code (lines 288, 334; also raymer_general_aviation_weights.py lines 246, 285) uses 'number of landing gear' (defaults 2 and 1), giving N_l = 3 and 1.5 instead of ~4.5, and inconsistent main/nose values — underestimating nose gear mass up to ~46%.

**Fix:** Add a gear_load_factor parameter (default 3.0) and compute N_l = gear_load_factor * 1.5; deprecate the n_gear-based load factor. Semantics change, so defer to v5.

### 46. GA mass_hydraulics uses fuselage width where Raymer's equation uses design gross weight

`aerosandbox/library/weights/raymer_general_aviation_weights.py:458` · bug · verified · backwards-compat risk: HIGH (defer to v5)

Raymer's GA hydraulics equation (5th Ed., Sec. 15.3.3) is W_hyd = K_h * W_dg^0.8 * M^0.5 with W_dg in lb (the K_h values quoted in the code's own comment come from that equation). The code substitutes fuselage_width^0.8, yielding ~0.01 lb for a typical light aircraft instead of a few lb — off by orders of magnitude.

**Fix:** Use (design_mass_TOGW/u.lbm)**0.8; requires adding a design_mass_TOGW parameter and deprecating fuselage_width.

### 47. InterpolatedModel.x_data and y_data use mismatched flattening orders, scrambling point pairing

`aerosandbox/modeling/interpolation.py:128` · bug · verified · backwards-compat risk: low

x_data is built from meshgrid(indexing='ij') flattened C-order (reshape(-1), line 124), but y_data uses np.ravel(order='F') (line 128). Verified numerically: for a 3x4 grid, x_data[i]/y_data[i] do not correspond to the same point. Any user (or plotting code) pairing these public attributes gets scrambled data for N>=2 dimensions.

**Fix:** Use C-order consistently: self.y_data = np.ravel(y_data_structured) (drop order='F'), matching the C-order x_data flattening.

### 48. Unsorted 1D point-cloud input constructs successfully but crashes at call time

`aerosandbox/modeling/interpolation_unstructured.py:92` · bug · verified · backwards-compat risk: low

The 1D shortcut passes raw x_data straight to InterpolatedModel, which requires strictly increasing coordinates. Verified: UnstructuredInterpolatedModel(x_data=unsorted 1D array, y_data) builds fine, then m(5.0) raises CasADi RuntimeError 'Gridpoints must be strictly increasing'. Unsorted data is the norm for the advertised unstructured/point-cloud use case.

**Fix:** In the 1D path, sort x (np.argsort) and reorder y before super().__init__(); raise a clear ValueError on duplicate x values.

### 49. 1D/structured shortcut path silently discards fill_value and interpolated_model_kwargs

`aerosandbox/modeling/interpolation_unstructured.py:93` · bug · verified · backwards-compat risk: none

The early-return path (lines 92-99) calls super().__init__() without fill_value or interpolated_model_kwargs. Verified: UnstructuredInterpolatedModel(x_data=1D array, y_data, fill_value=None) still returns NaN outside the domain instead of extrapolating; a 'method' override is likewise ignored. Also, x_data_raw_unstructured/y_data_raw attributes are never set on this path.

**Fix:** Pass fill_value and **interpolated_model_kwargs into the early super().__init__() call, and set x_data_raw_unstructured/y_data_raw before returning.

### 50. SurrogateModel.plot() crashes for 1D models with dict x_data: dict_keys is not subscriptable

`aerosandbox/modeling/surrogate_model.py:121` · bug · verified · backwards-compat risk: none

plot() uses self.x_data.keys()[0] and self.x_data.values()[0]; dict views are not subscriptable in Python 3 (Python 2 relic). Verified: any 1D FittedModel/InterpolatedModel built with dict x_data raises TypeError: 'dict_keys' object is not subscriptable on .plot(), a feature advertised in the FittedModel docstring.

**Fix:** Use next(iter(self.x_data.keys())) and next(iter(self.x_data.values())), or list(...)[0].

### 51. trapz deprecation message directs users to `asb.numpy.integrate_discrete_intervals`, which is not importable

`aerosandbox/numpy/__init__.py:55` · bug · verified · backwards-compat risk: none

aerosandbox/numpy/__init__.py never imports the `integrate_discrete` module, so `np.integrate_discrete_intervals` and `np.integrate_discrete_squared_curvature` raise AttributeError (verified). Yet calculus.py lines 274, 294, 301 (trapz docstring + PendingDeprecationWarning) tell users to migrate to exactly that name, and internal code must deep-import it (opti.py line 1225).

**Fix:** Add `from aerosandbox.numpy.integrate_discrete import integrate_discrete_intervals, integrate_discrete_squared_curvature` to aerosandbox/numpy/__init__.py.

### 52. mod() CasADi branch returns divisor instead of 0 when x1 is a negative exact multiple

`aerosandbox/numpy/arithmetic_dyadic.py:137` · bug · verified · backwards-compat risk: low

The sign correction `where(x1 < 0, out + x2, out)` keys on the sign of x1, not of the fmod remainder. When x1 < 0 divides evenly, fmod gives 0 and the code adds x2. Verified: mod(DM(-4), 2) returns 2; numpy gives 0. Breaks angle wrapping at exact period multiples under the CasADi backend.

**Fix:** Condition on the remainder: `out = where(out < 0, out + x2, out)` (or handle divisor sign: `out * sign(x2) < 0`).

### 53. mean(axis=None) on a 2D CasADi matrix returns a vector, not a scalar

`aerosandbox/numpy/arithmetic_monadic.py:92` · bug · verified · backwards-compat risk: none

The axis=None recursion is `mean(mean(x, axis=0), axis=1)`; the inner mean yields an n-by-1 column, and mean(col, axis=1) is a no-op, so the column is returned. Verified: mean(DM([[1,2],[3,4]])) returns [2, 3] instead of 2.5. Docstring promises a scalar for axis=None.

**Fix:** Use `return sum(x, axis=None) / (x.shape[0] * x.shape[1])` or recurse with axis=0 twice.

### 54. diff() with CasADi input ignores n: loop body uses `a` instead of `result`

`aerosandbox/numpy/calculus.py:65` · bug · verified · backwards-compat risk: none

In the CasADi branch, `for i in range(n): result = _cas.diff(a)` re-differences the original array each iteration, so any n >= 2 silently returns the first difference. Verified: diff(DM([1,4,9,16,25]), n=2) returns [3,5,7,9] (len 4) instead of [2,2,2] (len 3). Wrong values and wrong shape.

**Fix:** Change loop body to `result = _cas.diff(result)`.

### 55. `aerosandbox.numpy.integrate` attribute is scipy.integrate, shadowing the ASB submodule

`aerosandbox/numpy/integrate.py:4` · bug · verified · backwards-compat risk: low

integrate.py does `from scipy import integrate`; the star-import in aerosandbox/numpy/__init__.py line 55 then binds that leaked name over the submodule attribute. Verified: `np.integrate` is `scipy.integrate`, so `np.integrate.quad`/`np.integrate.solve_ivp` silently call scipy versions that crash on CasADi expressions, while `from aerosandbox.numpy.integrate import quad` gives the dual-backend one — two different functions at the same-looking path.

**Fix:** Rename to `from scipy import integrate as _integrate` in integrate.py (and use `_integrate` internally), restoring the submodule attribute.

### 56. solve_ivp CasADi backend silently ignores t_eval

`aerosandbox/numpy/integrate.py:286` · bug · verified · backwards-compat risk: low

The CasADi branch hardcodes `simtime_eval = np.linspace(0, 1, 100)`. The docstring says t_eval controls output times (with 100 points only 'if None'), but a user-supplied t_eval is dropped. Verified: requesting 7 eval points returns 100. Downstream shape mismatches when users align results with their t_eval.

**Fix:** If t_eval is not None, set `simtime_eval = (np.array(t_eval) - t0) / (tf - t0)`; keep 100-point default otherwise.

### 57. interp() with period and CasADi input mishandles x outside [0, period) (negative x)

`aerosandbox/numpy/interpolate.py:41` · bug · verified · backwards-compat risk: none

The CasADi branch uses `_cas.fmod(x, period)`, which returns negative values for negative x, so the point lands left of xp[0] and is flat/linearly extrapolated instead of wrapped. Verified: interp(DM(-90), xp=[0..360], fp=sin, period=360) returns -1.446 vs numpy's -0.955. Silent wrong results.

**Fix:** Use the package's own `mod()` (after fixing it) or `_cas.fmod(x, period) + where(... < 0, period, 0)` to wrap into [0, period).

### 58. norm() returns wrong matrix norms for ord=1 and ord=2 under CasADi backend

`aerosandbox/numpy/linalg.py:254` · bug · verified · backwards-compat risk: none

For a CasADi matrix with axis=None, ord=1 computes sum of all |entries| and ord=2 computes Frobenius-style sum(x**2)**0.5, but numpy.linalg.norm defines ord=1 as max column sum and ord=2 as spectral norm. Verified: norm([[1,2],[3,4]], ord=1) gives 6.0 (NumPy) vs 10 (CasADi); ord=2 gives 5.465 vs 5.477. Silent backend divergence.

**Fix:** In the CasADi branch, when input is a true matrix (axis=None), implement max-column-sum for ord=1 and raise NotImplementedError for ord=2, instead of silently returning entrywise reductions.

### 59. norm() CasADi branch silently ignores axis for ord=inf (and 'fro')

`aerosandbox/numpy/linalg.py:262` · bug · verified · backwards-compat risk: low

For ord='inf'/np.inf the code always calls `_cas.norm_inf(x)` on the whole matrix, ignoring a user-specified axis. Verified: norm(DM([[1,2],[3,4]]), ord='inf', axis=1) returns scalar 4; numpy returns [2, 4]. Same issue for ord='fro' with an axis.

**Fix:** When axis is not None and ord is inf, compute per-axis: `max(abs(x), axis=axis)` using the package's CasADi-aware max.

### 60. norm(..., keepdims=True) crashes with TypeError for CasADi matrix input

`aerosandbox/numpy/linalg.py:276` · bug · verified — found independently by 2 reviewers · backwards-compat risk: none

In the CasADi branch, when axis remains None (Frobenius/inf matrix norms), the keepdims path executes new_shape[axis] = 1 with axis=None. Verified: np.linalg.norm(cas.DM([[1,2],[3,4]]), keepdims=True) raises 'TypeError: list indices must be integers or slices, not NoneType', while the NumPy backend returns [[5.477]].

**Fix:** When axis is None and keepdims=True, reshape to (1, 1) instead of indexing new_shape with None.

### 61. geomspace with CasADi inputs returns linearly-spaced, not geometrically-spaced values

`aerosandbox/numpy/spacing.py:203` · bug · verified · backwards-compat risk: none

The CasADi branch computes `_onp.log10(10 ** linspace(start, stop, num))`, which is an identity (log10(10**x) == x), yielding linear spacing. Verified: geomspace(DM(1), DM(100), 5) returns [1, 25.75, 50.5, 75.25, 100] instead of [1, 3.16, 10, 31.6, 100]. Silent wrong results in any optimization using geomspace.

**Fix:** Use `10 ** linspace(_cas.log10(start), _cas.log10(stop), num)` (log the endpoints, not the result).

### 62. sigmoid(sigmoid_type="logistic") raises ValueError due to `== ("tanh" or "logistic")`

`aerosandbox/numpy/surrogate_model_tools.py:289` · bug · verified · backwards-compat risk: none

`("tanh" or "logistic")` evaluates to "tanh", so "logistic" — documented in the docstring, listed in the Literal type, and even named in the error message as valid — falls through to the else branch and raises ValueError. Verified crash.

**Fix:** Change to `if sigmoid_type in ("tanh", "logistic"):`.

### 63. Constraint declaration tracking off by one frame on Python 3.12+ for constraint lists

`aerosandbox/optimization/opti.py:421` · bug · verified · backwards-compat risk: none

subject_to recurses via a list comprehension with _stacklevel=_stacklevel+2, assuming the comprehension adds a stack frame. PEP 709 (Python 3.12) inlines comprehensions, so find_constraint_declaration and show_infeasibilities report the caller's caller, not the subject_to([...]) line. Verified: list-declared constraints report the wrong line; scalar constraints report correctly. Lists are the most common declaration pattern.

**Fix:** Use +1 when sys.version_info >= (3,12), else +2; or replace the comprehension with a plain for-loop and always use +1.

### 64. plate_buckling_critical_load double-counts the pi^2/(12(1-nu^2)) factor; results ~9.5% low

`aerosandbox/structures/buckling.py:192` · bug · verified · backwards-compat risk: none

The tabulated constants K=3.62/6.35/0.385 already embed pi^2/(12(1-nu^2)) at nu=0.3 (NACA TN 3781 gives k_c=4.00/6.98/0.425; e.g. 4.00*pi^2/(12*(1-0.09))=3.615). Line 192 multiplies by pi^2/(12(1-poissons_ratio^2)) again, so the returned critical load is ~0.905x the reference value (verified numerically). The formula contradicts its cited reference (Stress Analysis Manual Sec. 6.3 / NACA TN 3781).

**Fix:** Use the true k_c coefficients (4.00 pin-pin, 6.98 clamp-clamp, 0.425 free edge) with the explicit pi^2/(12(1-nu^2)) factor, so poissons_ratio stays meaningful.

### 65. TubeSparBendingStructure crashes on default (zero) or net-downward distributed load

`aerosandbox/structures/tube_spar_bending.py:260` · bug · verified · backwards-compat risk: none

scale=np.sum(np.trapz(distributed_force)*dy)*length**4/EI_guess is 0 for the default bending_distributed_force_function=0.0 and negative for any net-downward load; Opti.variable raises ValueError "The 'scale' argument must be a positive number." (opti.py:288). Verified: constructing with defaults or with bending_distributed_force_function=-100.0 both raise ValueError.

**Fix:** Wrap the scale expression in np.abs(...) and fall back to a positive heuristic (e.g. 1) when it is zero.

### 66. show_plot(legend_inline=True) crashes with IndexError if any axes has no lines (e.g., colorbar)

`aerosandbox/tools/pretty_plots/formatting.py:433` · bug · verified · backwards-compat risk: none

show_plot loops over all figure axes calling labelLines(lines=ax.get_lines()); labellines/core.py:196 does `lines[0].axes`, so any axes without lines (colorbar axes, empty subplot) raises IndexError. Verified: a plot with two labeled lines plus a colorbar crashes with `IndexError: list index out of range` when legend_inline=True.

**Fix:** Skip axes where len(ax.get_lines())==0 (or where no line has a non-underscore label) before calling labelLines; also guard in labelLines itself.

### 67. pie() crashes with ValueError when sort_by is a NumPy array (documented-valid input)

`aerosandbox/tools/pretty_plots/plots/pie.py:41` · bug · verified · backwards-compat risk: none

`elif sort_by == "values":` compares an ndarray to a string elementwise; `bool()` of the result raises "truth value of an array is ambiguous". The function's own error message documents "an array of numbers corresponding to each pie slice" as valid input. Verified: pie(values=[3,1,2], names=..., sort_by=np.array([2,0,1])) raises ValueError.

**Fix:** Check `isinstance(sort_by, str)` first, then compare string values inside that branch; treat non-str inputs as sort keys directly.

### 68. Dev dependency "ptest>=2.0.3" is a typo for pytest; installs unrelated abandoned package

`pyproject.toml:95` · bug · verified — found independently by 3 reviewers · backwards-compat risk: none

PyPI's 'ptest' is 'light test framework for Python', last released Nov 2020 — unrelated to pytest. It is resolved and locked in uv.lock (line 3085) and installed into every dev environment and every CI `uv run` (dev group is default-included). pytest is only present transitively via pytest-cov. This is also a typosquatting-shaped supply-chain hazard.

**Fix:** Replace "ptest>=2.0.3" with "pytest>=8.4.2" in [dependency-groups].dev and re-lock (uv lock).


#### Latent bugs & fragile code

### 69. bound_leg_YZ aliases vortex_bound_leg; in-place zeroing corrupts self.vortex_bound_leg after run()

`aerosandbox/aerodynamics/aero_3D/lifting_line.py:874` · latent-bug · verified · backwards-compat risk: none

`bound_leg_YZ = vortex_bound_leg` then `bound_leg_YZ[:, 0] = 0` mutates the same array stored as `self.vortex_bound_leg` (line 642). Results are correct within run() only because all other uses happen earlier, but any post-run consumer of the attribute silently gets x-components zeroed. Identical pattern in nonlinear_lifting_line.py lines 469-470 (attribute set at line 264).

**Fix:** Copy before mutating: `bound_leg_YZ = np.concatenate([np.zeros((N,1)), vortex_bound_leg[:, 1:]], axis=1)` or `vortex_bound_leg * np.array([[0, 1, 1]])`.

### 70. generate_polars crashes when cache_filename has no directory component

`aerosandbox/geometry/airfoil/airfoil.py:337` · latent-bug · verified · backwards-compat risk: none

`os.makedirs(os.path.dirname(cache_filename), exist_ok=True)` receives '' for a bare filename like 'polars.json', and `os.makedirs('')` raises FileNotFoundError (verified). So caching to the current directory crashes before XFoil even runs.

**Fix:** Guard: `dirname = os.path.dirname(cache_filename); if dirname: os.makedirs(dirname, exist_ok=True)`, or use `Path(cache_filename).parent.mkdir(parents=True, exist_ok=True)`.

### 71. Deprecated kwarg mutates get_kulfan_coordinates' shared mutable default array

`aerosandbox/geometry/airfoil/airfoil_families.py:226` · latent-bug · verified · backwards-compat risk: none

`lower_weights[0] = -1 * upper_weights[0]` under `enforce_continuous_LE_radius=True` writes into the module-level default `lower_weights=-0.2*np.ones(8)` when the caller omits lower_weights. All subsequent calls with default lower_weights return corrupted coordinates. Verified live: default output changes after one call with `upper_weights=0.35*ones(8), enforce_continuous_LE_radius=True`.

**Fix:** Use `lower_weights = np.copy(lower_weights)` before assignment; ideally make defaults None and construct arrays inside the function.

### 72. KulfanAirfoil.__init__ wipes all global warning filters via warnings.resetwarnings()

`aerosandbox/geometry/airfoil/kulfan_airfoil.py:39` · latent-bug · verified · backwards-compat risk: none

The IgnoreUserWarnings context manager's __exit__ calls `warnings.resetwarnings()`, which clears every filter processwide, not just the one it added. Verified live: a user's `warnings.simplefilter('error', FutureWarning)` is silently removed after `KulfanAirfoil('naca2412')`. This also breaks pytest filterwarnings config.

**Fix:** Replace the custom class with `with warnings.catch_warnings(): warnings.simplefilter('ignore', UserWarning)`, which restores prior filters on exit.

### 73. Deprecated `symmetric` kwarg check uses locals(); never fires, arg silently swallowed

`aerosandbox/geometry/fuselage.py:97` · latent-bug · verified — found independently by 2 reviewers · backwards-compat risk: none

`if "symmetric" in locals()` is always False because the argument lands in **kwargs. Verified: `Fuselage(symmetric=True)` is silently accepted with no error, instead of raising the intended deprecation error. Any typo'd kwarg to Fuselage/Wing is likewise silently ignored.

**Fix:** Check `"symmetric" in kwargs`; raise TypeError for any other unrecognized kwargs.

### 74. Polygon.perimeter() omits the closing edge for non-closed coordinate lists

`aerosandbox/geometry/polygon.py:142` · latent-bug · verified · backwards-compat risk: low

perimeter() uses np.diff without wrapping, while area()/centroid()/Ixx() treat the polygon as closed via np.roll. Verified: triangle [[0,0],[1,0],[0,1]] returns 2.414 instead of 3.414. Behavior/inconsistency is undocumented; for blunt-TE airfoils this silently excludes the TE base from wetted-area computations.

**Fix:** Close the loop (roll-based diff), or document the open-curve behavior; check Airfoil.perimeter/wetted-area callers before changing.

### 75. calculate_reduced_time silently truncates to integers when `time` is an int array

`aerosandbox/library/aerodynamics/unsteady.py:54` · latent-bug · verified · backwards-compat risk: none

reduced_time = np.zeros_like(time) inherits the input dtype. With integer time (e.g., np.arange(0,10)), each trapezoidal accumulation truncates: verified output [0,4,8,12,...] vs correct [0,4.22,8.89,14.0,...]. Wrong results with no warning.

**Fix:** Use np.zeros(np.size(time)) or zeros_like(time, dtype=float); better, vectorize with cumulative trapezoid instead of a Python loop.

### 76. field_length_analysis_torenbeek divides by unguarded climb angle; negative/zero T/W margin gives nonsense

`aerosandbox/library/field_lengths.py:180` · latent-bug · verified · backwards-compat risk: low

takeoff_airborne_distance = ... + obstacle_height / flight_path_angle_climb has no floor: if thrust_over_weight <= 1/(L/D), the climb angle is zero/negative, yielding division by zero or a negative field length silently. The sibling field_length_analysis() guards this exact case with np.softmax(angle, 0, softness=0.001) at line 328.

**Fix:** Apply the same softmax floor to flight_path_angle_climb (and the one-engine-out angle) before dividing, matching field_length_analysis.

### 77. In-place-mutation detection broken for dict x_data: zip iterates keys, not arrays

`aerosandbox/modeling/fitting.py:250` · latent-bug · verified · backwards-compat risk: low

When x_data is a dict, dict==dict raises ValueError, and the fallback zips x_data with x_data_original — iterating over KEYS, comparing equal key strings. Verified: a model doing x['x1'] *= 2 passes undetected; the documented TypeError never fires, the fit runs on mutated data, and mutated x_data is stored on self (re-mutated on every later __call__).

**Fix:** In the except branch, zip x_data.values() with x_data_original.values() when both are dicts.

### 78. interpn all-zero bspline workaround returns zeros with shape of xi, not of the result

`aerosandbox/numpy/interpolate.py:193` · latent-bug · verified · backwards-compat risk: none

The CasADi-bug workaround `return zeros_like(xi)` returns an (n_points, n_dims) array, while the normal path returns (n_points,). Verified: 2D grid of all zeros with xi shape (7,2) returns shape (7,2); nonzero values return (7,). Shape-dependent downstream code breaks only when the table happens to be all zeros.

**Fix:** Normalize xi first, then return `zeros_like(xi[:, 0])` (or `_onp.zeros(xi.shape[0])`).

### 79. interpn with fill_value=None mutates the caller's xi array in place

`aerosandbox/numpy/interpolate.py:223` · latent-bug · verified · backwards-compat risk: none

When fill_value is None, out-of-bounds clamping assigns into `xi[:, axis]`. Since `_onp.reshape` returns a view and 2D inputs are used directly, the caller's array is silently modified. Verified: passing xi=[[-0.5],[0.5],[1.5]] leaves it as [0, 0.5, 1] after the call.

**Fix:** Copy before clamping: `xi = _onp.array(xi)` (NumPy path) at the top of the CasADi/clamping branch, or build clamped values into a new variable.

### 80. log_transform variable with nonpositive lower_bound silently creates NaN constraint

`aerosandbox/optimization/opti.py:359` · latent-bug · verified · backwards-compat risk: none

np.log(lower_bound) with lower_bound <= 0 yields NaN (or -inf), producing a NaN constraint with only a NumPy RuntimeWarning. Verified: variable(init_guess=1, log_transform=True, lower_bound=-1) — a trivially satisfiable bound — creates fine, then solve() dies with an opaque CasADi/IPOPT RuntimeError. Same for upper_bound at line 364.

**Fix:** For log-transformed variables, skip bounds <= 0 for lower_bound (always satisfied) and raise a clear ValueError for upper_bound <= 0 (never satisfiable).

### 81. variable_categories_to_freeze as str breaks cache loading in solve() (iterates characters)

`aerosandbox/optimization/opti.py:645` · latent-bug · verified · backwards-compat risk: none

The __init__ hint allows Sequence[str] | str, and variable() handles the str case (line 295). But solve()'s load_frozen_variables_from_cache path iterates the value directly, so a str yields characters. Verified: variable_categories_to_freeze="Design" with load_frozen_variables_from_cache=True raises KeyError: 'D'. The documented "all" sentinel breaks the same way.

**Fix:** Normalize in __init__: if isinstance(variable_categories_to_freeze, str), wrap in a list (special-casing "all"); iterate the normalized list everywhere.

### 82. OperatingPoint.__init__ shares one mutable default Atmosphere across all instances

`aerosandbox/performance/operating_point.py:13` · latent-bug · verified · backwards-compat risk: low

atmosphere: Atmosphere = Atmosphere(altitude=0) is evaluated once at import. Verified: two default-constructed OperatingPoints share the same object; setting op1.atmosphere.altitude = 10000 changes op2.atmosphere.altitude too, silently corrupting unrelated instances (e.g., in sweeps or Dynamics subclasses).

**Fix:** Default to None and do `self.atmosphere = atmosphere if atmosphere is not None else Atmosphere(altitude=0)`; keep the type hint as Optional[Atmosphere].

### 83. indicated_airspeed() uses incompressible inversion of compressible impact pressure

`aerosandbox/performance/operating_point.py:308` · latent-bug · verified · backwards-compat risk: low

Computes sqrt(2*(P_t - P_s)/rho_0): total pressure is computed compressibly but inverted with incompressible Bernoulli. Standard calibrated airspeed is CAS = a0*sqrt((2/(gamma-1))*((qc/P0 + 1)**((gamma-1)/gamma) - 1)) (standard pitot equation). At M=0.8 sea level the current formula gives ~294 m/s vs. correct 272 m/s (~8% high); error grows with Mach.

**Fix:** Implement the compressible CAS relation above (a0, P0 from sea-level ISA); document that IAS is approximated as CAS.

### 84. Uses Axis.converter attribute, deprecated in Matplotlib 3.10 and removed in 3.12

`aerosandbox/tools/pretty_plots/labellines/core.py:272` · latent-bug · verified · backwards-compat risk: none

`isinstance(ax.xaxis.converter, DateConverter)` triggers MatplotlibDeprecationWarning on the installed mpl 3.11 ("deprecated in 3.10, will be removed in 3.12"). Once users upgrade to mpl 3.12, every labelLines()/show_plot(legend_inline=True) call raises AttributeError. Verified warning with warnings.simplefilter('error').

**Fix:** Use `ax.xaxis.get_converter()` when available: `conv = getattr(ax.xaxis, 'get_converter', lambda: ax.xaxis.converter)()`.

### 85. bootstrap_fits can loop forever if spline fits keep producing NaN

`aerosandbox/tools/statistics/time_series_uncertainty_quantification.py:205` · latent-bug · verified · backwards-compat risk: none

`while n_valid_splines < n_bootstraps` retries indefinitely when the fitted spline evaluates to NaN at the domain midpoint (e.g., pathological data/spline_degree). `n_attempted_splines` is incremented but never checked, so the process hangs with a stalled tqdm bar instead of raising.

**Fix:** Raise ValueError when n_attempted_splines exceeds e.g. 10*n_bootstraps, explaining that spline fitting is failing.

### 86. Import-time side effect: module globally overrides plotly's default renderer to 'browser'

`aerosandbox/visualization/plotly.py:7` · latent-bug · verified · backwards-compat risk: low

`pio.renderers.default = "browser"` runs at module import. aerosandbox/geometry/airfoil/airfoil.py:902 imports this module inside Airfoil.draw(backend='plotly'), so calling that method silently changes the user's global plotly renderer for the whole session — e.g., breaking inline figure rendering in Jupyter/VSCode notebooks for all subsequent plotly usage.

**Fix:** Remove the module-level assignment, or set it only inside spy()/plot_point_cloud, or only when pio.renderers.default is unset.

### 87. MassProperties.__array__ lacks NumPy 2.0 'copy' keyword; deprecated, will become TypeError

`aerosandbox/weights/mass_properties.py:199` · latent-bug · verified · backwards-compat risk: none

`__array__(self, dtype="O")` does not accept the `copy` keyword required by the NumPy 2.0 array protocol. Verified on numpy 2.4.6: np.array([mp, mp]) emits DeprecationWarning '__array__ implementation doesn't accept a copy keyword... must implement dtype and copy'. NumPy has stated this fallback will be removed, at which point any np.asarray(copy=...) on MassProperties raises TypeError.

**Fix:** Change signature to `def __array__(self, dtype=None, copy=None):` defaulting dtype to object internally; per protocol, honor/ignore copy explicitly.


#### Testing gaps

### 88. Python support matrix incomplete: 3.11/3.13/3.14 untested, classifiers omit versions

`.github/workflows/run-pytest.yml:20` · testing · not independently verified · backwards-compat risk: none

requires-python is >=3.10 with no ceiling, so pip installs on 3.11, 3.13, and 3.14, but CI only tests 3.10 and 3.12. Classifiers (pyproject.toml line 34) list only generic 'Programming Language :: Python :: 3', so PyPI shows no concrete supported versions. No 3.11+-only syntax found in the package, so 3.10 floor itself is sound.

**Fix:** Extend matrix to ['3.10','3.11','3.12','3.13'] (plus 3.14 if deps have wheels); add per-version trove classifiers.

### 89. "Without extras" CI job silently installs the dev dependency-group, so minimal install is never tested

`.github/workflows/run-pytest.yml:30` · testing · not independently verified — found independently by 2 reviewers · backwards-compat risk: none

`uv run --extra test` includes the dev group by default, which contains cadquery, pyvista, plotly, trimesh, sympy — the very optional deps the job claims to exclude. Evidence: test_cadquery_export runs (and fails) in this job instead of being skipped for missing cadquery. Headless-install breakage would not be caught. Also, push '**' plus pull_request '**' triggers duplicate runs per PR commit (runs 23448619799/23431688217).

**Fix:** Add `--no-default-groups` (or `--no-dev`) to the first pytest step; restrict push triggers to develop/master.

### 90. XFoil interface 10% covered; March 2026 list-input bugfix shipped with no regression test

`aerosandbox/aerodynamics/aero_2D/xfoil.py:599` · testing · not independently verified · backwards-compat risk: none

PR #177 fixed a TypeError in XFoil.alpha() with list input but added no test; xfoil.py sits at 10% coverage and mses.py at 11%. Input-normalization and output-parsing logic is pure Python and testable without the xfoil binary, so the same bug class can silently regress.

**Fix:** Add unit tests for XFoil.alpha() input handling (list/float/array, start_at paths) and output-file parsing using a canned polar fixture; skipif for binary-dependent paths.

### 91. AVL tests silently 'pass' when avl binary is absent and return non-None (pytest 9 break)

`aerosandbox/aerodynamics/aero_3D/test_aero_3D/test_avl.py:17` · testing · not independently verified · backwards-compat risk: none

Each test starts with 'if not avl_present: return', so CI always reports 4 green tests that executed nothing; a broken AVL interface would never be detected. They also have zero assertions and 'return analysis.run()', which raises PytestReturnNotNoneWarning today and fails under pytest 9 (dependabot PR already pending).

**Fix:** Use @pytest.mark.skipif(not avl_present, ...), assert on returned CL/CD values, and drop the return statements.

### 92. Core VLM solver validated only by 'assert aero is not None'; external-data validation not wired to pytest

`aerosandbox/aerodynamics/aero_3D/test_aero_3D/test_vortex_lattice_method.py:14` · testing · not independently verified · backwards-compat risk: none

All four VLM tests assert only that run() returns non-None — wrong lift/drag values would pass. Meanwhile the AeroBuildup low-AR validation against NASA TN D-3767 (Polhamus) data exists at test_aero_buildup/test_low_ar_wings_validation/low_ar_wings_validation.py but is not named test_* and has zero asserts, so it never runs in CI. (Good counterexamples exist: atmosphere vs ISA CSV, wing-moment vs 2D limit.)

**Fix:** Assert VLM CL/CDi against analytic references (e.g. flat plate CL_a = 2*pi/(1+2/AR), elliptic induced drag); convert the Polhamus study into a tolerance-based pytest test.

### 93. Five test_*.py files collect zero tests, giving phantom coverage

`aerosandbox/geometry/test_geometry/test_airplane.py:7` · testing · not independently verified · backwards-compat risk: none

pytest --collect-only confirms zero tests collected from: geometry/test_geometry/test_airplane.py (no test_ function), library/aerodynamics/test_aerodynamics/test_Cf_flat_plate.py (plot script only), geometry/airfoil/test_airfoil/test_airfoil_polar_generation.py (all under __main__), modeling/test_modeling/test_fitted_model.py (fixture only, no test), aero_3D/test_aero_3D/test_lifting_line.py (commented out).

**Fix:** Add real test functions with assertions, or rename/move plotting scripts out of test_* namespace so coverage gaps are visible.

### 94. library/, structures/, and visualization/ subsystems are ~0% tested

`aerosandbox/library/winds.py:1` · testing · not independently verified · backwards-compat risk: none

Coverage run: overall 51%; aerosandbox/library has ~20 modules at 0% (winds, power_solar, costs, mass_structural, field_lengths, all propulsion_*, all weights/raymer_* and torenbeek_weights), structures/buckling.py and tube_spar_bending.py 0%, visualization 0%, weights/mass_properties.py (top-level MassProperties class) only 33%. These are formula-heavy modules where regressions are invisible.

**Fix:** Add spot-check tests pinning known values from the cited references (Raymer, Torenbeek, ISA winds) and unit tests for MassProperties arithmetic (parallel-axis, __add__).


#### API design

### 95. List coordinates input silently discarded; shape validation via assert disappears under -O

`aerosandbox/geometry/airfoil/airfoil.py:103` · api-design · not independently verified · backwards-compat risk: low

Passing `coordinates=[[x0,y0],...]` (a natural array-like) hits `coordinates.shape` -> AttributeError, which is swallowed, leaving coordinates None with only a vague warning. Also, `assert len(shape)==2` validation is stripped when Python runs with -O, letting malformed arrays through.

**Fix:** Convert with `coordinates = np.asarray(coordinates)` first and raise explicit ValueError on bad shape instead of assert/pass.

### 96. Airplane.export_AVL's include_fuselages parameter has no effect

`aerosandbox/geometry/airplane.py:1016` · api-design · not independently verified · backwards-compat risk: none

export_AVL accepts `include_fuselages: bool = True` but never uses it; the AVL writer unconditionally writes BFILEs for all fuselages (avl.py:596). Passing include_fuselages=False silently exports fuselages anyway. Method also has no docstring, only TODO comments.

**Fix:** Wire the flag through to AVL (or strip fuselages from a copied airplane), and add a docstring.

### 97. No `__all__` in aerosandbox.numpy submodules: typing helpers leak into the public np namespace

`aerosandbox/numpy/__init__.py:49` · api-design · not independently verified · backwards-compat risk: low

The chained star-imports (lines 49-63) leak non-underscore module-level names. Verified present on `aerosandbox.numpy`: `Any`, `Callable`, `Literal`, `Sequence`, `cast` (typing.cast — especially confusing since NumPy 1.x had a different `np.cast`), `overload`, `OrderACF`, `OrderKACF`, and a circular self-reference `np.np`. These autocomplete as public API and invite accidental dependence.

**Fix:** Add explicit `__all__` to each aerosandbox/numpy submodule (or underscore-alias typing imports, e.g. `from typing import cast as _cast`).


#### Performance

### 98. Propulsor export runs an IPOPT optimization to compute two closed-form Euler angles

`aerosandbox/geometry/openvsp_io/asb_to_openvsp/propulsor_vspscript_generator.py:35` · performance · not independently verified · backwards-compat risk: none

For any non-[1,0,0] normal, generate_propulsor builds an asb.Opti problem and calls a nonlinear solver to find y/z rotations aligning [-1,0,0] with the normal. This is slow, requires solver availability, and the nonconvex maximize-dot-product from init 0 can land on a local optimum for aft-facing normals. Closed form exists. Docstring is also copy-pasted from generate_wing.

**Fix:** Use z_rot = -arcsin(n_y), y_rot = atan2(n_z, -n_x) (degrees); fix docstring.


#### Modernization

### 99. Publish workflow fires on every master push with no version guard, and uses a long-lived API token

`.github/workflows/publish-on-master-push.yml:7` · modernization · not independently verified · backwards-compat risk: none

Any master push without a version bump fails at `uv publish` (verified: run 15663835892, a README-only push, failed). Workflow also authenticates with secrets.PYPI_API_TOKEN instead of PyPI Trusted Publishing (OIDC), has no `permissions:` block, and pins setup-uv@v4 (current major is much newer).

**Fix:** Trigger on release/tag (or add `uv publish --check-url` guard), switch to Trusted Publishing with `id-token: write`, add least-privilege permissions, bump action versions.

### 100. Internal library code calls deprecated asb.numpy.trapz (9 call sites)

`aerosandbox/structures/tube_spar_bending.py:260` · modernization · not independently verified · backwards-compat risk: none

asb.numpy.trapz emits PendingDeprecationWarning on every call and is slated for removal (NumPy 2.x already removed np.trapz). Internal callers: tube_spar_bending.py lines 260, 266, 274, 282, 312, 318, 329; power_solar.py:417; airfoil_inviscid.py:141-142. Warning spam surfaces under pytest/-W. Also, trapz's 'See Also numpy.trapz' docstring is misleading: this function returns per-interval pieces, not a summed integral.

**Fix:** Migrate internal call sites to integrate_discrete_intervals(f, method='trapz'); clarify the trapz docstring semantics difference vs numpy.trapz.


#### Type hints

### 101. CDA_perpendicular_sheet_metal_joint Literal type omits 6 of 12 supported `kind` values

`aerosandbox/library/aerodynamics/components.py:219` · typing · not independently verified · backwards-compat risk: none

The Literal annotation lists only 6 kinds, but the docstring and CD_factors dict support 12 (bevel/rounded-bevel/flush variants missing). Type checkers (mypy/pyright) flag valid calls like kind="flush_lap_joint_forward_facing_step" as errors.

**Fix:** Extend the Literal to all 12 dict keys (after fixing the 'lap joint' typo).


#### Documentation & docstrings

### 102. Contributor setup instructions reference removed setup.py

`CONTRIBUTING.md:20` · docs · not independently verified · backwards-compat risk: none

Step 2.1 tells contributors to confirm they are in the repo root by looking for 'a file called `setup.py`', which no longer exists after the pyproject.toml/hatchling migration — a new contributor following the checklist literally will conclude they are in the wrong directory. The doc also predates the uv-based workflow used in CI (.github/workflows use uv).

**Fix:** Reference pyproject.toml as the root marker; optionally document `uv sync` alongside `pip install -e ".[full,test,docs]"`.

### 103. MSES docstrings are copy-pasted from XFoil: wrong class name, nonexistent params, broken usage example

`aerosandbox/aerodynamics/aero_2D/mses.py:90` · docs · not independently verified · backwards-compat risk: none

__init__ docstring says 'Interface to XFoil', documents params that don't exist (Re, mach, full_potential, xfoil_command, xfoil_repanel, verbose, timeout) and none of the actual params (mset_*, mses_*, use_xvfb, verbosity, timeouts, behavior_after_unconverged_run). The class usage example (lines 51-59) passes Re/mach to the constructor (TypeError) and calls ms.alpha(5), a method that doesn't exist. run() has no docstring.

**Fix:** Rewrite __init__ docstring for actual parameters, fix the usage example to use ms.run(alpha=..., Re=..., mach=...), and document run().

### 104. wing_aerodynamics/fuselage_aerodynamics docstrings state wrong moment reference point and a nonexistent arg

`aerosandbox/aerodynamics/aero_3D/aero_buildup.py:582` · docs · not independently verified · backwards-compat risk: none

wing_aerodynamics docstring says 'Moments are given with the reference at Wing [0, 0, 0]' but the code computes moments about self.xyz_ref (line 752). fuselage_aerodynamics (line 960) likewise claims 'reference at Fuselage [0, 0, 0]' while using self.xyz_ref (lines 1072, 1136, 1216). Both also document an `op_point` argument that is not a parameter (and wing's says 'analyze the fuselage at').

**Fix:** State that moments are about AeroBuildup.xyz_ref; remove the stale op_point Args entries; fix copy-paste wording.

### 105. critical_mach docstring off by factor of 2: parameter is L_n/d, not 2*L_n/d

`aerosandbox/aerodynamics/aero_3D/aero_buildup_submodels/fuselage_aerodynamics_utilities.py:15` · docs · not independently verified · backwards-compat risk: none

Docstring states "fineness_ratio_nose = 2 * L_n / d", but the code computes `2 * fineness_ratio_nose + b`, and the underlying fit (studies/FuselageCriticalMach/make_fits.py) was fit directly against Raymer's 2*L_n/d axis with no internal doubling. So the parameter must be L_n/d (as AeroBuildup passes it). A user following the docstring gets M_dd wrong by ~+0.07 at L_n/d=3.

**Fix:** Change docstring to fineness_ratio_nose = L_n / d, noting the factor of 2 (Raymer Fig 12.28 axis) is applied internally.

### 106. point_source docstring documents nonexistent parameters (copy-pasted from horseshoe function)

`aerosandbox/aerodynamics/aero_3D/singularities/point_source.py:33` · docs · not independently verified · backwards-compat risk: none

The Args section of calculate_induced_velocity_point_source documents x_left/y_left/z_left, x_right/y_right/z_right, gamma, and trailing_vortex_direction — none of which exist. The actual parameters x_source, y_source, z_source, and sigma are undocumented, and viscous_radius is described as a "Kaufmann vortex model" with reference to "the smallest bound leg", which is meaningless for a point source.

**Fix:** Rewrite Args to document x_source/y_source/z_source, sigma (volume flux strength), and viscous_radius as a source-desingularization length.

### 107. Public get_modes() has no docstring, untyped 'aero' dict, and undocumented accuracy caveats

`aerosandbox/dynamics/flight_dynamics/airplane.py:7` · docs · not independently verified · backwards-compat risk: none

get_modes() has no docstring, no return annotation, and 'aero' is untyped; callers cannot know it requires keys CD, CL, Cma, Cmq, Clp, Clb, Clr, CYb, CYr, Cnb, Cnr. Known discrepancies vs AVL (phugoid real part ~2x off, dutch-roll imag ~1.5x) are only noted in the __main__ block, invisible to users.

**Fix:** Add docstring citing FVA Eqs. 9.55-9.68, required aero keys, units, return structure, and accuracy caveats; annotate aero: dict[str, float] and g: float, plus return type.

### 108. Public function field_length_analysis has no docstring

`aerosandbox/library/field_lengths.py:272` · docs · not independently verified · backwards-compat risk: none

field_length_analysis (16 parameters, returns a 13-key dict) has no docstring at all, unlike its sibling field_length_analysis_torenbeek. Notably it returns balanced_field_length_accept/_reject instead of balanced_field_length, and the meaning of V_engine_failure_balanced_field_length is documented nowhere.

**Fix:** Add a docstring mirroring the torenbeek variant, documenting V_engine_failure_balanced_field_length and the accept/reject return keys.

### 109. Docstrings swap 'dependent' and 'independent': x_data called dependent, y_data independent

`aerosandbox/modeling/fitting.py:78` · docs · not independently verified · backwards-compat risk: none

FittedModel.__init__ documents x_data as 'Values of the dependent variable(s)' and y_data as 'Values of the independent variable' (lines 78, 84) — reversed; x is independent, y is dependent. The identical swapped wording appears in interpolation_unstructured.py lines 46 and 52. SurrogateModel.plot()'s axis_range docstring (surrogate_model.py:107-109) repeats the same error.

**Fix:** Swap the terms in all three files: x_data = independent variable(s), y_data = dependent variable.

### 110. Opti.minimize and Opti.maximize — core public API — have no docstrings

`aerosandbox/optimization/opti.py:492` · docs · not independently verified · backwards-compat risk: none

`Opti.minimize()`/`Opti.maximize()` are among the most-called methods in the library (every optimization problem uses one) yet have no docstring, so `help(asb.Opti.minimize)` and the API docs show nothing. This contradicts README's 'units are listed on all function docstrings' and CONTRIBUTING's 'every user-facing function... no exceptions' rule. `solve_sweep`, `save_solution`, and several other public Opti methods are also undocumented.

**Fix:** Add Google-style docstrings documenting `f`, sign convention of maximize, and objective-scaling notes.

### 111. derivative_of/constrain_derivative docstrings document integrator methods that raise exceptions

`aerosandbox/optimization/opti.py:1048` · docs · not independently verified · backwards-compat risk: none

Verified against integrate_discrete_intervals: "backwards euler" -> ValueError (only "backward euler" accepted); "midpoint" -> raises PendingDeprecationWarning as an exception; "runge-kutta", "rk4", "runge-kutta-3/8" -> ValueError, never implemented (yet documented with citations). Same wrong list is duplicated in constrain_derivative (lines 1174-1203). Only trapezoidal/trapezoid, forward/backward euler, simpson variants, cubic work.

**Fix:** Rewrite both docstrings to list actual supported methods; delete Runge-Kutta bullets or implement them. Also accept "backwards euler" as an alias.

### 112. torsion=True (the default) is silently ignored; docstring claims torsion is simulated

`aerosandbox/structures/legacy/beams.py:21` · docs · not independently verified · backwards-compat risk: low

The class docstring says it "simulates both bending and torsion" and gives the torsion governing equation, but the torsion implementation is commented out (lines 297-304, "Note: torsion feature is not yet implemented"). Users passing torsion=True (the default) get bending-only results with no warning, and phi/stress_shear never exist.

**Fix:** Emit a warning (or raise NotImplementedError for explicit torsion=True) and correct the docstring to state torsion is unimplemented.

### 113. Docstring says None is valid for load/modulus functions, but None crashes with TypeError

`aerosandbox/structures/tube_spar_bending.py:120` · docs · not independently verified · backwards-compat risk: low

The Args docs for bending_distributed_force_function and elastic_modulus_function both claim "None, in which case it's interpreted as a design variable to optimize over", but the code (lines 236-244) has no None branch for these two: None*np.ones_like(y) raises TypeError (verified). Only diameter_function and wall_thickness_function handle None. Type hints also omit None for these params.

**Fix:** Either add a None-to-opti.variable branch for both (additive), or remove the None bullet from those two docstrings.

### 114. autoapi_ignore misses *_derivations scratch directories, polluting published API docs

`docs/source/conf.py:47` · docs · not independently verified · backwards-compat risk: none

`autoapi_ignore = ["*/test_*.py", "*/ignore/*"]` excludes tests and ignore/ dirs but not `aerosandbox/numpy/integrate_discrete_derivations/` and `aerosandbox/numpy/derivative_discrete_derivations/` — sympy scratch scripts the repo's own ruff config classifies as non-library code. sphinx-autoapi will document them as if they were public API.

**Fix:** Add "*_derivations/*" (and optionally "*/in_progress/*") to autoapi_ignore.


#### Dead code

### 115. Airfoil optimizer script uses removed v3 API Airfoil.xfoil_cseq and stale result keys

`aerosandbox/aerodynamics/aero_2D/airfoil_optimizer/airfoil_optimizer.py:122` · dead-code · not independently verified · backwards-compat risk: none

`airfoil.xfoil_cseq(...)` does not exist on the v4 Airfoil class (no xfoil_* methods remain; grep confirms), and result keys 'Cd'/'Cm' are the v3 casing (v4 XFoil returns 'CD'/'CM'). The script crashes with AttributeError when executed. It also passes deprecated `enforce_continuous_LE_radius` to get_kulfan_coordinates.

**Fix:** Port the script to asb.XFoil(...).cl(...) with current keys, or delete/move it to a tutorials directory.


---

## 4. Low severity

Worth fixing opportunistically — grouped by category, ✓ = independently verified.


#### Bugs (wrong results or crashes)

- `aerosandbox/aerodynamics/aero_3D/linear_potential_flow.py:65` — **LinearPotentialFlow crashes on construction: dicts keyed by unhashable Wing/Fuselage instances** ✓ — `{wing: wing_model for wing in self.airplane.wings}` raises `TypeError: unhashable type: 'Wing'` because AeroSandboxObject defines `__eq__` (common.py:45) without `__hash__`. Confirmed by execution: the class cannot be instantiated even with all-default arguments. Same for fuselage_model, wing_options, fuselage_options. (Class does warn it is not ready for use.) *Fix:* Key the internal dicts by id(wing) or list index, or give AeroSandboxObject an identity-based `__hash__`.
- `aerosandbox/aerodynamics/aero_3D/singularities/point_source.py:68` — **Dimensional error in viscous_radius regularization: smoothing radius is sqrt(viscous_radius), not viscous_radius** ✓ — smoothed_x_15_inv is called with x = r_squared (units L^2), but the denominator adds x**2.5 (L^5) to viscous_radius**2.5 (L^2.5) — dimensionally inconsistent. Smoothing kicks in at physical distance r = sqrt(viscous_radius), not r = viscous_radius. LiftingLine (lifting_line.py:1017) and NonlinearLiftingLine (:693) pass viscous_radius=0.0001, so velocities are silently attenuated out to ~1 cm (50% error at 1 cm) instead of 0.1 mm. *Fix:* Change denominator to `x**2.5 + viscous_radius**5` so smoothing activates at r ~ viscous_radius, matching the documented meaning.
- `aerosandbox/atmosphere/atmosphere.py:118` — **Atmosphere.__len__ resets detected vector length to 1 on trailing length-1 arrays** ✓ — In the loop, a subscriptable value with np.length(v)==1 executes `length = 1`, clobbering a length already detected from an earlier variable. Verified: len(Atmosphere(altitude=np.array([1e3,2e3,3e3]), temperature_deviation=np.array([5.0]))) returns 1 instead of 3, so iteration/indexing over the instance is silently truncated. OperatingPoint.__len__ (operating_point.py:225) handles this correctly. *Fix:* Mirror OperatingPoint.__len__: skip values with np.length(v)==1 rather than assigning length=1.
- `aerosandbox/library/aerodynamics/components.py:285` — **Dict key typo: 'lap joint_...' makes a documented joint kind raise ValueError** ✓ — In CDA_perpendicular_sheet_metal_joint, the CD_factors key is "lap joint_forward_facing_step_with_rounded_bevel" (space instead of underscore). The docstring advertises "lap_joint_forward_facing_step_with_rounded_bevel"; passing it raises ValueError: Invalid `kind` of sheet metal joint (verified by execution). *Fix:* Rename key to "lap_joint_forward_facing_step_with_rounded_bevel"; optionally keep the misspelled key as an alias for one release.
- `aerosandbox/library/aerodynamics/unsteady.py:227` — **Wagner-function derivative coefficient typo: 0.00750075 should be 0.0075075** — dW_ds is the derivative of Jones' Wagner approximation used in wagners_function (line 71): d/ds[-0.165 e^{-0.0455 s}] = 0.165*0.0455 e^{-0.0455 s} = 0.0075075 e^{-0.0455 s}. Code has 0.00750075 (digit transposition), a ~0.1% error in that term of the Duhamel integral in calculate_lift_due_to_pitching_profile. *Fix:* Change 0.00750075 to 0.0075075 (or write 0.165 * 0.0455 explicitly).
- `aerosandbox/library/propulsion_propeller.py:102` — **style.use('seaborn') crashes on matplotlib >= 3.8** — The 'seaborn' style alias was removed in matplotlib 3.8 (renamed 'seaborn-v0_8'). Verified on the installed matplotlib 3.11: OSError "'seaborn' is not a valid ... style". Only reachable via the module's __main__ demo block, so the library import path is unaffected. *Fix:* Replace with `style.use('seaborn-v0_8')` or drop the style call from the demo.
- `aerosandbox/library/weights/raymer_cargo_transport_weights.py:39` — **Wing/vstab mass uses minimum t/c across sections; Raymer's equations specify root t/c** — mass_wing (line 39) and mass_vstab (line 156), plus the GA-file counterparts, take np.min of all section airfoil thicknesses. Raymer's (t/c) term is the root value; since transport wings are thickest at the root, min() selects the tip t/c and the (t/c)^-0.4 / (t/c)^-0.5 terms overestimate mass ~10-15% versus the cited reference. *Fix:* Use wing.xsecs[0].airfoil.max_thickness() (root section), as torenbeek_weights.py already does; document the choice.
- `aerosandbox/numpy/integrate_discrete.py:110` — **integrate_discrete_intervals(method="midpoint") raises instead of warning** ✓ — `raise PendingDeprecationWarning(...)` raises the warning class as an exception, so the documented "midpoint" alias (present in the Literal type and docstring) crashes instead of computing the trapezoidal result with a deprecation notice. Verified crash. *Fix:* Replace with `warnings.warn(..., PendingDeprecationWarning)` and let execution fall through to the trapezoidal computation.
- `aerosandbox/optimization/opti.py:38` — **freeze_style type hint says Literal["parameter", "frozen"] but code implements "float"** ✓ — The annotation advertises "frozen" as valid, but the implementation (lines 277, 304) checks for "float". Verified: Opti(freeze_style="frozen") raises ValueError("Bad value of `Opti.freeze_style`!") on the first frozen variable, while the undocumented "float" works. Users following the public type hint get a crash. *Fix:* Change annotation to Literal["parameter", "float"]; optionally accept "frozen" as an alias for compatibility with the old hint.
- `aerosandbox/optimization/opti.py:1118` — **derivative_of does not forward _stacklevel or category to self.variable()** — The derivative variable is created without _stacklevel (unlike the constrain_derivative call right below, which gets _stacklevel+1). Verified: find_variable_declaration for a derivative variable reports opti.py line 1118 instead of the user's derivative_of() call site. Also, the variable lands in "Uncategorized", so category-based freezing/caching misses derivative states. *Fix:* Pass _stacklevel=_stacklevel + 1 (and add an optional category kwarg forwarded through) in the self.variable() call inside derivative_of.
- `aerosandbox/optimization/opti.py:1455` — **OptiSol.show_infeasibilities crashes on problems with a single scalar constraint** ✓ — self(self.opti.g) returns a plain float when the problem has exactly one constraint, so len(g) raises TypeError: object of type 'float' has no len(). Verified. This debugging method is used precisely when solves fail, so single-constraint users hit a crash instead of the diagnostic. *Fix:* Apply np.atleast_1d to g, lbg, ubg (and constraint_violated) before iterating.
- `aerosandbox/structures/legacy/simple_beam_opt_daedalus_calibration.py:231` — **Hardcoded Windows user path in plt.savefig breaks the demo script on other machines** — plt.savefig("C:/Users/User/Downloads/beam.png") is a developer-specific absolute path; running this shipped module as a script raises FileNotFoundError on any machine without that directory (i.e., everyone), after the solve completes. *Fix:* Remove the savefig call or save to a relative path in the current working directory.
- `aerosandbox/tools/code_benchmarking.py:83` — **Timer.__enter__ does not return self, so `with Timer() as t:` binds t=None** ✓ — __enter__ calls self.tic() but returns None. Verified: `with Timer('x') as t: pass` leaves t=None, so users cannot read t.runtime after the block — the documented way to get the elapsed time programmatically. Even the module's own __main__ block uses `with Timer('a') as a:` and gets None. *Fix:* Add `return self` at the end of __enter__.
- `aerosandbox/tools/pretty_plots/labellines/__init__.py:3` — **labellines `__all__` contains function objects, not strings — star import raises TypeError** ✓ — `__all__ = [labelLine, labelLines]` lists the functions themselves. Verified: `from aerosandbox.tools.pretty_plots.labellines import *` raises `TypeError: Item in ...__all__ must be str, not function`. *Fix:* Change to `__all__ = ["labelLine", "labelLines"]`.
- `aerosandbox/tools/pretty_plots/labellines/core.py:251` — **labelLines ignores its `lines` argument and labels all legend handles instead** ✓ — The `lines` parameter is only used for `lines[0].axes`; the labeling loop iterates `all_lines` built from ax.get_legend_handles_labels(). Passing a subset of lines (per the docstring, "The lines to label") still labels every legend-labeled line on the axes. Upstream matplotlib-label-lines filters to the passed lines; this vendored copy dropped that. *Fix:* Filter: keep only handles that are in `lines` (upstream behavior), or document that all legend-labeled lines are used.
- `aerosandbox/tools/string_formatting.py:76` — **eng_string() drops the space before unit when exponent is outside SI-prefix range** ✓ — In the non-SI branch, `if add_space_after_number:` should be `if add_space_after_number is None:` (compare line 65). With the default add_space_after_number=None and a unit, None is falsy so no space is ever added. Verified: eng_string(1230, unit='N') -> '1.23 kN' but eng_string(1e30, unit='N') -> '1e30N' (inconsistent, unreadable). *Fix:* Change line 76 to `if add_space_after_number is None: add_space_after_number = unit != ""`.
- `aerosandbox/visualization/carpet_plot_utils.py:112` — **patch_nans bridging loop: `continue` where `break` was intended, so last pair overwrites first** — In the 'Bridging' stage, after filling array[i,j] from the first valid neighbor pair, `continue` is the last statement of the pairs loop — a no-op. Every subsequent valid pair overwrites the value, so the cell ends up bridged by the last valid pair (a diagonal) rather than the first (orthogonal, which the pair ordering prioritizes). Also mutates the input array in place (acknowledged TODO at line 62) and prints unconditionally with no verbose flag. *Fix:* Replace `continue` with `break`; add a `verbose=True` parameter gating the prints; operate on `array = np.copy(array)`.
- `aerosandbox/visualization/plotly.py:26` — **spy() destructively mutates the caller's matrix and plots dead all-ones data** ✓ — Line 26 writes log10 magnitudes into the input matrix in place, corrupting the caller's array (verified: input is modified after the call). Then line 33's `val = matrix[sparsity_pattern]` is immediately overwritten by `val = np.ones_like(i_index)` (line 34), so the heatmap z is constant 1 and the log10 computation is entirely dead code. *Fix:* Operate on a copy (matrix = np.array(matrix, dtype=float)); delete one of the two `val` assignments depending on intended intensity (log-magnitude vs. binary sparsity).

#### Latent bugs & fragile code

- `aerosandbox/aerodynamics/aero_2D/mses.py:368` — **MSES.run() crashes with KeyError('Ma') when zero runs converge** ✓ — If no run converges (or MSET failed and every MSES run is unconverged), runs_output is an empty dict and `runs_output.pop("Ma")` raises a bare KeyError instead of returning an empty result or an informative error. *Fix:* If runs_output is empty, return {} (or raise a descriptive RuntimeError stating no runs converged) before the pop('Ma') reordering.
- `aerosandbox/aerodynamics/aero_2D/xfoil.py:337` — **except subprocess.CalledProcessError block is dead code; curated XFoil crash diagnostics never fire** ✓ — Popen.communicate() never raises CalledProcessError (only subprocess.run(check=True) does), and the proc.poll() result on line 325 is discarded. So the segfault (rc 11), floating-point-exception (rc 8/136), and not-on-PATH (rc 1) XFoilError messages are unreachable; users get the generic 'no output file' error instead. A missing executable actually raises FileNotFoundError from Popen, also uncaught. *Fix:* After communicate(), inspect proc.returncode and raise the corresponding XFoilError; catch FileNotFoundError from Popen for the PATH message.
- `aerosandbox/aerodynamics/aero_2D/xfoil.py:399` — **UnboundLocalError masks intended XFoilError when polar output has no separator line** ✓ — If no line with 30+ dashes is found (empty/truncated output.txt, e.g. after a timeout kill), `raise IndexError` jumps to the except handler, which builds the XFoilError message from `title_line` and `columns` — both unbound at that point — producing UnboundLocalError instead of the helpful diagnostic. Lines 388/391 also use loop var `i` instead of `separator_line`. *Fix:* Build the error message without title_line/columns (or default them to None), and index with `separator_line` instead of `i`.
- `aerosandbox/aerodynamics/aero_3D/aero_buildup.py:750` — **Mirrored-section branch mutates shared aerodynamic_centers arrays in place** — `sect_AC_raw = aerodynamic_centers[sect_id]; sect_AC_raw[1] *= -1` flips the y-coordinate of the shared list element in place. Correct today only because the mirrored call always runs after the unmirrored call, exactly once per section; calling compute_section_aerodynamics again (or reordering the loop) would produce wrong signs on moments. *Fix:* Build a new array instead: `sect_AC_raw = aerodynamic_centers[sect_id] * np.array([1, -1, 1])` inside the mirror branch.
- `aerosandbox/aerodynamics/aero_3D/aero_buildup_submodels/softmax_scalefree.py:9` — **softmax_scalefree crashes on array-valued inputs under NumPy >= 1.24 (inhomogeneous array)** — `np.array([1e-6] + x)` prepends a scalar to the user's list. If entries of x are ndarrays (e.g., array-valued geometry, which Vectorizable typing permits), NumPy 2.x (project requires numpy>=2.2.6) raises ValueError for the inhomogeneous list. Also, an empty list x reaches np.softmax() with zero args and raises a confusing ValueError (e.g., AeroBuildup on an airplane with no wings/fuselages). *Fix:* Compute softness as `np.max(np.concatenate([np.ravel(np.array(xi)) for xi in ([1e-6] + x)]))*0.01` or per-element maximum; guard len(x)==0 with a clear error.
- `aerosandbox/aerodynamics/aero_3D/avl.py:774` — **parse_unformatted_data_output checks string bounds after indexing; IndexError/negative-index wraparound possible** — `while s[i] == " " and i <= len(s)` evaluates `s[i]` before the bounds check, and `i <= len(s)` permits i == len(s), raising IndexError if a value ends exactly at end-of-string. The backward key scan (`while s[i] == " " and i >= 0`) similarly allows i to reach -1, silently wrapping to the last character. *Fix:* Reorder to `while i < len(s) and s[i] ...` and `while i >= 0 and s[i] ...` in all four scan loops.
- `aerosandbox/aerodynamics/aero_3D/linear_potential_flow.py:78` — **issubclass() called on Wing/Fuselage instances raises TypeError when validating per-component options** ✓ — `all([issubclass(k, Wing) for k in wing_options.keys()])` receives Wing instances (per the documented `{Wing: {str: value}}` format), and `issubclass` on a non-class raises `TypeError: issubclass() arg 1 must be a class` instead of validating. Same at line 91 for fuselage_options. Should be `isinstance`. (Currently masked by the unhashable-Wing crash.) *Fix:* Replace `issubclass(k, Wing)` / `issubclass(k, Fuselage)` with `isinstance(k, ...)` in both validation checks.
- `aerosandbox/aerodynamics/aero_3D/singularities/uniform_strength_horseshoe_singularities.py:126` — **vortex_core_radius applied dimensionally inconsistently across Biot-Savart terms** — smoothed_inv(x) = x/(x^2 + r_c^2) is applied to norm_a (length), to norm_a*norm_b + a_dot_b (length^2, term1), and to norm_a - a_dot_u (~h^2/2|a|, terms 2-3). For term1 the effective bound-leg core radius is ~sqrt(r_c) (e.g. 1e-4 m for VLM's default r_c=1e-8 m), and for the trailing legs it grows as sqrt(2|a|*r_c) downstream — contradicting the docstring's claim that the parameter "governs the radius" of a Kaufmann vortex. *Fix:* Regularize distances instead: replace norms with sqrt(|a|^2 + r_c^2) etc., or scale the added constant per-term to have matching dimensions (r_c^4 for term1).
- `aerosandbox/aerodynamics/aero_3D/vortex_lattice_method.py:483` — **run_with_stability_derivatives uses locals() lookup and lacks the zero-division guard its siblings have** — `locals()[derivative_denominator]` to read the alpha/beta/p/q/r flags is fragile (breaks under rename/refactor; AeroBuildup and LiftingLine use an explicit dict). Also, x_np (line 525) and x_np_lateral (line 529) divide by CLa/CYb without the `np.where(... == 0, np.nan, ...)` guard AeroBuildup/LiftingLine apply, producing ZeroDivisionError or inf for degenerate configurations. *Fix:* Adopt the `do_analysis` dict pattern and the np.where zero-guard from aero_buildup.py lines 463-570.
- `aerosandbox/atmosphere/thermodynamics/gas.py:57` — **PerfectGas.__repr__ crashes for array-valued pressure/temperature** — pressure and temperature are typed Vectorizable, but __repr__ applies scalar format specs (f'{self.temperature:.6g}', eng_string on pressure). Verified: repr(PerfectGas(pressure=np.array([1e5,2e5]), temperature=np.array([300.,400.]))) raises ValueError ('truth value of an array... is ambiguous'). *Fix:* Wrap formatting in try/except (ValueError, TypeError) and fall back to plain str(), as Atmosphere.__repr__ (atmosphere.py:58-64) does.
- `aerosandbox/common.py:45` — **AeroSandboxObject defines __eq__ without __hash__, making all ASB objects unhashable** ✓ — Defining __eq__ sets __hash__ to None, so every Airplane, Wing, Airfoil, Atmosphere, etc. is unhashable. Verified: hash(asb.Airfoil('naca0012')) raises TypeError; sets, dict keys, and functools.lru_cache over these objects all crash. Introduced in commit f07007fc (2024-03-29), shipped in v4.2.x. *Fix:* Add `__hash__ = object.__hash__` to AeroSandboxObject (restores pre-2024 identity hashing; document that equal-by-value objects may hash differently, acceptable for mutable objects).
- `aerosandbox/common.py:490` — **ExplicitAnalysis.get_options returns geometry's options dict by reference in one branch** — When the analysis has no defaults for the geometry type but the geometry declares options, get_options returns geometry_object.analysis_specific_options[analysis_type] directly, while the other branch returns a deepcopy. A caller mutating the returned dict silently corrupts the geometry object's stored options, and behavior is inconsistent between branches. *Fix:* Return copy.deepcopy(geometry_options_for_this_analysis) in the else branch for symmetry.
- `aerosandbox/dynamics/point_mass/common_point_mass.py:242` — **__array__ missing NumPy 2 'copy' keyword; np.array(dyn) already fails under -W error** ✓ — NumPy 2 calls __array__(dtype, copy=...); this implementation only accepts dtype, so np.array(dyn) emits DeprecationWarning ('__array__ must implement dtype and copy keyword arguments') and will hard-fail (TypeError) in a future NumPy release. Verified warning on NumPy 2.4.6. *Fix:* Change signature to def __array__(self, dtype="O", copy=None) and ignore/honor copy per NumPy 2 migration guide.
- `aerosandbox/dynamics/point_mass/common_point_mass.py:301` — **constrain_derivatives: bad state name in 'which' leaks raw KeyError; friendly ValueError is dead code** — state_derivatives[state_var_name] at line 301 is outside the try block, so a typo in 'which' raises a raw KeyError instead of the intended "does not have a state named..." ValueError (verified). Also, 'raise ValueError(...)' at line 318 stringifies the original exception without 'from e', losing the traceback. *Fix:* Validate 'which' against self.state.keys() up front (raising the friendly ValueError), and chain re-raised exceptions with 'raise ... from e'.
- `aerosandbox/dynamics/point_mass/common_point_mass.py:627` — **Loop variable 'i' shadowed inside draw(): inner axes-color loop reuses outer trajectory index** — The outer loop 'for i in np.linspace(0, len(self)-1, n_vehicles_to_draw)' (line 555) has its index rebound by the inner 'for i, c in enumerate(["r","g","b"])' (line 627). Currently harmless because the outer body no longer uses i afterwards, but any future code appended after the axes block would silently read i=2. *Fix:* Rename the inner loop variable, e.g. 'for ax_index, c in enumerate(["r", "g", "b"])' and use rot[:, ax_index].
- `aerosandbox/dynamics/rigid_body/rigid_3D/body_euler.py:415` — **Rigid-body speed lacks the gradient-singularity guard used by point-mass classes** — DynamicsRigidBody3DBodyEuler.speed computes (u^2+v^2+w^2)**0.5 without the +1e-200 epsilon that DynamicsPointMass3DCartesian.speed (cartesian.py:106) deliberately adds. Under the CasADi backend, sqrt(0) yields NaN derivatives at the common all-zero initial guess, stalling the optimizer; alpha/beta arctan2d(0,0) at the origin compound this. *Fix:* Add the same +1e-200 term inside the square root, matching the point-mass implementation.
- `aerosandbox/geometry/airfoil/airfoil.py:363` — **make_symmetric_polars corrupts Cpmin, Xcpmin, Top_Xtr, Bot_Xtr in xfoil_data and cache** ✓ — XFoil.alpha() returns keys alpha, CL, CD, CDp, CM, Cpmin, Xcpmin, Chinge, Top_Xtr, Bot_Xtr (xfoil.py:410-424). Only CD/CDp/Re are kept symmetric; everything else is negated. For a symmetric airfoil at -alpha, Cpmin and Xcpmin are symmetric, and Top_Xtr/Bot_Xtr should swap with each other, not negate. Negative transition locations are written into self.xfoil_data and the JSON cache. *Fix:* Add Cpmin, Xcpmin, Chinge to the symmetric list and swap Top_Xtr/Bot_Xtr when mirroring, instead of negating.
- `aerosandbox/geometry/airfoil/airfoil.py:1085` — **Misparenthesized duplicate-point check in repanel(): np.all(np.diff(x)) > 0** — Intended `np.all(np.diff(x) > 0)` is written as `np.all(np.diff(x)) > 0`, which tests 'all diffs nonzero' then compares the bool to 0. It only behaves correctly by coincidence because cumulative arc-length diffs are nonnegative; the strict-monotonicity intent is not actually checked. *Fix:* Move the comparison inside: `np.all(np.diff(upper_distances_from_TE) > 0)` (and same for lower).
- `aerosandbox/library/aerodynamics/inviscid.py:86` — **oswalds_efficiency raises UnboundLocalError for unknown `method` instead of ValueError** ✓ — If method is anything other than "nita_scholz" or "kroo", `e` is never assigned and `return e` raises UnboundLocalError (verified with method='raymer'). Additionally, `method` has no type annotation and is undocumented in the Args section (as is fuselage_diameter_to_span_ratio). *Fix:* Add `else: raise ValueError(...)`; annotate method: Literal["nita_scholz", "kroo"] and document both missing args.
- `aerosandbox/library/aerodynamics/unsteady.py:51` — **assert used for input validation in public library functions (stripped under python -O)** ✓ — Public functions validate user input with assert: unsteady.py:51 (velocity/time length mismatch), :162 (ndarray angle_of_attack rejection), :212 (negative reduced_time), and carpet_plot_utils.py:155 (NaN-patch failure). Under `python -O` these vanish and invalid input proceeds to silently wrong results (e.g., mismatched-length velocity/time loop reads garbage). *Fix:* Replace with `if not cond: raise ValueError(...)` (or RuntimeError for carpet_plot_utils.py:155).
- `aerosandbox/library/aerodynamics/unsteady.py:70` — **wagners_function/kussners_function return NaN (with overflow warning) for large negative reduced time** — Both compute expr * np.where(s >= 0, 1, 0); since both operands are evaluated, exp(-0.3*s) overflows to inf for very negative s and inf*0 = NaN. Verified: wagners_function(-5000.0) returns nan plus RuntimeWarnings, instead of the intended 0. *Fix:* Clamp first (s = np.fmax(reduced_time, 0)) or use np.where(s >= 0, expr_with_clamped_s, 0).
- `aerosandbox/library/power_solar.py:232` — **solar_flux silently swallows unknown keyword arguments via **deprecated_kwargs** ✓ — `**deprecated_kwargs` is accepted but never inspected, so any misspelled keyword (e.g. panel_azimut_angle=90) or genuinely deprecated option is silently discarded and the flux is computed with defaults — silently wrong results instead of a TypeError. *Fix:* If deprecated_kwargs is non-empty, issue a DeprecationWarning naming known legacy keys and raise TypeError for unrecognized ones.
- `aerosandbox/library/propulsion_electric.py:369` — **mass_motor_electric silently returns None for an unrecognized `method`** ✓ — The if/elif chain over method ('burton', 'hobbyking', 'astroflight') has no else branch, so a typo like method='hobby_king' returns None with no error (verified). Downstream arithmetic then fails far from the cause, or None propagates silently. All comparable functions in this library raise ValueError for bad string options. *Fix:* Add `else: raise ValueError(f"Bad value of method: {method!r}")`.
- `aerosandbox/library/weights/raymer_cargo_transport_weights.py:94` — **mass_hstab classifies a stabilizer with zero control surfaces as all-moving (+14.3% mass)** — all_moving initializes to True and is only set False when a control surface is found with a partial hinge. A conventional fixed hstab modeled without ControlSurface objects (a very common simplification) therefore gets Raymer's K_uht=1.143 all-moving-tail factor, overestimating hstab mass by 14.3%. Also, the `break` only exits the inner loop. *Fix:* Treat hstabs with no control surfaces as not all-moving (or require explicit user flag); use a flag or for/else to break both loops.
- `aerosandbox/modeling/black_box.py:115` — **n_positional_args is always 0: len(parameters) minus len(parameters.values())** — len(signature.parameters) - len(signature.parameters.values()) is identically zero, so the lower-bound argument-count check never fires and the TypeError message 'Takes from 0 to N positional arguments' always reports a wrong lower bound. Intended value is clearly the count of parameters without defaults. Also, functions with *args/**kwargs silently miscount n_in at line 40. *Fix:* n_positional_args = sum(1 for p in signature.parameters.values() if p.default is p.empty); reject VAR_POSITIONAL/VAR_KEYWORD parameters explicitly.
- `aerosandbox/modeling/interpolation_unstructured.py:138` — **UnstructuredInterpolatedModel mutates the caller's x_data_resample dict in place** — The loop at lines 136-140 assigns linspace arrays into the user-supplied x_data_resample dict, replacing their int entries. Verified: after construction, the caller's dict {'x': 5} has become {'x': ndarray}. Reusing that dict for a second model or other logic gives surprising behavior. *Fix:* Copy first: x_data_resample = dict(x_data_resample) before the int-replacement loop.
- `aerosandbox/modeling/splines/bezier.py:55` — **quadratic_bezier_patch_from_tangents divides by zero when end tangents are parallel** — x_P1 divides by (dydx_a - dydx_b). For parallel tangents — e.g., the natural straight-line case dydx_a == dydx_b == chord slope — this yields inf/NaN control points and NaN outputs. The docstring warns about curvature flipping but not this singularity; under CasADi this silently embeds NaN into optimization graphs. *Fix:* Document the restriction and/or guard: when |dydx_a - dydx_b| is ~0, place (x_P1, y_P1) at the chord midpoint (degenerates to linear).
- `aerosandbox/modeling/surrogate_model.py:62` — **SurrogateModel.__call__ catches NameError but missing x_data raises AttributeError** ✓ — The type-check guard is wrapped in 'except NameError: pass' with comment 'If x_data does not exist', but self.x_data on an instance without that attribute raises AttributeError. Verified: a subclass that skips x_data (explicitly allowed by the class docstring: 'recommended but not required') crashes with AttributeError on every call that invokes super().__call__(x). *Fix:* Catch AttributeError instead of NameError (or use getattr(self, 'x_data', None) guard).
- `aerosandbox/numpy/__init__.py:67` — **`np.round` crashes on CasADi types, violating the dual-backend contract** ✓ — Line 67 force-imports NumPy's `round` with no CasADi dispatch. Verified: `np.round(cas.MX.sym('x'))` raises `TypeError: loop of ufunc does not support argument 0 of type MX which has no callable rint method`, while sibling wrapped functions (min, max, where, clip) all handle CasADi. The adjacent TODO ('check that min, max are properly imported') is stale — those are fine. *Fix:* Add a dual-backend `round` (e.g., `cas.floor(x + 0.5)` for CasADi branch, `_onp.round` otherwise) in array.py; drop the stale TODO.
- `aerosandbox/numpy/array.py:741` — **zeros_like/ones_like/full_like return wrong shape for 2D CasADi arrays** ✓ — The CasADi branch returns `_onp.zeros(shape=length(a))`, where length() returns only shape[0] (or shape[1] for row vectors). Verified: zeros_like(DM 3x2) returns a 1D array of shape (3,) instead of (3, 2). Same defect in ones_like (line 780) and thus full_like/empty_like. *Fix:* Return `_onp.zeros(a.shape)` (and `_onp.ones(a.shape)`), preserving the 2D CasADi shape.
- `aerosandbox/numpy/trig.py:12` — **Conversion constants _deg2rad and _rad2deg are named backwards** — `_deg2rad = 180.0 / _pi` (actually the rad-to-deg factor) and `_rad2deg = _pi / 180.0` (actually deg-to-rad). Current results are correct only because degrees()/radians() each use the wrongly-named constant; any future use of these private constants by name will produce inverted conversions. *Fix:* Swap the names: `_deg2rad = _pi / 180.0`, `_rad2deg = 180.0 / _pi`, and update the two usages.
- `aerosandbox/optimization/opti.py:658` — **freeze_style="float" plus cache loading crashes: floats never get is_manually_frozen attribute** ✓ — variable() sets var.is_manually_frozen inside try/except AttributeError (lines 340-343); float and ndarray frozen variables silently skip it. solve() then unconditionally reads var.is_manually_frozen. Verified: freeze_style="float" with load_frozen_variables_from_cache=True raises AttributeError: 'float' object has no attribute 'is_manually_frozen'. *Fix:* In solve(), use getattr(var, "is_manually_frozen", False); or track frozen-ness in a side dict/list instead of monkey-patched attributes.
- `aerosandbox/optimization/opti.py:675` — **Direct parameter_mapping size mismatch produces misleading 'cached solution' error** — The size check in solve() covers all parameter_mapping entries, including ones the user passed directly to solve(parameter_mapping=...), but the RuntimeError text only talks about 'loading cached solution' and re-running the original study — nonsensical when no cache is involved. *Fix:* Reword to a generic message stating the parameter's shape vs the supplied value's shape; mention the cache only when load_frozen_variables_from_cache is True.
- `aerosandbox/optimization/opti.py:727` — **Invalid behavior_on_failure value causes UnboundLocalError instead of clear ValueError** — solve() has if "raise"/elif "return_last" with no else, so a typo like "return-last" leaves sol unassigned. Verified: UnboundLocalError: cannot access local variable 'sol' at the return, giving no hint about the actual mistake. *Fix:* Add an else branch raising ValueError naming the invalid value and the allowed options.
- `aerosandbox/optimization/opti.py:1357` — **OptiSol.value does not propagate warn_on_unknown_types (or recursive) in recursive calls** — Recursive calls for list/tuple/set/dict/attribute contents call self.value(i) with defaults, dropping the warn_on_unknown_types flag. Verified: sol.value([unconvertible], warn_on_unknown_types=True) emits zero warnings while the same object at top level warns. *Fix:* Pass recursive=recursive and warn_on_unknown_types=warn_on_unknown_types through every internal self.value() call (lines 1357-1363, 1399).
- `aerosandbox/optimization/opti.py:1360` — **OptiSol.value on sets: array-valued members crash, frozenset silently becomes set** — The set branch builds {self.value(i) for i in x}; vector variables evaluate to np.ndarray, which is unhashable, raising TypeError. frozenset input also returns a mutable set, contradicting the 'preserve the type as best as possible' comment. *Fix:* Fall back to returning a list (or tuple) when converted members are unhashable; reconstruct frozenset via frozenset(...) when input was frozenset.
- `aerosandbox/structures/legacy/beams.py:168` — **Point loads added out of location order silently produce a non-monotonic mesh and wrong results** ✓ — setup() builds x by concatenating linspaces between consecutive point_load_locations in insertion order (lines 168-181). If loads are added with descending locations (e.g. add_point_load(20,...) then add_point_load(10,...)), segments reverse (negative dx) and overlap, so the FD constraints describe a nonsensical beam with no error raised. Point-force indexing (line 191) also assumes sorted order. *Fix:* Sort self.point_loads by numeric location at the start of setup(), or raise ValueError if locations are not ascending.
- `aerosandbox/structures/legacy/beams.py:306` — **TubeBeam1.setup() crashes with AttributeError when bending=False** ✓ — self.stress = self.stress_axial and the stress constraints (lines 306-312) sit outside the `if self.bending:` block, but self.stress_axial is only defined inside it (line 293). TubeBeam1(bending=False, torsion=True).setup() raises AttributeError: 'TubeBeam1' object has no attribute 'stress_axial' (verified), even though bending is a documented boolean option. *Fix:* Move the stress assignment and stress constraints inside the `if self.bending:` block.
- `aerosandbox/structures/tube_spar_bending.py:261` — **Variable-scaling heuristic is dimensionally inconsistent by one factor of length** ✓ — np.sum(np.trapz(distributed_force)*dy) is the total force F [N], so scale=F*length**4/EI_guess has units m^2, not m; a cantilever tip deflection scales as F*L^3/(3EI). All four scales (u, du, ddu, dEIddu at lines 260, 266, 274, 282) carry the same extra factor of length, inflating scales ~17x for a typical 17 m half-span and degrading IPOPT conditioning. Solution values are unaffected. *Fix:* Divide each scale by length (u: F*L^3/EI, du: F*L^2/EI, ddu: F*L/EI, dEIddu: F); compute F once.
- `aerosandbox/tools/inspect_tools.py:313` — **Argument-name parser truncates args containing ':' outside braces (slices, lambdas)** — The ':' handler assumes a type-hint and stops recording the current arg, but bracket level isn't tracked. Verified: get_function_argument_names_from_source_code('f(a[1:2], b)') returns ['a[1', 'b'], and lambdas give ['lambda t', ...]. Downstream, qp(x, y[0:5]) produces the wrong axis label 'y[0'. *Fix:* Track square-bracket nesting like braces, and only treat ':' as a type-hint separator at bracket level 0 (lambdas need similar handling).
- `aerosandbox/tools/pretty_plots/plots/contour.py:156` — **contour(z_log_scale=True) with constant Z crashes with cryptic geomspace error** — When Z is constant, Z_ratio=1 so log10(Z_ratio)=0; the division yields inf, `.astype(int)` casts to -9223372036854775808, and LogLocator setup raises "Number of samples, -9223372036854775807, must be non-negative". Verified. Also line 147's message says values must be "nonnegative" though the check `Z <= 0` requires strictly positive. *Fix:* Guard Z_ratio <= 1 (fall back to default locator) and reword the error message to "positive".
- `aerosandbox/tools/pretty_plots/plots/plot_smooth.py:148` — **plot_smooth raises UnboundLocalError instead of ValueError for invalid `function_of`** — The if/elif chain over function_of (None/'x'/'y') has no else branch, so any other value (e.g. 'z', a typo) falls through and later hits `UnboundLocalError: cannot access local variable 'x_resample'`. Verified. *Fix:* Add `else: raise ValueError(f"Invalid function_of: {function_of!r}; must be None, 'x', or 'y'")`.
- `aerosandbox/tools/python/importing.py:19` — **lazy_import uses importlib.util without importing importlib.util** ✓ — Module only does `import importlib`; `importlib.util` is a submodule not loaded by that import. Verified: `python -c "import importlib; importlib.util"` raises AttributeError on Python 3.12. It currently works only because importing the aerosandbox package transitively loads importlib.util. Also, find_spec() returning None (unknown module) yields AttributeError instead of ModuleNotFoundError. *Fix:* Change to `import importlib.util`; raise ModuleNotFoundError(name) when find_spec returns None.
- `aerosandbox/tools/python/io.py:22` — **convert_ipynb_to_py opens files without encoding='utf-8'** ✓ — .ipynb files are UTF-8 by specification (JSON), but both open() calls use the platform default encoding. On Windows (cp1252), any notebook containing non-ASCII characters raises UnicodeDecodeError on read, and the output .py can be written mis-encoded. *Fix:* Pass encoding='utf-8' to both open() calls.
- `aerosandbox/visualization/carpet_plot_utils.py:38` — **time_limit leaks the SIGALRM handler and rejects float durations** — The context manager never restores the previous SIGALRM handler on exit (verified: handler remains installed after the with-block), clobbering any user/library handler for the rest of the process. Also, signal.alarm() requires an int, so time_limit(0.5) raises TypeError (verified). Additionally the docstring example (line 24) catches `TimeoutException`, a class that does not exist — the code raises TimeoutError, so copy-pasting the documented usage gives NameError. *Fix:* Save old handler via signal.getsignal, restore in finally; use signal.setitimer(signal.ITIMER_REAL, seconds) for float support; fix docstring to `except TimeoutError`.
- `aerosandbox/visualization/carpet_plot_utils.py:62` — **patch_nans() mutates its input array in place and prints progress unconditionally** ✓ — patch_nans modifies the caller's array in place (author's own TODO on line 62: 'remove modification on incoming values') while also returning it, destroying the caller's data. It also prints 'Bridging' headers and per-iteration progress with no verbose flag. Similarly, visualization/plotly.py sparsity plot writes log10 values back into the caller's matrix. *Fix:* Start with `array = np.copy(array)`; gate the progress prints behind a `verbose: bool = False` parameter.
- `aerosandbox/visualization/plotly_Figure3D.py:72` — **Figure3D mirror=True crashes on tuple/list points, the exact format the docstrings show** ✓ — add_line/add_streamline/add_tri/add_quad accept iterables of length-3 iterables (docstring example: add_line([(0,0,0),(1,0,0)])), but with mirror=True each point is passed to the local reflect_over_XZ_plane (line 11), which requires a `.shape` attribute. Verified: f.add_line([(0,0,0),(1,0,0)], mirror=True) raises AttributeError: 'tuple' object has no attribute 'shape'. *Fix:* In reflect_over_XZ_plane, start with `input_vector = np.array(input_vector)`, or convert each point via np.array(point) before reflecting in the four add_* methods.
- `aerosandbox/weights/mass_properties.py:497` — **generate_possible_set_of_point_masses: dimensionally wrong radius estimate** — approximate_radius = (Ixx+Iyy+Izz)**0.5 / mass has units kg^-0.5·m, not m. The radius of gyration is sqrt(I/m), i.e., ((Ixx+Iyy+Izz)/mass)**0.5. For mass=100, Ixx=Iyy=Izz=1 the code yields 0.0173 m vs the correct 0.173 m (verified numerically), giving init guesses/scales off by sqrt(mass) and degrading IPOPT convergence for masses far from 1 kg. *Fix:* approximate_radius = ((self.Ixx + self.Iyy + self.Izz) / self.mass) ** 0.5 + 1e-16

#### Testing gaps

- `aerosandbox/atmosphere/thermodynamics/test/test_gas.py:6` — **Thermodynamics test file is an empty stub: 'def test_isentropic(): pass'** — The only test in the thermodynamics package is a pass-stub; PerfectGas is merely instantiated at module import. The module reports a green test in CI while its property/process math is entirely unverified. *Fix:* Implement the isentropic-relations test (e.g. verify p*v^gamma constant, T ratios vs textbook values) or delete the stub so the gap is visible.
- `aerosandbox/geometry/test_geometry/test_wingxsec.py:6` — **test_wingxsec has 'TODO actually test this', no assertions, and exercises only deprecated kwargs** — test_init constructs a WingXSec with legacy control_surface_* arguments and asserts nothing, so the modern ControlSurface-based construction path of a core geometry class is untested here and any constructor regression passes. *Fix:* Assert geometric properties (chord, twist, xyz_le) after construction and add a case using the current control_surfaces=[ControlSurface(...)] API.

#### API design

- `aerosandbox/__init__.py:1` — **Top-level namespace leaks `asb.Path` and `asb.version`; no `__all__`** — `from pathlib import Path` (line 1) and `from importlib.metadata import version` (line 57) leak into the package namespace — verified `asb.Path` is pathlib.Path and `asb.version` is importlib.metadata.version (a function needing a distribution name, easily confused with `asb.__version__`). No `__all__` is defined, so `from aerosandbox import *` exports these too. *Fix:* Alias as `from pathlib import Path as _Path`, `from importlib.metadata import version as _version`, and add an explicit `__all__` of intended exports.
- `aerosandbox/aerodynamics/aero_3D/singularities/uniform_strength_horseshoe_singularities.py:64` — **assert_equal_shape rejects valid broadcasting the docstring explicitly promises** — The docstring says the function "can be vectorized as desired assuming input dimensions/broadcasting are compatible", but assert_equal_shape requires x_field/y_field/z_field to have identical shapes. Passing x_field as an array with scalar y_field/z_field (valid broadcast, e.g., sampling along the x-axis) raises ValueError. The sibling point_source function performs no such check, so the two are inconsistent. *Fix:* Use np.broadcast_shapes-style compatibility checking (or drop the assert), and document shape requirements; also document that trailing_vortex_direction must be a single 3-vector (linear_potential_flow.py:254 mis-annotates it as Nx3).
- `aerosandbox/atmosphere/atmosphere.py:56` — **_valid_altitude_range stored but never enforced; silent extrapolation beyond 0-80 km** — __init__ sets self._valid_altitude_range = (0, 80000), but no library code checks it (only the test file reads it). Users querying e.g. 150 km silently get extrapolated fit values with no warning, despite the model being documented for 0-100 km MAE only. *Fix:* Either document the extrapolation behavior in the class docstring or emit a warnings.warn for NumPy-backend altitudes outside the range (skip for CasADi symbolics).
- `aerosandbox/atmosphere/atmosphere.py:148` — **Invalid `method` errors say "Bad value of 'type'!" and are deferred to first property call** — pressure() and temperature() (lines 148, 161) raise ValueError("Bad value of 'type'!") but the parameter is named `method`. Also, an invalid method string passes __init__ silently and only fails later when pressure()/temperature() is called, far from the construction site. *Fix:* Fix message to reference 'method' and list valid options; optionally validate `method` in __init__ (non-breaking for valid inputs).
- `aerosandbox/dynamics/point_mass/common_point_mass.py:797` — **potential_energy is a property with an unusable 'g' parameter and misleading Args docstring** — The property getter takes g=9.81, but properties cannot receive arguments; dyn.potential_energy(g=...) raises TypeError ('float' object is not callable). The docstring's 'Args: g' actively invites this misuse. g is silently fixed at 9.81. *Fix:* Remove the g parameter and Args section from the property (documenting g=9.81), or keep property and note that custom g requires mass_props.mass * g * dyn.altitude.
- `aerosandbox/dynamics/point_mass/point_3D/cartesian.py:225` — **add_force default 'axes' differs per subclass ('earth'/'wind'/'body'), contradicting base-class signature** — The abstract base declares axes: Literal[...] = "wind", but DynamicsPointMass3DCartesian defaults to "earth", speed-gamma classes to "wind", rigid-body to "body". Code written against the base-class docs silently applies forces in the wrong frame when switching dynamics classes. Subclass overrides also drop the Literal annotation entirely. *Fix:* Document each subclass's native default in its docstring, restore Literal annotations now; unify defaults (or require explicit axes) in v5 since changing them silently alters physics.
- `aerosandbox/geometry/airplane.py:360` — **Airplane.draw(): plotly show_kwargs set inside face loop; matplotlib branch returns None** — In the plotly branch, `show_kwargs = {"show": show, **show_kwargs}` sits inside the per-face loop, so an airplane whose mesh has zero faces never gets the show flag. Separately, the matplotlib branch returns None while the docstring promises "The plotted object" for all backends. *Fix:* Hoist the show_kwargs line above the loop; return the axis/figure from the matplotlib branch.
- `aerosandbox/geometry/nosecone_shapes/haack.py:4` — **nosecone_shapes package exports nothing and haack functions lack docstrings/typing** — nosecone_shapes/__init__.py is empty, so `haack_series`, `karman`, `LV_haack`, and `tangent` are reachable only via the full submodule path. None of the four public functions have docstrings, the `C` parameter is unannotated, and there are no return annotations. (Formula itself checks out against the standard Haack-series equation.) *Fix:* Re-export the functions in __init__.py; add docstrings (units, C-parameter meaning, normalization R=1 at x/L=1) and type hints.
- `aerosandbox/library/propulsion_electric.py:273` — **electric_propeller_propulsion_analysis returns locals(), contradicting dict[str, float] annotation** — `return locals()` leaks every local name including op_point (an object), the imported function propeller_shaft_power_from_thrust, and undocumented intermediates, contradicting the `-> dict[str, float]` annotation and making the returned schema unstable under refactoring. mass_wires (line 527) has the same pattern, returning loop-scoped names. *Fix:* Build and return an explicit dict of documented keys (or at minimum del non-float locals and annotate dict[str, Any]); keep existing keys for compatibility.
- `aerosandbox/modeling/fitting.py:231` — **Model-evaluation failure re-raised as bare Exception, discarding the original error** — except Exception: raise Exception('...generic advice...') drops the original exception type/message and breaks exception chaining is-cause linkage (no 'from e'), so users debugging their model see only boilerplate. Raising bare Exception also forces callers into overly broad except clauses. *Fix:* except Exception as e: raise ValueError(<same guidance>) from e — preserves traceback chaining; ValueError is still caught by existing 'except Exception' callers.
- `aerosandbox/structures/buckling.py:135` — **plate_buckling_critical_load `length` parameter is documented but never used** — length is a required positional arg documented as "The length of the plate, in m." but is never referenced in the computation; the K constants are long-plate (a/b >> 1) asymptotic values. Users of short plates get the asymptotic coefficient with no warning, and the required arg misleads callers into thinking length affects the result. *Fix:* Document the long-plate assumption and that length is unused (or warn when length < ~3*width). Removing the parameter is breaking; defer to v5.
- `aerosandbox/tools/pretty_plots/__init__.py:33` — **pretty_plots mutates global seaborn theme and mpl rcParams as an import side effect** — Importing the package runs `sns.set_theme(...)` (line 33) and overwrites `mpl.rcParams` figure.dpi/useoffset/negative_linestyle (lines 37-39). Any transitive import restyles a user's unrelated figures with no opt-out or way to re-apply after `mpl.rcdefaults()`. The behavior is the package's purpose, so keep the default, but it should also be exposed as a callable. *Fix:* Wrap the theming in a public `set_theme()`/`apply_style()` function called once at import, and document the side effect in the module docstring.
- `aerosandbox/tools/statistics/time_series_uncertainty_quantification.py:150` — **Unconditional print() in library code paths without verbose gating** — Of 59 print() calls outside __main__ blocks, most are properly gated by `if self.verbose`. Ungated exceptions: bootstrap_fits prints estimated noise stdev (lines 150, 153; the function has no verbose parameter), mses.py:250-251 dumps subprocess stdout/stderr on error regardless of verbosity, xfoil.py:583 prints an interactive-mode notice. *Fix:* Add a `verbose: bool = True` parameter to bootstrap_fits; attach the mses stdout/stderr to the raised exception message instead of printing.
- `aerosandbox/weights/__init__.py:2` — **Shape mass-properties helpers not re-exported despite sibling being top-level API** — Only `mass_properties_from_radius_of_gyration` is re-exported (and lifted to `asb.`), while the equally public `mass_properties_of_ellipsoid`, `mass_properties_of_sphere`, `mass_properties_of_rectangular_prism`, and `mass_properties_of_cube` in the same module require deep imports — tutorials (e.g. tutorial/03/.../04) resort to `from aerosandbox.weights.mass_properties_of_shapes import ...`. *Fix:* Re-export the four `mass_properties_of_*` functions in aerosandbox/weights/__init__.py and aerosandbox/__init__.py.

#### Performance

- `aerosandbox/library/winds.py:61` — **winds.py loads .npy/.csv datasets and builds interpolation models at import time** — Lines 61-134 and 156-187 run np.load x3, np.genfromtxt, array reshaping, and construct two InterpolatedModel objects at module import. Any `import aerosandbox.library.winds` pays this I/O cost even if only wind_speed_conus_summer_99 is needed, and a missing/corrupt dataset breaks import entirely. *Fix:* Lazily build winds_95_world_model / tropopause_altitude_model on first call (cached factory or module __getattr__), keeping existing names as accessors.

#### Modernization

- `.github/workflows/run-pytest.yml:6` — **CI workflow modernization: duplicate runs, fail-fast cancellation, deprecated actions, Python 3.13 untested** — Triggering on both push '**' and pull_request '**' double-runs every same-repo PR; default fail-fast cancels the sibling Python job (seen in run 23448619799); actions/checkout@v4 and setup-uv@v4 are on deprecated Node 20 (GitHub forcing Node 24 since June 2026); matrix tests only 3.10/3.12 though requires-python has no upper bound; the tutorial-notebook step never runs when the package step fails. *Fix:* Limit push trigger to master/develop, set fail-fast: false, bump to checkout@v5/setup-uv@v7, add 3.13, split tutorials into a separate job.
- `INSTALLATION.md:102` — **Troubleshooting advice recommends 'sudo pip install', which fails on modern Linux (PEP 668)** — Lines 99-109 tell users to run `sudo pip install aerosandbox[full]`. On Debian 12+/Ubuntu 23.04+ this aborts with an 'externally-managed-environment' error, and sudo-pip into the system interpreter is long-deprecated guidance. The unquoted `aerosandbox[full]` also fails glob expansion under zsh (macOS default shell). The Anaconda-centric walkthrough is likewise dated. *Fix:* Replace sudo advice with `python -m venv` / `--user` instructions and quote extras: `pip install "aerosandbox[full]"`.
- `aerosandbox/aerodynamics/aero_2D/airfoil_inviscid.py:141` — **AirfoilInviscid internally calls asb.numpy.trapz, which is slated for removal and warns on every call** — _enforce_governing_equations uses np.trapz (the asb midpoint-average, not integration) to get panel midpoints; the function emits PendingDeprecationWarning on each call and its own docstring says it will be removed. Library-internal use guarantees future breakage and confuses readers since it is not a trapezoidal integral here. *Fix:* Replace with explicit midpoints: `(airfoil.x()[1:] + airfoil.x()[:-1]) / 2` (same for y).
- `aerosandbox/aerodynamics/aero_3D/aero_buildup_submodels/softmax_scalefree.py:5` — **Duplicate, undocumented softmax_scalefree shadows the public asb.numpy.softmax_scalefree** — aerosandbox.numpy already exposes softmax_scalefree (aerosandbox/numpy/surrogate_model_tools.py:135) with a documented *args/relative_softness API (norm-based scaling). This submodule version has no docstring, a different signature (list argument), and subtly different semantics (max-based scaling with a 1e-6 floor). Two same-named public-ish functions with different behavior is a maintenance and confusion hazard. *Fix:* Add a docstring now; in v5 (or after validating numerics), delegate to np.softmax_scalefree or rename to clarify the max-based, floored variant.
- `aerosandbox/common.py:182` — **Deprecation warnings library-wide lack `stacklevel=2` (and one lacks a category)** — warnings.warn calls for deprecations omit stacklevel: common.py line 182 (substitute_solution), library/aerodynamics/viscous.py lines 205-446 (8 sites), tools/pretty_plots/utilities/natural_univariate_spline.py lines 58-70, numpy/calculus.py line 299. Warnings therefore point at library internals, and since Python 3.7 DeprecationWarning is hidden unless triggered from __main__ — users never see them. viscous.py line 179 also warns with no category at all. *Fix:* Add `stacklevel=2` to all deprecation warns; give viscous.py line 179 an explicit `DeprecationWarning` category. (Wing/Fuselage xyz_le shims already do this correctly.)
- `aerosandbox/common.py:557` — **ImplicitAnalysis.initialize wrapper loses subclass __init__ signature/docstring** — init_wrapped is not decorated with functools.wraps(init_method), so help(), Sphinx autodoc, and IDE signature hints for every ImplicitAnalysis subclass show (*args, opti=None, **kwargs) with no docstring instead of the real constructor signature. The injected `opti` kwarg is also undiscoverable from the signature. *Fix:* Apply @functools.wraps(init_method) to init_wrapped; document the injected `opti` kwarg in the wrapper's behavior.
- `aerosandbox/library/aerodynamics/viscous.py:179` — **Cl_flat_plate deprecation of Re_c uses bare UserWarning without category or stacklevel** — Cl_flat_plate warns of Re_c deprecation via warn(msg) — a UserWarning pointing at library code — while sibling functions (Cl_2412 etc.) correctly use DeprecationWarning. None of the module's warnings pass stacklevel=2, so users see the warning attributed to aerosandbox internals rather than their call site. *Fix:* Use warnings.warn(msg, DeprecationWarning, stacklevel=2) here and add stacklevel=2 to the other deprecation warnings.
- `aerosandbox/library/winds.py:10` — **Invalid escape sequences in docstrings emit SyntaxWarning on Python 3.12+ (4 files)** — Windows paths in docstrings contain invalid escapes ('\P', '\s'): winds.py:10, mass_structural.py:85, propulsion_electric.py:359, propulsion_propeller.py:77. These raise SyntaxWarning on Python 3.12+ at import/compile and are scheduled to become SyntaxError in a future CPython release. *Fix:* Make the affected docstrings raw strings (r"""...""") or escape the backslashes.
- `aerosandbox/numpy/linalg.py:268` — **Exception chaining lost in 37 raise-without-from sites; worst prints exception to stdout** — Ruff B904 flags 37 `raise` statements inside `except` blocks without `from e`, losing tracebacks. Worst example: linalg.py:268 does `print(e)` to stdout then raises a ValueError with no chain, so the real cause is not attached to the exception. Also 29 B028 warnings.warn calls without stacklevel, which misattribute warning locations to library internals. *Fix:* Use `raise ValueError(...) from e` (remove print(e)); add stacklevel=2 to warns; consider enabling ruff B rules in pyproject.toml.
- `aerosandbox/tools/pretty_plots/colors.py:78` — **get_last_line_color reads private Line2D._color instead of get_color()** — Accesses the private `_color` attribute of matplotlib Line2D, which is not stable public API and can also return unresolved specs (e.g. 'C0') rather than the drawn RGB. The public accessor has existed forever and is guaranteed stable. *Fix:* Return `line.get_color()`.
- `pyproject.toml:11` — **License metadata uses deprecated table form instead of PEP 639 SPDX string** — `license = { text = "MIT" }` plus the 'License :: OSI Approved :: MIT License' classifier is the pre-PEP-639 style; current hatchling deprecates both in favor of an SPDX expression and license-files, and future build backends will warn or error. *Fix:* Use `license = "MIT"` and `license-files = ["LICENSE.txt"]`; drop the license classifier.
- `pyproject.toml:102` — **No [tool.pytest.ini_options] config and pytest-cov installed but coverage never measured in CI** — There is no pytest configuration: no testpaths, minversion, or filterwarnings policy (93 warnings pass silently, including the pytest-9-breaking return-not-None pattern). pytest-cov is a declared test dependency but CI never runs coverage, so the 51% overall figure is invisible to maintainers. *Fix:* Add [tool.pytest.ini_options] with testpaths='aerosandbox', selected filterwarnings=error entries, and add --cov=aerosandbox with a report/upload step in CI.

#### Type hints

- `aerosandbox/aerodynamics/aero_2D/mses.py:80` — **behavior_after_unconverged_run accepts any string; typos silently disable both behaviors** — The parameter is typed plain `str` and never validated. A typo (e.g. 'reinit') matches neither branch, so unconverged runs are silently skipped with no reinitialization and no termination — a third, undocumented behavior. Also, working_directory is typed `str | None` but XFoil accepts `Path | str | None`. *Fix:* Type as Literal['reinitialize', 'terminate'], validate in __init__ raising ValueError; widen working_directory to Path | str | None.
- `aerosandbox/aerodynamics/aero_3D/lifting_line.py:58` — **model_size typed as bare str instead of Literal; NonlinearLiftingLine spanwise_resolution unannotated** — LiftingLine's `model_size: str = "medium"` accepts any string and is forwarded to NeuralFoil where invalid values fail deep in the stack; AeroBuildup already uses `Literal["small", "large"]` style. nonlinear_lifting_line.py line 58 leaves `spanwise_resolution=8` unannotated with a '# TODO document' marker. *Fix:* Annotate model_size with the Literal of valid NeuralFoil sizes and add `spanwise_resolution: int = 8` plus its docstring entry.
- `aerosandbox/aerodynamics/aero_3D/singularities/point_source.py:13` — **Typing gaps: viscous_radius untyped, fuselage_form_factor missing return annotation** — Public functions have incomplete annotations: point_source.py:13 `viscous_radius=0` has no type hint (should be float); fuselage_aerodynamics_utilities.py:97 fuselage_form_factor lacks a return annotation (-> float); uniform_strength_horseshoe_singularities.py:17 annotates vortex_core_radius as float although the np.all(vortex_core_radius == 0) check implies array support, and the horseshoe/point-source pair use different names (vortex_core_radius vs viscous_radius) for the same concept. *Fix:* Add `viscous_radius: float = 0`, `-> float` on fuselage_form_factor; align naming/typing of the desingularization parameter across the two singularity functions.
- `aerosandbox/atmosphere/atmosphere.py:139` — **Public Atmosphere/OperatingPoint methods lack parameter and return type annotations** — pressure(), temperature(), density(), speed_of_sound(), dynamic_viscosity(), mean_free_path(), knudsen(length) in atmosphere.py, and dynamic_pressure(), mach(), reynolds(reference_length), total_pressure(), etc. in operating_point.py have no annotations, while the codebase elsewhere (e.g., aerosandbox/numpy/trig.py) uses Vectorizable -> Array hints. *Fix:* Annotate inputs/returns with Vectorizable (from aerosandbox.numpy.typing), matching existing convention; purely additive, no runtime change.
- `aerosandbox/common.py:38` — **_asb_metadata annotated as dict[str, str] but defaulted to None** — The class attribute annotation `_asb_metadata: dict[str, str] = None` is type-incorrect (None is not a dict[str, str]); mypy/pyright flag this on the base class of the entire library. *Fix:* Annotate as `dict[str, str] | None = None`.
- `aerosandbox/geometry/airfoil/airfoil.py:240` — **Shared mutable ndarray defaults and missing annotations on public signatures** — generate_polars (alphas, Res), plot_polars, local_camber/local_thickness, max_camber/max_thickness, and KulfanAirfoil.upper/lower_coordinates all use module-level np arrays as defaults (shared mutable state; one in-place edit corrupts all future calls). generate_polars' alphas/Res are also unannotated and crash on plain lists (`Res.min()`, `attached_alphas_to_use.min()`). *Fix:* Use None defaults constructed inside functions (or tuples), add type hints, and `np.asarray` list inputs in generate_polars.
- `aerosandbox/modeling/fitting.py:341` — **goodness_of_fit: 'type' parameter shadows builtin and lacks Literal/return annotations** — def goodness_of_fit(self, type='R^2') shadows the builtin, has no type annotation despite accepting only 9 fixed strings, and no return annotation. Renaming the parameter would break keyword callers (goodness_of_fit(type='L1')), so the rename itself is a v5 item; annotations can be added now. *Fix:* Annotate now: type: Literal['R^2', 'mean_absolute_error', ...] = 'R^2', -> float. Defer parameter rename (e.g., to 'metric') to v5.
- `aerosandbox/structures/tube_spar_bending.py:22` — **Minor typing gaps on TubeSparBendingStructure public surface** — assume_thin_tube=True lacks a `: bool` annotation; volume(), total_force(), and draw(show=True) have no parameter or return annotations. The class is a documented public analysis, so these gaps hurt IDE/static-analysis support. *Fix:* Add `assume_thin_tube: bool = True`, `show: bool = True`, and return annotations (float-like for volume/total_force, None for draw).

#### Documentation & docstrings

- `CONTRIBUTING.md:117` — **Docstring style is inconsistent across public API: Google, numpydoc, reST, and freeform mixed** — CONTRIBUTING prescribes Google-style, and most modules comply (opti.py, wing.py, aero_buildup.py), but common.py and numpy/calculus.py use numpydoc, while library/aerodynamics/viscous.py, structures/legacy/beams.py, and OperatingPoint.reynolds use Sphinx reST `:param:` fields. Sampled ~25 public functions/20 modules: many public members are freeform or missing (Airfoil.get_aero_from_neuralfoil, MassProperties.inertia_tensor, most DynamicsPointMass3DCartesian methods). *Fix:* Incrementally convert reST/numpydoc docstrings to Google style; napoleon already renders both so this is zero-risk.
- `README.md:117` — **README typo 'discplines' and inconsistent master/develop branch links** — Line 117: 'Among many other discplines:' (should be 'disciplines'). Also, tutorial hyperlinks inconsistently target branches: lines 98/104/110 link to master-branch notebooks while lines 184/190 link to develop-branch paths — develop links shown to PyPI users can reference unreleased or reorganized content and break independently of releases. *Fix:* Fix the typo; point all tutorial links at master (or use relative links, which GitHub resolves per-branch).
- `README.md:207` — **README claims base install is 'headless with minimal dependencies' but base deps include matplotlib+seaborn** — README line 207 advertises `pip install aerosandbox` as lightweight/headless with 'optional visualization dependencies skipped', but pyproject.toml lines 37-48 make matplotlib and seaborn hard requirements — and until the airfoil.py plotly bug is fixed, even plotly is effectively required to import the package. *Fix:* Update README wording to say base install includes matplotlib/seaborn and `[full]` adds plotly/pyvista/trimesh/cadquery/shapely/sympy.
- `aerosandbox/README.md:74` — **Developer README 'Map' references removed in_progress folder and omits three real modules** — The module map ends with `in_progress`: 'Here be dragons' — that directory no longer exists in the package. Meanwhile the map omits `/dynamics/`, `/weights/`, and `/performance/`, all of which exist and export top-level public classes (DynamicsPointMass*, MassProperties, OperatingPoint). README.md links here as 'the developer README', so new contributors get a stale picture. *Fix:* Delete the in_progress bullet; add one-line entries for dynamics/, weights/, and performance/.
- `aerosandbox/__init__.py:64` — **`asb.docs()` opens the GitHub source tree instead of the hosted docs site that pyproject declares** — docs() opens `github.com/peterdsharpe/AeroSandbox/tree/master/aerosandbox` with a TODO to point at hosted docs — but pyproject.toml already declares Homepage `https://peterdsharpe.github.io/AeroSandbox/`, and a `docs` extra (sphinx/autoapi/furo) exists. The top-level package also has no module docstring, so `help(aerosandbox)` shows nothing about the library. *Fix:* Point docs() at https://peterdsharpe.github.io/AeroSandbox/ and add a short package docstring to aerosandbox/__init__.py.
- `aerosandbox/aerodynamics/aero_2D/airfoil_polar_functions.py:17` — **airfoil_coefficients_post_stall docstring lists wrong args and type hint understates accepted input** — Docstring Args section lists 'airfoil' and 'op_point', but the second parameter is `alpha` (degrees); Returns section is empty though it returns (CL, CD, CM). `alpha: float` is too narrow — Airfoil.plot_polars and KulfanAirfoil call it with arrays, which is the intended vectorized use. *Fix:* Document alpha [deg] and the (CL, CD, CM) return tuple; annotate alpha as float | np.ndarray (Vectorizable).
- `aerosandbox/aerodynamics/aero_3D/avl.py:208` — **Public AVL.run() docstring says 'Private function to run AVL'** — run() is the primary public entry point (used in the class usage example) but its docstring begins 'Private function to run AVL', a leftover from the removed _run_avl() abstraction (commit b726d4d1). The __init__ docstring also omits the xyz_ref, ground_effect, and ground_effect_height parameters. *Fix:* Reword the run() docstring and document the missing constructor parameters.
- `aerosandbox/aerodynamics/aero_3D/lifting_line.py:374` — **LiftingLine docstrings: 'alp ha' split mid-word, nonexistent 'opti' arg documented, wrong class names** — run_with_stability_derivatives Args list renders '- alp / ha (bool)' (word broken across lines 374-375). __init__ docstring (lines 80-86) documents an `opti` parameter that does not exist for this explicit analysis, and omits model_size/spanwise_resolution/etc. calculate_streamlines docstring (lines 1050, 1070) references 'VortexLatticeFilaments' and 'VortexLatticeMethod.streamlines' instead of LiftingLine. *Fix:* Fix the alpha typo, remove the opti paragraph, document actual constructor params, and correct class names in copied docstrings.
- `aerosandbox/atmosphere/atmosphere.py:93` — **IndexError message missing space: 'while theparent has length'** — The f-string concatenation in Atmosphere.__getitem__ (lines 93-94) and the identical code in OperatingPoint.__getitem__ (operating_point.py:212-213) renders as '...it has length N while theparent has length M.' in user-facing exceptions. *Fix:* Add the missing space ('while the parent has length') in both files, ideally deduplicating the shared get_item_of_attribute helper.
- `aerosandbox/atmosphere/atmosphere.py:173` — **density_altitude() docstring omits `method` arg and unimplemented 'exact' option** — The public method accepts method: Literal['approximate','exact'] but has no Args section; 'exact' raises NotImplementedError, which is undocumented. Also ratio_of_specific_heats() (line 238) is public with no docstring or return annotation. *Fix:* Document the method arg, state 'exact' is not yet implemented, cite the Wikipedia formula used; add a docstring to ratio_of_specific_heats().
- `aerosandbox/atmosphere/thermodynamics/choked_flow.py:4` — **mass_flow_rate() has no docstring, units, reference, or type hints** — Public function computing compressible-flow mass flow (formula verified correct: mdot = A*P_t*sqrt(gamma/(R*T_t))*M*(1+(gamma-1)/2*M^2)^(-(gamma+1)/(2(gamma-1)))) documents nothing: no units (SI assumed), no arg descriptions, no return description, no annotations. *Fix:* Add docstring with SI units per arg, return units [kg/s], a reference (e.g., Anderson, Modern Compressible Flow), and Vectorizable/float annotations.
- `aerosandbox/atmosphere/thermodynamics/gas.py:193` — **process() docstring says inplace=True returns nothing, but code returns self** — Docstring: 'If `inplace` is True, nothing is returned.' The implementation (line 329) does `return self`. The docstring also omits 'isenthalpic' from the process list even though it is in the Literal (it raises NotImplementedError). *Fix:* Update docstring: inplace=True mutates and returns self; note 'isenthalpic' is accepted but not yet implemented.
- `aerosandbox/dynamics/point_mass/common_point_mass.py:394` — **add_gravity_force docstring says '-z direction' but the force is applied in +z_e (down, NED)** — The code correctly adds Fz = +m*g in Earth axes (z_e points down; altitude = -z_e, and DynamicsPointMass1DVertical integrates it with correct falling behavior), but the docstring claims a force 'in the -z direction', contradicting the implementation and the library's NED Earth-axes convention. *Fix:* Reword docstring: 'adds a downward force (+z in Earth/NED axes) equal to the weight of the aircraft'; annotate g: float.
- `aerosandbox/dynamics/point_mass/point_1D/vertical.py:19` — **Docstring errors in point-mass classes: wrong axis label, nonexistent method name, u/v typo** — vertical.py:19 says 'Fz_e: Force along the Earth-x axis' (should be Earth-z). horizontal.py:12 and vertical.py:12 reference '.add_gravity()', but the method is add_gravity_force(). point_3D/cartesian.py:21 says 'v_e: v-velocity' (should be y-velocity). *Fix:* Correct the three docstrings: Earth-z axis, .add_gravity_force(), and 'y-velocity, in Earth axes'.
- `aerosandbox/geometry/airfoil/airfoil.py:1439` — **rotate() docstring contradicts itself: summary says clockwise, behavior is counterclockwise** — Summary line says 'Rotates the airfoil clockwise', but the Args section says counterclockwise and np.rotation_matrix_2D is documented as 'counterclockwise rotation' (aerosandbox/numpy/rotations.py:32). A user trusting the summary gets a sign error in geometry. *Fix:* Change summary to 'Rotates the airfoil counterclockwise by the specified angle, in radians.'
- `aerosandbox/geometry/airfoil/airfoil.py:1526` — **write_dat() and draw() docstrings misstate return values** — write_dat() docstring says 'Returns: None' but always returns the .dat contents as a str (line 1542) and lacks a return annotation. draw() (line 876) also says 'Returns: None' though the plotly backend returns a Figure when show=False, and the matplotlib backend never returns the annotated Figure. *Fix:* Document write_dat -> str (annotate `-> str`); document draw's backend-dependent return, and don't document draw_markers param gap.
- `aerosandbox/geometry/airfoil/airfoil_families.py:462` — **least_squares unknown-vector packing comment states upper-first; code packs lower-first** — The explanatory block says the solution vector packs 'upper_weights ... lower_weights ...', but the A matrix (lines 517-525) and the extraction (lines 532-533) put lower_weights in the first block and upper_weights second. Misleading for maintainers editing this solver. *Fix:* Correct the comment to lower_weights first, then upper_weights, then leading_edge_weight, then TE thickness.
- `aerosandbox/geometry/fuselage.py:626` — **FuselageXSec docstring superellipse equation wrong by factor of 2** — Docstring states `abs(y / width) ^ shape + abs(z / height) ^ shape = 1`, but width/height are full extents: get_3D_coordinates places points at y = ±width/2, z = ±height/2. The correct implicit equation is `abs(2y/width)^shape + abs(2z/height)^shape = 1`. *Fix:* Correct the equation to use width/2 and height/2.
- `aerosandbox/geometry/polygon.py:92` — **Polygon.rotate() docstring self-contradicts: "clockwise" summary vs "counterclockwise" Args** — The summary line says "Rotates a Polygon clockwise by the specified amount", while the `angle` arg doc says "counterclockwise". The implementation uses np.rotation_matrix_2D, a standard counterclockwise rotation, so the summary is wrong. *Fix:* Change the summary to "counterclockwise".
- `aerosandbox/geometry/propulsor.py:28` — **Public Propulsor.__init__ docstring is a placeholder ("TODO add docs")** — asb.Propulsor is exported public API, but its constructor docstring is literally "TODO add docs" — no description of xyz_normal orientation convention (default [-1,0,0], points 'backwards'), radius/length semantics, or color. get_disk_3D_coordinates has no docstring at all. *Fix:* Document all constructor args (mirroring FuselageXSec style) and get_disk_3D_coordinates.
- `aerosandbox/geometry/wing.py:610` — **Broken reference URL in Wing.aerodynamic_center docstring ("downloattttd")** — The cited methodology link reads `https://core.ac.uk/downloattttd/pdf/79175663.pdf` — a keyboard-mash typo that breaks the reference. The correct URL appears in mean_aerodynamic_chord (line 476): `https://core.ac.uk/download/pdf/79175663.pdf`. *Fix:* Fix "downloattttd" to "download".
- `aerosandbox/library/aerodynamics/viscous.py:25` — **Docstring gaps: empty Returns + stray TODO in Cd_cylinder; Korn functions missing param docs** — Cd_cylinder's docstring has an empty "Returns:" section with a developer TODO rendered into user docs. In transonic.py, Cd_wave_Korn omits the `mach` parameter from its docstring and mach_crit_Korn has an empty Returns; indicial_gust_response (unsteady.py:127) documents `velocity` but the parameter is `plate_velocity`. *Fix:* Fill in Returns sections, move TODO to a code comment, document `mach`, rename `velocity` doc entry to `plate_velocity`.
- `aerosandbox/modeling/black_box.py:18` — **black_box public API has empty docstring args and undocumented fd_step/fd_step_iter** — Exported black_box documents 'function:', 'n_in:', 'n_out:' and 'Returns:' as blank; fd_step and fd_step_iter (mapped to CasADi fd_options h/h_iter) are absent from the docstring entirely. Type hints are also wrong: Callable[[Any], float] declares a single-argument callable, but the wrapper accepts arbitrary positional/keyword args (should be Callable[..., float]). *Fix:* Fill in the Args/Returns sections (n_in semantics, fd_step -> CasADi 'h', fd_step_iter -> 'h_iter') and change annotations to Callable[..., float].
- `aerosandbox/modeling/surrogate_model.py:47` — **Minor docstring errors: 'R^1 -> R^2' typo and invalid usage example in FittedModel** — SurrogateModel.__call__ docstring says 'in the case of a 1-dimensional input (R^1 -> R^2)' — should be R^1 -> R^1 (class maps to R^1). Related: fitting.py line 131 example '>>> y = FittedModel(x)' calls the class instead of an instance; should be my_fitted_model(x). *Fix:* Fix 'R^2' to 'R^1' in surrogate_model.py:47; change fitting.py:131-133 examples to use an instance variable.
- `aerosandbox/numpy/array.py:49` — **array() docstring says dtype is ignored for CasADi contents, but code crashes instead** — The docstring states dtype 'is ignored' when input contains CasADi types, but the `or dtype is not None` clause routes CasADi-containing lists to `_onp.array(..., dtype=...)`. Verified: array([MX.sym('x'), 2], dtype=float) raises ValueError. interpolate.interp calls array(xp, dtype=float), so a list of CasADi scalars as xp would crash. *Fix:* Drop `or dtype is not None` so CasADi-containing input takes the vertcat path (dtype ignored), matching the docstring.
- `aerosandbox/optimization/opti.py:39` — **Opti.__init__ parameters entirely undocumented (marked '# TODO document')** — The constructor of the package's flagship class documents none of its six parameters. cache_filename, load_frozen_variables_from_cache, save_to_cache_on_solve, ignore_violated_parametric_constraints, and freeze_style drive the whole freeze/cache workflow, which is currently only discoverable via the variable() docstring's examples. *Fix:* Add an Args section to __init__ documenting each parameter and the interaction between cache_filename, variable_categories_to_freeze, and freeze_style.
- `aerosandbox/optimization/opti.py:742` — **solve_sweep is public but has no docstring and incomplete type annotations** — solve_sweep has zero docstring; update_initial_guesses_between_solves and verbose lack annotations; the declared return type np.ndarray | Callable[[cas.MX], np.ndarray] hides that the array contains OptiSol objects (or None for failed runs), which callers must know to use it. *Fix:* Add a docstring documenting parameters, the object-array-of-OptiSol/None return, and return_callable semantics; annotate the two bare parameters as bool.
- `aerosandbox/performance/operating_point.py:10` — **OperatingPoint, a top-level asb export, has no class docstring** — `asb.OperatingPoint` is one of the primary user-facing classes (paired with Airplane in every aero analysis) but the class has no docstring, so `help(asb.OperatingPoint)` and the generated API docs show nothing about alpha/beta being in degrees, velocity units, or axes conventions — exactly the unit pitfalls the README promises docstrings will clarify. *Fix:* Add a class docstring covering constructor parameters, units (alpha/beta in degrees), and axes conventions.
- `aerosandbox/structures/buckling.py:139` — **poissons_ratio missing from docstring Args; buckling functions lack return-type annotations** — plate_buckling_critical_load's docstring omits poissons_ratio from its Args list. All three public functions in buckling.py annotate inputs as float but accept arrays/CasADi expressions (dual-backend) and have no return annotations. Also, docstring option order for column_buckling_critical_load differs from the Literal. *Fix:* Document poissons_ratio; annotate params as float | np.ndarray and add return annotations.
- `aerosandbox/structures/legacy/simple_beam_opt.py:43` — **TODO triage: 76 TODO/HACK comments; author-flagged wrong physics in legacy beam models** — 76 TODO/FIXME/HACK comments in library code. Highest-value: simple_beam_opt.py:43 and simple_beam_opt_daedalus_calibration.py:42 use isotropic G = E/2/(1+nu) with nu=0.5 for CFRP with the author's own note 'TODO fix this!!! CFRP is not isotropic!' (composite shear modulus is not derivable from the isotropic relation); both live in structures/legacy/. *Fix:* Add a docstring warning about the isotropic-G approximation in the legacy beam modules, or deprecate them formally; burn down remaining TODOs opportunistically.
- `aerosandbox/visualization/plotly_Figure3D.py:114` — **Figure3D docstrings reference nonexistent add_face() and wrong examples** — add_tri and add_quad docstring examples both say 'E.g. add_face(...)' — no add_face method exists — and add_quad's example shows only 3 points for a method that requires exactly 4 (raises ValueError). add_streamline's docstring says 'Adds a line' (copy-paste from add_line). Figure3D itself has no class docstring. *Fix:* Fix examples to add_tri/add_quad with correct point counts; correct add_streamline summary; add a class docstring.
- `aerosandbox/weights/mass_properties.py:150` — **Malformed user-facing error messages in MassProperties.__getitem__ and __len__** — The IndexError message at lines 150-151 concatenates adjacent f-strings without a space, producing '...while theparent has length...'. The ValueError at line 195 reads 'State variables are appear vectorized, but of different lengths!' (grammar). *Fix:* Add the missing space ('while the parent') and change to 'State variables appear vectorized, but with different lengths!'.
- `docs/source/conf.py:53` — **conf.py points at nonexistent _static/_templates dirs and has stale 2023 copyright** — `html_static_path = ["_static"]` and `templates_path = ["_templates"]` reference directories that do not exist under docs/source/, producing 'html_static_path entry does not exist' warnings on every build (fatal if -W is ever enabled). `copyright = "2023, Peter Sharpe"` is stale. *Fix:* Remove the two unused path settings (or create the dirs); use a dynamic copyright year.
- `pyproject.toml:41` — **neuralfoil<0.4.0 upper bound is justified but undocumented** — NeuralFoil minor releases change network weights/API surface — CHANGELOG 4.2.7 shows explicit compatibility work was needed for 0.3.0 — so the cap is rationally defensive (latest on PyPI is 0.3.2; no 0.4 exists yet). But nothing records this rationale, and when 0.4.0 ships, co-installation with other packages wanting it will break with an opaque resolver error. *Fix:* Add a TOML comment explaining the cap (weights/API change per minor) and track a compat issue for 0.4.

#### Dead code

- `.gitignore:1` — **Stale local build artifacts: dist/ holds old releases; AeroSandbox.egg-info/ is a setuptools-era leftover** — dist/ (aerosandbox-4.2.8 wheel, 4.2.9 sdist) and AeroSandbox.egg-info/ exist in the repo root. Both are correctly gitignored and untracked, so no repo pollution, but egg-info will never be regenerated under hatchling and a stale one can feed wrong metadata to pip during editable installs. CI publishes from a clean checkout, so releases are unaffected. *Fix:* Delete dist/ and AeroSandbox.egg-info/ locally; keep the .gitignore entries.
- `aerosandbox/aerodynamics/aero_2D/IBL2.py:21` — **IBL2 is an empty stub: __init__ body is `pass`, silently doing nothing** — IBL2's __init__ accepts streamwise_coordinate, edge_velocity, viscosity, theta_0, H_0 and does nothing with them; instantiating it silently succeeds and produces a useless object. It is not exported in aero_2D/__init__.py but is importable and looks like a working analysis. *Fix:* Raise NotImplementedError in __init__ (with a docstring note that it is a placeholder), or remove the stub.
- `aerosandbox/aerodynamics/aero_2D/mses.py:81` — **MSES constructor param mset_alpha is stored but never used** — self.mset_alpha is assigned in __init__ but run() always meshes at alphas[0] (line 248) and re-meshes at next_alpha; the user-supplied mset_alpha has no effect, silently ignoring an apparently functional public parameter. *Fix:* Use self.mset_alpha for the initial mset() call (falling back to alphas[0] if None), or deprecate the parameter with a warning.
- `aerosandbox/atmosphere/atmosphere.py:18` — **Unused module constant effective_collision_diameter; unused PerfectGas attribute** — effective_collision_diameter in atmosphere.py is referenced nowhere in the repo (mean_free_path uses the viscosity-based formula instead). Similarly PerfectGas.effective_collision_diameter (gas.py:51) is stored and copied through process() but never used in any computation. *Fix:* Remove the atmosphere.py constant (or mark deprecated since it is importable); either implement PerfectGas.mean_free_path using the attribute or document it as reserved.
- `aerosandbox/dynamics/rigid_body/rigid_3D/body_euler.py:317` — **sincos() helper duplicated verbatim; alpha/beta/speed re-declarations duplicate base class** — The ~25-line sincos() snap-to-cardinal helper is copy-pasted twice (state_derivatives line 168, convert_axes line 317), and body_euler.py lines 417-425 re-define alpha/beta identically to _DynamicsRigidBodyBaseClass (common_rigid_body.py:52-60). Divergence risk on future edits; the bare 'except Exception' in sincos also masks real errors. *Fix:* Extract sincos() to a module-level private function (or use it from one place), delete the duplicated alpha/beta properties, and narrow the except clause.
- `aerosandbox/library/aerodynamics/unsteady.py:232` — **Dead no-op statement inside calculate_lift_due_to_pitching_profile integrand** — The integrand contains `if dW_ds(sigma) < 0: dW_ds(sigma)` — the body discards its result, so the whole block is a no-op (leftover debugging), and it doubles dW_ds evaluations inside scipy quad. *Fix:* Delete the if-block; return dW_ds(sigma) * AoA_function(s - sigma) directly.
- `aerosandbox/library/weights/torenbeek_weights.py:407` — **torenbeek mass_fuselage is an unfinished stub that always raises NotImplementedError** — Public function mass_fuselage computes two fineness_ratio() results, discards them, then unconditionally raises NotImplementedError; it has no docstring and appears complete in autocomplete/docs alongside working siblings. power_nuclear_rtg.po210_specific_power has the same discarded-expression stub pattern. *Fix:* Either finish the Torenbeek Appendix D implementation or remove the dead statements and add a docstring stating it is not yet implemented, pointing to mass_fuselage_simple.
- `aerosandbox/numpy/calculus.py:13` — **Ruff triage: 3 F401 unused imports are the only default-rule violations** — `uv run ruff check aerosandbox --statistics` reports only 3 F401s: unused `typing.cast` (calculus.py:13), unused `_CasADiType` (linalg.py:12), plus one in a test file (out of scope). All autofixable with --fix. *Fix:* Run `ruff check aerosandbox --fix` to remove the two library-file unused imports.
- `aerosandbox/structures/legacy/simple_beam_opt.py:151` — **stress_von_mises_squared is misnamed, not von Mises, and dead code** — In both simple_beam_opt.py (line 151) and simple_beam_opt_daedalus_calibration.py (line 139), stress_von_mises_squared = np.sqrt(stress_axial**2 + 0*stress_shear**2) is (a) a square root, not a square; (b) not the von Mises criterion (which is sqrt(sigma^2 + 3*tau^2)); (c) has shear zeroed out; and (d) is never used (stress = stress_axial). *Fix:* Delete the variable in both scripts, or implement sqrt(sigma^2 + 3*tau^2) and actually use it.
- `aerosandbox/tools/statistics/time_series_uncertainty_quantification.py:181` — **Dead code in bootstrap_fits: no-op expression and unreachable branch** — Line 181 `x_noise_stdev / x_rng` computes and discards a value (looks like a lost `x_stdev_normalized =` assignment; x noise is correctly added in un-normalized space at line 211, so the value is unneeded). Also lines 240-243 re-check `if fit_points is None` inside the else-branch of the same test at line 235 — unreachable, including its ValueError. *Fix:* Delete line 181 and the unreachable `if fit_points is None` block at lines 240-243.
- `qodana.yaml:29` — **qodana.yaml is orphaned — no workflow runs Qodana** — .github/workflows/ contains only run-pytest.yml and publish-on-master-push.yml; nothing references Qodana, so this starter-profile config (with an unpinned jetbrains/qodana-python:latest linter) is dead configuration, and ruff already covers linting per pyproject. *Fix:* Delete qodana.yaml, or add a qodana workflow if JetBrains analysis is still wanted.

---

## 5. Deferred to v5 (requires breaking changes)

3 finding(s) above are marked with high backwards-compat risk; they are repeated here for planning:

- `aerosandbox/library/weights/raymer_cargo_transport_weights.py:288` — **n_gear misread as gear count; Raymer's N_gear is the gear load factor (~3)** ✓ — Raymer defines N_l = N_gear * 1.5 where N_gear is the gear load factor from Table 11.5 (typically ~3), an aircraft-level constant. Code (lines 288, 334; also raymer_general_aviation_weights.py lines 246, 285) uses 'number of landing gear' (defaults 2 and 1), giving N_l = 3 and 1.5 instead of ~4.5, and inconsistent main/nose values — underestimating nose gear mass up to ~46%. *Fix:* Add a gear_load_factor parameter (default 3.0) and compute N_l = gear_load_factor * 1.5; deprecate the n_gear-based load factor. Semantics change, so defer to v5.
- `aerosandbox/library/weights/raymer_general_aviation_weights.py:458` — **GA mass_hydraulics uses fuselage width where Raymer's equation uses design gross weight** ✓ — Raymer's GA hydraulics equation (5th Ed., Sec. 15.3.3) is W_hyd = K_h * W_dg^0.8 * M^0.5 with W_dg in lb (the K_h values quoted in the code's own comment come from that equation). The code substitutes fuselage_width^0.8, yielding ~0.01 lb for a typical light aircraft instead of a few lb — off by orders of magnitude. *Fix:* Use (design_mass_TOGW/u.lbm)**0.8; requires adding a design_mass_TOGW parameter and deprecating fuselage_width.
- `aerosandbox/dynamics/point_mass/point_3D/cartesian.py:225` — **add_force default 'axes' differs per subclass ('earth'/'wind'/'body'), contradicting base-class signature** — The abstract base declares axes: Literal[...] = "wind", but DynamicsPointMass3DCartesian defaults to "earth", speed-gamma classes to "wind", rigid-body to "body". Code written against the base-class docs silently applies forces in the wrong frame when switching dynamics classes. Subclass overrides also drop the Literal annotation entirely. *Fix:* Document each subclass's native default in its docstring, restore Literal annotations now; unify defaults (or require explicit axes) in v5 since changing them silently alters physics.

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
