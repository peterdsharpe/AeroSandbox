# Contributing

Wow! So you're interested in contributing - first of all, thank you so much! Here's what you need to know before contributing:

1. We use the Git branching model [posted here](https://nvie.com/posts/a-successful-git-branching-model/), which is by far the most common model in open-source software development. Main points:

      1. Never commit directly to `master`!
      2. Add features in new branches named `feature/insert-name-here`. Be sure to branch off of `develop`, not off of `master`!
      3. When your feature is ready in your feature branch:
          1. First, rebase your feature branch on `develop` and check that all unit tests still pass (run `pytest` in terminal in the project root directory).
          2. Submit a pull request to merge it back into `develop` and wait for a core developer to approve.

2. As far as code style goes, we use PEP8 naming conventions:

      1. `variable_names_use_snake_case`
      2. `function_names_also_use_snake_case`
      3. `ClassNamesUsePascalCase`

3. Use long, descriptive variable names. Use `temperature` instead of `T`. Use `wing_tip_coordinate_x` instead of `wtx`. In the age of modern IDEs with autocomplete and one-click refactoring, there is no reason not to.

4. All new classes should extend one of the classes in the top-level file `common.py`. In particular, all explicit analyses (e.g. workbook-style aero buildups) should extend `ExplicitAnalysis` and all implicit analyses (i.e. analyses that involve iteratively solving nonlinear systems of equations) should extend `ImplicitAnalysis`. All other classes should extend `AeroSandboxObject`.

5. All engineering quantities (i.e. quantities with units) used anywhere in AeroSandbox are expressed in base metric units, or derived units thereof (meters, newtons, meters per second, kilograms, etc.). This is true even for quantities that are usually expressed in non-base-metric units: `battery_capacity` is in units of joules (not watt-hours), `temperature` is in Kelvin (not Celsius), and`elastic_modulus` is in units of pascals (not GPa). The only exception is when units are explicitly noted as a suffix in a variable name: for example,and `battery_capacity_watt_hours` is in units of watt-hours, and `altitude_ft` is in units of feet.

6. When writing math, use NumPy functions everywhere where possible. If this throws an error during testing, replace the offending function with its equivalent from `aerosandbox.optimization.math`. If you do not find the function you need here, notify a core developer.

7. Every function must be documented by a docstring of some sort, with no exceptions. [Google-style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) are slightly preferred but not required. It is highly recommended (but not required) that you also do the following:
    
    * Document the purpose, function, and expected input type(s) of every input parameter within this docstring. 
    * [Type hint](https://realpython.com/lessons/type-hinting/) all functions that you write.
    * Include usage examples in runnable Python in for each function in its docstring (demarcated by `>>>`) 
    
8. With *very* rare exceptions, do not write the same sequence of characters more than twice. For example:

    ``` python
    ### This is discouraged
    print(sol.value(x))
    print(sol.value(y))
    print(sol.value(z))
    
    ### Instead, do this:
    for var in [x, y, z]:
        print(sol.value(var))
    ```

9. Spread long mathematical expressions across multiple lines based on natural groupings of ideas in the equation. For example:

    ```python
    ### This is discouraged
    distance = ((x_start - x_end) ** 2 + (y_start - y_end) ** 2) ** 0.5
    
    ### Instead, do this:
    distance = (
    	(x_start - x_end) ** 2 +
        (y_start - y_end) ** 2
    ) ** 0.5
    ```

## Code of Conduct

There is just one rule when interacting with this project:

1. Be **respectful** to **everyone**.

Breaking this up:

* "respectful": 
	* Use kind and welcoming language. 
	* Respect different viewpoints and the experiences that lead to them. 
	* Gracefully accept constructive criticism. 
	* Do not waste others' time by trolling or arguing in bad faith.
	* Do not insult, harass, or dox others.
* "everyone": Everyone means everyone, regardless of age, body
	size, disability, ethnicity, sex characteristics, gender identity and expression,
	level of experience, education, socio-economic status, nationality, personal
	appearance, race, religion, or sexual identity and orientation.

### Attribution

This Code of Conduct is adapted from the [Contributor Covenant](https://www.contributor-covenant.org/version/1/4/code-of-conduct.html).