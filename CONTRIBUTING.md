# Contributing

Wow! So you're interested in contributing - thank you so much!

First, read the AeroSandbox roadmap in `AeroSandbox/aerosandbox/README.md`.

After reading that, here's what else you need to know before contributing:

## On Installation and Git Conventions

If you're developing, install AeroSandbox in editable mode. In other words:

1. First:
	* For most developers (i.e. those that haven't yet been officially added as collaborators to the ASB repository):
		1. Fork the repository on GitHub.
		2. Clone your forked repository from GitHub to your computer.
	* For developers that have been officially added as collaborators to the ASB repository on GitHub:
		1. Clone the AeroSandbox repository from GitHub to your computer.
2. Then, do these steps:
	1. On your computer, open up a terminal in the AeroSandbox root directory. (To check that you're in the right place, view your current directory with either `dir` or `ls` depending on your OS; you should see a file called `setup.py`.)
	2. If you already have an AeroSandbox installation on your computer, first uninstall that (`pip uninstall aerosandbox`).
	3. Install the cloned copy of your repository in editable mode (`-e`), and with all optional dependencies (`pip install -e .[full,test,docs]`).
	4. Switch to the develop branch for normal use (`git checkout develop`)
	5. *While on the develop branch*, create a new branch if you want to make changes (`git checkout -b feature/insert-your-feature-name-here`)
3. From here, you can make your changes. After you are finished:
	1. Make sure that your branch passes tests by running [pytest](https://docs.pytest.org/) from terminal (`pytest`).
	2. Verify that your code (at least loosely) follows the coding standards in this document.
	3. Make sure your work is committed and pushed.
	4. Create a pull request on GitHub targeting the `develop` branch of the main ASB repository.

We use the Git branching model [posted here](https://nvie.com/posts/a-successful-git-branching-model/) (this is by far the most common model Git branching model, so if you've used Git before on a collaborative project, you probably already know this). Main points:

1. Never commit directly to `master`!
2. Add features in new branches named `feature/insert-name-here`. Be sure to branch these features off of `develop`, not off of `master`!
3. When your feature is ready in your feature branch:
	1. First, merge `develop` into your feature branch to get latest changes and check that all unit tests still pass (run `pytest` in terminal in the project root directory).
	2. Submit a pull request to merge it back into `develop` and wait for a core developer to approve.

Some other guidelines:

* Commit often. In general, commit after every "unit" of work or anytime the code base returns to a "working" (i.e. tests passing) state. If you find yourself working for more than an hour or writing >100 lines since your last commit, consider whether a "unit" of work has been done. (Perhaps it has, perhaps it hasn't - just think about it.)

* Write concise but descriptive commit messages - think "Google search" type language. Examples:

	* "add method is_red to class House in house.py" (preferred)
	* "add is_red to House" (acceptable)
	* "add methods to house.py" (acceptable)
	* "blah" (not acceptable)

  One exception to the rule: if you're just adding/changing comments or documentation with no code changes, you can just write "docs" (or similar) as your commit message and be done.

## On Project Structure

* Some AeroSandbox-specific notes:

	* Use `aerosandbox.numpy` functions everywhere where possible. If this throws an error during testing or if you do not find the function you need here, notify a core developer.

	* You should never need to import CasADi or try to manually determine array type anywhere outside of `aerosandbox/numpy/`. If for some reason you need to do CasADi-specific things, do them within a function inside `aerosandbox/numpy` - this keeps the split between engineering code and numerics code clean.

	* AeroSandbox code is extensively documented; if you're ever not sure how to use something, call `help()` on it in console! E.g. `help(asb.Airplane)`.

* All new classes should extend one of the classes in the top-level file `common.py`. In particular, all explicit analyses (e.g. workbook-style aero buildups) should extend `ExplicitAnalysis` and all implicit analyses (i.e. analyses that involve iteratively solving nonlinear systems of equations) should extend `ImplicitAnalysis`. All other classes should extend `AeroSandboxObject`. Also, all user-facing classes should contain a `__repr__` method.

* Write unit tests for all new functionality using `pytest`. Make sure the tests pass before submitting a pull request.

## On Code Style for Scientific Computing

This is all pretty standard across all scientific computing in Python:

* As far as code style goes, we use standard Python PEP8 naming conventions:

    1. `variable_names_use_snake_case`
    2. `def function_names_also_use_snake_case():`
    3. `class ClassNamesUsePascalCase:`

* Use long, descriptive variable names. Use `temperature` instead of `T`. Use `wing_tip_coordinate_x` instead of `wtx`. In the age of modern IDEs with autocomplete and one-click refactoring, there is no reason not to obfuscate meaning with short variable names. Long variable names also force you to split complicated expressions onto multiple lines; this is a good thing (see the following point).

  * Spread expressions across multiple lines based on natural groupings of ideas. Generally, all functions with multiple input parameters should have each parameter on a new line unless it exceptionally short (something like `range(3,10)`, for example). Some examples of discouraged and encouraged coding standards:

      ```python
      ### This is discouraged:
      distance = ((x_start - x_end) ** 2 + (y_start - y_end) ** 2) ** 0.5
    
      ### Instead, do this:
      distance = (
          (x_start - x_end) ** 2 +
          (y_start - y_end) ** 2
      ) ** 0.5
      ```

      ```python
      ### This is discouraged:
      plt.plot(time, temperature, ".-", color='lightgrey', alpha=0.6, label="Temperature over time")
    
      ### Instead, do this:
      plt.plot(
              time, 
            temperature, 
            ".-", 
            color='lightgrey', 
            alpha=0.6, 
            label="Temperature over time"
      )
      ```

* All engineering quantities (i.e. quantities with units) used anywhere in AeroSandbox are expressed in base metric units, or derived units thereof (meters, newtons, meters per second, kilograms, etc.).

    * This is true even for quantities that are traditionally expressed in non-base-metric units. Some common "gotchas":
        * `battery_capacity` is in units of joules (not watt-hours)
        * `temperature` is in Kelvin (not Celsius)
        * `elastic_modulus` is in units of pascals (not GPa)
        * `altitude` is in meters (not feet)
        * `radio_frequency` is in Hz (not MHz)

  The only exception is when units are explicitly noted as a suffix in a variable name: for example `battery_capacity_watt_hours` is in units of watt-hours, and `altitude_ft` is in units of feet.

* Every new user-facing function is required to be documented by a docstring of some sort, with no exceptions. [Google-style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) are preferred but not required - as long as your docstring is intelligible to an average engineer who might come across it, you're fine. It is highly recommended (but not required) that you also do the following:

    * Document the purpose, function, and expected input type(s) of every input parameter within this docstring.
    * [Type hint](https://realpython.com/lessons/type-hinting/) all functions that you write.

  We illustrate both of these requirements with the following example:
    ```python
    from typing import List, Tuple, Union  
  # Note: List, Tuple, Dict, Union, etc. (note capitalization) need to be imported from the built-in "typing"
    
   
    def is_my_dog_happy(
        temperature: float,  # <-- this is a type hint for a parameter!
        n_boats: int,  # note that Python doesn't enforce the type you specify, they're just "hints" for the user
        coffee_brands: Union[List[str], Tuple[str]]  # You can denote multiple acceptable inputs with "Union", imported from "typing"
    ) -> bool:  # <-- this is a type hint for a return!
        """
        This function tells me if my dog is happy today.
  
        Example:
        >>> if is_my_dog_happy(300, 10, ["Illy", "Peet's"]):
        >>>     celebrate()
  
        Args:
            temperature: The temperature outside, in Kelvin [float]
            n_boats: The number of boats I see on the Charles River [int]
            coffee_brands: An iterable of the names of various brands of coffee [List[str]]
  
        Returns:
            Whether or not my dog is happy [bool]
        """
        return True  # Of course my dog is always happy!
    ```
    * Also notice that in this example above, we put each parameter on its own line. Generally, do this.
    * Include usage examples in runnable Python in for each function in its docstring (demarcated by `>>>`)

* With rare exceptions, do not type the same sequence of characters more than twice. For example:

    ``` python
    ### This is discouraged:
    print(sol(x))
    print(sol(y))
    print(sol(z))
    
    ### Instead, do this:
    for var in [x, y, z]:
        print(sol(var))
    ```
  
* Use keyword arguments for functions, unless those functions are exceptionally simple, well-known, or self-explanatory. For example:

    ```python
  import aerosandbox as asb
  
  ### This is discouraged:
  atmosphere = asb.Atmosphere(7000)
  
  ### Instead, do this:    
  atmosphere = asb.Atmosphere(altitude=7000)
    ```

* When defining a new function: never, never make the default value of a parameter a mutable data type. Common culprits are empty lists `[]` and empty dicts `{}`. Debugging this common mistake is often very difficult and wastes other devs' time. For an explanation of why this is such bad practice, read this article: ["Python Mutable Defaults are the Source of All Evil"](https://florimond.dev/en/posts/2018/08/python-mutable-defaults-are-the-source-of-all-evil/).

	Any IDE worth its salt will likely yell at you if you do this - listen to it!

    * As an example of what **NOT** to do:
	``` python
	def bad_function(
			my_parameter=[],  # Don't do this
			another_param={},  # Or this!!!
	):
		pass
	
	
	### Instead, do this:
	def good_function(
			my_parameter: List=None,
			another_param: Dict=None,
	):
		### Set defaults
		if my_parameter is None:
			my_parameter = []
		if another_param is None:
			another_param = {}
	```

## Intellectual Property Ownership

As a condition of contribution to this repository, contributors agree to transfer all right, title, and interest in their contributions to the Core Developers of this repository. The Core Developers are defined as the set of individuals who, at any time, simultaneously fulfill all three criteria: a) They are authorized as a collaborator on this GitHub repository, b) their name is listed in the `README.md` as an author, and c) their total number of Git commits to this repository is at least 20% that of the individual with the highest number of commits to this repository. This is not intended to impede the ability of contributors to use their own contributions, but rather to allow Core Developers to make timely decisions for the project without tracking down all contributors who have made minor commits many years ago. Notably, the Core Developers grant both the contributor and other entities the broad rights described in `LICENSE.txt`.

# Code of Conduct

There is just one rule when interacting with this project:

1. Be **respectful** to **everyone**.

Breaking this up:

* "respectful":
	* Use kind and welcoming language.
	* Respect different viewpoints and the experiences that lead to them.
	* Gracefully accept constructive criticism.
	* Do not waste others' time by trolling or arguing in bad faith.
	* Do not insult, harass, or dox others.
	* Stay respectful even when others may be disrespectful.
* "everyone": Everyone means everyone, regardless of age, body size, disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Attribution

This Code of Conduct is loosely adapted from the [Contributor Covenant](https://www.contributor-covenant.org/version/1/4/code-of-conduct.html).
