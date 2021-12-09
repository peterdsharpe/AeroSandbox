# [AeroSandbox](https://peterdsharpe.github.io/AeroSandbox/) :airplane:

by [Peter Sharpe](https://peterdsharpe.github.io) (<pds [at] mit [dot] edu>)

[![Downloads](https://pepy.tech/badge/aerosandbox)](https://pepy.tech/project/aerosandbox)
[![Monthly Downloads](https://pepy.tech/badge/aerosandbox/month)](https://pepy.tech/project/aerosandbox)
[![Build Status](https://github.com/peterdsharpe/AeroSandbox/workflows/Tests/badge.svg)](https://github.com/peterdsharpe/AeroSandbox/actions/workflows/run-pytest.yml)

## Overview

**AeroSandbox is a Python package for design optimization of engineered systems such as aircraft.**

At its heart, AeroSandbox is an optimization suite that combines the ease-of-use of [familiar NumPy syntax](aerosandbox/numpy) with the power of [modern automatic differentiation](./tutorial/10%20-%20Miscellaneous/03%20-%20Resources%20on%20Automatic%20Differentiation.md).

This automatic differentiation dramatically improves optimization performance on large problems: **design problems with tens of thousands of decision variables solve in seconds on a laptop**.

AeroSandbox also comes with dozens of end-to-end-differentiable aerospace physics models, allowing you to **simultaneously optimize an aircraft's aerodynamics, structures, propulsion, mission trajectory, stability, and more.**

![VLM Image](media/images/vlm3_with_control_surfaces.png)
*VLM simulation of a glider, aileron deflections of +-30°. Runtime of 0.35 sec on a typical laptop (i7-8750H).*

![PANEL Image](media/images/panel1_naca4412.png)
*Panel simulation of a wing (extruded NACA2412, α=15°, AR=4). Note the strong three-dimensionality of the flow near the tip.*

## Getting Started

### Installation

Use `pip install aerosandbox[full]` for a complete install.

For a lightweight installation with minimal dependencies, use `pip install aerosandbox`. All optimization, numerics, and physics models are included this headless install, but some visualization dependencies are not installed.

### Tutorials, Examples, and Documentation

To get started, [check out the tutorials folder here](./tutorial/)! All tutorials are viewable in-browser, or you can open them as Jupyter notebooks by cloning this repository.

For a more detailed and theory-heavy introduction to AeroSandbox, [please see this thesis](./tutorial/sharpe-pds-sm-AeroAstro-2021-thesis.pdf).

For a yet-more-detailed developer-level description of AeroSandbox modules, [please see the developer README](aerosandbox/README.md).

You can print documentation and examples for any AeroSandbox object by using the built-in `help()` function (e.g., `help(asb.Airplane)`). AeroSandbox code is also documented *extensively* in the source and contains hundreds of unit test examples, so examining the source code can also be useful.

### Usage Details

One final point to note: as we're all sensible and civilized here, **all inputs and outputs to AeroSandbox are expressed in base SI units, or derived units thereof** (e.g, m, N, kg, m/s, J, Pa).

The only exception to this rule is when units are explicitly noted via variable name suffix. For example:

* `battery_capacity` -> Joules
* `battery_capacity_watt_hours` -> Watt-hours.

All angles are in radians, except for α and β which are in degrees due to long-standing aerospace convention. (In any case, units are marked on all function docstrings.)

If you wish to use other units, consider using `aerosandbox.tools.units` to convert easily.

## Project Details

### Contributing

Please feel free to join the development of AeroSandbox - contributions are always so welcome! If you have a change you'd like to make, the easiest way to do that is by submitting a pull request.

The text file [`CONTRIBUTING.md`](./CONTRIBUTING.md) has more details for developers and power users.

If you've already made several additions and would like to be involved in a more long-term capacity, please message me!
Contact information can be found next to my name near the top of this README.

### Donating

If you like this software, please consider donating to support development [via PayPal](https://paypal.me/peterdsharpe)
or [GitHub Sponsors](https://github.com/sponsors/peterdsharpe/)! I'm a grad student, so every dollar that you donate helps wean me off my diet of instant coffee and microwaved ramen noodles.

### Bugs

Please, please report all bugs by creating a new issue at [https://github.com/peterdsharpe/AeroSandbox/issues](https://github.com/peterdsharpe/AeroSandbox/issues)!

### Versioning

AeroSandbox loosely uses [semantic versioning](https://semver.org/), which should give you an idea of whether or not you can probably expect backward-compatibility and/or new features from any given update. However, the code is a work in progress and things change rapidly - for the time being, please freeze your version of AeroSandbox for any serious deployments. Commercial users: I'm more than happy to discuss consulting work for active AeroSandbox support if this package proves helpful!

### Citation

If you find AeroSandbox useful in a research publication, please cite it using the following BibTeX snippet:

```bibtex
@mastersthesis{aerosandbox,
    title = {AeroSandbox: A Differentiable Framework for Aircraft Design Optimization},
    author = {Sharpe, Peter D.},
    school = {Massachusetts Institute of Technology},
    year = {2021}
}
```

### License

[MIT License, full terms here](LICENSE.txt).

## Stargazers over time

[![Stargazers over time](https://starchart.cc/peterdsharpe/AeroSandbox.svg)](https://starchart.cc/peterdsharpe/AeroSandbox) 
