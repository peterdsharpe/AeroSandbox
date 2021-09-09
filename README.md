# [AeroSandbox](https://peterdsharpe.github.io/AeroSandbox/) :airplane:
by [Peter Sharpe](https://peterdsharpe.github.io) (<pds [at] mit [dot] edu>)

[![Downloads](https://pepy.tech/badge/aerosandbox)](https://pepy.tech/project/aerosandbox)
[![Monthly Downloads](https://pepy.tech/badge/aerosandbox/month)](https://pepy.tech/project/aerosandbox)
[![Build Status](https://github.com/peterdsharpe/AeroSandbox/workflows/Tests/badge.svg)](https://github.com/peterdsharpe/AeroSandbox/actions/workflows/run-pytest.yml)

## Overview

**AeroSandbox is a Python package for design optimization of engineered systems, with additional tools for aircraft design applications.**



ASB has  collection of end-to-end automatic-differentiable models and analysis tools for aircraft design applications. This property of automatic-differentiability dramatically improves performance on large problems; **design problems with thousands or tens of thousands of decision variables solve in seconds on a laptop**. Using AeroSandbox, you can **simultaneously optimize an aircraft's aerodynamics, structures, propulsion, mission trajectory, stability, and more.** 

AeroSandbox has powerful aerodynamics solvers (VLM, 3D panel) written from the ground up, and AeroSandbox can also be used as a standalone aerodynamics solver if desired. Like all other modules, these solvers are end-to-end automatic-differentiable. Therefore, **in half a second, you can calculate not only the aerodynamic performance of an airplane, but also the sensitivity of aerodynamic performance with respect to an arbitary number of design variables.**

![VLM Image](media/images/vlm3_with_control_surfaces.png)
*VLM simulation of a glider, aileron deflections of +-30°. Runtime of 0.35 sec on a typical laptop (i7-8750H).*

![PANEL Image](media/images/panel1_naca4412.png)
*Panel simulation of a wing (extruded NACA2412, α=15°, AR=4). Note the strong three-dimensionality of the flow near the tip.*

## Getting Started

### Installation and Tutorials

Install with `pip install AeroSandbox`. Requires Python 3.7+.

Alternatively, clone from `master` on GitHub and install with `pip install .`.

To get started, check out the `tutorials` folder [here](./tutorial/)!

### Dependencies

The fastest way to ensure that all dependencies are satisfied is by simply running "pip install AeroSandbox" in your command prompt. However, you can also install dependencies on your own if you'd like: see "requirements.txt" for the list.

## Donating
If you like this software, please consider donating to support development [via PayPal](https://paypal.me/peterdsharpe) or [GitHub Sponsors](https://github.com/sponsors/peterdsharpe/)! I'm a grad student, so every dollar that you donate helps wean me off my diet of instant coffee and microwaved ramen noodles.

## Bugs
Please, please report all bugs by creating a new issue at [https://github.com/peterdsharpe/AeroSandbox/issues](https://github.com/peterdsharpe/AeroSandbox/issues)!

## Details
One final point to note: as we're all sensible and civilized human beings here, **all inputs and outputs to AeroSandbox are expressed in base metric units, or derived units thereof** (meters, newtons, meters per second, kilograms, etc.). The only exception to this rule is when units are explicitly noted in a variable name: for example, `battery_capacity` would be in units of joules, `elastic_modulus` would be in units of pascals, and `battery_capacity_watt_hours` would be in units of watt-hours.

## Versioning
AeroSandbox loosely uses [semantic versioning](https://semver.org/), which should give you an idea of whether or not you can probably expect backward-compatibility and/or new features from any given update. However, the code is a work in progress and things change rapidly - for the time being, please freeze your version of AeroSandbox for any serious deployments. Commercial users: I'm more than happy to discuss consulting work for active AeroSandbox support if this package proves helpful!

## Contributing
Please feel free to join the development of AeroSandbox - contributions are always so welcome! If you have a change you'd like to make, the easiest way to do that is by submitting a pull request. 

The text file [`CONTRIBUTING.md`](./CONTRIBUTING.md) has more details for developers and power users.

If you've already made several additions and would like to be involved in a more long-term capacity, please message me! Contact information can be found next to my name near the top of this README.

## Citation

If you find AeroSandbox useful in a research publication, please cite using the following BibTeX snippet:

```bibtex
@mastersthesis{aerosandbox,
    title = {AeroSandbox: A Differentiable Framework for Aircraft Design Optimization},
    author = {Sharpe, Peter D.},
    school = {Massachusetts Institute of Technology},
    year = {2021}
}
```

## License

MIT License

Copyright (c) 2020 Peter Sharpe

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Stargazers over time

[![Stargazers over time](https://starchart.cc/peterdsharpe/AeroSandbox.svg)](https://starchart.cc/peterdsharpe/AeroSandbox) 
