# [AeroSandbox](https://peterdsharpe.github.io/AeroSandbox/) :airplane:

by [Peter Sharpe](https://peterdsharpe.github.io) (<pds [at] mit [dot] edu>)

[![Downloads](https://pepy.tech/badge/aerosandbox)](https://pepy.tech/project/aerosandbox)
[![Monthly Downloads](https://pepy.tech/badge/aerosandbox/month)](https://pepy.tech/project/aerosandbox)
[![Build Status](https://github.com/peterdsharpe/AeroSandbox/workflows/Tests/badge.svg)](https://github.com/peterdsharpe/AeroSandbox/actions/workflows/run-pytest.yml)
[![PyPI](https://img.shields.io/pypi/v/aerosandbox.svg)](https://pypi.python.org/pypi/aerosandbox)
[![Documentation Status](https://readthedocs.org/projects/aerosandbox/badge/?version=master)](https://aerosandbox.readthedocs.io/en/master/?badge=master)

## Overview

**AeroSandbox is a Python package for design optimization of engineered systems such as aircraft.**

At its heart, AeroSandbox is an optimization suite that combines the ease-of-use of [familiar NumPy syntax](aerosandbox/numpy) with the power of [modern automatic differentiation](./tutorial/10%20-%20Miscellaneous/03%20-%20Resources%20on%20Automatic%20Differentiation.md).

This automatic differentiation dramatically improves optimization performance on large problems: **design problems with tens of thousands of decision variables solve in seconds on a laptop**.

AeroSandbox also comes with dozens of end-to-end-differentiable aerospace physics models, allowing you to **simultaneously optimize an aircraft's aerodynamics, structures, propulsion, mission trajectory, stability, and more.**

## Examples

Use AeroSandbox to design and optimize entire aircraft:

<table style="table-layout: fixed; width: 100%;">
    <tr>
        <td style="width:50%;">
            <p align="center">
                <a href="https://github.com/peterdsharpe/Feather-RC-Glider"><i>Feather</i> (an ultra-lightweight 1-meter-class RC motor glider)</a>
            </p>
            <img src="https://raw.githubusercontent.com/peterdsharpe/Feather-RC-Glider/master/CAD/feather.png" width="500" alt="Feather first page">
        </td>
        <td style="width:50%;">
            <p align="center">
                <a href="https://github.com/peterdsharpe/solar-seaplane-preliminary-sizing"><i>SEAWAY-Mini</i> (a solar-electric, 13' wingspan seaplane)</a>
            </p>
            <img src="https://raw.githubusercontent.com/peterdsharpe/solar-seaplane-preliminary-sizing/main/CAD/renders/seaway_mini_packet_Page_1.png" width="500" alt="Seaway-Mini first page">
        </td>
    </tr>
</table>

Use AeroSandbox to support real-world aircraft development programs, all the way from your very first sketch to your first-flight and even beyond:

<table style="table-layout: fixed; width: 100%;">
    <tr>
        <td style="width:50%;">
            <p align="center" width="500">
                <a href="https://github.com/peterdsharpe/DawnDesignTool">Initial concept sketches + sizing of <i>Dawn</i> (a solar-electric airplane for climate science research) in AeroSandbox, Spring 2020</a>
            </p>
            <img src="./media/images/dawn1-first-sketch.png" width="500" alt="Dawn initial design">
        </td>
        <td style="width:50%;">
            <p align="center">
                <a href="https://youtu.be/CyTzx9UCvyo"><i>Dawn</i> (later renamed <i>SACOS</i>) in first flight, Fall 2022</a>
            </p>
            <p align="center"><a href="https://www.electra.aero/news/sacos-first-flight">(Thanks to so, so many wonderful people!)</a></p>
            <img src="./media/images/SACOS%20First%20Flight.jpg" width="500" alt="SACOS first flight">
        </td>
    </tr>
</table>

Use AeroSandbox to explore counterintuitive, complicated design tradeoffs, all at the earliest stages of conceptual design *where these insights make the most difference*:

<table>
	<tr>
		<td>
			<p align="center">
				<a href="https://github.com/peterdsharpe/DawnDesignTool">Exploring how big a solar airplane needs to be to fly, as a function of seasonality and latitude</a>
			</p>
			<img src="https://github.com/peterdsharpe/DawnDesignTool/raw/master/docs/30kg_payload.svg" width="500" alt="Dawn seasonality latitude tradespace">
		</td>
		<td>
			<p align="center">
				<a href="https://www.popularmechanics.com/military/aviation/a13938789/mit-developing-mach-08-rocket-drone-for-the-air-force/">Exploring how the mission range of <i>Firefly</i>, a Mach 0.8 rocket drone, changes if we add an altitude limit, simultaneously optimizing aircraft design and trajectories</a>
			</p>
			<img src="./media/images/firefly-range-ceiling-trade.png" width="500" alt="Firefly range ceiling trade">
		</td>
	</tr>
</table>

Use AeroSandbox as a pure aerodynamics toolkit:

<table>
	<tr>
		<td>
			<p align="center">
				VLM simulation of a glider, aileron deflections of +-30°
			</p>
			<img src="./media/images/vlm3_with_control_surfaces.png" width="500" alt="VLM simulation">
		</td>
		<td>
			<p align="center">
				Panel simulation of a wing (extruded NACA2412, α=15°, AR=4)
			</p>
			<img src="./media/images/panel1_naca4412.png" width="500" alt="Panel simulation">
		</td>
	</tr>
</table>

Use AeroSandbox as a pure structural toolkit:

<table>
	<tr>
		<td>
			<p align="center">
				Structural optimization of a composite tube spar
			</p>
			<img src="./media/images/beam-optimization.png" width="300" alt="Beam optimization">
		</td>
	</tr>
</table>

## Getting Started

### Installation

In short:

* `pip install aerosandbox[full]` for a complete install.

* `pip install aerosandbox` for a lightweight (headless) installation with minimal dependencies. All optimization, numerics, and physics models are included, but optional visualization dependencies are skipped.

For more installation details (e.g., if you're new to Python), [see here](./INSTALLATION.md).

### Tutorials, Examples, and Documentation

To get started, [check out the tutorials folder here](./tutorial/)! All tutorials are viewable in-browser, or you can open them as Jupyter notebooks by cloning this repository.

For a more detailed and theory-heavy introduction to AeroSandbox, [please see this thesis](./tutorial/sharpe-pds-sm-AeroAstro-2021-thesis.pdf).

For a yet-more-detailed developer-level description of AeroSandbox modules, [please see the developer README](aerosandbox/README.md).

For fully-detailed API documentation, see [the documentation website](https://aerosandbox.readthedocs.io/en/master/).

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
or [GitHub Sponsors](https://github.com/sponsors/peterdsharpe/)! Proceeds will go towards more coffee for the grad students.

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

[MIT License, terms here](LICENSE.txt). Basically: use AeroSandbox for anything you want; no warranty express or implied.

## Stargazers over time

[![Stargazers over time](https://starchart.cc/peterdsharpe/AeroSandbox.svg)](https://starchart.cc/peterdsharpe/AeroSandbox) 
