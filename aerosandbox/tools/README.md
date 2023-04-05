# Miscellaneous Tools

**Note: all contents in this folder are miscellaneous and do not interact in any way with the rest of AeroSandbox. If you are trying to learn AeroSandbox, you should completely ignore this folder and move on.**

Some of them may use libraries which are not required as part of a base install of AeroSandbox, so many of these scripts won't work without extra installation on users' ends. Things in this folder may break, without warning - they are just gists stored in a common place for the benefit of a small core group of developers on various aircraft design projects at MIT.

Think of this as a "Gists" folder - random snippets and scripts from scientific computing work go here. Basically, these are a few snippets of "bundled software".

A summary of a few of these:

- `aerosandbox.tools.pretty_plots`: A set of plotting functions for making plots look pretty, and for producing various plots that are useful for engineering design optimization. Built on top of Matplotlib and Seaborn.
- `aerosandbox.tools.string_formatting`: A set of functions for formatting strings in a variety of ways, including LaTeX formatting. Useful for making pretty plot labels.
- `aerosandbox.tools.units`: A set of scalars that represents various units. (Note: AeroSandbox uses base SI units everywhere internally - these are just for user convenience.)
- `aerosandbox.tools.inspect_tools`: This is where some Python black magic happens - basically, it's Python interpreting its own source code. Has functions that will return their own source code (as a string), at any arbitrary level of the call stack. Can take a Python object and generate source code (as a string) that attempts to reconstruct it by parsing its constructor. Has functions that can tell you from what file and line of code they were called from.
- `aerosandbox.tools.webplotdigitizer_reader`: A function that reads in a [WebPlotDigitizer](https://github.com/ankitrohatgi/WebPlotDigitizer) CSV file and returns a dictionary of the data. Useful for reading in data from old, image-based data tables (e.g., wind tunnel charts) and reconstructing it digitally.