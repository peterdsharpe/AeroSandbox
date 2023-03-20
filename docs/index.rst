.. AeroSandbox documentation master file, created by
   sphinx-quickstart on Mon Mar 20 11:33:53 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AeroSandbox Documentation
=======================================

by `Peter Sharpe <https://peterdsharpe.github.io>`_ (<pds [at] mit [dot] edu>)

.. image:: https://pepy.tech/badge/aerosandbox
.. image:: https://pepy.tech/badge/aerosandbox/month
.. image:: https://github.com/peterdsharpe/AeroSandbox/workflows/Tests/badge.svg
.. image:: https://img.shields.io/pypi/v/aerosandbox.svg

**AeroSandbox is a Python package for design optimization of engineered systems such as aircraft.**

At its heart, AeroSandbox is an optimization suite that combines the ease-of-use of familiar NumPy syntax with the power of modern automatic differentiation.

This automatic differentiation dramatically improves optimization performance on large problems: **design problems with tens of thousands of decision variables solve in seconds on a laptop**.

AeroSandbox also comes with dozens of end-to-end-differentiable aerospace physics models, allowing you to **simultaneously optimize an aircraft's aerodynamics, structures, propulsion, mission trajectory, stability, and more.**


.. toctree::
   :maxdepth: 10



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
