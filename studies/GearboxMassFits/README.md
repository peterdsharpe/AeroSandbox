# GearboxMassFits
By Peter Sharpe

## Overview

Refitting data from a NASA technical report.

It baffles me that the original authors took data that's a straight line on a log-log plot and fit it to a linear equation instead of a power law - that's not how logarithms work at all!

This distinction doesn't particularly matter if you're using the equation within their data range, but if you extrapolate the authors' model even slightly beyond the data range, you start to predict negative gearbox mass.

Fit comparison:

![Fit Display](gearboxmassfit.svg)

Data from NASATM—2009-215680.

## TL;DR: The Model

The following model is found, where `x = log10(Re)`:

The model:
```python
log10(mass_lbs) = 
p1 * log10(beta) + p2
```
where:
```
beta = (power_hp / rpm_out) ** 0.75 * (rpm_in / rpm_out) ** 0.15
```

Constants:
```
p1 = 1.0445171124733774
p2 = 2.0083615496306910
```
