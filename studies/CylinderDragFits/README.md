# CylinderDragFits
By Peter Sharpe

## Overview

A single, universally-applicable curve fit for the drag coefficient of a cylinder in crossflow for a wide range of Reynolds numbers ranging from 1e-1 to 1e7. The fit also extrapolates sensibly beyond this range.

Why doesn't this exist already in the literature?!

![Fit Display](cylinderdragfit.svg)

Data from Wiley, "Incompressible Flow" (4th Ed.).

## TL;DR: The Model

The following model is found, where `x = log10(Re)`:

The model:

Matlab-ish syntax:
```matlab
log10(CD) = 
(log10(10^(csub0*x+csub1)+csub2+csub3*x))*(1-1/(1+exp(-csigh*(x-csigc)))) + (csup0+csupscl/csuph*log(exp(csuph*(csupc-x))+1))*(1/(1+exp(-csigh*(x-csigc))))
```

Python-ish syntax:
```python
log10(CD) = 
(log10(10 ** (csub0*x+csub1)+csub2+csub3*x))*(1-1/(1+exp(-csigh*(x-csigc)))) + (csup0+csupscl/csuph*log(exp(csuph*(csupc-x))+1))*(1/(1+exp(-csigh*(x-csigc))))
```
Constants:
```
csigc = 5.5766722118597247
csigh = 23.7460859935990563
csub0 = -0.6989492360435040
csub1 = 1.0465189382830078
csub2 = 0.7044228755898569
csub3 = 0.0846501115443938
csup0 = -0.0823564417206403
csupc = 6.8020230357616764
csuph = 9.9999999999999787
csupscl = -0.4570690347113859
```
