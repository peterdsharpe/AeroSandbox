# Multi-Boom Spar Mass Study
Peter Sharpe

## Description
Finds the mass of the spar for a wing on a single- or multi-boom lightweight aircraft. Model originally designed for solar aircraft.

Assumptions (you can change all of these in `sweep.py`):
* Elliptical lift distribution
* Constraint that local wing dihedral/anhedral angle must not exceed 10 degrees anywhere in the ultimate load case.
* If multi-boom, assumes roughly static-aerostructurally-optimal placement of the outer booms and equal boom weights.

1. Run `sweep.py`. Tweak options before running, if desired.