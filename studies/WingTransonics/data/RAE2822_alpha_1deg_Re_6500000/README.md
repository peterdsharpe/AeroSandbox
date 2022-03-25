# Data Series

Aerodynamics data computed with a variety of computational methods.

Case info:

* 2D study
* RAE2822 airfoil
* Angle of attack (alpha) of 1.00 degree.
* Chord-referenced Reynolds number: 6.5 million

Data series:

* `mses.csv`: Data computed with MSES. (Euler + IBL via transpiration)
* `su2_series1`: Data computed with SU2. (RANS) 
  * Full details at [peterdsharpe/TransonicWingAerodynamicsDatabase](https://github.com/peterdsharpe/transonic-wing-aerodynamics-database)
  * Meshes generated with PyAero. 
    * Wall y+ of 1 essentially everywhere.
    * Inflation layer meshed with growth ratio of 1.10, out to a total thickness of 10x the computed flat-plate equivalent BL thickness.
    * 300 streamwise points around the entire airfoil perimeter.
  * Numerics:
    * Steady
      * Convergence criterion: Cauchy bound of drag coefficient at 1e-6 over 100 iterations.
    * SA turbulence model, without the BC transition model.
      * Profile drag discrepancy against XFoil is observed in the data at medium-subsonic mach numbers (~0.3-0.7); XFoil is likely more correct here.
        * Verified that adding BC transition model doesn't fix this.
        * Verified that switching to SST turbulence model doesn't fix this.
        * [Seems that viscous drag error vs. XFoil, experiment is reported by other SU2 users?](https://www.cfd-online.com/Forums/su2/167046-high-drag-airfoil-compared-xfoil-wind-tunnel-data.html) Concerning.
        * Mesh independence study not yet conducted (simply due to time limits). TODO.
    * JST (centered) flux scheme
      * MUSCL for mass, momentum
      * 1st order upwinding for `nu_tilde` (nondimensional turbulent viscosity)
      * Verified that AUSM doesn't fix profile drag computation issue.
    * Full numerics details in `su2.cfg`
* `xfoil6.csv`: Data computed with XFoil v6.99 (public). (Laplace + Prandtl-Glauert correction + IBL via transpiration.)
* `xfoil7-pg.csv`: Data computed with XFoil v7.02 (not public). (Laplace + Prandtl-Glauert correction + IBL via transpiration.)
* TODO: `xfoil7-fp.csv`: Data computed with XFoil v7.02 (not public). (Full potential solution + IBL)
