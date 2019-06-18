from Classes import *
import ExampleAirplanes

a = ExampleAirplanes.simple_airplane()
# a.wings[0].sections[0].chordwise_spacing = 'uniform'
# a.wings[0].sections[0].spanwise_spacing = 'uniform'
a.set_paneling_everywhere(50,50)
#a.draw()

ap = AeroProblem(
    airplane=a,
    op_point=OperatingPoint(velocity=10,
                            alpha=5,
                            beta=0),
)
ap.make_vlm1_problem()

# a.draw()
# ap.draw_panels()

# Answer you should get: (XFLR5)
# CL = 0.316
# CD = 0.008
# CL/CD = 39.892
# Cm = -0.073

# Dimensionalized:
# L = 19.355 N
# D = 0.49 N

# From Oswald
# AR = 4
# CDi = 0.316 ^ 2 / (pi * 4 * 0.7)
# CDi = 0.011351