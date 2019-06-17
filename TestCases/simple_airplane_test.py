from Classes import *
import ExampleAirplanes

a = ExampleAirplanes.simple_airplane()
a.set_paneling_everywhere(12,40)
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

# Answer you should get:
# CL = 0.316
# CD = 0.008
# CL/CD = 39.892
# Cm = -0.073

# Dimensionalized:
# L = 19.355 N
# D = 0.49 N