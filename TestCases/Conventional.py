from Classes import *
import ExampleAirplanes

a = ExampleAirplanes.conventional()

a.wings=[a.wings[0],a.wings[1],a.wings[2]]

ap = AeroProblem(
    airplane=a,
    op_point=OperatingPoint(velocity=10,
                            alpha=5,
                            beta=0),
)
ap.vlm1()

# a.draw()
# ap.draw_panels(draw_forces= True, draw_vortex_strengths=True)
