from Classes import *
import ExampleAirplanes

a = ExampleAirplanes.conventional()
# a.set_paneling_everywhere(20,20)

# a.wings=[a.wings[0],a.wings[2]]

ap = vlm1(
    airplane=a,
    op_point=OperatingPoint(velocity=10,
                            alpha=5,
                            beta=0),
)
ap.run()

# a.draw()
# ap.draw(draw_forces= True, draw_vortex_strengths=True)
