from Classes import *
import ExampleAirplanes

a = ExampleAirplanes.XFLR_default()
ap = AeroProblem(
    airplane=a,
    op_point=OperatingPoint(velocity=10,
                            alpha=5,
                            beta=0),
)
ap.make_vlm1_problem()

# a.draw()
# ap.draw_panels(draw_forces= True, draw_vortex_strengths=True)
