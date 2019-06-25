from Classes import *
import ExampleAirplanes

a = ExampleAirplanes.conventional()

ap = vlm1(
    airplane=a,
    op_point=OperatingPoint(velocity=10,
                            alpha=5,
                            beta=5),
)
ap.run()
ap.draw()