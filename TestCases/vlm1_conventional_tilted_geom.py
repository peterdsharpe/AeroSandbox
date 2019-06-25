from Classes import *
import ExampleAirplanes

a = ExampleAirplanes.conventional()

a.wings[0].sections[0].twist=10
a.wings[0].sections[1].twist=10
a.wings[0].sections[2].twist=10

ap = vlm1(
    airplane=a,
    op_point=OperatingPoint(velocity=10,
                            alpha=0,
                            beta=0),
)
ap.run()
ap.draw()
