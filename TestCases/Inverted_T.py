from Classes import *
import ExampleAirplanes

a = ExampleAirplanes.inverted_T_test()
a.set_paneling_everywhere(1,1)
# a.wings=[a.wings[0]]

ap = vlm1(
    airplane=a,
    op_point=OperatingPoint(velocity=10,
                            alpha=5,
                            beta=0),
)
ap.run()

ap.calculate_Vij(ap.c[2])