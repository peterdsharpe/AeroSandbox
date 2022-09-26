import aerosandbox.numpy as np
import aerosandbox as asb

opti = asb.Opti()

n_vars = 2
alpha = opti.variable(n_vars=n_vars)
beta = np.linspace(0,2,n_vars)

vlm_results = asb.VortexLatticeMethod(
    airplane=airplane,
    op_point=asb.OperatingPoint(
        atmosphere=asb.Atmosphere(),
        alpha = alpha,
        beta = beta
    )
).run()

opti.subject_to([
    vlm_results["F_g"][1] == 100,
])

sol = opti.solve()