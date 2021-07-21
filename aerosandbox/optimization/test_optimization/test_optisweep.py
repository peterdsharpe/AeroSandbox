import aerosandbox as asb
import aerosandbox.numpy as np
import pathos.multiprocessing as mp

opti = asb.Opti()

x = opti.variable(init_guess=0)
p = opti.parameter(value=5)

f = (x - 2 * p) ** 2

opti.subject_to(f < p)
opti.minimize(f)

if __name__ == '__main__':
    with mp.Pool() as pool:
        o = opti.sweep(
            pool=pool,
            parameter_mapping={
                p: np.linspace(5, -5, 11)
            },
            outputs=[
                x
            ],
            filename="C:/Users/User/Downloads/test.csv",
            parallel=True
        )
        print(o)

# def func(pval):
#     opti.set_value(p, pval)
#     return opti.solve(verbose=False).value(x)
#
#
# if __name__ == '__main__':
#
#     ps = np.linspace(5, -5, 11)
#
#     with mp.Pool() as pool:
#         for res in pool.imap_unordered(
#                 func=func,
#                 iterable=ps,
#         ):
#             print(res)
