from aerosandbox import *
import dill as pickle
with open(r"C:\Projects\GitHub\AeroSandbox\aerosandbox\tools\airfoil_fitter\func.pkl", "rb") as f:
    func = pickle.load(f)
print(
    func(0, 1e6)
)