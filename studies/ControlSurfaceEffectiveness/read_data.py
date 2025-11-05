import pandas as pd
import aerosandbox.numpy as np

df = pd.read_csv("data.csv", index_col=0)


a = df.alpha.values
d = df.deflection.values
hf = 1 - df.hinge_x.values
r = df.Re.values
da = df.dalpha.values


mask = (d != 0) & (d + a < 20)

a = a[mask]
d = d[mask]
hf = hf[mask]
r = r[mask]
da = da[mask]

lr = np.log(r)

eff = da / d
