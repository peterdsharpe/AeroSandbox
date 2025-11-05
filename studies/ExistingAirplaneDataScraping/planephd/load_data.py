import pandas as pd

df = pd.read_csv("airplanes_data.csv", index_col=0)
df.set_index("name", inplace=True)
