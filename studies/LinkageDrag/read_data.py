from aerosandbox.tools.webplotdigitizer_reader import read_webplotdigitizer_csv

raw_data = read_webplotdigitizer_csv("./wpd_datasets.csv")
linkage_types = [
    '55mm, covered, bottom',
    '55mm, bottom',
    '85mm, bottom',
    '55mm, covered, top',
    '55mm, top',
    '85mm, top'
]

import pandas as pd

df = pd.DataFrame(
    data={
        float(k.replace('Re', '')): v[:, 1] * 0.01
        for k, v in raw_data.items()
    },
    index=linkage_types
)

df = df.transpose().sort_index().transpose()

##### Dataframe description:
'''
Example:

df =
                       100000.0  200000.0
55mm, covered, bottom  0.001011  0.000613
55mm, bottom           0.001448  0.000875
85mm, bottom           0.001168  0.000812
55mm, covered, top     0.001137  0.000709
55mm, top              0.001629  0.000757
85mm, top              0.001622  0.000883

Columns are Reynolds numbers w.r.t. airfoil chord (200 mm in this experiment).
Rows are linkage type, size, and placement.
Values are drag area [m^2].

Flap was 23% of wing chord. Pushrods were standard 2mm diameter.

Wing lift coefficient:
    For Re=1e5 case, CL=0.75
    For Re=2e5 case, CL=0.30 

(Constant wing loading)

'''