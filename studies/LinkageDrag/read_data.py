from aerosandbox.tools.webplotdigitizer_reader import read_webplotdigitizer_csv

raw_data = read_webplotdigitizer_csv("./wpd_datasets.csv")
linkage_types = [
    "55mm, covered, bottom",
    "55mm, bottom",
    "85mm, bottom",
    "55mm, covered, top",
    "55mm, top",
    "85mm, top",
]

import pandas as pd

raw_df = pd.DataFrame(
    data={f"CD @ {k}": v[:, 1] * 0.01 for k, v in raw_data.items()}, index=linkage_types
)

raw_df = raw_df.transpose().sort_index().transpose()

data = {
    k: []
    for k in [
        "CDA",
        "Re_l",
        "linkage_length",
        "is_covered",
        "is_top",
    ]
}

for col in raw_df.columns:
    Re_cref = float(str(col).split("Re")[-1])

    for name, CDA in zip(raw_df.index, raw_df[col]):
        linkage_length = float(name.split("mm")[0]) * 1e-3

        Re_l = Re_cref * linkage_length / 0.200
        # Note: in original raw data, Re is referenced to airfoil chord, not linkage length (200 mm in this
        # experiment).

        for k, v in dict(
            CDA=CDA,
            Re_l=Re_l,
            linkage_length=linkage_length,
            is_covered="covered" in name,
            is_top="top" in name,
        ).items():
            data[k].append(v)

df = pd.DataFrame(data)

##### Dataframe description:
"""
Example:

df =
         CDA     Re_l  linkage_length  is_covered  is_top
0   0.001011  27500.0           0.055        True   False
1   0.001448  27500.0           0.055       False   False
2   0.001168  42500.0           0.085       False   False
3   0.001137  27500.0           0.055        True    True
4   0.001629  27500.0           0.055       False    True
5   0.001622  42500.0           0.085       False    True
6   0.000613  55000.0           0.055        True   False
7   0.000875  55000.0           0.055       False   False
8   0.000812  85000.0           0.085       False   False
9   0.000709  55000.0           0.055        True    True
10  0.000757  55000.0           0.055       False    True
11  0.000883  85000.0           0.085       False    True

Columns are:
    
    * Measured drag area (CDA), [m^2].
    
    * Reynolds numbers w.r.t. linkage length
    
    * Linkage length [m]
    
    * Whether or not the linkage has an aerodynamic fairing.
    
    * Whether or not the measurement was for a top-side linkage (True) or a bottom-side one (False). Differences in 
    local boundary layer and inviscid effects cause local velocity changes. 

Flap was 23% of wing chord. Pushrods were standard 2mm diameter.

Wing lift coefficient:
    For Re_cref=1e5 cases, CL=0.75
    For Re_cref=2e5 cases, CL=0.30 

(Constant wing loading)

"""
