import pandas as pd
from pathlib import Path

file = Path(__file__).parent / "NACA-RM-A50K27.xlsx"

CL = pd.read_excel(file, sheet_name="CL")
CD = pd.read_excel(file, sheet_name="CD")
Cm = pd.read_excel(file, sheet_name="CM")

from scipy import interpolate

CL_to_alpha = interpolate.interp1d(CL["CL"], CL["alpha"], fill_value="extrapolate")

Cm["alpha"] = CL_to_alpha(Cm["CL"])
