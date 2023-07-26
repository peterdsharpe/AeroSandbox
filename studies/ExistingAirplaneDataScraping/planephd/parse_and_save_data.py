from scrape_data import airplanes_data
import polars as pl
import re
import numpy as np

# Initialize an empty list to store the rows of the DataFrame
rows = []

# For each airplane in the airplanes data
for airplane, data in airplanes_data.items():
    # Initialize an empty dictionary to store the row
    row = {}

    # For each key-value pair in the data
    for k, v in data.items():
        k = re.sub(r'[:,()=\n]', '', k)
        k = re.sub(r'\s+', '_', k).lower()

        if "total_variable_cost" in k:
            k = "total_variable_cost"
        elif "fuel_cost_per_hour" in k:
            k = "fuel_cost_per_hour"
        elif k == "fuel_burn":
            k = "fuel_burn_@_75%"

        if k in ["manufacturer", "model"]:
            unit = None
        elif "$" in v:
            unit = "USD".strip().lower()
        else:  # converts "15 lb" to "lb"
            unit = re.sub(r'[^a-zA-Z]+', '', v).strip().lower()

        if unit is not None:
            k += f"_{unit}"

        v = re.sub(r'[^0-9.]+', '', v).strip()
        if k not in ["manufacturer", "model"]:
            v = float(v)

        if k in row:
            print(f"Warning: {k} already in row!")
        row[k] = v

    # Add the airplane name to the row
    parts = airplane.strip().split(" ")
    row["start_year"] = parts[0]
    row["end_year"] = parts[2]
    row["name"] = " ".join(parts[3:]).title()

    # Add the row to the list of rows
    rows.append(row)

# Convert the list of rows to a DataFrame
df = pl.DataFrame(rows)
df = df.rename({
    "best_range_i_nm": "best_range_nm",
})

dfp = df.to_pandas()

# Print the DataFrame
print(df)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots()

    mask = (
            (dfp["best_cruise_speed_kias"] >= 140) &
            (dfp["best_range_nm"] >= 650) &
            (dfp["maximum_payload_lbs"] >= 2600)
    )

    for i, row in dfp[mask].iterrows():
        x = row["best_cruise_speed_kias"]
        y = row["fuel_burn_@_75%_gph"]


        ### If graph has a dot here already, continue and skip
        def is_already_plotted():
            for line in ax.lines:
                if np.allclose(
                        [line.get_xdata()[0], line.get_ydata()[0]],
                        [x, y],
                        atol=0.1
                ):
                    return True
            return False


        if len(ax.lines) > 0:
            if is_already_plotted():
                continue

        plt.plot(
            [x], [y], ".k", alpha=0.4, markeredgewidth=0
        )
        ax.annotate(
            f"{row['name'].title()}",
            xy=(x, y),
            xytext=(0, -10),
            textcoords="offset points",
            ha='center',
            va='center',
            fontsize=7,
            alpha=0.4
        )
    plt.xlim(right=300)
    plt.ylim(bottom=0, top=150)
    # plt.plot(
    #     dfp["best_cruise_speed_kias"][mask],
    #     dfp["fuel_burn_@_75%_gph"][mask],
    #     ".k"
    # )
    p.show_plot()
