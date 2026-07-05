from pathlib import Path
import json


def convert_ipynb_to_py(
    input_file: Path,
    output_file: Path,
) -> None:
    """
    Read an input Jupyter notebook (.ipynb) and convert it to a Python file (.py).

    Tried using `jupyter nbconvert`, but that is SO SLOW, like 3 seconds per notebook! It's
    just json parsing, this should *not* take more than a few milliseconds - come on, Jupyter!

    Parameters
    ----------
    input_file : Path
        File path.
    output_file : Path
        File path.

    Returns
    -------
    None
    """
    with open(input_file, "r", encoding="utf-8") as f:
        ipynb_contents = json.load(f)
    with open(output_file, "w+", encoding="utf-8") as f:
        for cell in ipynb_contents["cells"]:
            if cell["cell_type"] == "code":
                f.writelines(cell["source"])
                f.write("\n")
