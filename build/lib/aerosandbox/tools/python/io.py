from pathlib import Path
import json


def convert_ipynb_to_py(
        input_file: Path,
        output_file: Path,
) -> None:
    """
    Reads an input Jupyter notebook (.ipynb) and converts it to a Python file (.py)

    Tried using `jupyter nbconvert`, but that is SO SLOW, like 3 seconds per notebook! It's just json parsing,
    this should *not* take more than a few milliseconds - come on, Jupyter!

    Args:
        input_file: File path
        output_file: File path

    Returns: None

    """
    with open(input_file, "r") as f:
        ipynb_contents = json.load(f)
    with open(output_file, "w+") as f:
        for cell in ipynb_contents['cells']:
            if cell['cell_type'] == "code":
                f.writelines(cell['source'])
                f.write("\n")
