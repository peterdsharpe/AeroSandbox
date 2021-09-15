"""
################################################################################
README:

Ignore this file; this is just here so that all tutorials are automatically run by PyTest on each build (in order
to ensure that they don't throw any errors).
"""

import os
import sys
from pathlib import Path
import tempfile
import json


def convert_ipynb_to_py(
        input_file: Path,
        output_file: Path,
):
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


def run_python_file(path: Path):
    """
    Executes a Python file from a path.
    Args:
        path: File path

    Returns: None

    """
    sys.path.append(str(path.parent))
    __import__(os.path.splitext(path.name)[0])


def run_all_python_files(path: Path, recursive=True) -> None:
    """
    Executes all Python files and Jupyter Notebooks in a directory.
    Args:
        path: A Path-type object (Path from built-in pathlib) representing a filepath
        recursive: Executes recursively (e.g. searches all subfolders too)

    Returns: None

    """
    # Exclusions:
    if path == Path(os.path.abspath(__file__)):  # Don't run this file
        return
    if "ignore" in str(path).lower():  # Don't run any file or folder with the word "ignore" in the name.
        return

    if path.is_file():

        ### Run the file if it's a Python file
        if path.suffix == ".py":
            run_python_file(path)

        ### Run the file if it's a Jupyter notebook
        if path.suffix == ".ipynb":
            with tempfile.TemporaryDirectory() as tempdir:
                tempdir = Path(tempdir)
                notebook = path
                output_dir = tempdir
                output_filename = path.name.strip(path.suffix) + ".py"
                python_file = output_dir / output_filename

                # convert_command = f'jupyter nbconvert --to python "{notebook}" --output-dir "{output_dir}" --output "{output_filename}"'
                # os.system(convert_command)

                convert_ipynb_to_py(notebook, python_file)

                run_python_file(python_file)

    ### Recurse through a directory if directed to
    if recursive and path.is_dir():
        for subpath in path.iterdir():
            run_all_python_files(subpath)


def test_all_tutorials():
    tutorial_dir_path = Path(os.path.abspath(__file__)).parent
    run_all_python_files(tutorial_dir_path)


if __name__ == '__main__':
    test_all_tutorials()

    # import pytest
    # pytest.main()
