from pathlib import Path
import json, sys, os


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


def run_python_file(path: Path) -> None:
    """
    Executes a Python file from a path.
    Args:
        path: File path

    Returns: None

    """
    sys.path.append(str(path.parent))
    __import__(path.with_suffix("").name)


def run_all_python_files(path: Path, recursive=True, verbose=True) -> None:
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
            if verbose:
                print(f"##### Running file: {path}")
            run_python_file(path)

        ### Run the file if it's a Jupyter notebook
        if path.suffix == ".ipynb":
            if verbose:
                print(f"##### Converting file: {path}")
            notebook = path
            python_file = path.with_suffix(".py")

            convert_ipynb_to_py(notebook, python_file)

            run_all_python_files(python_file, recursive=False, verbose=verbose)

    elif path.is_dir() and recursive:
        if verbose:
            print(f"##### Opening directory: {path}")
        for subpath in path.iterdir():
            run_all_python_files(subpath, recursive=recursive, verbose=verbose)
