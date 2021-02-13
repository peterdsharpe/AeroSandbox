"""
Ignore this file; this is just here so that all tutorials are automatically run by PyTest on each build (in order
to ensure that they don't throw any errors).
"""

import os
from pathlib import Path


def run_all_python_files(path: Path, recursive=True) -> None:
    """
    Executes all Python files in a directory.
    Args:
        path: A Path-type object (Path from built-in pathlib) representing a filepath
        recursive: Executes recursively (e.g. searches all subfolders too)

    Returns: None

    """
    # Exclusions:
    if path == Path(os.path.abspath(__file__)):  # Don't run this file
        return
    if "Tutorials in Progress - Ignore for Now" in str(path):  # Don't run this folder
        return

    ### Run the file if it's a Python file
    if path.is_file():
        if path.suffix == ".py":
            exec(open(str(path)).read())

    ### Recurse through a directory if directed to
    if recursive and path.is_dir():
        for subpath in path.iterdir():
            run_all_python_files(subpath)


def test_all_tutorials():
    tutorial_dir_path = Path(os.path.abspath(__file__)).parent
    run_all_python_files(tutorial_dir_path)


if __name__ == '__main__':
    # test_all_tutorials()

    import pytest
    pytest.main()