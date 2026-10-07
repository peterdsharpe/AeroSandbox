import subprocess
import sys
from pathlib import Path

import pytest

import aerosandbox.tools.python.importing as importing_module


def test_lazy_import():
    from aerosandbox.tools.python.importing import lazy_import

    json_module = lazy_import("json")
    assert json_module.dumps({"a": 1}) == '{"a": 1}'


def test_lazy_import_standalone():
    """
    `importlib.util` is a submodule that a bare `import importlib` does not
    load; the module must import it explicitly rather than relying on some
    other package having loaded it transitively. Verified by executing the
    module source in a fresh interpreter (where nothing else has imported
    importlib.util) and calling lazy_import().
    """
    module_path = Path(importing_module.__file__)

    test_code = (
        "source = open(r'''{path}''').read()\n"
        "namespace = {{}}\n"
        "exec(source, namespace)\n"
        "json_module = namespace['lazy_import']('json')\n"
        "assert json_module.dumps({{'a': 1}}) == '{{\"a\": 1}}'\n"
    ).format(path=module_path)

    result = subprocess.run(
        [sys.executable, "-c", test_code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_convert_ipynb_to_py_non_ascii(tmp_path):
    """
    .ipynb files are UTF-8 by specification; conversion should handle
    non-ASCII content regardless of the platform's default encoding.
    """
    import json

    from aerosandbox.tools.python.io import convert_ipynb_to_py

    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["alpha_deg = 5  # Angle of attack α [°]\n"],
            },
            {
                "cell_type": "markdown",
                "source": ["# Some text\n"],
            },
        ]
    }

    input_file = tmp_path / "notebook.ipynb"
    output_file = tmp_path / "notebook.py"
    input_file.write_text(json.dumps(notebook), encoding="utf-8")

    convert_ipynb_to_py(input_file, output_file)

    output = output_file.read_text(encoding="utf-8")
    assert "α [°]" in output
    assert "Some text" not in output  # Markdown cells are skipped


if __name__ == "__main__":
    pytest.main([__file__])
