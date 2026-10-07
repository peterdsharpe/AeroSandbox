import warnings
from pathlib import Path

import pytest

import aerosandbox.library

LIBRARY_DIR = Path(aerosandbox.library.__file__).parent


@pytest.mark.parametrize(
    "module_filename",
    [
        "propulsion_electric.py",
        "propulsion_propeller.py",
        "mass_structural.py",
    ],
)
def test_no_invalid_escape_sequences(module_filename: str):
    """
    Regression test: Windows paths in docstrings (e.g. 'C:\\Projects\\...')
    used to live in non-raw strings, emitting 'invalid escape sequence'
    SyntaxWarnings on Python 3.12+ (slated to become SyntaxErrors).
    """
    source_path = LIBRARY_DIR / module_filename
    source = source_path.read_text()

    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        compile(source, str(source_path), "exec")


if __name__ == "__main__":
    pytest.main([__file__])
