import pytest

from aerosandbox.tools.string_formatting import eng_string


def test_eng_string_si_suffixes():
    assert eng_string(1230.0) == "1.23k"
    assert eng_string(-1230000.0) == "-1.23M"
    assert eng_string(1230.0, unit="N") == "1.23 kN"
    assert eng_string(-1230000.0, unit="N") == "-1.23 MN"


def test_eng_string_space_before_unit_outside_si_range():
    """
    For exponents outside the SI-prefix range (|exp| > 24), the unit should
    still be separated by a space by default, consistent with the SI branch.
    """
    assert eng_string(1e30, unit="N") == "1e30 N"
    assert eng_string(1e-30, unit="N") == "1e-30 N"

    # Without a unit, no trailing space should be added
    assert eng_string(1e30) == "1e30"

    # Explicit overrides
    assert eng_string(1e30, unit="N", add_space_after_number=False) == "1e30N"


def test_eng_string_zero_and_nan():
    assert eng_string(0) == "0"
    assert eng_string(float("nan")) == "NaN"


if __name__ == "__main__":
    pytest.main([__file__])
