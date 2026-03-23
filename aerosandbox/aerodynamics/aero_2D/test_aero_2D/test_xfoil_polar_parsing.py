import pytest
import numpy as np
from unittest.mock import patch, mock_open, MagicMock
import aerosandbox as asb


# 9-column polar as written by XFoil 6.97 with cinc active:
# Cpmin is added to each data row but is absent from the header line.
_POLAR_XFOIL_697_WITH_CINC = """\

       XFOIL         Version 6.97

 Calculated polar for: naca2412

 1 1 Reynolds number fixed          Mach number fixed

 xtrf =   1.000 (top)        1.000 (bottom)
 Mach =   0.000     Re =     0.300 e 6     Ncrit =   9.000

   alpha    CL        CD       CDp       CM     Chinge   Top_Xtr  Bot_Xtr
  ------ -------- --------- --------- -------- --------- -------- --------
   2.000   0.4869   0.00839   0.00312  -0.0623  -0.7969   0.00456   0.6726   0.9999
   0.000   0.1714   0.00774   0.00321  -0.0506  -0.5082   0.00231   0.8262   0.8693
  -2.000  -0.0554   0.01002   0.00398  -0.0593  -1.0355   0.00360   0.9274   0.3749
"""


def _make_xfoil():
    af = asb.Airfoil("naca2412").repanel(n_points_per_side=100)
    return asb.XFoil(
        airfoil=af,
        Re=3e5,
        xfoil_command="xfoil",
    )


def _run_with_mock_polar(xf, polar_text):
    mock_proc = MagicMock()
    mock_proc.communicate.return_value = ("", "")
    mock_proc.poll.return_value = 0

    with patch("subprocess.Popen", return_value=mock_proc), \
         patch("os.remove"), \
         patch("builtins.open", mock_open(read_data=polar_text)):
        return xf._run_xfoil(run_command="")


def test_xfoil_697_cinc_column_mismatch():
    """
    XFoil 6.97 with cinc active writes Cpmin to data rows but not to the
    header line, producing 9 data columns against 8 header columns.
    Previously this raised XFoilError due to a column count mismatch.
    The parser should remap columns correctly and return valid results.
    """
    result = _run_with_mock_polar(_make_xfoil(), _POLAR_XFOIL_697_WITH_CINC)

    assert len(result["alpha"]) == 3
    assert "Cpmin" in result
    assert len(result["Cpmin"]) == 3


def test_xfoil_polar_sort_with_unpopulated_fields():
    """
    Regression test: the sort step in XFoil.alpha() should not raise
    IndexError when some output fields were not populated by XFoil.

    With XFoil 6.97 and cinc active, Xcpmin is absent from both the header
    and data rows. The output dict is initialised with all expected keys,
    leaving Xcpmin as an empty array after parsing. Previously the sort step
    raised:

        IndexError: index 2 is out of bounds for axis 0 with size 0

    The fix drops unpopulated fields before sorting.
    """
    xf = _make_xfoil()
    mock_proc = MagicMock()
    mock_proc.communicate.return_value = ("", "")
    mock_proc.poll.return_value = 0

    with patch("subprocess.Popen", return_value=mock_proc), \
         patch("os.remove"), \
         patch("builtins.open", mock_open(read_data=_POLAR_XFOIL_697_WITH_CINC)):
        result = xf.alpha([-2.0, 0.0, 2.0], start_at=None)
    
    # alpha() sorts results ascending
    assert list(result["alpha"]) == pytest.approx([-2.0, 0.0, 2.0])
    assert list(result["CL"]) == pytest.approx([-0.0554, 0.1714, 0.4869])
    assert "Xcpmin" not in result or len(result["Xcpmin"]) == 0

if __name__ == "__main__":
    # pass
    pytest.main()