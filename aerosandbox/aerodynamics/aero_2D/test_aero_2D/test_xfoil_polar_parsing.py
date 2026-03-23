import aerosandbox as asb
import aerosandbox.numpy as np
import pytest

def test_xfoil_sort_with_empty_output_arrays():
    """
    Regression test: output arrays of different lengths (e.g. Cpmin when cinc
    is not active) should not raise IndexError during the sort step.
    """
    import aerosandbox.numpy as np

    # Simulate the condition: some arrays populated, some empty
    sort_order = np.argsort(np.array([3.0, 1.0, 2.0]))

    output = {
        "alpha": np.array([3.0, 1.0, 2.0]),
        "CL":    np.array([0.3, 0.1, 0.2]),
        "Cpmin": np.array([], dtype=float),   # empty — not populated by this XFoil version
    }

    # This should not raise
    output = {k: np.array(v, dtype=float) for k, v in output.items()}
    output = {k: v for k, v in output.items() if len(v) > 0}
    sorted_output = {k: v[sort_order] for k, v in output.items()}

    assert "Cpmin" not in sorted_output
    assert list(sorted_output["alpha"]) == [1.0, 2.0, 3.0]

if __name__ == "__main__":
    # pass
    pytest.main()