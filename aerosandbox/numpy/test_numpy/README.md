# Math Module Test Guide
Peter Sharpe

## Basic Idea
The file `test_all_operations_run` simply tests that all (or almost all) possible mathematical operations compute without errors for all possible combinations of input types; they have no regard for correctness, only that no errors are raised.

The file `test_array` tests that array-like objects can be created out of individual scalars.

All other individual files test the *correctness* of specific calculations against known values computed with NumPy as a reference. 