import pandas as pd

from pipelines.utility.dichotomize_count_var import dichotomize_count_var


def test_dichotomize_splits_on_threshold():
    # Known
    counts = pd.Series([0, 1, 5, 0, 3])
    # Validation of threshold dichotomization
    result = dichotomize_count_var(counts, threshold=0)
    # Assert test
    expected = pd.Series([0, 1, 1, 0, 1])
    pd.testing.assert_series_equal(result, expected, check_dtype=False)

def test_threshold_is_exclusive():
    # values EQUAL to the threshold should become 0, not 1 (function uses >)
    counts = pd.Series([2, 3, 4])
    result = dichotomize_count_var(counts, threshold=3)
    expected = pd.Series([0, 0, 1])  # only 4 > 3
    pd.testing.assert_series_equal(result, expected, check_dtype=False)

def test_negative_values_raise():
    counts = pd.Series([0, -1, 2])
    try:
        dichotomize_count_var(counts)
        assert False, "expected ValueError for negative counts"
    except ValueError:
        pass  # Error checks work
