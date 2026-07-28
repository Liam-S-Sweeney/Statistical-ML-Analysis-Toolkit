import pandas as pd

from config import IMPOSSIBLE_ZERO_VARS
from pipelines.data_organizers.impossible_var_cleaner import clean_impossible_var


def test_only_remove_impossible_zeros():
    # Known
    test_df = pd.DataFrame({key: [0, 1, 2, 3] for key in IMPOSSIBLE_ZERO_VARS})
    # Validation
    result_df = clean_impossible_var(test_df, *IMPOSSIBLE_ZERO_VARS)
    zeros_remaining = (result_df[IMPOSSIBLE_ZERO_VARS] == 0).to_numpy().sum()
    assert zeros_remaining == 0, f"{zeros_remaining} impossible zeros survived cleaning"
    # The all-zero row was dropped, leaving exactly the 3 valid rows
    assert len(result_df) == 3, f"expected 3 surviving rows, got {len(result_df)}"

def test_valid_values_are_preserved():
    # A non-zero value must never be removed
    test_df = pd.DataFrame({key: [1, 2, 3, 4] for key in IMPOSSIBLE_ZERO_VARS})
    result_df = clean_impossible_var(test_df, *IMPOSSIBLE_ZERO_VARS)
    assert len(result_df) == 4
    assert (result_df[IMPOSSIBLE_ZERO_VARS] == 0).to_numpy().sum() == 0
