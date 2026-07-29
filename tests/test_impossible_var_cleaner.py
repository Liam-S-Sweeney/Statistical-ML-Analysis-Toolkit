import pandas as pd

from pipelines.data_organizers.impossible_var_cleaner import clean_impossible_var

# Declared locally rather than imported from config: the cleaner now resolves
# this list at runtime, so the test must supply its own fixture instead of
# depending on a global that changes per dataset.
IMPOSSIBLE_ZERO_VARS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def test_only_remove_impossible_zeros():
    test_df = pd.DataFrame({key: [0, 1, 2, 3] for key in IMPOSSIBLE_ZERO_VARS})
    result_df = clean_impossible_var(
        test_df, *IMPOSSIBLE_ZERO_VARS, impossible_zero_vars=IMPOSSIBLE_ZERO_VARS
    )
    zeros_remaining = (result_df[IMPOSSIBLE_ZERO_VARS] == 0).to_numpy().sum()
    assert zeros_remaining == 0, f"{zeros_remaining} impossible zeros survived cleaning"
    # The all-zero row was dropped, leaving exactly the 3 valid rows
    assert len(result_df) == 3, f"expected 3 surviving rows, got {len(result_df)}"


def test_valid_values_are_preserved():
    test_df = pd.DataFrame({key: [1, 2, 3, 4] for key in IMPOSSIBLE_ZERO_VARS})
    result_df = clean_impossible_var(
        test_df, *IMPOSSIBLE_ZERO_VARS, impossible_zero_vars=IMPOSSIBLE_ZERO_VARS
    )
    assert len(result_df) == 4
    assert (result_df[IMPOSSIBLE_ZERO_VARS] == 0).to_numpy().sum() == 0


def test_empty_impossible_list_is_a_noop():
    """With no impossible-zero columns configured, zeros must survive."""
    test_df = pd.DataFrame({"a": [0, 1, 2], "b": [3, 4, 5]})
    result_df = clean_impossible_var(test_df, "a", "b", impossible_zero_vars=[])
    assert len(result_df) == 3
