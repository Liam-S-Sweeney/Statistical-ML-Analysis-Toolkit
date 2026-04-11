import pandas as pd
from scipy.stats import chi2_contingency
from config import ID_VAR
from pipelines.data_organizers.impossible_var_cleaner import endo_exo_clean_impossible_var
from pipelines.data_organizers.file_pathways import MULTI_VAR_ANALYSIS_OUTPUT_FOLDER

def run_chi_sqr(endo, exo, id_var=ID_VAR):
    df = endo_exo_clean_impossible_var(endo, *exo, id_var)

    out_dir = MULTI_VAR_ANALYSIS_OUTPUT_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    endo_str = endo[0] if isinstance(endo, list) else endo

    # Chi Square Results
    for exo_var in exo:
        contgency_table = pd.crosstab(df[endo_str], df[exo_var])
        chi2, p, dof, expected_freq = chi2_contingency(contgency_table)

        # Chi Output Summary
        summary = pd.DataFrame([{
            'chi2': chi2,
            'p': p,
            'dof': dof,
        }])
        expected_df = pd.DataFrame(
            expected_freq,
            index=contgency_table.index,
            columns=contgency_table.columns,
        )

        cols_title = f'{endo_str}-{exo_var}'
        summary.to_csv(out_dir / f"{cols_title}_chi2.csv", index=False)
        expected_df.to_csv(out_dir / f"{cols_title}_chi2_expected.csv", index=True)
    return None

