import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from config import ID_VAR
from pipelines.data_organizers.impossible_var_cleaner import endo_exo_clean_impossible_var
from pipelines.data_organizers.file_pathways import MULTI_VAR_ANALYSIS_OUTPUT_FOLDER

def cramers_v_label(v):
    if np.isnan(v):   
        return np.nan
    elif v < 0.10:      
        return 'negligible'
    elif v < 0.20:      
        return 'small'
    elif v < 0.40:      
        return 'medium'
    else:
        return 'large'

def run_chi_sqr(endo, exo, id_var=ID_VAR):
    df = endo_exo_clean_impossible_var(endo, *exo, id_var)

    out_dir = MULTI_VAR_ANALYSIS_OUTPUT_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    endo_str = endo[0] if isinstance(endo, list) else endo

    # Chi Square Results
    for exo_var in exo:
        contingency_table = pd.crosstab(df[endo_str], df[exo_var])
        chi2, p, dof, expected_freq = chi2_contingency(contingency_table)
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (contingency_table.values.sum() * min_dim)) if min_dim > 0 else np.nan
        cv_label = cramers_v_label(cramers_v)
        n_cells = expected_freq.size
        n_below_5 = (expected_freq < 5).sum().sum()
        pct_below_5 = n_below_5 / n_cells

        # Chi Output Summary
        summary = pd.DataFrame([{
            'chi2': chi2,
            'p': p,
            'dof': dof,
            'cramers_v': cramers_v,
            'cv_label': cv_label,
            'pct_cells_below_5': round(pct_below_5, 4),
            'assumption_met': pct_below_5 <= 0.20,
        }])
        
        expected_df = pd.DataFrame(
        expected_freq,
        index=contingency_table.index,
        columns=contingency_table.columns,
        )

        cols_title = f'{endo_str}-{exo_var}'
        summary.to_csv(out_dir / f"{cols_title}_chi2.csv", index=False)
        expected_df.to_csv(out_dir / f"{cols_title}_chi2_expected.csv", index=True)
    return None

