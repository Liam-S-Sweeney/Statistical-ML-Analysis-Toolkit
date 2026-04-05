import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.regression.mixed_linear_model import MixedLM
from pipelines.data_organizers.csv_loader import load_clean
from config import ID_VAR
from pipelines.data_organizers.impossible_var_cleaner import clean_impossible_var

def olr_mlm(*cols,id_var=ID_VAR):
    clean_df = load_clean()
    df = clean_impossible_var(clean_df, id_var, *cols)

    df_long = pd.melt(
        df,
        id_vars=id_var,
        value_vars=cols,
        var_name='wave',
        value_name='score'
    )

    if df_long.empty:
        print("No subjects with complete data across all selected waves.")
        return
    
    print(f'Subjects with complete data: {df_long[id_var].nunique()}')

    # Linear Regression
    # ordinal_anova_result = OrderedModel(
    #     endog= pass,
    #     exog=pass,
    # )

    # ICC
    icc_result = pg.intraclass_corr(
        data=df_long,
        targets=id_var,
        raters='wave',
        ratings='score'
    )

    out_dir = CORRELATIONAL_OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)
    cols_title = '-'.join(cols)

    rm_anova_result.to_csv(out_dir / f"{cols_title}-rm-anova.csv",index=False)
    icc_result.to_csv(out_dir / f"{cols_title}-icc.csv",index=False)
    return 

