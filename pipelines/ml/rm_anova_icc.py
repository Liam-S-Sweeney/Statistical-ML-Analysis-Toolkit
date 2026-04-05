import pingouin as pg
import pandas as pd
from pipelines.data_organizers.csv_loader import load_clean
from config import ID_VAR
from pipelines.data_organizers.impossible_var_cleaner import clean_impossible_var
from pipelines.data_organizers.file_pathways import MULTI_VAR_ANALYSIS_OUTPUT_FOLDER

def rm_anova_icc(*cols,id_var=ID_VAR):
    clean_df = load_clean()
    df = clean_impossible_var(clean_df, id_var, *cols)

    df_long = pd.melt(
        df,
        id_vars=id_var,
        value_vars=cols,
        var_name='wave',
        value_name='score'
    )

    complete_subjects = (
        df_long.groupby(id_var)['score']
        .apply(lambda x: x.notna().all())
    )
    complete_ids = complete_subjects[complete_subjects].index
    df_long = df_long[df_long[id_var].isin(complete_ids)]

    if df_long.empty:
        print("No subjects with complete data across all selected waves.")
        return
    
    print(f'Subjects with complete data: {df_long[id_var].nunique()}')

    # RM Anova
    rm_anova_result = pg.rm_anova(
        data=df_long,
        dv='score',
        within='wave',
        subject=id_var,
        detailed=True
    )

    # ICC
    icc_result = pg.intraclass_corr(
        data=df_long,
        targets=id_var,
        raters='wave',
        ratings='score'
    )

    out_dir = MULTI_VAR_ANALYSIS_OUTPUT_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    cols_title = '-'.join(cols)

    rm_anova_result.to_csv(out_dir / f"{cols_title}-rm-anova.csv",index=False)
    icc_result.to_csv(out_dir / f"{cols_title}-icc.csv",index=False)
    return 