
import matplotlib.pyplot as plt
import seaborn as sns
from pipelines.data_organizers.csv_loader import load_clean
from config import HUE_COL, SIZE_COL, PALETTE
from pipelines.data_organizers.impossible_var_cleaner import clean_impossible_var
from pipelines.data_organizers.file_pathways import HM_VIS

def heatmap_visualizations(*cols, hue_col=HUE_COL, size_col=SIZE_COL, palette=PALETTE):

    clean_df = load_clean() 

    # build df from selected cols 
    df = clean_impossible_var(clean_df, *cols)
    
    if len(cols) < 2:
        raise ValueError("Select at least two variables.")

    # add hue + size columns if they exist
    if hue_col in clean_df.columns:
        df[hue_col] = clean_df[hue_col]
    else:
        hue_col = None  

    if size_col in clean_df.columns:
        df[size_col] = clean_df[size_col]
    else:
        size_col = None

    # attempt to add a palette
    try:
        sns.color_palette(palette, as_cmap=True)
    except:
        pass
    
# ---- Heatmap (correlation matrix) ----
    corr = df.corr(numeric_only=True)

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        square=True
    )
    output_dir = HM_VIS
    cols_title = '-'.join(cols)
    plt.savefig(output_dir / f"{cols_title}-HM.png", dpi=300, bbox_inches='tight')

    plt.show()