import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pathlib import Path
from data_loader import load_clean
from config import HUE_COL, SIZE_COL, PALETTE, MULTIVARIATE_ANALYSIS_PATH, OUTPUT_PNGS_PATH
from cdfs import compute_descriptives_for_series
from impossible_var_cleaner import clean_impossible_var

def explore_multi_variables(*cols, palette=PALETTE):
    clean_df = load_clean()

    # build df from selected cols 
    df = clean_impossible_var(clean_df, *cols)

    descriptive_dict = [
        compute_descriptives_for_series(df[col], name=col, position=df.columns.get_loc(col))
        for col in cols
        ]
    
    out_dir = MULTIVARIATE_ANALYSIS_PATH
    cols_title = '-'.join(cols)
    pd.DataFrame(descriptive_dict).to_csv(out_dir / f"{cols_title}-multivar_analysis.csv",index=False)

    for col in cols:
        # non_na df of cols
        plot_vals = df[col]
        
        # attempt to add a palette
        try:
            sns.color_palette(palette, as_cmap=True)
        except:
            pass

        # Shape of Distribution per Variable
        fig, (ax_probplot, ax_hist, ax_box, ax_violin, ax_swarm) = plt.subplots(1,5, figsize=(16, 6))

        # Probability Plot
        stats.probplot(
            plot_vals,
            dist='norm',
            plot=ax_probplot
            )
        
        points, line = ax_probplot.get_lines()

        # Style the Points
        points.set_color("steelblue")
        points.set_marker("o")
        points.set_markersize(5)
        points.set_alpha(0.7)

        # Style the fit line
        line.set_color("red")
        line.set_linewidth(2)
        line.set_linestyle("-")

        ax_probplot.set_title(f'Probability Plot')

        # Histogram
        plot_vals = plot_vals[np.isfinite(plot_vals)]
        mu, sigma = stats.norm.fit(plot_vals)
        counts, bins, _ = ax_hist.hist(
            plot_vals,
            bins='fd',
            edgecolor='black',
            alpha=0.6,
            )
        x_dense = np.linspace(plot_vals.min(), plot_vals.max(), 500)
        pdf_dense = stats.norm.pdf(x_dense, mu, sigma)
        pdf_dense_scaled = pdf_dense * len(plot_vals) * (bins[1] - bins[0])

        ax_hist.plot(x_dense, pdf_dense_scaled, 'r-', lw=2,
                    label=f'Normal fit ($\\mu={mu:.2f}$, $\\sigma={sigma:.2f}$)')
        ax_hist.set_title('Histogram')
        ax_hist.legend(fontsize=7,)

        # Box Plot
        sns.boxplot(x=plot_vals, ax=ax_box).set(xlabel=None)
        ax_box.set_title("Box Plot")

        # Violin Plot
        sns.violinplot(x=plot_vals, ax=ax_violin).set(xlabel=None)
        ax_violin.set_title('Violin Plot')

        # Swarm Plot
        sns.swarmplot(x=plot_vals, ax=ax_swarm).set(xlabel=None)
        ax_swarm.set_title('Swarm Plot')

        plt.subplots_adjust(hspace=0.25)
        fig.suptitle(f"{col} Visualizations")

        output_dir = OUTPUT_PNGS_PATH
        plt.savefig(output_dir / f"{col}-Visualizations.png", dpi=300, bbox_inches='tight')

        plt.show()
    
############

def multivariate_visualizations(*cols, hue_col=HUE_COL, size_col=SIZE_COL, palette=PALETTE):

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

    g = sns.PairGrid(data=df, diag_sharey=False, hue=hue_col)

    g.map_diag(sns.histplot)
    g.map_lower(sns.scatterplot, size=df[size_col] if size_col else None)
    g.map_upper(sns.kdeplot)

    if hue_col:
        g.add_legend(title=hue_col)

    output_dir = OUTPUT_PNGS_PATH
    cols_title = '-'.join(cols)
    plt.savefig(output_dir / f"{cols_title}-Multivar_Visualization.png", dpi=300, bbox_inches='tight')

    plt.show()

#########

def correlational_analysis(*cols, hue_col=HUE_COL, palette=PALETTE):

    clean_df = load_clean() 

    # clean selected columns
    cols = [c for c in cols if c]

    if len(cols) < 2:
        raise ValueError("Select at least two variables for correlation analysis.")

    # build df from selected cols that actually exist
    missing = [c for c in cols if c not in clean_df.columns]
    if missing:
        raise KeyError(f"Missing columns in dataset: {missing}")

    clean_df_copy = clean_df[cols].copy()

    # build df from selected cols 
    df = clean_impossible_var(clean_df_copy, *cols)

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

    # ---- Pairplot ----
    hue_to_use = None
    if hue_col and hue_col in clean_df.columns:
        df[hue_col] = clean_df[hue_col]
        hue_to_use = hue_col

    sns.pairplot(
        data=df,
        hue=hue_to_use,     
        kind="reg"
    )

    output_dir = OUTPUT_PNGS_PATH
    cols_title = '-'.join(cols)
    plt.savefig(output_dir / f"{cols_title}-Correlational_Visualization.png", dpi=300, bbox_inches='tight')

    plt.show()
     
