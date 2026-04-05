import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pipelines.data_organizers.csv_loader import load_clean
from config import PALETTE
from pipelines.data_organizers.file_pathways import DESC_VIS

def desc_visualization(*cols, df=load_clean(), palette = PALETTE):
    for col in cols:
        # non_na df of cols
        plot_vals = df[col]
        plot_vals = plot_vals[np.isfinite(plot_vals)]
        
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

        output_dir = DESC_VIS
        plt.savefig(output_dir / f"{col}-Desc_Visualizations.png", dpi=300, bbox_inches='tight')

        plt.show()