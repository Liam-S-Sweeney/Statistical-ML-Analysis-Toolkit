# Statistical ML Analysis Toolkit

A Streamlit-based GUI for data analysis and machine learning on quantitative CSV datasets. Built in Python - designed as a free, open-source alternative to SPSS for research purposes.

> **Includes a demo dataset:** Pima Indians Diabetes Dataset (`files/master_csvs/diabetes.csv`)

---

![App Screenshot](docs/st_display.png)

---

## What It Does

Select variables from the sidebar, then run any analysis with one click. All outputs save automatically to `outputs/` as CSVs and 300 DPI PNGs.

### Statistical Analyses
| Analysis | What You Get |
|---|---|
| **Uni/Multivariate Exploration** | Per-variable descriptive CSV: mean, median, mode, variance, std, skew, kurtosis, SE, IQR, range, frequency counts |
| **RM-ANOVA & ICC** | Repeated measures ANOVA and intraclass correlation coefficient via `pingouin`; two CSVs saved per run |

### ML Analyses
| Analysis | What You Get |
|---|---|
| **GMM Analysis** | Full unsupervised ML pipeline — see below |
| **OLR & Multilevel Modeling** | Ordinal logistic regression + mixed-effects model *(in progress)* |

### Data Visualizations
| Visualization | What You Get |
|---|---|
| **Descriptive Visualization** | 5-panel per-variable plot: probability plot, histogram with normal fit, boxplot, violin plot, swarm plot |
| **Heatmap Visualization** | Pearson correlation heatmap |
| **PairGrid Visualization** | PairGrid with histograms, scatter plots, and KDE; optional grouping variable encoding |
| **PairPlot Visualization** | Seaborn pairplot with optional hue/size grouping |

### Full-Data Generators
| Tool | What It Does |
|---|---|
| **Master Descriptive CSV** | Single CSV covering every variable in the dataset |
| **All Variable Descriptives** | Per-variable descriptive CSV for every column |
| **File Type Converter** | Converts files in `files/non_csvs/` to CSV format |
| **CSV Merger** | Outer-joins all CSVs in `files/unmerged_csvs/` into a single longitudinal dataset |

---

## GMM Pipeline

1. **Load & clean** — Apply missing-code replacement; replace physiologically impossible zeros with NaN for configurable columns (`IMPOSSIBLE_ZERO_VARS`)
2. **Standardize** — Z-score all features via `StandardScaler`
3. **LDA check** — 5-fold cross-validated LDA establishes a supervised upper bound before unsupervised clustering
4. **PCA** — Reduce to components explaining ≥ 95% cumulative variance; fit all GMMs in PCA space
5. **Model selection** — Fit 40 models (K = 1–10 × 4 covariance types: full, tied, diag, spherical); select best by BIC within 1-std acceptable range
6. **Evaluate** — NMI and ARI against a diagnosis label column (`DX`); row-normalized crosstab heatmap
7. **Export** — CSV with LDA accuracy, optimal PCA components, best covariance type and K, cluster counts and proportions, NMI, and ARI

---

## Quickstart

```bash
git clone https://github.com/Liam-S-Sweeney/ML-GUI.git
cd ML-GUI

python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\Activate.ps1       # Windows PowerShell

pip install -r requirements.txt
streamlit run main_gui.py
```

The app launches in your browser. Variables are selected from the sidebar on the left. The tool ships pre-configured for the bundled [Pima Indians Diabetes Dataset](https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv).

---

## Working with Longitudinal Data

For multi-wave datasets, place one CSV per wave in `files/unmerged_csvs/` named by wave label (e.g. `W1.csv`, `W2.csv`, `W3.csv`). Each file must contain a column matching `ID_VAR` in `config.py`.

Then click **CSV Merger** in the app, or run it directly:

```python
from pipelines.data_organizers.csv_merger import merge_csv
merge_csv()
```

This performs an outer join across all wave files, prefixing each column with its wave label (`W1_variable`, `W2_variable`, etc.), and saves the result to `files/master_csvs/`. Update `MAIN_CSV_NAME` in `config.py` to point to the output file.

---

## Configuration

All parameters live in `config.py`. Update when switching datasets.

| Parameter | Description |
|---|---|
| `MAIN_CSV_NAME` | Filename of your CSV inside `files/master_csvs/` |
| `ID_VAR` | Subject identifier column — used by RM-ANOVA, ICC, and the CSV merger |
| `DX` | Diagnosis / outcome column — used by GMM for cluster alignment evaluation |
| `IMPOSSIBLE_ZERO_VARS` | Columns where zero is physiologically impossible; zeros replaced with NaN before analysis |
| `MISSING_CODES` | Numeric missing-data codes (e.g. `-99`, `-999`, `-9999`) — replaced with NaN on load |
| `HUE_COL` / `SIZE_COL` / `PALETTE` | Grouping columns for PairGrid and PairPlot encoding — set to a non-existent column name to disable |

---

## Project Structure

```
Statistical-ML-Analysis-Toolkit/
├── main_gui.py                              # Entry point — Streamlit app
├── config.py                                # All configurable parameters
├── requirements.txt
├── pyproject.toml
│
├── files/
│   ├── master_csvs/                         # Place your merged/ready dataset here
│   ├── unmerged_csvs/                       # Place per-wave CSVs here for merging
│   └── non_csvs/                            # Place SPSS or other source files here for conversion
│
├── outputs/                                 # All auto-generated outputs land here
│   ├── all_var_desc_analysis_output/        # Per-variable descriptive CSVs
│   ├── master_var_desc_output/              # Master descriptive CSV
│   ├── multi_var_analysis_output/           # Multivariate exploration + RM-ANOVA/ICC CSVs
│   ├── gmm_analysis_output/                 # GMM model selection results and cluster CSVs
│   ├── regression_analysis_output/          # OLR/MLM outputs (in progress)
│   └── figure_pngs_output/                  # 300 DPI visualization PNGs
│       ├── bic_aic_vis/                     # BIC/AIC model selection plots
│       ├── cev_pca/                         # Cumulative explained variance plots
│       ├── desc_vis/                        # Descriptive visualizations
│       ├── gmm_hm_vis/                      # GMM cluster heatmaps
│       ├── gmm_pp_vis/                      # GMM pairplots
│       ├── hm_vis/                          # Correlation heatmaps
│       ├── pg_vis/                          # PairGrid plots
│       └── pp_vis/                          # PairPlots
│
└── pipelines/
    ├── data_organizers/
    │   ├── csv_loader.py                    # Load CSV and replace missing codes with NaN
    │   ├── csv_merger.py                    # Outer-join wave CSVs for longitudinal data
    │   ├── file_pathways.py                 # Centralized path registry (do not edit)
    │   ├── impossible_var_cleaner.py        # Zero-imputation and NaN cleaning for analysis
    │   └── type_converter.py               # Convert non-CSV files to CSV
    ├── ml/
    │   ├── gmm_analysis.py                  # Full GMM pipeline
    │   ├── rm_anova_icc.py                  # RM-ANOVA + ICC via pingouin
    │   └── olr_mlm.py                       # OLR & MLM (in progress)
    ├── statistics/
    │   ├── master_descriptive_gen.py        # Full-dataset descriptive CSV generator
    │   ├── all_single_var_desc_gen.py       # Per-variable descriptive CSV generator
    │   ├── multivar_desc_gen.py             # Selected-variable exploration
    │   └── png_generators/
    │       ├── desc_gen.py                  # Descriptive visualizations
    │       ├── hm_gen.py                    # Correlation heatmap
    │       ├── pg_gen.py                    # PairGrid
    │       └── pp_gen.py                    # PairPlot
    └── utility/
        └── cdfs.py                          # Core descriptive statistics engine
```

---

## Interpreting GMM Output

| Result | Interpretation |
|---|---|
| **ARI / NMI near 0** | Clusters don't align with diagnostic labels — GMM finds density modes, not label boundaries. Cross-reference LDA accuracy. |
| **ARI > 0.2** | Moderate cluster–diagnosis alignment; diagnostic groups occupy partially distinct regions of feature space |
| **Flat BIC curve** | No strong cluster signal — data may not contain separable subpopulations for the selected features |
| **K = 1 selected** | Single density mode; consider a different feature set |
| **High LDA accuracy, low ARI** | Features are linearly separable under supervision but don't form natural density clusters |

---

## Requirements

```
pandas numpy matplotlib seaborn scipy statsmodels scikit-learn
imbalanced-learn pingouin pyreadstat streamlit requests
```
