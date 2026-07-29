# Statistical & ML Analysis Toolkit

A Streamlit app for end-to-end statistical and machine learning analysis of tabular data. Upload a dataset, pick variables and settings in the sidebar, and every analysis renders in the browser with a download button — no scripting, no config editing, no files placed on the server.

Developed by Liam Sweeney.
<p>
  <a href="https://statistical-ml-analysis-toolkit-6ksufkeieztrnrdfvzkvf5.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit">
  </a>
</p>

---

<img width="1876" height="775" alt="blank_display" src="https://github.com/user-attachments/assets/90764caa-7038-47df-8f16-4968a379beea" />

---

## Features

### Statistical Analyses
- **Uni/Multivariate Exploration** — Descriptive statistics and distributional summaries across one or more variables
- **Repeated Measures ANOVA & ICC** — Within-subjects ANOVA via `pingouin`, with intraclass correlation assessments and six post-hoc correction methods (uncorrected, Bonferroni, Šidák, Holm, FDR-BH, FDR-BY). Subjects missing any selected wave are dropped listwise.
- **Chi-Square Test** — Association testing between a categorical dependent variable and one or more independent variables, with Cramér's V effect sizes and magnitude labels

### ML Analyses
- **Gaussian Mixture Model (GMM)** — Unsupervised clustering with automated model selection; standardizes and PCA-reduces the data (≥ 95% cumulative explained variance), fits all four covariance types (`full`, `tied`, `diag`, `spherical`) across the configured K range, and selects the best configuration via BIC (1-SD rule); outputs NMI, ARI, cluster proportions, and diagnostic alignment against the configured outcome column
- **Linear Discriminant Analysis (LDA)** — Supervised classification against the configured diagnostic column, with cross-validated accuracy, a summary table, and a confusion matrix heatmap
- **Ordinary Least Squares Regression (OLSR)** — Linear regression with residual diagnostics
- **OLS Moderated Regression (MRA) + Simple Slopes** — Mean-centered interaction modeling (`Y ~ X + W + X·W`) with Breusch-Pagan heteroscedasticity check and simple slope probing at ±1 SD of the moderator
- **Binary Logistic Regression (BLR)** — Logistic regression with auto-dichotomization of count DVs (any occurrence vs. none), odds ratios with 95% CIs, and assumption warnings (class imbalance, events-per-predictor, multicollinearity)
- **BLR Moderated Regression (MRA) + Simple Slopes** — Interaction modeling on a binary outcome using the same simple slopes framework as OLS MRA, with VIF checks and confusion-matrix accuracy
- **Ordinal Logistic Regression (OLR)** — Logit-link ordered categorical outcome modeling via `statsmodels` `OrderedModel`
- **Negative Binomial Regression (NBR)** — For overdispersed count outcomes

### Data Visualizations
- **Descriptive** — Distribution plots per selected variable
- **Heatmap** — Correlation heatmap across selected variables
- **PairGrid** — Pairwise relationship grid with KDE overlays
- **PairPlot** — Seaborn pairplot with KDE on diagonals

### Full-Data Generators
- **Master Descriptive CSV Generator** — One summary CSV covering all variables in the dataset
- **All Variable Descriptives Generator** — Per-variable descriptive CSVs for the full dataset

### Data Ingestion
- **Multi-format upload** — CSV, Excel (`.xlsx`/`.xls`), SPSS (`.sav`), and JSON are parsed directly in the browser; parsing is cached on file content so reruns are instant
- **In-app merging** — Upload several files at once and supply an ID column; they are outer-joined on that column, with each file's columns prefixed by its filename to preserve wave/source provenance (files lacking the ID column are skipped with a warning)

---

## Images of Toolkit

<p align="center">
<img width="1467" height="438" alt="blank_dataset_display" src="https://github.com/user-attachments/assets/86c4500e-8b40-457a-975d-bc6382c841d9" />
</p>

<p align="center">
<img width="1518" height="769" alt="full_dataset_display" src="https://github.com/user-attachments/assets/de475de3-4bf4-4df5-8757-c217e87f2e0c" />
</p>

<p align="center">
<img width="206" height="886" alt="sidebar_display" src="https://github.com/user-attachments/assets/38985650-3fed-4f6d-be21-d4ca1f7628a0" />
</p>

---

## Project Structure

```
Statistical-ML-Analysis-Toolkit/
├── main_gui.py                         # Streamlit app entry point
├── config.py                           # Packaged fallback defaults (not dataset-specific)
├── conftest.py                         # Pytest path configuration
├── requirements.txt
├── pyproject.toml
├── app_styles/
│   ├── load_css.py
│   └── main.css
├── app_utils/
│   └── results.py                      # Runs a pipeline and renders what it produced
├── docs/
│   └── st_display.png                  # Interface screenshot
├── files/
│   ├── master_csvs/                    # Local-dev CSV location (diabetes.csv sample included)
│   ├── unmerged_csvs/
│   └── non_csvs/
├── outputs/                            # Written at runtime; ephemeral on hosted deploys
│   ├── all_var_desc_analysis_output/
│   ├── gmm_analysis_output/
│   ├── lda_analysis_output/
│   ├── master_var_desc_output/
│   ├── multi_var_analysis_output/
│   ├── regression_analysis_output/
│   └── figure_pngs_output/
│       ├── bic_aic_vis/
│       ├── cev_pca/
│       ├── desc_vis/
│       ├── gmm_hm_vis/
│       ├── gmm_pp_vis/
│       ├── lda_hm_vis/
│       ├── hm_vis/
│       ├── pg_vis/
│       └── pp_vis/
├── pipelines/
│   ├── data_organizers/
│   │   ├── session_data.py             # Active dataset + runtime settings store
│   │   ├── csv_loader.py               # Resolves session data → path → disk; recodes missing values
│   │   ├── csv_merger.py               # Disk-based CSV merge (local use; not wired to the UI)
│   │   ├── file_pathways.py            # Central path registry (do not modify)
│   │   ├── impossible_var_cleaner.py   # Zero-as-missing handling
│   │   └── type_converter.py           # Disk-based .sav/.json/.xlsx/.xml → .csv (local use)
│   ├── ml/
│   │   ├── blr.py                      # Binary logistic regression
│   │   ├── gmm_analysis.py             # GMM clustering pipeline
│   │   ├── lda.py                      # Linear discriminant analysis
│   │   ├── mblr.py                     # BLR moderated regression + simple slopes
│   │   ├── mols.py                     # OLS moderated regression + simple slopes
│   │   ├── nbr.py                      # Negative binomial regression
│   │   ├── olr.py                      # Ordinal logistic regression
│   │   ├── ols.py                      # OLS regression
│   │   └── rm_anova_icc.py             # RM-ANOVA & ICC
│   ├── statistics/
│   │   ├── all_single_var_desc_gen.py
│   │   ├── chi_sqr.py
│   │   ├── master_descriptive_gen.py
│   │   ├── multivar_desc_gen.py
│   │   └── png_generators/
│   │       ├── desc_gen.py
│   │       ├── hm_gen.py
│   │       ├── pg_gen.py
│   │       └── pp_gen.py
│   └── utility/
│       ├── cdfs.py
│       ├── dichotomize_count_var.py
│       ├── log_ssa.py                  # Simple slopes (logistic)
│       ├── ols_ssa.py                  # Simple slopes (OLS)
│       ├── overlap_checker.py          # Variable-slot overlap validation
│       └── probe_vals.py
└── tests/
    ├── test_dichotomize_count_var.py
    ├── test_impossible_var_cleaner.py
    ├── test_log_ssa.py
    └── test_ols_ssa.py
```

---
## Getting Started Remotely
- Click the link below to access the toolkit via Streamlit.app
<p>
  <a href="https://statistical-ml-analysis-toolkit-6ksufkeieztrnrdfvzkvf5.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit">
  </a>
</p>

## Getting Started Locally

### 1. Requirements

- Python ≥ 3.8
- pip

### 2. Clone and install

```bash
git clone https://github.com/Liam-S-Sweeney/Statistical-ML-Analysis-Toolkit.git
cd Statistical-ML-Analysis-Toolkit
pip install -r requirements.txt
```

(Optional, for a clean environment:)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run main_gui.py
```

The app opens in your browser (default: `http://localhost:8501`). Runtime folders are created automatically on startup, and log records are written to the console.

### 4. Upload your data

Open the **Dataset** expander at the top of the page and upload one or more files (CSV, Excel, SPSS `.sav`, JSON).

- **One file** — used as-is.
- **Several files** — enter an ID column in the merge field; the files are outer-joined on it and each file's columns are prefixed with its filename.

Once a dataset is active, the app shows its shape and the first 20 rows, and the analysis buttons unlock. Uploading a new dataset replaces the active one; variable selections that are no longer valid columns reset automatically.

There is no need to place files on disk or edit `config.py`. For local development, a CSV in `files/master_csvs/` matching `MAIN_CSV_NAME` in `config.py` is used as a fallback when nothing has been uploaded; the bundled `diabetes.csv` is there as a sample dataset.

### 5. Configure the analysis

All runtime settings live in the sidebar and are stored per session — nothing is written back to `config.py`.

| Sidebar group | Setting | Used by |
|---|---|---|
| Data cleaning | Subject ID column | Chi-square, RM-ANOVA, all regressions |
| Data cleaning | Missing-value codes | All analyses (recoded to `NaN` at load) |
| Data cleaning | Variables where 0 is impossible | All analyses (zeros treated as missing) |
| GMM / LDA | Diagnostic / outcome column | GMM, LDA |
| GMM / LDA | K range | GMM |
| GMM / LDA | `n_init`, random seed | GMM |
| GMM / LDA | Figure DPI | GMM, LDA figures |

`config.py` holds the packaged fallback values for these settings plus a few that have no sidebar control yet: `DATA_THRESHOLD` (count-DV dichotomization), `REG_COVAR` (GMM covariance regularization), `CV` (LDA cross-validation folds), and the plotting options `HUE_COL`, `SIZE_COL`, and `PALETTE`.

---

## Variable Selection

The sidebar exposes four variable slots used by different analyses:

| Slot | Role |
|---|---|
| **General** | Exploration, RM-ANOVA, GMM, LDA, and visualizations |
| **Dependent / Endogenous** | Outcome variable for regression models and chi-square |
| **Independent / Exogenous** | Predictor variables for regression models and chi-square |
| **Moderator** | Interaction term for MRA pipelines |

The app validates slot usage per analysis (e.g., regressions enforce exactly one dependent variable; MRA pipelines enforce exactly one focal predictor and one moderator) and flags overlapping selections across slots before running.

---

## Results

Every analysis runs through a wrapper that snapshots the output tree, calls the pipeline, and renders whatever appeared:

| Artifact | Rendered as |
|---|---|
| CSV | Interactive table + **Download CSV** |
| PNG | Inline image + **Download image** |
| `.txt` / `.log` | Code block + **Download text** |
| Returned DataFrame | Interactive table + **Download summary CSV** |
| Anything else | Download button |

Failures are caught and shown as an error message with an expandable traceback, so a bad variable selection never takes the app down.

Artifacts are also written to disk under `outputs/`, organized by analysis:

| Analysis | Output Location |
|---|---|
| GMM | `gmm_analysis_output/` + `figure_pngs_output/` (BIC/AIC, CEV-PCA, heatmap, pairplot) |
| LDA | `lda_analysis_output/` + `figure_pngs_output/lda_hm_vis/` |
| Regression (BLR, OLR, OLSR, NBR) and MRA + Simple Slopes | `regression_analysis_output/` |
| RM-ANOVA & ICC | `multi_var_analysis_output/` (one CSV per correction method) |
| Multivariate Exploration, Chi-Square | `multi_var_analysis_output/` |
| Master Descriptives | `master_var_desc_output/` |
| All Variable Descriptives | `all_var_desc_analysis_output/` |
| Visualizations | `figure_pngs_output/<type>/` |

On a hosted deployment that filesystem is ephemeral — download anything you want to keep before the session ends.

---

## Testing

Unit tests cover the dichotomization utility, impossible-value cleaning, and both simple slopes implementations (OLS and logistic). The session store falls back to a plain dict outside a Streamlit runtime, so the suite runs without launching the app.

```bash
pip install -e .[dev]     # pytest + ruff
pytest
ruff check .
```

---

## Dependencies

```
pandas
numpy
requests
matplotlib
seaborn
scipy
statsmodels
scikit-learn
imbalanced-learn
pingouin
pyreadstat
streamlit
```

Install all via `pip install -r requirements.txt`.

---

## Notes

- The active dataset and all runtime settings live in Streamlit session state (`pipelines/data_organizers/session_data.py`). Every pipeline reaches its data through `csv_loader.load_clean()`, which resolves the session dataset first and falls back to disk, so pipelines work identically whether data was uploaded or placed in `files/`.
- Missing value codes are recoded to `NaN` at load time; variables flagged as impossible-zero have their zeros treated as missing prior to analysis.
- The GMM pipeline standardizes features, applies PCA retaining ≥ 95% cumulative explained variance, then selects the best model via the BIC 1-SD rule across all four covariance structures and the configured K range.
- MRA pipelines mean-center the focal predictor and moderator before computing the interaction term to reduce multicollinearity, and probe simple slopes at ±1 SD of the moderator.
- BLR auto-dichotomizes count DVs (any occurrence vs. none, threshold configurable via `DATA_THRESHOLD`) and validates that the outcome is binary before fitting; warnings are emitted for class imbalance and insufficient events-per-predictor.
- `csv_merger.py` and `type_converter.py` operate on the `files/` directories and are no longer surfaced in the UI — merging and format conversion now happen at upload time. They remain available for local batch use.
- `pipelines/data_organizers/file_pathways.py` is the central path registry — modifying it can break most features.
