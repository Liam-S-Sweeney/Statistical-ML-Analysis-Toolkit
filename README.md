# Statistical & ML Analysis Toolkit

A Streamlit-based GUI for end-to-end statistical and machine learning analysis. Designed for researchers working with tabular CSV data, the toolkit chains data ingestion, cleaning, analysis, and visualization into a single point-and-click interface -- No scripting required at runtime.

Developed by Liam Sweeney.

---

## Features

### Statistical Analyses
- **Uni/Multivariate Exploration** -- Descriptive statistics and distributional summaries across one or more variables, exported to CSV
- **Repeated Measures ANOVA & ICC** -- Within-subjects ANOVA via `pingouin`, with intraclass correlation assessments and six post-hoc correction methods (Bonferroni, Šidák, Holm, FDR-BH, FDR-BY, uncorrected)
- **Chi-Square Test** -- Association testing between a categorical dependent variable and one or more independent variables

### ML Analyses
- **Gaussian Mixture Model (GMM)** -- Unsupervised clustering with automated model selection; standardizes and PCA-reduces the data (≥ 95% cumulative explained variance), fits all four covariance types (`full`, `tied`, `diag`, `spherical`) across a configurable K range, and selects the best configuration via BIC (1-SD rule); outputs NMI, ARI, cluster proportions, and diagnostic alignment against a configured outcome column
- **Linear Discriminant Analysis (LDA)** -- Supervised classification against the configured diagnostic column, with cross-validated accuracy, a summary table, and a confusion matrix heatmap
- **Ordinary Least Squares Regression (OLSR)** -- Linear regression with residual diagnostics
- **OLS Moderated Regression (MRA) + Simple Slopes** -- Mean-centered interaction modeling (`Y ~ X + W + X·W`) with Breusch-Pagan heteroscedasticity check and simple slope probing at ±1 SD of the moderator
- **Binary Logistic Regression (BLR)** -- Logistic regression with auto-dichotomization of count DVs (any occurrence vs. none), odds ratios with 95% CIs, and assumption warnings (class imbalance, events-per-predictor, multicollinearity)
- **BLR Moderated Regression (MRA) + Simple Slopes** -- Interaction modeling on a binary outcome using the same simple slopes framework as OLS MRA, with VIF checks and confusion-matrix accuracy
- **Ordinal Logistic Regression (OLR)** -- Logit-link ordered categorical outcome modeling via `statsmodels` `OrderedModel`
- **Negative Binomial Regression (NBR)** -- For overdispersed count outcomes

### Data Visualizations
- **Descriptive** -- Distribution plots per selected variable
- **Heatmap** -- Correlation heatmap across selected variables
- **PairGrid** -- Pairwise relationship grid with KDE overlays
- **PairPlot** -- Seaborn pairplot with KDE on diagonals

### Full-Data Generators
- **Master Descriptive CSV Generator** -- One summary CSV covering all variables in the dataset
- **All Variable Descriptives Generator** -- Per-variable descriptive CSVs for the full dataset

### Data Preparation Utilities
- **File Type Converter** -- Converts SPSS (`.sav`), JSON, Excel (`.xlsx`/`.xls`), and XML files placed in `files/non_csvs/` into CSVs saved to `files/unmerged_csvs/`
- **CSV Merger** -- Outer-joins all CSVs in `files/unmerged_csvs/` on the configured `ID_VAR`, prefixing each file's columns with its filename (useful for longitudinal wave data), and saves the merged master CSV to `files/master_csvs/`. Files lacking the ID column are skipped with a warning.

---

## Project Structure

```
Statistical-ML-Analysis-Toolkit/
├── main_gui.py                         # Streamlit app entry point
├── config.py                           # Dataset-specific configuration
├── conftest.py                         # Pytest path configuration
├── requirements.txt
├── pyproject.toml
├── app_styles/
│   ├── load_css.py
│   └── main.css
├── files/
│   ├── master_csvs/                    # Place your primary CSV here
│   ├── unmerged_csvs/                  # CSVs to be merged
│   └── non_csvs/                       # Non-CSV files for conversion
├── outputs/
│   ├── all_var_desc_analysis_output/
│   ├── gmm_analysis_output/
│   ├── lda_analysis_output/            # Created at first LDA run
│   ├── master_var_desc_output/
│   ├── multi_var_analysis_output/
│   ├── regression_analysis_output/
│   └── figure_pngs_output/
│       ├── bic_aic_vis/
│       ├── cev_pca/
│       ├── desc_vis/
│       ├── gmm_hm_vis/
│       ├── gmm_pp_vis/
│       ├── lda_hm_vis/                 # Created at first LDA run
│       ├── hm_vis/
│       ├── pg_vis/
│       └── pp_vis/
├── pipelines/
│   ├── data_organizers/
│   │   ├── csv_loader.py               # Load CSV + recode missing values to NaN
│   │   ├── csv_merger.py               # Outer-join CSVs on ID_VAR
│   │   ├── file_pathways.py            # Central path registry (do not modify)
│   │   ├── impossible_var_cleaner.py   # Zero-as-missing handling
│   │   └── type_converter.py           # .sav/.json/.xlsx/.xml → .csv
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

## Getting Started

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

### 3. Configure for your dataset

Edit `config.py` before running:

```python
MAIN_CSV_NAME = 'your_data.csv'        # Filename of your primary CSV (in files/master_csvs/)
ID_VAR = 'SubjectID'                   # Subject/row identifier column (used by merger & regressions)

MISSING_CODES = [-99, -999, -9999]     # Values recoded to NaN at load time
IMPOSSIBLE_ZERO_VARS = [               # Variables where 0 is physiologically impossible
    'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'
]

HUE_COL = None                         # Optional grouping variable for visualizations
SIZE_COL = None
PALETTE = None

# MRA
DATA_THRESHOLD = 1                     # Dichotomization threshold for count DVs

# GMM
DX = 'Outcome'                         # Diagnostic/outcome column for cluster alignment
DPI = 300                              # Figure resolution
K_MIN = 1                              # Minimum cluster count to test
K_MAX = 11                             # Maximum cluster count (exclusive)
N_INIT = 10                            # GMM initializations per fit
RAND_STATE = 0                         # Random seed
REG_COVAR = 1e-3                       # Covariance regularization

# LDA
CV = 5                                 # Cross-validation folds
```

### 4. Load your data

- Place your primary CSV in `files/master_csvs/` and make sure `MAIN_CSV_NAME` matches its filename.
- If your data is split across multiple CSVs (e.g., longitudinal waves), place them in `files/unmerged_csvs/` and use the **CSV Merger** button in the app.
- Non-CSV files (SPSS `.sav`, JSON, Excel, XML) go in `files/non_csvs/` — use the **File Type Converter** button, then merge.

### 5. Run the app

```bash
streamlit run main_gui.py
```

The app opens in your browser (default: `http://localhost:8501`). All required runtime folders are created automatically on startup, and every analysis output is saved to the appropriate subdirectory under `outputs/`. A `toolkit.log` file records run-time logging.

If no CSV is found in `files/master_csvs/`, the analysis buttons are disabled and the app will prompt you to prepare data via the converter/merger utilities.

---

## Variable Selection

The sidebar exposes four variable slots used by different analyses:

| Slot | Role |
|---|---|
| **General Variables** | Used by exploration, RM-ANOVA, GMM, LDA, and visualization analyses |
| **Dependent Variables** | Outcome/endogenous variable for regression models and chi-square |
| **Independent Variables** | Predictor/exogenous variables for regression models and chi-square |
| **Moderator Variable** | Interaction term for MRA pipelines |

The app validates slot usage per analysis (e.g., regressions enforce exactly one dependent variable; MRA pipelines enforce exactly one focal predictor and one moderator) and flags overlapping selections across slots before running.

---

## Output Files

| Analysis | Output Location | Format |
|---|---|---|
| GMM | `gmm_analysis_output/` + `figure_pngs_output/` | CSV + PNGs (BIC/AIC, CEV-PCA, heatmap, pairplot) |
| LDA | `lda_analysis_output/` + `figure_pngs_output/lda_hm_vis/` | Summary table + confusion matrix PNG |
| Regression (BLR, OLR, OLSR, NBR) | `regression_analysis_output/` | `.txt` summary files |
| MRA + Simple Slopes (OLS & BLR) | `regression_analysis_output/` | `.txt` summary files |
| RM-ANOVA & ICC | `multi_var_analysis_output/` | CSV per correction method |
| Multivariate Exploration | `multi_var_analysis_output/` | CSV |
| Chi-Square | `multi_var_analysis_output/` | CSV |
| Master Descriptives | `master_var_desc_output/` | CSV |
| All Variable Descriptives | `all_var_desc_analysis_output/` | CSV |
| Visualizations | `figure_pngs_output/<type>/` | PNG (300 DPI) |

---

## Testing

Unit tests cover the dichotomization utility, impossible-value cleaning, and both simple slopes implementations (OLS and logistic).

```bash
pip install .[dev]     # installs pytest
pytest
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

- Missing value codes defined in `config.py` are recoded to `NaN` at load time; variables listed in `IMPOSSIBLE_ZERO_VARS` have zero values treated as missing prior to analysis.
- The GMM pipeline standardizes features, applies PCA retaining ≥ 95% cumulative explained variance, then selects the best model via the BIC 1-SD rule across all four covariance structures and the configured K range.
- MRA pipelines mean-center the focal predictor and moderator before computing the interaction term to reduce multicollinearity, and probe simple slopes at ±1 SD of the moderator.
- BLR auto-dichotomizes count DVs (any occurrence vs. none, threshold configurable via `DATA_THRESHOLD`) and validates that the outcome is binary before fitting; warnings are emitted for class imbalance and insufficient events-per-predictor.
- The CSV Merger prefixes each merged file's columns with its filename stem, preserving wave/source provenance in longitudinal merges.
- `pipelines/data_organizers/file_pathways.py` is the central path registry — modifying it can break most features.
