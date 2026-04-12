# Statistical & ML Analysis Toolkit 🧠

A Python-based GUI toolkit for statistical and machine learning analysis, built on Streamlit. Designed for clinical and longitudinal research data, with a focus on messy, real-world datasets.

Developed by [Liam Sweeney](https://github.com/Liam-S-Sweeney)

---

## Features

### Statistical Analyses
- **Uni/Multivariate Exploration** — descriptive statistics across one or more selected variables
- **Repeated Measures ANOVA & ICC** — within-subject analysis across waves via `pingouin`, paired as a single analysis unit

### ML Analyses
- **Gaussian Mixture Model (GMM)** — unsupervised clustering with automatic model selection via BIC/AIC across covariance types and K values; includes PCA dimensionality reduction, LDA cross-validation, NMI/ARI alignment scoring, and cluster probability output
- **Ordinal Logistic Regression (OLR)** — for ordinal dependent variables, fit via maximum likelihood with logit link
- **Ordinary Least Squares (OLS) Regression** — for continuous dependent variables with standard diagnostics

### Data Visualizations
- **Descriptive Visualization** — distribution plots per variable
- **Heatmap Visualization** — correlation heatmap across selected variables
- **PairGrid Visualization** — pairwise grid plots
- **PairPlot Visualization** — pairwise scatter + KDE plots

### Full-Data Generators
- **Master Descriptive CSV Generator** — descriptive statistics for every variable in the dataset
- **All Variable Descriptives Generator** — individual descriptive outputs per variable
- **File Type Converter** — converts non-CSV files (e.g. `.sav`) to CSV
- **CSV Merger** — merges all CSVs in the `unmerged_csvs/` folder into a single master CSV

---

## Project Structure

```
Statistical-ML-Analysis-Toolkit/
├── main_gui.py                        # Streamlit GUI entry point
├── config.py                          # Dataset configuration
├── requirements.txt
├── app_styles/
│   ├── load_css.py
│   └── main.css
├── files/
│   ├── master_csvs/                   # Place your dataset here
│   ├── unmerged_csvs/                 # CSVs to be merged
│   └── non_csvs/                      # Non-CSV files for conversion
├── outputs/
│   ├── all_var_desc_analysis_output/
│   ├── figure_pngs_output/            # All generated visualizations
│   ├── gmm_analysis_output/
│   ├── master_var_desc_output/
│   ├── multi_var_analysis_output/     # RM-ANOVA & ICC outputs
│   └── regression_analysis_output/   # OLR & OLS outputs
└── pipelines/
    ├── data_organizers/
    │   ├── csv_loader.py
    │   ├── csv_merger.py
    │   ├── file_pathways.py
    │   ├── impossible_var_cleaner.py
    │   └── type_converter.py
    ├── ml/
    │   ├── gmm_analysis.py
    │   ├── olr.py
    │   ├── ols.py
    │   └── rm_anova_icc.py
    ├── statistics/
    │   ├── master_descriptive_gen.py
    │   ├── all_single_var_desc_gen.py
    │   ├── multivar_desc_gen.py
    │   └── png_generators/
    │       ├── desc_gen.py
    │       ├── hm_gen.py
    │       ├── pg_gen.py
    │       └── pp_gen.py
    └── utility/
        └── cdfs.py
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/Liam-S-Sweeney/Statistical-ML-Analysis-Toolkit.git
cd Statistical-ML-Analysis-Toolkit
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your dataset

Edit `config.py`:

```python
MAIN_CSV_NAME = 'your_dataset.csv'   # filename of your CSV
ID_VAR = 'subject_id'                # subject/participant ID column
MISSING_CODES = [-99, -999, -9999]   # values to treat as missing
IMPOSSIBLE_ZERO_VARS = []            # columns where 0 is physiologically impossible
DX = 'diagnosis_column'              # diagnostic label column for GMM
```

Place your dataset in `files/master_csvs/`.

### 5. Install the package in editable mode
```bash
pip install -e .
```
This registers the project root on Python's path so all internal imports resolve correctly.

### 6. Launch the app
```bash
streamlit run main_gui.py
```

---

## Usage

**Variable Selection (Sidebar)**
- **General Variables** — used for descriptive stats, GMM, RM-ANOVA, and visualizations
- **Dependent Variables** — endogenous variable for OLR and OLS (select one)
- **Independent Variables** — exogenous predictors for OLR and OLS

**Running Analyses**

Select variables in the sidebar, then click the corresponding button in the main panel. All outputs are saved automatically to the relevant subfolder in `outputs/`.

---

## Data Notes

- Missing values are handled automatically using the codes defined in `config.py`
- Variables listed in `IMPOSSIBLE_ZERO_VARS` have zero values replaced with NaN before analysis
- The CSV Merger supports longitudinal data with wave-prefixed column naming (e.g. `W1_varname`, `W2_varname`)
- RM-ANOVA and ICC require variables representing the same measure across multiple timepoints/waves

---

## Dependencies

- `pandas`, `numpy`, `scipy`
- `statsmodels` — OLR, OLS
- `scikit-learn` — GMM, PCA, LDA, StandardScaler
- `pingouin` — RM-ANOVA, ICC
- `matplotlib`, `seaborn` — visualizations
- `pyreadstat` — `.sav` file conversion
- `streamlit` — GUI

---

## Roadmap

- [ ] Supervised ML — Logistic Regression, Random Forest
- [ ] Multilevel Modeling (MLM) with wide-to-long reshape pipeline
- [ ] PyTorch integration
- [ ] SOLID/class-based refactor into a proper package structure
