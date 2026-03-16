# CSV Analyzer GUI

A desktop data analysis tool built with Python and Tkinter for exploratory analysis, visualization, and unsupervised machine learning on CSV datasets. Designed for clinical and behavioral research data.

---

## Features

- **Descriptive Statistics** — Generates per-variable and master summary CSVs covering central tendency, variability, distribution shape, and frequency
- **Multivariate Exploration** — Per-variable distribution visualizations including probability plots, histograms with normal fit, boxplots, violin plots, and swarm plots
- **Multivariate Visualization** — PairGrid plots with scatter, KDE, and histogram layers; hue and size encoding via configurable columns
- **Correlational Analysis** — Pearson correlation heatmap and regression pairplot across selected variables
- **GMM Analysis** — Full Gaussian Mixture Model pipeline including data cleaning, PCA dimensionality reduction, BIC/AIC model selection across covariance types, cluster evaluation, and CSV export

---

## Project Structure

```
CSV-Analyzer-GUI/
│
├── main_gui.py                      # Tkinter GUI — entry point
├── config.py                        # All configurable parameters
├── data_loader.py                   # CSV loading and cleaning
├── cdfs.py                          # Descriptive statistics engine
├── global_descriptive_generator.py  # Master and per-variable CSV generators
├── multivariate_exploration.py      # Exploration, visualization, correlation
├── gmm_analysis.py                  # GMM pipeline
│
├── data_files/                      # Place your dataset here
│   └── your_dataset.csv
├── single_var_descriptives/         # Auto-generated per-variable CSVs
├── multivariate_analysis/           # Auto-generated multivariate output CSVs
└── requirements.txt
```

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Liam-S-Sweeney/CSV-Analyzer-GUI.git
cd CSV-Analyzer-GUI
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv

# Windows (PowerShell)
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your Dataset

Place your CSV file inside the `data_files/` folder and update `DATA_PATH` in `config.py`:

```python
DATA_PATH = DATA_DIR / "your_dataset.csv"
```

### 5. Run the Application

```bash
python main_gui.py
```

---

## Configuration (`config.py`)

All key parameters are set in `config.py` — update this file when switching datasets.

| Parameter | Description |
|---|---|
| `DATA_PATH` | Path to your CSV file inside `data_files/` |
| `DX` | Column name for your diagnosis / outcome variable |
| `IMPOSSIBLE_ZERO_VARS` | Columns where a value of 0 is physiologically impossible (e.g. Glucose, BMI) — zeros are replaced with NaN before GMM analysis |
| `MISSING_CODES` | Numeric codes used to represent missing data (e.g. `-99`, `-999`) — replaced with NaN on load |
| `HUE_COL` / `SIZE_COL` | Columns used for color and size encoding in PairGrid visualizations |
| `PALETTE` | Seaborn palette name for visualizations |

---

## How to Use

### Selecting Variables

Use the spinbox at the top of the GUI to set the number of variables, then use the searchable dropdowns to select columns from your dataset. Dropdowns filter as you type.

### Available Analyses

**Multivariate Exploration**
Requires 1+ variables. Generates a descriptive statistics CSV and produces a 5-panel visualization for each selected variable: probability plot, histogram with normal fit, boxplot, violin plot, and swarm plot.

**Multivariate Visualization**
Requires 2+ variables. Produces a PairGrid with histograms on the diagonal, scatter plots on the lower triangle, and KDE on the upper triangle. Encodes a grouping variable via hue and size if configured.

**Multivariate Correlation**
Requires 2+ variables. Produces a Pearson correlation heatmap and a regression pairplot across selected variables.

**GMM Analysis**
Requires 2+ variables. Runs the full pipeline described below and exports results to `multivariate_analysis/`.

**Master Descriptive CSV Generator**
No variable selection needed. Computes descriptive statistics for every column in the dataset and saves to `data_files/master_descriptives.csv`.

**All Single Var Descriptive CSV Generator**
No variable selection needed. Saves one descriptive CSV per variable to `single_var_descriptives/`. Warns before overwriting existing files.

---

## GMM Pipeline

The GMM analysis runs automatically in sequence when triggered from the GUI.

### 1. Data Cleaning
Columns listed in `IMPOSSIBLE_ZERO_VARS` are checked for zero values. Any column with zeros is cleaned by replacing them with NaN. Rows with remaining NaN values are dropped. The resulting sample size is printed.

### 2. Standardization
All selected features are z-scored (mean=0, std=1) using `StandardScaler`. This ensures no single feature dominates covariance estimation due to scale differences.

### 3. LDA Sanity Check
A supervised Linear Discriminant Analysis is cross-validated (5-fold) against the outcome column (`DX`). This establishes whether the selected features can linearly separate diagnostic groups at all — setting an upper bound on what GMM can recover unsupervised.

### 4. PCA
Full PCA is fit to determine the number of components needed to explain ≥95% of variance (`optimal_n`). A reduced PCA is then fit with `optimal_n` components. All downstream GMM fitting and prediction uses the PCA-transformed data.

### 5. Model Selection (BIC/AIC)
GMMs are fit for K = 1 to 10 across all four covariance types (`full`, `tied`, `diag`, `spherical`) — 40 models total. BIC and AIC are recorded for each. The model with the lowest BIC is selected. All models within one standard deviation of the BIC minimum are reported as statistically acceptable alternatives.

### 6. Cluster Assignment
The best model assigns each subject a cluster label (hard) and per-cluster probability (soft). Probabilities are written back to the main dataframe.

### 7. Evaluation
- **Crosstab** — Row-normalized cross-tabulation of diagnosis × cluster, displayed as a heatmap
- **NMI** — Normalized Mutual Information between cluster assignments and diagnosis labels
- **ARI** — Adjusted Rand Index between cluster assignments and diagnosis labels (chance-corrected)

### 8. Output
Results are saved to `multivariate_analysis/ml_<variables>-gmm_analysis.csv` containing:

| Field | Description |
|---|---|
| LDA CV accuracy | Mean ± std from 5-fold LDA cross-validation |
| Optimal Components | Number of PCA components retained |
| Acceptable Combinations | All (covariance type, K) pairs within 1 std of minimum BIC |
| Best Covariance / Best K | Selected model configuration |
| Cluster Counts / Proportions | Size of each cluster |
| NMI / ARI | Cluster-diagnosis alignment metrics |
| DX Length / Non-null / Unique | Outcome variable diagnostics |

---

## Interpreting GMM Results

**ARI and NMI near 0** — Clusters do not align with diagnostic labels. This may mean the classes are interleaved in feature space (not a failure of the code — GMM finds density modes, not label boundaries). Confirm with LDA accuracy: if LDA is high but ARI is near 0, the features are linearly separable supervised but do not form distinct density clusters.

**ARI > 0.2** — Moderate alignment. Clusters partially correspond to diagnosis, suggesting the diagnostic groups occupy somewhat distinct regions of feature space.

**Flat BIC curve / many acceptable models** — No strong cluster signal at any K. The data does not contain clearly separated subpopulations for the selected features.

**K=1 selected** — All data forms a single density mode. GMM is not the right tool for this dataset and feature set.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scipy
statsmodels
scikit-learn
imbalanced-learn
requests
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Notes

- The tool is currently configured for the [Pima Indians Diabetes Dataset](https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv) by default. To use a different dataset, update `config.py`.
- `IMPOSSIBLE_ZERO_VARS` is dataset-specific. Review and update it for any new dataset — zeros in columns like `BloodPressure` or `BMI` are missing data coded as zero in some clinical datasets, not true measurements.
- Output directories (`data_files/`, `single_var_descriptives/`, `multivariate_analysis/`) are created automatically on first run if they do not exist.
