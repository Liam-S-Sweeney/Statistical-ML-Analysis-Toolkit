import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from pipelines.data_organizers.impossible_var_cleaner import clean_impossible_var
from config import DX, CV, DPI
from pipelines.data_organizers.csv_loader import load_clean
from pipelines.data_organizers.file_pathways import LDA_HM_VIS, LDA_OUTPUT_FOLDER

def lda_model(
        *cols: list,
        dx_col_:str = DX, 
        cv_:int = CV,
        dpi_:int = DPI,
    ):
    """
    Linear Discriminant Analysis (LDA):\n
    - A Supervised classification analysis method that seperates two or more classes by reducing dimensionality\n\n
    ---\n\n
    Assumptions:\n
    - Gaussian Distribution\n
    - Equal Covariance Matrices\n
    - Linear Separability\n\n
    ---\n\n
    Parameters\n
    ---\n
    * cols*       : list of variables
    * dx_col_     : diagnostic variable 
    * cv_         : cross-validation splitting strategy 
    * dpi:        : Dots Per Inch \n
    ---\n
    Output:\n
    - Summary Table
    - Confusion Matrix Heatmap
    """

    # Initial load of csv
    clean_df = load_clean()

    # Basic config checking
    if not isinstance(dpi_, int) or dpi_ < 1:
        raise ValueError(f'dpi_ must be a positive integer, got {dpi_}\n - Check config')
    if not isinstance(cv_, int) or cv_ < 2:
        raise ValueError(f'cv_ must be an integer >= 2, got {cv_}\n - Check config')
    if dx_col_ not in clean_df.columns:
        raise ValueError(f'dx_col_ ({dx_col_}) not found in DataFrame\n - Check config')

    # build df from selected cols 
    df = clean_impossible_var(clean_df, *cols)

    # Standardize data (z-score)
    X_df = df[list(cols)].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    # Align DX labels
    dx_arr = clean_df.loc[df.index, dx_col_].to_numpy()
    mask = ~np.isnan(dx_arr)
    X_df_masked = X_df.iloc[mask]
    X_masked = X_scaled[mask]
    y_masked = dx_arr[mask]

    # Chance Level
    classes, counts = np.unique(y_masked, return_counts=True)
    chance = counts.max() / counts.sum()

    # CV Accuracy
    lda = LinearDiscriminantAnalysis()
    cv_scores = cross_val_score(lda, X_masked, y_masked, cv=cv_)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

     # Fit masked to LDA 
    lda.fit(X_masked, y_masked)

    # Feature Loading
    loadings = pd.DataFrame(
        lda.coef_,
        columns=cols,
        index=[f'LD{i+1}' for i in range(lda.coef_.shape[0])]
    ).T.sort_values('LD1', ascending=False)                     # T = transpose features to rows and LDs to columns

    # Explained variance ratio 
    if hasattr(lda, 'explained_variance_ratio_'):
        evr = lda.explained_variance_ratio_
    else:
        evr = None

    # Class priors
    print(f"\nClass Priors: { {int(c): round(p, 3) for c, p in zip(lda.classes_, lda.priors_)} }")

    # Confusion Matrix via cross_val_predict
    y_pred = cross_val_predict(lda, X_masked, y_masked, cv=cv_)
    cm = confusion_matrix(y_masked, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f'True_{int(c)}' for c in classes],
        columns=[f'Pred_{int(c)}' for c in classes],
    )

    # Per-class recall
    per_class_recall = dict(zip(
    [f'Recall_class_{int(c)}' for c in classes],
    (cm.diagonal() / cm.sum(axis=1)).round(3).tolist()
    ))

    # Confusion Matrix Heatmap
    output_dir = LDA_HM_VIS
    cols_title = '-'.join(cols)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm_df, annot=True, fmt='d')
    plt.title(f'LDA Confusion Matrix ({cv_}-Fold CV)')
    plt.tight_layout()
    plt.savefig(output_dir / f"{cols_title}-LDA_CM.png", dpi=dpi_, bbox_inches='tight')
    plt.close()

    # Output 
    output = pd.DataFrame([{
        'CV Accuracy': f'{cv_mean:.3f} +/- {cv_std:.3f}',
        'Chance Level': round(chance, 3),
        'Class Priors': str({int(c): round(float(p), 3) for c, p in zip(lda.classes_, lda.priors_)}),
        'Explained Variance Ratio': str(np.round(evr, 3).tolist()) if evr is not None else 'N/A',
        'Feature Loadings (LD1)': str({k: round(v,4) for k,v in loadings['LD1'].to_dict().items()}),
        **per_class_recall,
    }])

    warnings = []
    for col in cols:
        for cls in classes:
            vals = X_df_masked[col][y_masked == cls]
            if 3 <= len(vals) <= 5000:
                _, p = stats.shapiro(vals)
                if p < 0.05:
                    warnings.append(f"[WARNING] {col} non-normal in class {int(cls)} (Shapiro-Wilk p={p:.3f})")
    
    output_dir = LDA_OUTPUT_FOLDER
    with open(output_dir / f'{cols_title}-lda.txt','w') as f:
        f.write('--- Linear Discriminant Analysis ---\n\n')
        for col, val in output.iloc[0].items():
            f.write(f'{col}: {val}\n')
        if warnings:
            f.write("\n--- Assumption Checks ---\n")
            for w in warnings:
                f.write(f'{w}\n')

    return None