import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from config import DPI, DX, K_MAX, K_MIN, N_INIT, RAND_STATE, REG_COVAR
from pipelines.data_organizers.csv_loader import load_clean
from pipelines.data_organizers.file_pathways import (
    BIC_AIC_VIS,
    CEV_PCA_VIS,
    GMM_ANALYSIS_OUTPUT_FOLDER,
    GMM_HM_VIS,
    GMM_PP_VIS,
)
from pipelines.data_organizers.impossible_var_cleaner import clean_impossible_var


def gmm_analysis(
        *cols: str,
        dx_col_: str = DX,
        dpi_: int = DPI,
        k_min: int = K_MIN,
        k_max: int = K_MAX,
        n_init:int = N_INIT,
        rand_state: int = RAND_STATE,
        reg_covar: float = REG_COVAR,
        ):
    """
    Gaussian Mixture Model (GMM):\n
    - A probalistic clustering technique that models data as a combination of multiple Gaussian distributions enabling flexible grouping\n\n
    ---\n\n
    Assumptions:\n
    - Gaussian Distributions\n
    - Independent and Identically Distributed (I.I.D) Observations\n
    - Elliptical Clusters\n
    - Limited Outliers\n\n
    ---\n\n
    Parameters\n
    ---\n
    * cols*               : list of variables
    * dx_col_             : diagnostic variable 
    * dpi                 : Dots Per Inch 
    * k_min               : Minimum k clusters
    * k_min               : Maximum k clusters
    * n_init              : Number of gmm initializations
    * rand_state          : Seed used for analysis
    * reg_covar           : Non-negative regularization added to the diagonal of covariance
    ---\n
    Output:\n
    - Summary Table
    - BIC + AIC Tables
    - Heatmap
    """

    # Initial load of csv
    clean_df = load_clean()

    # Logging
    logger = logging.getLogger(__name__)

    # Adjusting for range parameters of k_max
    k_max = k_max+1

    # Basic config checking
    for arg_name, arg in {'dpi_':dpi_, 'n_init':n_init}.items():
        if not isinstance(arg, int) or arg < 1:
            raise ValueError(f'{arg_name} must be a positive integer, got {arg}\n - Check config')
    if dx_col_ not in clean_df.columns:
        raise ValueError(f'dx_col_ ({dx_col_}) not found in DataFrame\n - Check config')
    if k_min < 1 or k_max <= k_min:
        raise ValueError(f'k_min must be >= 1 and k_max must be > k_min, got k_min={k_min}, k_max={k_max}')
    if not isinstance(rand_state, int):
        raise ValueError(f'rand_state must be an integer, got {rand_state}\n - Check config')
    if not isinstance(reg_covar, float):
        raise ValueError(f'reg_covar must be a float value, got {reg_covar}\n - Check config')

    # build df from selected cols
    df = clean_impossible_var(clean_df, *cols)

    # Standardize data (z-score)
    X_df = df[list(cols)].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    # Apply and cumulative Full PCA
    pca_full = PCA(n_components=len(cols))
    pca_full.fit(X_scaled)
    cumulative_full = pca_full.explained_variance_ratio_.cumsum()
    optimal_n = np.argmax(cumulative_full >= 0.95) + 1
    optimal_n = optimal_n if cumulative_full[optimal_n - 1] >= 0.95 else len(cols)
    logger.info(f"Optimal number of components: {optimal_n}")

    # Apply PCA based on optimal_n
    pca = PCA(n_components=optimal_n)
    X_pca = pd.DataFrame(pca.fit_transform(X_scaled))

    # Plot cumulative explained variance
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(cumulative_full) + 1),
             cumulative_full,
             marker='o', linestyle='--', color='b'
             )
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.tight_layout()

    output_dir = CEV_PCA_VIS
    cols_title = '-'.join(cols)
    plt.savefig(output_dir / f"{cols_title}-CEV_PCA.png", dpi=dpi_, bbox_inches='tight')

    plt.close()

    # Visual Representation of standardized data
    g = sns.pairplot(
        data=X_pca,
        diag_kind='kde',
        )
    g.map_upper(sns.kdeplot)
    output_dir = GMM_PP_VIS
    g.savefig(output_dir / f"{cols_title}-GMM_PP.png", dpi=dpi_, bbox_inches='tight')
    plt.close()

    # Model selection via BIC/AIC
    bic, aic = [], []
    gmms = {}
    all_keys = []

    cov_types = ['full', 'tied', 'diag', 'spherical']

    for cov in cov_types:
        for k in range(k_min,k_max):
            gmm_k = GaussianMixture(n_components=k,
                                    covariance_type=cov,
                                    n_init=n_init,
                                    random_state=rand_state,
                                    reg_covar=reg_covar,
                                    )
            gmm_k.fit(X_pca)
            bic.append(gmm_k.bic(X_pca))
            aic.append(gmm_k.aic(X_pca))
            gmms[(cov,k)] = gmm_k
            all_keys.append((cov,k))

    bic = np.array(bic)
    aic = np.array(aic)

    acceptable_keys = []

    for cov in cov_types:
        cov_indices = [i for i, (c,k) in enumerate(all_keys) if c == cov]
        cov_bics = bic[cov_indices]

        cov_min = cov_bics.min()
        cov_std = cov_bics.std()
        threshold = cov_min + cov_std

        for i in cov_indices:
            if bic[i] <= threshold:
                acceptable_keys.append(all_keys[i])

    best_idx = np.argmin(bic)
    best_cov, best_k = all_keys[best_idx]
    best_gmm = gmms[(best_cov, best_k)]

    aic_best_idx = np.argmin(aic)
    aic_best_cov, aic_best_k = all_keys[aic_best_idx]

    logger.info(f"BIC: Best covariance: {best_cov} | Best K: {best_k}\n"
          f"AIC: Best covariance: {aic_best_cov} | Best K: {aic_best_k}")

    # BIC/AIC Selection Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    n_k = k_max - k_min

    for i, cov in enumerate(cov_types):
        sl = slice(i * n_k, (i + 1) * n_k)
        axes[0].plot(range(k_min, k_max), bic[sl], marker='o', label=cov)
        axes[1].plot(range(k_min, k_max), aic[sl], marker='o', label=cov)

    for ax, title in zip(axes, ['BIC vs K', 'AIC vs K']):
        ax.set_xlabel("K")
        ax.set_ylabel("Score (lower is better)")
        ax.set_title(title)
        ax.set_xticks(range(k_min, k_max))
        ax.legend()

    plt.tight_layout()
    output_dir = BIC_AIC_VIS
    plt.savefig(output_dir / f"{cols_title}-BIC_AIC.png", dpi=dpi_, bbox_inches='tight')
    plt.close()

    # Save clusters/probabilities from best_gmm
    best_labels = pd.Series(
        best_gmm.predict(X_pca),
        index=df.index
    )
    best_probs = pd.DataFrame(
        best_gmm.predict_proba(X_pca),
        index=df.index,
        columns=[f'cluster_{i}_prob' for i in range(best_k)]
    )

    for col in best_probs.columns:
        clean_df.loc[df.index, col] = best_probs[col]

    cluster_counts = np.bincount(best_labels.to_numpy(), minlength=best_k)
    logger.info(f"cluster counts: {cluster_counts}")
    cluster_proportions = cluster_counts / cluster_counts.sum()
    logger.info(f"cluster proportions: {cluster_proportions}")

    # Pull diagnoses for same rows used in GMM
    dx = clean_df.loc[df.index, dx_col_]

    # Crosstab diagnosis x cluster
    ct = pd.crosstab(
        dx,
        best_labels,
        normalize="index"
        )
    logger.info(ct)

    # Heatmap
    plt.figure(figsize=(8,5))
    sns.heatmap(ct, annot=True)
    plt.title("DX_GROUP vs GMM cluster (row-normalized)")
    plt.ylabel("DX_GROUP")
    plt.xlabel("GMM cluster")
    output_dir = GMM_HM_VIS
    plt.savefig(output_dir / f"{cols_title}-GMM_Heatmap.png", dpi=dpi_, bbox_inches='tight')

    plt.close()

    # Quantify Alignment
    dx_arr = clean_df.loc[df.index, dx_col_].to_numpy()
    mask = ~np.isnan(dx_arr)

    nmi = normalized_mutual_info_score(dx_arr[mask], best_labels.to_numpy()[mask])
    logger.info(f"NMI: {nmi}")

    ari = adjusted_rand_score(dx_arr[mask], best_labels.to_numpy()[mask])
    logger.info(f"ARI: {ari}")

    logger.info(f"dx length: {len(dx)} | dx non-null: {dx.notna().sum()} | dx unique: {dx.nunique(dropna=True)}")
    dx_length = len(dx)
    dx_non_null = dx.notna().sum()
    dx_unique = dx.nunique(dropna=True)

    logger.info(f"dx value counts:\n{dx.value_counts(dropna=False).head(10)}")
    dx_val_counts = dx.value_counts(dropna=True)

    # Output DF
    output = pd.DataFrame([{
        'Optimal Number of Components': optimal_n,
        'Acceptable (cov, K) Combinations': str(acceptable_keys),
        'Best covariance': best_cov,
        'Best K': best_k,
        'Cluster Counts': str(cluster_counts.tolist()),
        'Cluster Proportions': str(cluster_proportions.round(3).tolist()),
        'NMI': round(nmi,3),
        'ARI': round(ari,3),
        'DX Length': dx_length,
        'DX Non-null': dx_non_null,
        'DX Unique': dx_unique,
        'DX Val Counts': str(dx_val_counts.to_dict()),
    }])

    # Output DF -> CSV
    output_dir = GMM_ANALYSIS_OUTPUT_FOLDER
    output.to_csv(output_dir / f"ml_{cols_title}-gmm_analysis.csv",index=False)
    return output
