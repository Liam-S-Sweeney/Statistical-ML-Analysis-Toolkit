import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from pathlib import Path
from config import DX, IMPOSSIBLE_ZERO_VARS
from data_loader import load_clean

def gmm_analysis(*cols,dx_col=DX, impossible_zero_vars = IMPOSSIBLE_ZERO_VARS):

    # Initial load of csv
    clean_df = load_clean()

###
    # build df from selected cols - 
    df = clean_df[list(cols)].dropna().copy()

    impossible_zero_cols = [c for c in impossible_zero_vars if c in df.columns]

    threshold = 0
    skewed_val = 0

    for col in impossible_zero_cols:
        proportion = (df[col] == 0).mean()
        if proportion > threshold:
            print(f"{col}: {proportion:.1%} {skewed_val} detected — replacing with NaN")
            df[col] = df[col].replace(0, np.nan)

    df = df.dropna()
    print(f"Final df shape after cleaning: {df.shape}")
###

    # Standardize data (z-score)
    X_df = df[list(cols)].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    # LDA
    dx_arr_full = clean_df.loc[df.index, dx_col].to_numpy()
    mask_full = ~np.isnan(dx_arr_full)

    lda = LinearDiscriminantAnalysis()
    lda_scores = cross_val_score(lda, X_scaled[mask_full], dx_arr_full[mask_full], cv=5)
    print(f"LDA CV accuracy: {lda_scores.mean():.3f} ± {lda_scores.std():.3f}")
    lda_output = f'{lda_scores.mean():.3f} ± {lda_scores.std():.3f}'

    # Apply and cumulative Full PCA
    pca_full = PCA(n_components=len(cols))
    pca_full.fit(X_scaled)
    cumalative_full = pca_full.explained_variance_ratio_.cumsum()
    optimal_n = np.argmax(cumalative_full >= 0.95) + 1
    print(f"Optimal number of components: {optimal_n}")

    # Apply PCA based on optimal_n
    pca = PCA(n_components=optimal_n)
    X_pca = pd.DataFrame(pca.fit_transform(X_scaled))

    # Plot cumulative explained variance
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(cumalative_full) + 1), 
             cumalative_full,
             marker='o', linestyle='--', color='b'
             )
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Visual Representation of standardized data
    g = sns.pairplot(
        data=X_pca, 
        diag_kind='kde',
        )
    g.map_upper(sns.kdeplot)
    plt.show()

    # Model selection via BIC/AIC
    bic, aic = [], []
    gmms = {}
    all_keys = []

    cov_types = ['full', 'tied', 'diag', 'spherical']

    for cov in cov_types:
        for k in range(1, 11):
            gmm_k = GaussianMixture(n_components=k, 
                                    covariance_type=cov, 
                                    n_init=10, 
                                    random_state=0,
                                    reg_covar=1e-3,
                                    )
            gmm_k.fit(X_pca)
            bic.append(gmm_k.bic(X_pca))
            aic.append(gmm_k.aic(X_pca))
            gmms[(cov,k)] = gmm_k
            all_keys.append((cov,k))

    bic = np.array(bic)
    aic = np.array(aic)

    bic_min = bic.min()
    bic_std = bic.std()

    acceptable_idx = np.where(bic <= bic_min + bic_std)[0]
    acceptable_keys = [all_keys[i] for i in acceptable_idx]
    print(f"Acceptable (cov, K) combinations: {acceptable_keys}")

    best_idx = np.argmin(bic)
    best_cov, best_k = all_keys[best_idx]
    best_gmm = gmms[(best_cov, best_k)]

    print(f"Best covariance: {best_cov} | Best K: {best_k}")

    # BIC/AIC Selection Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for i, cov in enumerate(cov_types):
        sl = slice(i * 10, (i + 1) * 10)
        axes[0].plot(range(1, 11), bic[sl], marker='o', label=cov)
        axes[1].plot(range(1, 11), aic[sl], marker='o', label=cov)

    for ax, title in zip(axes, ['BIC vs K', 'AIC vs K']):
        ax.set_xlabel("K")
        ax.set_ylabel("Score (lower is better)")
        ax.set_title(title)
        ax.set_xticks(range(1, 11))
        ax.legend()

    plt.tight_layout()
    plt.show()

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

    # Predict a new subject using the BEST model - DEMONSTRATION ONLY
    new_subject = df.sample(1, random_state=0)
    new_scaled = scaler.transform(new_subject[list(cols)])
    new_pca = pca.transform(new_scaled)

    print("new_cluster:", best_gmm.predict(new_pca))
    new_pt_cluster = best_gmm.predict(new_pca)
    print("new_prob:", best_gmm.predict_proba(new_pca))
    new_pt_prob = best_gmm.predict_proba(new_pca)

    cluster_counts = np.bincount(best_labels, minlength=best_k)
    print(f"cluster counts: {cluster_counts}")
    cluster_proportions = cluster_counts / cluster_counts.sum()
    print(f"cluster proportions: {cluster_proportions}")

    # Pull diagnoses for same rows used in GMM
    dx = clean_df.loc[df.index, dx_col]

    # Crosstab diagnosis x cluster
    ct = pd.crosstab(
        dx, 
        best_labels,
        normalize="index"
        )
    print(ct)

    # Heatmap
    plt.figure(figsize=(8,5))
    sns.heatmap(ct, annot=True)
    plt.title("DX_GROUP vs GMM cluster (row-normalized)")
    plt.ylabel("DX_GROUP")
    plt.xlabel("GMM cluster")
    plt.show()

    # Quantify Alignment
    dx_arr = clean_df.loc[df.index, dx_col].to_numpy()
    mask = ~np.isnan(dx_arr)
    print("NMI:", normalized_mutual_info_score(dx_arr[mask], best_labels[mask])) # Normalized Mutual Information
    nmi = normalized_mutual_info_score(dx_arr[mask], best_labels[mask])
    print("ARI:", adjusted_rand_score(dx_arr[mask], best_labels[mask])) # Adjusted Rand Index
    ari = adjusted_rand_score(dx_arr[mask], best_labels[mask])
    print("dx length:", len(dx), " | dx non-null:", dx.notna().sum(), " | dx unique:", dx.nunique(dropna=True))
    dx_length = len(dx)
    dx_non_null = dx.notna().sum()
    dx_unique = dx.nunique(dropna=True)
    print("dx value counts:\n", dx.value_counts(dropna=False).head(10))
    dx_val_counts = dx.value_counts(dropna=True)

    # Output DF
    output_db = pd.DataFrame([{
        'LDA CV accuracy': lda_output,
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
        'TEST | New Pt Cluster': new_pt_cluster,
        'TEST | New Pt Probability': new_pt_prob,
    }])

    # Output DF -> CSV
    output_dir = Path('multivariate_analysis')
    cols_title = '-'.join(cols)
    output_db.to_csv(output_dir / f"ml_{cols_title}-gmm_analysis.csv",index=False)
    
