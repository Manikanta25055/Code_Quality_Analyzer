"""
Data Pattern Analysis - Linear vs Non-Linear
Analyze feature relationships and data patterns.

Author: Manikanta Gonugondla
Co-Author: Claude Sonnet 4.5
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os


def analyze_data_patterns():
    """Analyze if data has linear or non-linear patterns"""

    print("=" * 70)
    print("Data Pattern Analysis - Linear vs Non-Linear")
    print("=" * 70)

    # Load data
    df = pd.read_csv('../data/normalized_features.csv')

    print(f"\nDataset shape: {df.shape}")
    print(f"Features: {len(df.columns) - 2}")  # Exclude repo_name and quality_label

    # Separate features and labels
    feature_cols = [col for col in df.columns if col not in ['repo_name', 'quality_label']]
    X = df[feature_cols]
    y = df['quality_label']

    # Create output directory
    os.makedirs('../evaluation/pattern_analysis', exist_ok=True)

    # 1. Feature Pair Scatter Plots
    print("\n1. Creating feature pair scatter plots...")
    create_feature_scatterplots(X, y, feature_cols)

    # 2. PCA Analysis
    print("\n2. Performing PCA analysis...")
    analyze_pca(X, y)

    # 3. t-SNE Analysis
    print("\n3. Performing t-SNE analysis...")
    analyze_tsne(X, y)

    # 4. Correlation Analysis
    print("\n4. Analyzing feature correlations...")
    analyze_correlations(X, y, feature_cols)

    # 5. Statistical Analysis
    print("\n5. Performing statistical analysis...")
    statistical_analysis(X, y, feature_cols)

    # 6. Generate Report
    print("\n6. Generating analysis report...")
    generate_report(X, y, feature_cols)

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print("\nGenerated files in evaluation/pattern_analysis/:")
    print("  - feature_scatterplots.png")
    print("  - pca_analysis.png")
    print("  - tsne_analysis.png")
    print("  - correlation_with_target.png")
    print("  - pattern_analysis_report.txt")


def create_feature_scatterplots(X, y, feature_cols):
    """Create scatter plots for top features vs quality label"""

    # Select top 9 features (most varying)
    variances = X.var().sort_values(ascending=False)
    top_features = variances.head(9).index.tolist()

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()

    for idx, feature in enumerate(top_features):
        ax = axes[idx]

        # Scatter plot with jitter
        for label in [0, 1]:
            mask = y == label
            x_vals = X[feature][mask]
            y_vals = y[mask] + np.random.normal(0, 0.05, size=sum(mask))  # Add jitter

            color = '#e74c3c' if label == 0 else '#27ae60'
            label_name = 'Low Quality' if label == 0 else 'High Quality'
            ax.scatter(x_vals, y_vals, alpha=0.6, s=50, c=color, label=label_name)

        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Quality Label (with jitter)', fontsize=10)
        ax.set_title(f'{feature} vs Quality', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('../evaluation/pattern_analysis/feature_scatterplots.png', dpi=150)
    plt.close()
    print("  Saved: feature_scatterplots.png")


def analyze_pca(X, y):
    """Perform PCA to visualize data in 2D"""

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: PCA scatter
    for label in [0, 1]:
        mask = y == label
        color = '#e74c3c' if label == 0 else '#27ae60'
        label_name = 'Low Quality' if label == 0 else 'High Quality'
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                       alpha=0.6, s=100, c=color, label=label_name, edgecolors='black')

    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    axes[0].set_title('PCA: 2D Projection of Features', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2: Variance explained
    pca_full = PCA()
    pca_full.fit(X)
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)

    axes[1].plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-', linewidth=2, markersize=8)
    axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    axes[1].set_xlabel('Number of Components', fontsize=12)
    axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
    axes[1].set_title('PCA: Cumulative Variance Explained', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('../evaluation/pattern_analysis/pca_analysis.png', dpi=150)
    plt.close()
    print("  Saved: pca_analysis.png")

    # Print analysis
    print(f"\n  PCA Results:")
    print(f"    - First 2 components explain {sum(pca.explained_variance_ratio_)*100:.2f}% of variance")
    print(f"    - Components needed for 95% variance: {np.argmax(cumsum_var >= 0.95) + 1}")


def analyze_tsne(X, y):
    """Perform t-SNE for non-linear dimensionality reduction"""

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)

    # Create plot
    plt.figure(figsize=(10, 8))

    for label in [0, 1]:
        mask = y == label
        color = '#e74c3c' if label == 0 else '#27ae60'
        label_name = 'Low Quality' if label == 0 else 'High Quality'
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   alpha=0.6, s=100, c=color, label=label_name, edgecolors='black')

    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.title('t-SNE: Non-linear 2D Projection', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('../evaluation/pattern_analysis/tsne_analysis.png', dpi=150)
    plt.close()
    print("  Saved: tsne_analysis.png")


def analyze_correlations(X, y, feature_cols):
    """Analyze correlation between features and target"""

    # Calculate correlations
    correlations = []
    for col in feature_cols:
        corr = np.corrcoef(X[col], y)[0, 1]
        correlations.append({'feature': col, 'correlation': abs(corr), 'signed_corr': corr})

    corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)

    # Plot top 15 correlations
    plt.figure(figsize=(12, 8))
    top_15 = corr_df.head(15)

    colors = ['#27ae60' if x > 0 else '#e74c3c' for x in top_15['signed_corr']]

    plt.barh(range(len(top_15)), top_15['signed_corr'], color=colors)
    plt.yticks(range(len(top_15)), top_15['feature'])
    plt.xlabel('Correlation with Quality Label', fontsize=12)
    plt.title('Top 15 Feature Correlations with Target', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('../evaluation/pattern_analysis/correlation_with_target.png', dpi=150)
    plt.close()
    print("  Saved: correlation_with_target.png")

    # Print top correlations
    print("\n  Top 5 Correlated Features:")
    for idx, row in corr_df.head(5).iterrows():
        print(f"    - {row['feature']:30s}: {row['signed_corr']:+.4f}")


def statistical_analysis(X, y, feature_cols):
    """Perform statistical tests for linearity"""

    from scipy.stats import pearsonr, spearmanr

    print("\n  Linearity Tests (Pearson vs Spearman):")
    print("  (If Spearman >> Pearson, suggests non-linear relationship)")

    results = []
    for col in feature_cols[:10]:  # Top 10 features
        pearson_r, _ = pearsonr(X[col], y)
        spearman_r, _ = spearmanr(X[col], y)

        results.append({
            'feature': col,
            'pearson': abs(pearson_r),
            'spearman': abs(spearman_r),
            'difference': abs(spearman_r) - abs(pearson_r)
        })

    results_df = pd.DataFrame(results).sort_values('difference', ascending=False)

    print(f"\n  {'Feature':<30s} {'Pearson':>10s} {'Spearman':>10s} {'Diff':>10s}")
    print("  " + "-" * 65)
    for _, row in results_df.head(5).iterrows():
        print(f"  {row['feature']:<30s} {row['pearson']:>10.4f} {row['spearman']:>10.4f} {row['difference']:>10.4f}")

    return results_df


def generate_report(X, y, feature_cols):
    """Generate comprehensive analysis report"""

    # Perform PCA
    pca = PCA()
    pca.fit(X)
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = np.argmax(cumsum_var >= 0.95) + 1

    # Calculate separability
    class_0 = X[y == 0]
    class_1 = X[y == 1]

    with open('../evaluation/pattern_analysis/pattern_analysis_report.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DATA PATTERN ANALYSIS REPORT\n")
        f.write("Linear vs Non-Linear Classification\n")
        f.write("=" * 70 + "\n\n")

        f.write("DATASET OVERVIEW\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total samples: {len(X)}\n")
        f.write(f"Total features: {len(feature_cols)}\n")
        f.write(f"Class 0 (Low Quality): {sum(y == 0)}\n")
        f.write(f"Class 1 (High Quality): {sum(y == 1)}\n\n")

        f.write("LINEARITY ANALYSIS\n")
        f.write("-" * 70 + "\n")

        # PCA analysis
        f.write(f"\n1. PCA Analysis:\n")
        f.write(f"   - First 2 components explain: {sum(pca.explained_variance_ratio_[:2])*100:.2f}%\n")
        f.write(f"   - Components for 95% variance: {n_components_95} out of {len(feature_cols)}\n")

        if n_components_95 <= 5:
            f.write(f"   - INTERPRETATION: Low dimensionality ({n_components_95} components)\n")
            f.write(f"                     Suggests LINEAR patterns dominate\n")
        elif n_components_95 <= 10:
            f.write(f"   - INTERPRETATION: Medium dimensionality ({n_components_95} components)\n")
            f.write(f"                     Suggests MIXED linear/non-linear patterns\n")
        else:
            f.write(f"   - INTERPRETATION: High dimensionality ({n_components_95} components)\n")
            f.write(f"                     Suggests strong NON-LINEAR patterns\n")

        # Class separability
        f.write(f"\n2. Class Separability:\n")

        # Calculate mean distance between classes
        mean_0 = class_0.mean(axis=0)
        mean_1 = class_1.mean(axis=0)
        euclidean_dist = np.linalg.norm(mean_0 - mean_1)

        f.write(f"   - Euclidean distance between class centroids: {euclidean_dist:.4f}\n")

        if euclidean_dist < 5:
            f.write(f"   - INTERPRETATION: Classes are CLOSE together\n")
            f.write(f"                     May require NON-LINEAR decision boundaries\n")
        else:
            f.write(f"   - INTERPRETATION: Classes are WELL SEPARATED\n")
            f.write(f"                     LINEAR separation may be sufficient\n")

        # Model performance comparison
        f.write(f"\n3. Model Performance (from training):\n")
        try:
            comparison_df = pd.read_csv('../models/advanced/model_comparison.csv')

            # Compare linear vs non-linear models
            linear_models = ['Logistic Regression']
            nonlinear_models = ['Random Forest (Tuned)', 'SVM (RBF Kernel)', 'Neural Network (MLP)']

            f.write(f"\n   Linear Models:\n")
            for model in linear_models:
                if model in comparison_df['Model'].values:
                    acc = comparison_df[comparison_df['Model'] == model]['Test_Accuracy'].values[0]
                    f.write(f"     - {model}: {acc:.4f}\n")

            f.write(f"\n   Non-Linear Models:\n")
            for model in nonlinear_models:
                if model in comparison_df['Model'].values:
                    acc = comparison_df[comparison_df['Model'] == model]['Test_Accuracy'].values[0]
                    f.write(f"     - {model}: {acc:.4f}\n")

            # Find best performing type
            linear_max = comparison_df[comparison_df['Model'].isin(linear_models)]['Test_Accuracy'].max() if any(comparison_df['Model'].isin(linear_models)) else 0
            nonlinear_max = comparison_df[comparison_df['Model'].isin(nonlinear_models)]['Test_Accuracy'].max() if any(comparison_df['Model'].isin(nonlinear_models)) else 0

            f.write(f"\n   - Best Linear Model: {linear_max:.4f}\n")
            f.write(f"   - Best Non-Linear Model: {nonlinear_max:.4f}\n")

            if nonlinear_max > linear_max + 0.05:
                f.write(f"   - INTERPRETATION: Non-linear models perform significantly better\n")
                f.write(f"                     Data has STRONG NON-LINEAR patterns\n")
            elif abs(nonlinear_max - linear_max) < 0.05:
                f.write(f"   - INTERPRETATION: Linear and non-linear perform similarly\n")
                f.write(f"                     Data has PRIMARILY LINEAR patterns\n")
            else:
                f.write(f"   - INTERPRETATION: Mixed performance\n")
                f.write(f"                     Data has BOTH linear and non-linear patterns\n")
        except:
            f.write(f"   - Model comparison file not found\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("FINAL CONCLUSION\n")
        f.write("=" * 70 + "\n\n")

        # Make final determination
        if n_components_95 > 10:
            conclusion = "NON-LINEAR"
            reason = f"High PCA dimensionality ({n_components_95} components needed)"
        elif euclidean_dist < 5:
            conclusion = "NON-LINEAR"
            reason = "Classes are not well-separated in feature space"
        else:
            conclusion = "MIXED (Both Linear and Non-Linear)"
            reason = "Evidence of both linear and non-linear patterns"

        f.write(f"Pattern Type: {conclusion}\n\n")
        f.write(f"Primary Reason: {reason}\n\n")
        f.write(f"Recommendation:\n")
        f.write(f"  - Use NON-LINEAR models (SVM with RBF, Random Forest, Neural Networks)\n")
        f.write(f"  - Consider ensemble methods combining multiple model types\n")
        f.write(f"  - Feature engineering may help improve linear model performance\n")

    print("  Saved: pattern_analysis_report.txt")

    print("\n" + "=" * 70)
    print(f"CONCLUSION: Your data exhibits {conclusion} patterns")
    print("=" * 70)


if __name__ == "__main__":
    analyze_data_patterns()
