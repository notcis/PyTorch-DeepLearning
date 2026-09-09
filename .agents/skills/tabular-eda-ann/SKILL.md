---
name: tabular-eda-ann
description: Performs comprehensive Exploratory Data Analysis (EDA) on tabular CSV datasets using pandas, seaborn, and matplotlib. Prompts the user for a CSV path and analyzes missing values, data types, target distribution/skewness, multicollinearity, and feature scaling needs for ANN data preparation.
---

# Tabular Dataset EDA for ANN Guidelines

When instructed to perform EDA on a tabular CSV dataset to prepare it for Neural Network (ANN) training and evaluation, follow this structured workflow:

1. **Path Acquisition & Loading**:
   - Accept the CSV path from user prompt/input. Validate file existence.
   - Load dataset using `pd.read_csv()`.
2. **Data Sanity & Structural Health Check**:
   - Inspect `.info()`, `.shape`, data types (`object`, `category`, `numeric`).
   - Check missing values count and percentages per feature.
   - Detect duplicate rows.
3. **Target Analysis (Classification or Regression)**:
   - If Classification: Audit class frequencies and percentage balance (determines if `stratify=y` or class weighting is needed).
   - If Regression: Check target skewness and normality (determines if log/power transform is needed).
4. **Feature Distribution & Outlier Screening**:
   - Plot histograms/KDE and boxplots for continuous variables.
   - Check value ranges/magnitudes to guide feature scaling (`StandardScaler` / `MinMaxScaler`).
5. **Relationship & Multicollinearity Audit**:
   - Compute and plot a Pearson/Spearman correlation matrix heatmap.
   - Highlight strong multicollinearity (|r| > 0.85).
6. **Data Preparation Recommendations for ANN**:
   - Suggest categorical encodings (One-Hot / Target encoding).
   - Suggest scaling strategies (StandardScaler fitted strictly on Train split).
   - Suggest handling strategy for missing values / outliers.

---

## Reference Implementation

```python
import os
from pathlib import Path
from typing import List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def run_tabular_eda(
    csv_path: Union[str, Path],
    target_col: Optional[str] = None,
    max_numeric_plots: int = 12
) -> pd.DataFrame:
    """
    Executes full EDA pipeline on a CSV dataset for ANN data preparation.
    """
    path = Path(csv_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"❌ File not found at: {path}")

    df = pd.read_csv(path)
    sns.set_theme(style="whitegrid")

    print(f"\n=======================================================")
    print(f"📊 DATASET OVERVIEW: {path.name}")
    print(f"=======================================================")
    print(f"- Rows: {df.shape[0]:,} | Columns: {df.shape[1]:,}")
    print(f"- Duplicates: {df.duplicated().sum():,} rows")

    # 1. Missing Values & Types Summary
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    summary_table = pd.DataFrame({
        "Dtype": df.dtypes,
        "Non-Null Count": df.notnull().sum(),
        "Missing Values": missing_counts,
        "Missing (%)": missing_pct.round(2),
        "Unique Values": df.nunique()
    })
    print("\n📋 Column Summary & Data Quality:")
    print(summary_table.to_string())

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"\n- Numerical Features ({len(num_cols)}): {num_cols}")
    print(f"- Categorical/Text Features ({len(cat_cols)}): {cat_cols}")

    # 2. Target Variable Analysis
    if target_col and target_col in df.columns:
        print(f"\n🎯 TARGET AUDIT: '{target_col}'")
        plt.figure(figsize=(6, 4))
        is_discrete = (not pd.api.types.is_numeric_dtype(df[target_col])) or (df[target_col].nunique() <= 10)
        
        if is_discrete:
            counts = df[target_col].value_counts(dropna=False)
            sns.barplot(x=counts.index.astype(str), y=counts.values, palette="mako")
            plt.title(f"Target Class Distribution (Classification)", fontweight="bold")
            plt.ylabel("Count")
            for idx, val in enumerate(counts.values):
                plt.text(idx, val, f"{val} ({val/len(df)*100:.1f}%)", ha="center", va="bottom", fontsize=9)
            print(f"Target Value Counts:\n{counts}")
        else:
            sns.histplot(df[target_col].dropna(), kde=True, color="teal")
            plt.title(f"Target Distribution (Regression) | Skew: {df[target_col].skew():.2f}", fontweight="bold")
            plt.xlabel(target_col)
            print(f"Target Skewness: {df[target_col].skew():.3f} | Kurtosis: {df[target_col].kurt():.3f}")

        plt.tight_layout()
        plt.show()

    # 3. Numeric Features Distribution (Scale & Range Check for ANN)
    features_to_plot = [c for c in num_cols if c != target_col][:max_numeric_plots]
    if features_to_plot:
        n_plots = len(features_to_plot)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        axes = axes.flatten() if n_plots > 1 else [axes]

        for i, col in enumerate(features_to_plot):
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="royalblue")
            min_val, max_val = df[col].min(), df[col].max()
            axes[i].set_title(f"{col}\n[{min_val:.1f} to {max_val:.1f}]", fontsize=9)
            axes[i].set_xlabel("")

        for j in range(n_plots, len(axes)):
            axes[j].axis("off")

        plt.suptitle("Numeric Features Distribution (Audit for Scaling / Outliers)", fontsize=13, y=1.02, fontweight="bold")
        plt.tight_layout()
        plt.show()

    # 4. Correlation Heatmap (Multicollinearity Check)
    if len(num_cols) > 1:
        corr_matrix = df[num_cols].corr()
        plt.figure(figsize=(min(12, len(num_cols) + 3), min(9, int(len(num_cols) * 0.8) + 2)))
        sns.heatmap(corr_matrix, annot=len(num_cols) <= 15, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
        plt.title("Numeric Features Correlation Matrix", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.show()

    # 5. Summary Guidance for ANN Data Prep
    print("\n🧠 ANN DATA PREPARATION CHECKLIST:")
    print("-------------------------------------------------------")
    if cat_cols:
        print(f"• Categorical Encoding: Need One-Hot Encoding / Label Encoding for: {cat_cols}")
    print("• Feature Scaling: Features vary across ranges; apply StandardScaler or MinMaxScaler fitted strictly on Train split.")
    if summary_table["Missing Values"].sum() > 0:
        print("• Imputation: Missing values detected; apply SimpleImputer or KNNImputer before model feeding.")
    if target_col and target_col in df.columns and is_discrete:
        print("• Train/Test Splitting: Use stratified splitting (stratify=y) to preserve class balance.")
    print("-------------------------------------------------------\n")

    return df


if __name__ == "__main__":
    # Prompt CSV path from user
    user_csv = input("📁 Enter CSV dataset path: ").strip().strip('"').strip("'")
    user_target = input("🎯 Enter target column name (leave empty if none): ").strip()
    
    target = user_target if user_target else None
    dataset = run_tabular_eda(csv_path=user_csv, target_col=target)