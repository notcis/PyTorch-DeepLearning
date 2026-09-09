---
name: tabular-ann-dataprep
description: Prompts the user for a CSV dataset path, performs data preprocessing (missing value handling, feature normalization, categorical encoding, train/val/test splitting), wraps data into TensorDatasets, and returns train_dataloader, val_dataloader, and test_dataloader for ANN model training using PyTorch.
---

# PyTorch Tabular ANN Data Preparation Guidelines

When instructed to prepare tabular data from a CSV path for ANN training, follow these structured standards:

1. **Path Acquisition & Loading**:
   - Prompt the user for the CSV file path and target column name.
   - Load the dataset safely using `pandas`.
2. **Preprocessing Pipeline (Best Practices)**:
   - Separate features ($X$) and target ($y$).
   - Identify numerical and categorical columns automatically.
   - Apply `StandardScaler` (or `PowerTransformer` for skewed features) **strictly fitted on the training split only** to prevent Data Leakage[cite: 1, 14].
   - Encode categorical features (One-Hot Encoding or Label Encoding).
3. **Dataset Splitting**:
   - Split data into Train, Validation, and Test sets (e.g., 70%/15%/15% or 80%/10%/10%).
   - Support stratification for classification tasks if necessary.
4. **PyTorch Tensor & DataLoader Construction**:
   - Convert processed numpy arrays into PyTorch `TensorDataset`.
   - Wrap datasets into PyTorch `DataLoader` objects (`shuffle=True` for train; `shuffle=False` for val/test)[cite: 3].

---

## Reference Implementation

```python
import os
from pathlib import Path
from typing import Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset


def prepare_tabular_ann_dataloaders(
    csv_path: Union[str, Path],
    target_col: str,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    is_classification: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Prompts/Accepts a CSV path, preprocesses features and target, 
    splits the dataset, converts to TensorDataset, and returns DataLoaders.
    """
    path = Path(csv_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"❌ File not found at path: {path}")

    # 1. Load Dataset
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"❌ Target column '{target_col}' not found in CSV columns: {list(df.columns)}")

    print(f"\n📂 Loading dataset from: {path.name} | Shape: {df.shape}")

    # 2. Separate Features (X) and Target (y)
    X_raw = df.drop(columns=[target_col])
    y_raw = df[target_col].values

    # Handle target formatting for PyTorch
    if is_classification:
        # If classification, ensure labels are integers starting from 0
        if y_raw.dtype == object or not np.issubdtype(y_raw.dtype, np.integer):
            y_raw = pd.factorize(y_raw)[0]
        y_tensor_dtype = torch.long
    else:
        y_tensor_dtype = torch.float32

    # 3. Handle Numerical & Categorical Columns
    num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f" - Numerical Features ({len(num_cols)}): {num_cols}")
    print(f" - Categorical Features ({len(cat_cols)}): {cat_cols}")

    # Process features using manual imputation/scaling and one-hot encoding
    processed_dfs = []
    
    if num_cols:
        # Fill missing numeric values with median
        X_num = X_raw[num_cols].fillna(X_raw[num_cols].median()).values
        processed_dfs.append(X_num)

    if cat_cols:
        # Fill missing categorical values with mode
        X_cat_raw = X_raw[cat_cols].fillna(X_raw[cat_cols].mode().iloc[0])
        encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        X_cat = encoder.fit_transform(X_cat_raw)
        processed_dfs.append(X_cat)

    X_processed = np.hstack(processed_dfs) if len(processed_dfs) > 1 else processed_dfs[0]

    # 4. Train / Val / Test Split (Strictly before scaling to prevent data leakage)
    val_test_ratio = val_split + test_split
    stratify_param = y_raw if is_classification else None

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_processed, y_raw,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=stratify_param
    )

    test_ratio_relative = test_split / val_test_ratio
    stratify_temp = y_temp if is_classification else None

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_ratio_relative,
        random_state=seed,
        stratify=stratify_temp
    )

    # 5. Feature Normalization (StandardScaler fitted strictly on Train split)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 6. Convert to PyTorch Tensors & TensorDatasets
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=y_tensor_dtype))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val, dtype=y_tensor_dtype))
    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=y_tensor_dtype))

    # 7. Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    input_dim = X_train_scaled.shape[1]
    print(f"\n✅ DataLoaders Created Successfully!")
    print(f" - Train Batches: {len(train_dataloader)} | Samples: {len(train_dataset):,}")
    print(f" - Val Batches:   {len(val_dataloader)} | Samples: {len(val_dataset):,}")
    print(f" - Test Batches:  {len(test_dataloader)} | Samples: {len(test_dataset):,}")
    print(f" - Input Dimension for ANN First Layer: {input_dim}\n")

    return train_dataloader, val_dataloader, test_dataloader, input_dim


if __name__ == "__main__":
    # Example Prompt interaction loop
    user_csv_path = input("📁 Enter CSV file path: ").strip().strip('"').strip("'")
    user_target_col = input("🎯 Enter target column name: ").strip()
    
    train_loader, val_loader, test_loader, in_dim = prepare_tabular_ann_dataloaders(
        csv_path=user_csv_path,
        target_col=user_target_col,
        batch_size=64,
        is_classification=True
    )