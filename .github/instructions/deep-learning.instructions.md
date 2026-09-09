# Deep Learning Architecture & Coding Standards

1. **PyTorch & Lightning Standards**:
   - Always use `pytorch_lightning.LightningModule` for model definitions and `LightningDataModule` for data pipelines.
   - Decouple data preparation, model architecture, training logic, and evaluation metrics clearly.
   - Use `torchmetrics` instead of writing custom metric calculations manually inside the training loop.
   - Always log metrics using `self.log()` or `self.log_dict()` with proper `on_step` and `on_epoch` parameters.

2. **Code Robustness & Hardware Optimization**:
   - Use type hinting for tensor shapes and data structures (e.g., `x: torch.Tensor` with shape comments `# [B, C, H, W]`).
   - Set device-agnostic execution (`torch.cuda.is_available()`, MPS, or Lightning `accelerator="auto"`).
   - Use `DataLoader` with configurable `num_workers`, `pin_memory=True`, and explicit `persistent_workers` setting.
   - Support Automatic Mixed Precision (AMP) using `precision=16-mixed` or `bf16-mixed` in the Trainer.

3. **Reproducibility & Best Practices**:
   - Always set seeds using `lightning.pytorch.seed_everything(seed)` at the start of scripts.
   - Include checkpointing (`ModelCheckpoint`) and early stopping (`EarlyStopping`) callbacks in the Trainer config.
   - Do not use hardcoded paths; use `pathlib.Path` or environment variables.