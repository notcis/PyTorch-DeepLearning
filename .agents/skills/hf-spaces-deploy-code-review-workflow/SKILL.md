---
name: hf-spaces-deploy-code-review
description: Performs an automated pre-deployment code review and audit for Hugging Face Spaces (Gradio/Streamlit), checking for CPU compatibility, hardcoded secrets/paths, file size/Git LFS, app.py entrypoints, and requirements.txt soundness.
---

# Hugging Face Spaces Deployment Code Review Guidelines

When asked to review, audit, or inspect a project/repository before deploying to **Hugging Face Spaces (Gradio / Streamlit)**, conduct a comprehensive multi-point inspection following this review protocol:

---

## 1. Core Review Checklist

### A. Environment & Hardware Compatibility (Free Tier CPU Fallback)
- [ ] **No Hardcoded CUDA**: Ensure `.cuda()` or `.to('cuda')` is not hardcoded. Look for device fallback:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```

- [ ] Weight Loading map_location: Ensure torch.load() includes map_location=device or map_location='cpu'.

- [ ] Inference Optimization: Check that model inference is wrapped in torch.inference_mode() or torch.no_grad() and set to model.eval().

### B. Spaces Configuration & Structure
- [ ] README.md Metadata: Verify YAML frontmatter exists at the very top of README.md:

  - title, emoji, sdk (gradio/streamlit), sdk_version, app_file (usually app.py), and pinned.

- [ ] Entrypoint File: Verify app.py exists in the root directory (or matches app_file in metadata).

- [ ] Launch Method: Check that demo.launch() or app.launch() is called without hardcoded local absolute paths or blocking host binds.

- ### C. Dependencies & Packaging (requirements.txt)
- [ ] No Local/GPU-only Wheels: Strip torch wheels pointing to local paths or specific CUDA extra-index URLs unless configured via pre-install scripts.

- [ ] Essential vs Bloated Dependencies: Check that heavy unnecessary packages (e.g., jupyter, tensorboard, torchvision if only text/tabular is used) are excluded.

[ ] Required Packages Present: Ensure gradio (or streamlit), torch, pillow, numpy, etc. are explicitly pinned or declared.

- ### D. File Size, Weights & Git LFS Compliance
- [ ] Model Weight Strategy:

  - Files > 10MB to 500MB require Git LFS (git lfs track "*.pt" "*.onnx" "*.safetensors").

  - Avoid committing entire training checkpoints (optimizer_states, scheduler_states). Recommend exporting clean state_dict or TorchScript.

- [ ] Data & Cache Bloat: Ensure __pycache__/, .git/, .env, raw datasets, and logs (lightning_logs/) are listed in .gitignore.

- ### E. Security & Privacy Audit
- [ ] No Hardcoded Tokens/Keys: Check for leaked Hugging Face tokens (hf_...), API keys, database URLs, or private personal paths (e.g., C:/Users/... or /home/...).

- [ ] Use os.environ.get("HF_TOKEN") or Spaces Secrets for private access.

## 2. Review Output Format
Provide structured review feedback in the following format:

```markdown
## 🔍 Hugging Face Spaces Pre-Deployment Audit Report

### 🚦 Deployment Readiness: [🟢 READY / 🟡 MINOR FIXES / 🔴 BLOCKED]

### 1. ⚠️ Critical Blockers (Must Fix)
- [List any issues that will cause the Space to crash or fail build]

### 2. ⚡ Performance & CPU Compatibility
- [Device management, inference mode, batch size for free CPU tier]

### 3. 📦 Dependency & Configuration Review (`README.md` / `requirements.txt`)
- [YAML metadata check, missing or conflicting libraries]

### 4. 🗂️ File Size & Git LFS Verification
- [Model weight format (.pt/.safetensors), Git LFS tracking, large assets]

### 5. 💡 Refactoring & Code Suggestions
```python
# Provide clean code snippets for required fixes
```

```