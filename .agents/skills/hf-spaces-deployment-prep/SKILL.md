---
name: hf-spaces-deployment-prep
description: Prepares, packages, and verifies all required files, assets, model checkpoints, and configuration for deploying a PyTorch/Gradio application to Hugging Face Spaces.
---

# Hugging Face Spaces Deployment Preparation Guidelines

When preparing a PyTorch or PyTorch Lightning model and Gradio application for deployment to **Hugging Face Spaces (Gradio SDK)**, follow these standardization and packaging steps:

## 1. Directory Structure Requirements

Ensure the deployment folder follows this standard structure:

```text
hf_space_deployment/
├── app.py                  # Main entry point running the Gradio app
├── model.py                # Standalone model architecture definition
├── requirements.txt        # Exact dependencies (pinned lightweight versions)
├── class_names.json        # Target label classes metadata
├── README.md               # Space metadata header (YAML frontmatter)
├── assets/                 # (Optional) Sample images or examples
│   ├── sample1.jpg
│   └── sample2.jpg
└── weights/
    └── model_weights.pt    # State dict or TorchScript (avoid full pickled objects)
```

---

## 2. Standard Templates for Key Files

### A. Space Metadata (README.md)

Always include the YAML frontmatter at the top of README.md to configure the Space:
```yaml
---
title: Deep Learning Model Demo
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Deep Learning Model Demo
This Space hosts a PyTorch computer vision model demo deployed via Gradio.
```

### B. Dependency Specifications (requirements.txt)

Keep dependencies clean, CPU-compatible by default (HF free tier uses CPU), and avoid oversized libraries:
```text
torch
torchvision
gradio>=4.44.0
pillow
numpy
```

### C. Weight Export Script (export_for_hf.py)

Export only the state_dict or TorchScript to keep file sizes minimal and portable:
```python
import torch
import json
from pathlib import Path

def export_model_for_hf(
    model: torch.nn.Module,
    class_names: list,
    output_dir: str = "./hf_space_deployment"
):
    """
    Exports state_dict and class metadata for Hugging Face Space deployment.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "weights").mkdir(exist_ok=True)

    # 1. Save Model Weights
    model.eval()
    weights_path = out_path / "weights" / "model_weights.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"✅ Weights saved to: {weights_path}")

    # 2. Save Class Names
    classes_path = out_path / "class_names.json"
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print(f"✅ Class names saved to: {classes_path}")
```

### D. Production Entrypoint (app.py)
```python
import json
from pathlib import Path
import torch
import gradio as gr
from PIL import Image
from torchvision import transforms

# 1. Load Configurations & Labels
DEPLOY_DIR = Path(__file__).parent
with open(DEPLOY_DIR / "class_names.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)

# 2. Load Model & Weights
from model import create_model # Import your model constructor

DEVICE = torch.device("cpu") # Free HF Spaces run on CPU
model = create_model(num_classes=len(class_names))
weights_path = DEPLOY_DIR / "weights" / "model_weights.pt"

if weights_path.exists():
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 3. Preprocessing Pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 4. Predict Function
def predict(image: Image.Image):
    if image is None:
        return {}
    img_rgb = image.convert("RGB")
    tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    return {class_names[i]: float(probs[i].item()) for i in range(len(class_names))}

# 5. Build Gradio App
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Input Image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="Model Deployment on Hugging Face Spaces",
    description="Upload an image to classify."
)

if __name__ == "__main__":
    demo.launch()

```
